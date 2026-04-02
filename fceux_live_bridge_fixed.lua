local job_path = os.getenv("FCEUX_LIVE_JOB_FIXED")
if not job_path or job_path == "" then
  error("FCEUX_LIVE_JOB_FIXED is not set")
end

local job = dofile(job_path)

local log_path = job.control_dir .. "\\bridge_log.txt"
local function log_line(message)
  local fh = io.open(log_path, "a")
  if fh then
    fh:write(message .. "\n")
    fh:close()
  end
end

local function path_exists(path)
  local fh = io.open(path, "rb")
  if fh then
    fh:close()
    return true
  end
  return false
end

local function write_text(path, text)
  local fh = assert(io.open(path, "w"))
  fh:write(text)
  fh:close()
end

local function read_lines(path)
  local rows = {}
  local fh = assert(io.open(path, "r"))
  for line in fh:lines() do
    rows[#rows + 1] = line
  end
  fh:close()
  return rows
end

local function set_readonly(state)
  if FCEU ~= nil and FCEU.setreadonly ~= nil then
    FCEU.setreadonly(state)
  else
    emu.setreadonly(state)
  end
end

local function mask_to_joypad(mask)
  return {
    right = AND(mask, 0x80) ~= 0,
    left = AND(mask, 0x40) ~= 0,
    down = AND(mask, 0x20) ~= 0,
    up = AND(mask, 0x10) ~= 0,
    start = AND(mask, 0x08) ~= 0,
    select = AND(mask, 0x04) ~= 0,
    B = AND(mask, 0x02) ~= 0,
    A = AND(mask, 0x01) ~= 0,
  }
end

local function mask_to_token(mask)
  local parts = {}
  if AND(mask, 0x80) ~= 0 then table.insert(parts, "R") end
  if AND(mask, 0x40) ~= 0 then table.insert(parts, "L") end
  if AND(mask, 0x20) ~= 0 then table.insert(parts, "D") end
  if AND(mask, 0x10) ~= 0 then table.insert(parts, "U") end
  if AND(mask, 0x08) ~= 0 then table.insert(parts, "T") end
  if AND(mask, 0x04) ~= 0 then table.insert(parts, "S") end
  if AND(mask, 0x02) ~= 0 then table.insert(parts, "B") end
  if AND(mask, 0x01) ~= 0 then table.insert(parts, "A") end
  if #parts == 0 then
    return "NONE"
  end
  return table.concat(parts, "")
end

local function smb_x_pos()
  return memory.readbyte(0x006D) * 0x100 + memory.readbyte(0x0086)
end

local function smb_state_lines(total_steps, bootstrap_frame)
  return {
    "framecount=" .. tostring(movie.framecount()),
    "total_steps=" .. tostring(total_steps),
    "bootstrap_frame=" .. tostring(bootstrap_frame),
    "world=" .. tostring(memory.readbyte(0x075F) + 1),
    "stage=" .. tostring(memory.readbyte(0x075C) + 1),
    "life=" .. tostring(memory.readbyte(0x075A)),
    "status=" .. tostring(memory.readbyte(0x0756)),
    "x_pos=" .. tostring(smb_x_pos()),
    "y_pos=" .. tostring(memory.readbyte(0x03B8)),
  }
end

local function capture_state(control_dir, total_steps, bootstrap_frame)
  local frame_blob = gui.gdscreenshot()
  local frame_path = control_dir .. "\\current_frame.gd"
  local frame_fh = assert(io.open(frame_path, "wb"))
  frame_fh:write(frame_blob)
  frame_fh:close()

  local meta_path = control_dir .. "\\current_meta.txt"
  write_text(meta_path, table.concat(smb_state_lines(total_steps, bootstrap_frame), "\n") .. "\n")
  write_text(control_dir .. "\\state_ready.flag", "ready\n")
end

local function wait_for_actions(control_dir)
  local action_path = control_dir .. "\\actions.txt"
  local quit_path = control_dir .. "\\quit.flag"
  while true do
    if path_exists(quit_path) then
      log_line("quit flag detected")
      return nil, true
    end
    if path_exists(action_path) then
      local rows = read_lines(action_path)
      os.remove(action_path)
      os.remove(control_dir .. "\\state_ready.flag")
      log_line("actions consumed: " .. tostring(#rows))
      return rows, false
    end
  end
end

local function exact_movie_bootstrap()
  if not emu.loadrom(job.rom_path) then
    error("Failed to load ROM: " .. job.rom_path)
  end
  log_line("rom loaded")

  if not movie.play(job.trace_path, true) then
    error("Failed to load movie: " .. job.trace_path)
  end
  movie.playbeginning()
  log_line("movie playback started")

  local target_frame = tonumber(job.bootstrap_raw_frame) or 0
  while movie.framecount() < target_frame do
    emu.frameadvance()
    local mode = movie.mode()
    if mode ~= "playback" and mode ~= "finished" then
      error("Movie playback ended before reaching bootstrap_raw_frame=" .. tostring(target_frame))
    end
    if mode == "finished" and movie.framecount() < target_frame then
      error("Movie finished at frame " .. tostring(movie.framecount()) .. " before target " .. tostring(target_frame))
    end
  end

  movie.stop()
  set_readonly(false)
  log_line("movie stopped at frame " .. tostring(movie.framecount()))
  return movie.framecount()
end

local control_dir = job.control_dir
local total_steps = 0

if job.visual_debug then
  emu.speedmode("normal")
else
  emu.speedmode("maximum")
end

local bootstrap_frame = exact_movie_bootstrap()
local current_action_token = "BOOTSTRAP"

while true do
  gui.text(8, 8, "action: " .. current_action_token)
  capture_state(control_dir, total_steps, bootstrap_frame)
  log_line("state captured total_steps=" .. tostring(total_steps))
  local rows, should_quit = wait_for_actions(control_dir)
  if should_quit or rows == nil then
    log_line("bridge exiting from wait loop")
    break
  end
  for _, row in ipairs(rows) do
    local mask = tonumber(row)
    if mask ~= nil then
      current_action_token = mask_to_token(mask)
      joypad.set(1, mask_to_joypad(mask))
      gui.text(8, 8, "action: " .. current_action_token)
      emu.frameadvance()
      total_steps = total_steps + 1
      if (job.max_total_steps or 0) > 0 and total_steps >= job.max_total_steps then
        log_line("max_total_steps reached: " .. tostring(total_steps))
        should_quit = true
        break
      end
    end
  end
  if should_quit then
    log_line("bridge exiting after action loop")
    break
  end
end

if job.visual_debug and (job.debug_exit_delay or 0) > 0 then
  for _ = 1, job.debug_exit_delay do
    emu.frameadvance()
  end
end

emu.exit()
