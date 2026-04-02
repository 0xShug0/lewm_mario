local job_path = os.getenv("FCEUX_EXPORT_JOB")
if not job_path or job_path == "" then
  error("FCEUX_EXPORT_JOB is not set")
end

local job = dofile(job_path)

local function write_metadata(path, values)
  local fh = assert(io.open(path, "w"))
  for _, row in ipairs(values) do
    fh:write(string.format("%s=%s\n", row[1], tostring(row[2])))
  end
  fh:close()
end

local function current_mode()
  local mode = movie.mode()
  if mode == nil then
    return "nil"
  end
  return mode
end

emu.setreadonly(true)

if not emu.loadrom(job.rom_path) then
  error("Failed to load ROM: " .. job.rom_path)
end

if not movie.play(job.trace_path, true) then
  error("Failed to load movie: " .. job.trace_path)
end

movie.playbeginning()

local raw_fh = assert(io.open(job.output_capture_path, "wb"))
local captured_frames = 0
local first_blob_size = 0
local current_frame = 0
local stride = tonumber(job.save_every) or 1
local max_frames = tonumber(job.max_frames) or 0
local visual_debug = job.visual_debug == true
local debug_exit_delay = tonumber(job.debug_exit_delay) or 0
local capture_initial_frame = job.capture_initial_frame == true

if visual_debug then
  emu.speedmode("normal")
else
  emu.speedmode("maximum")
end

if capture_initial_frame then
  local blob = gui.gdscreenshot()
  if first_blob_size == 0 then
    first_blob_size = string.len(blob)
  end
  raw_fh:write(blob)
  captured_frames = captured_frames + 1
end

while true do
  emu.frameadvance()
  local mode = current_mode()
  if mode ~= "playback" and mode ~= "finished" then
    break
  end

  current_frame = movie.framecount()
  if current_frame > 0 and current_frame % stride == 0 then
    local blob = gui.gdscreenshot()
    if first_blob_size == 0 then
      first_blob_size = string.len(blob)
    end
    raw_fh:write(blob)
    captured_frames = captured_frames + 1
  end

  if mode == "finished" then
    break
  end
  if max_frames > 0 and current_frame >= max_frames then
    break
  end
end

raw_fh:close()

write_metadata(job.output_metadata_path, {
  { "trace_path", job.trace_path },
  { "rom_path", job.rom_path },
  { "movie_name", movie.getname() },
  { "movie_length", movie.length() },
  { "movie_mode_end", current_mode() },
  { "captured_frames", captured_frames },
  { "last_movie_frame", current_frame },
  { "save_every", stride },
  { "max_frames", max_frames },
  { "blob_size", first_blob_size },
  { "rom_hash_base64", rom.gethash("base64") },
  { "visual_debug", tostring(visual_debug) },
  { "debug_exit_delay", debug_exit_delay },
  { "capture_initial_frame", tostring(capture_initial_frame) },
})

if visual_debug and debug_exit_delay > 0 then
  for _ = 1, debug_exit_delay do
    emu.frameadvance()
  end
end

emu.exit()
