from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mario_lewm.dataset import MarioTraceDataset, discover_episodes
from mario_lewm.fm2 import FM2_BUTTONS, build_action_library, fm2_row_to_nes_action, unblock_action_sequence
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.planning import plan_to_goal

GD_HEADER_SIZE = 11
FRAME_HEIGHT = 240
FRAME_WIDTH = 256
CHANNELS_ARGB = 4
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated LeWM live goal demo: run exact FM2 bootstrap, export start/goal JPGs, and render 720p video."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-name", type=str, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--goal-index", type=int, required=True)
    parser.add_argument("--trace-root", type=Path, default=Path("traces"))
    parser.add_argument("--fceux-dir", type=Path, default=Path("fceux"))
    parser.add_argument("--rom", type=Path, default=Path("fceux") / "SMB.nes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--visual-debug", action="store_true")
    parser.add_argument("--debug-exit-delay", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> tuple[LeWorldModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LeWorldModelConfig(**checkpoint["config"])
    model = LeWorldModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_episode_metadata(episode) -> dict:
    if episode.frames_npz is None:
        return {}
    with np.load(episode.frames_npz, allow_pickle=False) as data:
        if "metadata_json" not in data:
            return {}
        return json.loads(str(data["metadata_json"]))


def action_sequences_for_library(episodes) -> list[np.ndarray]:
    sequences = []
    for episode in episodes:
        metadata = episode.metadata or {}
        if metadata.get("lewm_terminal_pad_action"):
            num_blocks = int(metadata.get("lewm_num_action_blocks", max(0, len(episode.actions) - 1)))
            sequences.append(np.asarray(episode.actions[:num_blocks], dtype=np.float32))
        else:
            sequences.append(np.asarray(episode.actions, dtype=np.float32))
    return sequences


def load_or_rebuild_action_library(checkpoint: dict, episodes, expected_action_dim: int, device: torch.device) -> torch.Tensor:
    saved = checkpoint.get("action_library")
    if saved is not None:
        action_library = torch.as_tensor(saved, dtype=torch.float32, device=device)
        if action_library.ndim == 2 and action_library.shape[1] == expected_action_dim:
            return action_library
        print(
            f"warning=rebuilding_action_library saved_action_dim="
            f"{action_library.shape[1] if action_library.ndim == 2 else 'invalid'} "
            f"expected_action_dim={expected_action_dim}"
        )
    rebuilt = build_action_library(action_sequences_for_library(episodes))
    return torch.as_tensor(rebuilt, dtype=torch.float32, device=device)


def parse_meta_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def meta_int(meta: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(meta.get(key, str(default)))
    except ValueError:
        return default


def live_meta_is_gameplay(meta: dict[str, str]) -> bool:
    world = meta_int(meta, "world")
    stage = meta_int(meta, "stage")
    status = meta_int(meta, "status")
    life = meta_int(meta, "life")
    x_pos = meta_int(meta, "x_pos")
    return 0 < world <= 8 and 0 < stage <= 4 and 0 <= status < 4 and life > 0 and 0 <= x_pos < 5000


def decode_gd_blob(raw_bytes: bytes) -> np.ndarray:
    pixel_bytes = FRAME_HEIGHT * FRAME_WIDTH * CHANNELS_ARGB
    if len(raw_bytes) != GD_HEADER_SIZE + pixel_bytes:
        raise ValueError(f"Unexpected frame blob size {len(raw_bytes)}.")
    argb = np.frombuffer(raw_bytes[GD_HEADER_SIZE:], dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
    return argb[..., 1:4].copy()


def decode_current_frame(path: Path) -> np.ndarray:
    return decode_gd_blob(path.read_bytes())


def wait_for_state(
    control_dir: Path,
    poll_seconds: float,
    previous_total_steps: int,
    process: subprocess.Popen,
    timeout_seconds: float = 30.0,
) -> tuple[np.ndarray, dict[str, str]]:
    ready_path = control_dir / "state_ready.flag"
    frame_path = control_dir / "current_frame.gd"
    meta_path = control_dir / "current_meta.txt"
    start = time.time()
    while True:
        if process.poll() is not None:
            raise RuntimeError("FCEUX exited before a new live state was produced.")
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for a new live state from FCEUX.")
        if not ready_path.exists() or not frame_path.exists() or not meta_path.exists():
            time.sleep(max(0.01, poll_seconds))
            continue
        meta = parse_meta_file(meta_path)
        total_steps = int(meta.get("total_steps", "-1") or -1)
        if total_steps == previous_total_steps:
            time.sleep(max(0.01, poll_seconds))
            continue
        frame = decode_current_frame(frame_path)
        return frame, meta


def send_actions(control_dir: Path, action_rows: np.ndarray) -> None:
    action_path = control_dir / "actions.txt"
    temp_path = control_dir / "actions.tmp"
    masks = [str(fm2_row_to_nes_action(row)) for row in action_rows]
    temp_path.write_text("\n".join(masks) + "\n", encoding="utf-8")
    temp_path.replace(action_path)


def request_quit(control_dir: Path) -> None:
    temp_path = control_dir / "quit.tmp"
    quit_path = control_dir / "quit.flag"
    temp_path.write_text("quit\n", encoding="utf-8")
    temp_path.replace(quit_path)


def fm2_row_to_token(row: np.ndarray) -> str:
    row = np.asarray(row, dtype=np.float32).reshape(len(FM2_BUTTONS))
    pressed = [button for button, value in zip(FM2_BUTTONS, row) if value > 0.5]
    return "".join(pressed) if pressed else "NONE"


def filter_block_action_library(action_library: torch.Tensor, block_size: int) -> torch.Tensor:
    if action_library.ndim != 2:
        raise ValueError("action_library must be [num_actions, action_dim].")
    base = len(FM2_BUTTONS)
    expected = block_size * base
    if action_library.shape[1] != expected:
        raise ValueError(
            f"action_library shape does not match block_size: "
            f"got action_dim={action_library.shape[1]}, expected {expected}"
        )
    rows = action_library.view(action_library.size(0), block_size, base)
    idx = {button: FM2_BUTTONS.index(button) for button in FM2_BUTTONS}
    keep = (
        (rows[:, :, idx["T"]] < 0.5).all(dim=1)
        & (rows[:, :, idx["S"]] < 0.5).all(dim=1)
        & ~((rows[:, :, idx["R"]] > 0.5) & (rows[:, :, idx["L"]] > 0.5)).any(dim=1)
        & ~((rows[:, :, idx["U"]] > 0.5) & (rows[:, :, idx["D"]] > 0.5)).any(dim=1)
    )
    filtered = action_library[keep]
    if filtered.numel() == 0:
        raise ValueError("Filtering gameplay action blocks removed the entire action library.")
    return filtered


def write_live_job(
    path: Path,
    *,
    rom_path: Path,
    trace_path: Path,
    control_dir: Path,
    frames_dir: Path,
    visual_debug: bool,
    debug_exit_delay: int,
    max_total_steps: int,
    bootstrap_raw_frame: int,
) -> None:
    content = "\n".join(
        [
            "return {",
            f"  rom_path = [[{rom_path}]],",
            f"  trace_path = [[{trace_path}]],",
            f"  control_dir = [[{control_dir}]],",
            f"  frames_dir = [[{frames_dir}]],",
            f"  visual_debug = {str(bool(visual_debug)).lower()},",
            f"  debug_exit_delay = {int(debug_exit_delay)},",
            f"  max_total_steps = {int(max_total_steps)},",
            f"  bootstrap_raw_frame = {int(bootstrap_raw_frame)},",
            "}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def save_rgb_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path, format="JPEG", quality=95)


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    scale = max(scale, 1e-6)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(new_size, resample=Image.Resampling.NEAREST)


def draw_labeled_panel(
    canvas: Image.Image,
    image_rgb: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=14, fill=(26, 26, 26), outline=(60, 60, 60))
    draw.text((x + 12, y + 10), label, fill=(255, 255, 255), font=font)
    image = Image.fromarray(image_rgb)
    fitted = fit_image(image, width - 20, height - 48)
    px = x + (width - fitted.width) // 2
    py = y + 36 + (height - 48 - fitted.height) // 2
    canvas.paste(fitted, (px, py))


def control_mode_for_token(action_token: str) -> str:
    if action_token in {"TRACE", "END"}:
        return "Replay"
    return "Model Control"


def render_frame_720p(
    frame_rgb: np.ndarray,
    action_token: str,
    title: str,
    start_ref_rgb: np.ndarray,
    goal_ref_rgb: np.ndarray,
) -> np.ndarray:
    frame = Image.fromarray(frame_rgb)
    gameplay_scale = min(768 / FRAME_WIDTH, OUTPUT_HEIGHT / FRAME_HEIGHT)
    gameplay_size = (
        max(1, int(round(FRAME_WIDTH * gameplay_scale))),
        max(1, int(round(FRAME_HEIGHT * gameplay_scale))),
    )
    scaled = frame.resize(gameplay_size, resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (18, 18, 18))

    x0 = (OUTPUT_WIDTH - scaled.width) // 2
    y0 = (OUTPUT_HEIGHT - scaled.height) // 2
    canvas.paste(scaled, (x0, y0))

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    mode = control_mode_for_token(action_token)
    is_model = mode == "Model Control"
    mode_fill = (34, 92, 210) if is_model else (92, 92, 92)
    action_fill = (255, 200, 72) if is_model else (220, 220, 220)

    draw.rounded_rectangle((24, 20, 620, 132), radius=14, fill=(0, 0, 0))
    draw.text((40, 34), title, fill=(255, 255, 255), font=font)
    draw.rounded_rectangle((40, 60, 180, 92), radius=10, fill=mode_fill)
    draw.text((54, 70), mode, fill=(255, 255, 255), font=font)
    draw.rounded_rectangle((40, 98, 430, 124), radius=10, fill=(28, 28, 28), outline=action_fill)
    draw.text((54, 106), f"ACTION: {action_token}", fill=action_fill, font=font)

    side_panel_width = 220
    side_panel_height = 280
    panel_y = (OUTPUT_HEIGHT - side_panel_height) // 2
    draw_labeled_panel(canvas, start_ref_rgb, x=16, y=panel_y, width=side_panel_width, height=side_panel_height, label="Start", font=font)
    draw_labeled_panel(
        canvas,
        goal_ref_rgb,
        x=OUTPUT_WIDTH - side_panel_width - 16,
        y=panel_y,
        width=side_panel_width,
        height=side_panel_height,
        label="Goal",
        font=font,
    )
    return np.asarray(canvas)


def render_video_from_manifest(
    frames_dir: Path,
    video_path: Path,
    title: str,
    fps: int,
    start_ref_rgb: np.ndarray,
    goal_ref_rgb: np.ndarray,
) -> int:
    manifest_path = frames_dir / "frames_manifest.tsv"
    hold_manifest_path = frames_dir / "hold_manifest.tsv"
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("No recorded frames found to render.")

    hold_after_frame_idx: dict[int, int] = {}
    if hold_manifest_path.exists():
        with hold_manifest_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                frame_idx = int(row["frame_idx"])
                hold_frames = int(row["hold_frames"])
                hold_after_frame_idx[frame_idx] = max(0, hold_after_frame_idx.get(frame_idx, 0) + hold_frames)

    temp_dir = frames_dir / "_render_720p"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        render_idx = 0
        for row in rows:
            frame_idx = int(row["frame_idx"])
            action_token = row["action_token"]
            gd_path = frames_dir / f"frame_{frame_idx:06d}.gd"
            frame_rgb = decode_gd_blob(gd_path.read_bytes())
            rendered = render_frame_720p(frame_rgb, action_token, title, start_ref_rgb, goal_ref_rgb)
            repeat_count = 1 + hold_after_frame_idx.get(frame_idx, 0)
            for _ in range(repeat_count):
                png_path = temp_dir / f"frame_{render_idx:06d}.png"
                cv2.imwrite(str(png_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                render_idx += 1

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {completed.stderr}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return render_idx


def next_run_index(output_dir: Path) -> int:
    max_idx = 0
    for path in output_dir.glob("demo_*.mp4"):
        stem = path.stem
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


def run_once(
    args: argparse.Namespace,
    *,
    model: LeWorldModel,
    device: torch.device,
    episodes,
    episode,
    eval_dataset: MarioTraceDataset,
    action_library: torch.Tensor,
    block_size: int,
) -> None:
    if args.output_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.output_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    history = model.config.history_size
    metadata = load_episode_metadata(episode)
    trace_name = str(metadata.get("trace_name", "")).strip()
    trace_path = args.trace_root / trace_name
    bootstrap_raw_frame = args.start_index * block_size
    goal_pixels = eval_dataset.get_frame_tensor(episode, args.goal_index).unsqueeze(0).unsqueeze(0).to(device)

    run_index = next_run_index(args.output_dir)
    run_tag = f"{run_index:03d}"
    control_dir = args.output_dir / f"control_{run_tag}"
    frames_dir = args.output_dir / f"captured_frames_{run_tag}"
    control_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    hold_manifest_path = frames_dir / "hold_manifest.tsv"
    hold_manifest_path.write_text("frame_idx\thold_frames\thold_seconds\n", encoding="utf-8")

    start_jpg = args.output_dir / "start_frame.jpg"
    goal_jpg = args.output_dir / "goal_frame.jpg"
    video_path = args.output_dir / f"demo_{run_tag}.mp4"
    plan_log = args.output_dir / f"plan_log_{run_tag}.json"
    job_path = control_dir / "live_job.lua"
    write_live_job(
        job_path,
        rom_path=args.rom.resolve(),
        trace_path=trace_path.resolve(),
        control_dir=control_dir.resolve(),
        frames_dir=frames_dir.resolve(),
        visual_debug=args.visual_debug,
        debug_exit_delay=max(0, args.debug_exit_delay),
        max_total_steps=max(0, args.max_steps * block_size),
        bootstrap_raw_frame=bootstrap_raw_frame,
    )

    fceux_exe = args.fceux_dir / "fceux64.exe"
    if not fceux_exe.exists():
        raise FileNotFoundError(fceux_exe)
    if not args.rom.exists():
        raise FileNotFoundError(args.rom)

    start_ref_rgb = eval_dataset.get_frame_tensor(episode, args.start_index).permute(1, 2, 0).mul(255).byte().numpy()
    goal_ref_rgb = eval_dataset.get_frame_tensor(episode, args.goal_index).permute(1, 2, 0).mul(255).byte().numpy()
    save_rgb_image(start_ref_rgb, start_jpg)
    save_rgb_image(goal_ref_rgb, goal_jpg)

    env = dict(os.environ)
    env["FCEUX_LIVE_JOB_RECORD"] = str(job_path.resolve())
    command = [
        str(fceux_exe.resolve()),
        "-lua",
        str((Path.cwd() / "fceux_live_bridge_record.lua").resolve()),
        str(args.rom.resolve()),
    ]
    process = subprocess.Popen(command, cwd=args.fceux_dir, env=env)

    plan_records = []
    context_frames: deque[torch.Tensor] = deque(maxlen=history)
    current_block_index = args.start_index
    last_seen_total_steps = -1

    try:
        while current_block_index < min(args.start_index + args.max_steps, len(episode.actions) - 1) and process.poll() is None:
            frame, live_meta = wait_for_state(control_dir, args.poll_seconds, last_seen_total_steps, process)
            last_seen_total_steps = int(live_meta.get("total_steps", last_seen_total_steps))
            if not live_meta_is_gameplay(live_meta):
                time.sleep(max(0.01, args.poll_seconds))
                continue

            if not context_frames:
                ctx_start = args.start_index - history + 1
                for frame_idx in range(ctx_start, args.start_index):
                    context_frames.append(eval_dataset.get_frame_tensor(episode, frame_idx))
                context_frames.append(eval_dataset._preprocess_npz_frames(frame[None])[0])
            else:
                context_frames.append(eval_dataset._preprocess_npz_frames(frame[None])[0])

            init_pixels = torch.stack(list(context_frames), dim=0).unsqueeze(0).to(device)
            plan_start = time.perf_counter()
            actions, cost = plan_to_goal(
                model=model,
                init_pixels=init_pixels,
                goal_pixels=goal_pixels,
                action_library=action_library,
                horizon=max(history, args.horizon),
                population=args.population,
                iterations=args.iterations,
                elite_frac=args.elite_frac,
            )
            planning_seconds = max(0.0, time.perf_counter() - plan_start)
            hold_frames = max(0, int(math.ceil(planning_seconds * args.fps)) - 1)
            if hold_frames > 0:
                display_frame_idx = bootstrap_raw_frame + last_seen_total_steps
                with hold_manifest_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{display_frame_idx}\t{hold_frames}\t{planning_seconds:.6f}\n")
            block_chunk = actions[0].detach().cpu().numpy().astype(np.float32)[: max(1, args.replan_every)]
            raw_chunk = unblock_action_sequence(block_chunk, block_size)
            send_actions(control_dir, raw_chunk)
            current_block_index += len(block_chunk)
            record = {
                "current_block_index": current_block_index,
                "goal_index": args.goal_index,
                "bootstrap_raw_frame": bootstrap_raw_frame,
                "predicted_cost": float(cost.detach().cpu()),
                "x_pos": meta_int(live_meta, "x_pos", -1),
                "planning_seconds": planning_seconds,
                "chunk_masks": [fm2_row_to_nes_action(row) for row in raw_chunk],
                "chunk_tokens": [fm2_row_to_token(row) for row in raw_chunk],
            }
            plan_records.append(record)
            print(json.dumps(record))

        request_quit(control_dir)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        if process.poll() is None:
            request_quit(control_dir)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    plan_log.write_text(json.dumps(plan_records, indent=2), encoding="utf-8")
    title = f"{args.episode_name} | start {args.start_index} -> goal {args.goal_index}"
    rendered_frames = render_video_from_manifest(frames_dir, video_path, title, args.fps, start_ref_rgb, goal_ref_rgb)
    summary = {
        "run_index": run_index,
        "episode_name": args.episode_name,
        "start_index": args.start_index,
        "goal_index": args.goal_index,
        "bootstrap_raw_frame": bootstrap_raw_frame,
        "video_path": str(video_path),
        "start_frame_path": str(start_jpg),
        "goal_frame_path": str(goal_jpg),
        "control_dir": str(control_dir),
        "frames_dir": str(frames_dir),
        "rendered_frames": rendered_frames,
        "output_resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "fps": args.fps,
    }
    (args.output_dir / f"summary_{run_tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1.")

    model, checkpoint = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    episodes = discover_episodes(args.dataset_root)
    by_name = {episode.name: episode for episode in episodes}
    if args.episode_name not in by_name:
        raise KeyError(f"Episode {args.episode_name!r} not found. Available: {sorted(by_name)}")
    episode = by_name[args.episode_name]
    metadata = load_episode_metadata(episode)
    dataset_block_size = int(metadata.get("lewm_action_block_size", 1))
    if dataset_block_size <= 0:
        raise ValueError("Invalid lewm_action_block_size in dataset metadata.")
    checkpoint_action_dim = int(model.config.action_dim)
    if checkpoint_action_dim % len(FM2_BUTTONS) != 0:
        raise ValueError(
            f"Checkpoint action_dim={checkpoint_action_dim} is not divisible by {len(FM2_BUTTONS)}."
        )
    checkpoint_block_size = checkpoint_action_dim // len(FM2_BUTTONS)
    if checkpoint_block_size != dataset_block_size:
        raise ValueError(
            "Checkpoint/dataset mismatch: "
            f"checkpoint expects block_size={checkpoint_block_size}, "
            f"dataset uses block_size={dataset_block_size}. "
            "Use a checkpoint trained on mario_dataset_lewm."
        )

    history = model.config.history_size
    if args.start_index < history - 1:
        raise ValueError("start-index must be at least history_size - 1 for exact local-goal evaluation.")
    if args.goal_index <= args.start_index:
        raise ValueError("goal-index must be greater than start-index.")
    if args.goal_index >= len(episode.actions):
        raise ValueError("goal-index is outside the episode.")

    trace_name = str(metadata.get("trace_name", "")).strip()
    if not trace_name:
        raise ValueError("Dataset metadata is missing trace_name, cannot do exact FM2 bootstrap.")
    trace_path = args.trace_root / trace_name
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file from metadata was not found: {trace_path}")

    eval_dataset = MarioTraceDataset(
        [episode],
        history_size=history,
        num_preds=1,
        image_size=model.config.image_size,
        stride=1,
    )

    action_library = load_or_rebuild_action_library(checkpoint, episodes, checkpoint_action_dim, device)
    action_library = filter_block_action_library(action_library, dataset_block_size)

    overwrite_once = args.overwrite
    for run_idx in range(args.num_runs):
        run_args = argparse.Namespace(**vars(args))
        run_args.overwrite = overwrite_once and run_idx == 0
        print(
            json.dumps(
                {
                    "run_number": run_idx + 1,
                    "num_runs": args.num_runs,
                    "episode_name": args.episode_name,
                    "start_index": args.start_index,
                    "goal_index": args.goal_index,
                    "output_dir": str(args.output_dir),
                }
            )
        )
        run_once(
            run_args,
            model=model,
            device=device,
            episodes=episodes,
            episode=episode,
            eval_dataset=eval_dataset,
            action_library=action_library,
            block_size=dataset_block_size,
        )


if __name__ == "__main__":
    main()
