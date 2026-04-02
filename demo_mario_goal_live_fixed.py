from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from mario_lewm.dataset import MarioTraceDataset, discover_episodes
from mario_lewm.fm2 import FM2_BUTTONS, build_action_library, fm2_row_to_nes_action, unblock_action_sequence
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.planning import plan_to_goal

GD_HEADER_SIZE = 11
FRAME_HEIGHT = 240
FRAME_WIDTH = 256
CHANNELS_ARGB = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live goal-conditioned Mario control with exact FM2 bootstrap before LeWM takeover."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-name", type=str, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--goal-index", type=int, required=True)
    parser.add_argument("--trace-root", type=Path, default=Path("traces"))
    parser.add_argument("--fceux-dir", type=Path, default=Path("fceux"))
    parser.add_argument("--rom", type=Path, default=Path("fceux") / "SMB.nes")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum number of action blocks to execute.")
    parser.add_argument("--visual-debug", action="store_true")
    parser.add_argument("--debug-exit-delay", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--control-dir", type=Path, default=Path("fceux") / "_wm_goal_live_fixed")
    parser.add_argument("--plan-log", type=Path, default=None)
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


def decode_current_frame(path: Path) -> np.ndarray:
    raw_bytes = path.read_bytes()
    pixel_bytes = FRAME_HEIGHT * FRAME_WIDTH * CHANNELS_ARGB
    if len(raw_bytes) != GD_HEADER_SIZE + pixel_bytes:
        raise ValueError(f"Unexpected frame blob size {len(raw_bytes)} at {path}.")
    argb = np.frombuffer(raw_bytes[GD_HEADER_SIZE:], dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
    return argb[..., 1:4].copy()


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
    visual_debug: bool,
    debug_exit_delay: int,
    max_total_steps: int,
    bootstrap_raw_frame: int,
) -> None:
    # The fixed bridge uses exact FM2 movie playback for bootstrap so the live
    # state matches the exported dataset much more closely than manual joypad replay.
    content = "\n".join(
        [
            "return {",
            f"  rom_path = [[{rom_path}]],",
            f"  trace_path = [[{trace_path}]],",
            f"  control_dir = [[{control_dir}]],",
            f"  visual_debug = {str(bool(visual_debug)).lower()},",
            f"  debug_exit_delay = {int(debug_exit_delay)},",
            f"  max_total_steps = {int(max_total_steps)},",
            f"  bootstrap_raw_frame = {int(bootstrap_raw_frame)},",
            "}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
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
    block_size = dataset_block_size

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

    bootstrap_raw_frame = args.start_index * block_size
    goal_pixels = eval_dataset.get_frame_tensor(episode, args.goal_index).unsqueeze(0).unsqueeze(0).to(device)

    action_library = load_or_rebuild_action_library(checkpoint, episodes, checkpoint_action_dim, device)
    action_library = filter_block_action_library(action_library, block_size)

    if args.control_dir.exists():
        shutil.rmtree(args.control_dir)
    args.control_dir.mkdir(parents=True, exist_ok=True)
    job_path = args.control_dir / "live_job.lua"
    write_live_job(
        job_path,
        rom_path=args.rom.resolve(),
        trace_path=trace_path.resolve(),
        control_dir=args.control_dir.resolve(),
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

    env = dict(os.environ)
    env["FCEUX_LIVE_JOB_FIXED"] = str(job_path.resolve())
    command = [
        str(fceux_exe.resolve()),
        "-lua",
        str((Path.cwd() / "fceux_live_bridge_fixed.lua").resolve()),
        str(args.rom.resolve()),
    ]
    process = subprocess.Popen(command, cwd=args.fceux_dir, env=env)

    plan_records = []
    context_frames: deque[torch.Tensor] = deque(maxlen=history)
    current_block_index = args.start_index
    last_seen_total_steps = -1

    try:
        while current_block_index < min(args.start_index + args.max_steps, len(episode.actions) - 1) and process.poll() is None:
            frame, live_meta = wait_for_state(args.control_dir, args.poll_seconds, last_seen_total_steps, process)
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
            block_chunk = actions[0].detach().cpu().numpy().astype(np.float32)[: max(1, args.replan_every)]
            raw_chunk = unblock_action_sequence(block_chunk, block_size)
            send_actions(args.control_dir, raw_chunk)
            current_block_index += len(block_chunk)
            record = {
                "current_block_index": current_block_index,
                "goal_index": args.goal_index,
                "bootstrap_raw_frame": bootstrap_raw_frame,
                "predicted_cost": float(cost.detach().cpu()),
                "x_pos": meta_int(live_meta, "x_pos", -1),
                "chunk_masks": [fm2_row_to_nes_action(row) for row in raw_chunk],
                "chunk_tokens": [fm2_row_to_token(row) for row in raw_chunk],
            }
            plan_records.append(record)
            print(json.dumps(record))

        request_quit(args.control_dir)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        if process.poll() is None:
            request_quit(args.control_dir)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    if args.plan_log is not None:
        args.plan_log.parent.mkdir(parents=True, exist_ok=True)
        args.plan_log.write_text(json.dumps(plan_records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
