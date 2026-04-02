from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mario_lewm.fm2 import FM2_BUTTONS, block_action_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LeWM-style blocked Mario dataset from per-frame exports.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_episode_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"], dtype=np.uint8)
        actions = np.asarray(data["actions"], dtype=np.float32)
        metadata = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(str(data["metadata_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
    return frames, actions, metadata


def build_blocked_episode(frames: np.ndarray, actions: np.ndarray, frame_skip: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive.")
    if len(actions) < frame_skip:
        raise ValueError("Not enough actions to build blocked transitions.")

    # Collapse per-frame controls into LeWM-style action blocks. With frame_skip=5,
    # one model step corresponds to five raw emulator frames.
    blocked_actions = block_action_sequence(actions, frame_skip)
    num_blocks = len(blocked_actions)

    if len(frames) == len(actions) + 1:
        # exact alignment when initial observation was exported
        blocked_frames = frames[: num_blocks * frame_skip + 1 : frame_skip]
    else:
        # best-effort fallback for older exports without an initial frame
        blocked_frames = frames[: num_blocks * frame_skip : frame_skip]
        if len(blocked_frames) < num_blocks + 1:
            raise ValueError(
                f"Episode has frames={len(frames)} and actions={len(actions)}; "
                "LeWM blocked transitions need frames == actions + 1. "
                "Re-export with --capture-initial-frame using the fixed exporter."
            )
        blocked_frames = blocked_frames[: num_blocks + 1]

    if len(blocked_frames) != num_blocks + 1:
        usable = min(num_blocks, max(0, len(blocked_frames) - 1))
        blocked_actions = blocked_actions[:usable]
        blocked_frames = blocked_frames[: usable + 1]
        num_blocks = usable

    if num_blocks <= 0:
        raise ValueError("Blocked episode would be empty.")

    # The trainer expects action and frame arrays to share the same leading length,
    # so we append one dummy terminal block that is never executed in the demo.
    pad = np.zeros((1, blocked_actions.shape[1]), dtype=np.float32)
    blocked_actions_padded = np.concatenate([blocked_actions, pad], axis=0)
    return blocked_frames.astype(np.uint8), blocked_actions_padded.astype(np.float32)


def main() -> None:
    args = parse_args()
    episode_paths = sorted(args.dataset_root.glob("*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No .npz episodes found under {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for idx, episode_path in enumerate(episode_paths, start=1):
        out_path = args.output_dir / episode_path.name
        if out_path.exists() and not args.overwrite:
            print(f"skip_existing index={idx}/{len(episode_paths)} name={episode_path.stem}")
            continue

        frames, actions, metadata = load_episode_npz(episode_path)
        print(
            f"build_input index={idx}/{len(episode_paths)} name={episode_path.stem} "
            f"frames={len(frames)} actions={len(actions)}"
        )
        blocked_frames, blocked_actions = build_blocked_episode(frames, actions, args.frame_skip)
        metadata.update(
            {
                "lewm_frame_skip": args.frame_skip,
                "lewm_action_block_size": args.frame_skip,
                "lewm_base_action_dim": len(FM2_BUTTONS),
                "lewm_block_action_dim": int(blocked_actions.shape[1]),
                "lewm_blocked_frames": int(len(blocked_frames)),
                "lewm_blocked_actions": int(len(blocked_actions)),
                "lewm_num_action_blocks": int(len(blocked_actions) - 1),
                "lewm_terminal_pad_action": True,
            }
        )
        np.savez_compressed(
            out_path,
            frames=blocked_frames,
            actions=blocked_actions,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        record = {
            "name": episode_path.stem,
            "blocked_frames": int(len(blocked_frames)),
            "blocked_actions": int(len(blocked_actions)),
            "action_blocks_without_pad": int(len(blocked_actions) - 1),
            "frame_skip": args.frame_skip,
        }
        summary.append(record)
        print(json.dumps(record))

    (args.output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
