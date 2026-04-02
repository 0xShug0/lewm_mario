from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path("mario_dataset_lewm")
DISPLAY_SCALE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show blocked-dataset start/goal frames side by side. Usage: python show_goal_frames.py TWR 1 2"
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("trace", type=str, help="Episode name / .npz stem inside mario_dataset_lewm")
    parser.add_argument("start_index", type=int, help="Blocked-frame start index")
    parser.add_argument("goal_index", type=int, help="Blocked-frame goal index")
    return parser.parse_args()


def load_episode_frames(dataset_root: Path, trace_name: str) -> np.ndarray:
    episode_path = dataset_root / f"{trace_name}.npz"
    if not episode_path.exists():
        available = sorted(path.stem for path in dataset_root.glob("*.npz"))
        raise FileNotFoundError(f"{episode_path} not found. Available episodes include: {available[:20]}")
    with np.load(episode_path, allow_pickle=False) as data:
        return np.asarray(data["frames"], dtype=np.uint8)


def add_label(frame_rgb: np.ndarray, text: str) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    labeled = frame_bgr.copy()
    cv2.rectangle(labeled, (0, 0), (220, 28), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def main() -> None:
    args = parse_args()
    frames = load_episode_frames(args.dataset_root, args.trace)
    if not (0 <= args.start_index < len(frames)):
        raise IndexError(f"start_index={args.start_index} is out of range for {args.trace} with {len(frames)} frames.")
    if not (0 <= args.goal_index < len(frames)):
        raise IndexError(f"goal_index={args.goal_index} is out of range for {args.trace} with {len(frames)} frames.")

    start_frame = add_label(frames[args.start_index], f"start {args.start_index}")
    goal_frame = add_label(frames[args.goal_index], f"goal {args.goal_index}")
    spacer = np.full((start_frame.shape[0], 12, 3), 24, dtype=np.uint8)
    canvas = np.concatenate([start_frame, spacer, goal_frame], axis=1)
    canvas = cv2.resize(
        canvas,
        (canvas.shape[1] * DISPLAY_SCALE, canvas.shape[0] * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    window_name = f"{args.trace}: {args.start_index} -> {args.goal_index}"
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
