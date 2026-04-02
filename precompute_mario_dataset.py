from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from mario_lewm.dataset import discover_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute resized Mario dataset frames for faster training.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resize_frames_uint8(frames: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    if tensor.shape[-2] != image_size or tensor.shape[-1] != image_size:
        tensor = TF.resize(
            tensor,
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    tensor = (tensor * 255.0).round().clamp(0, 255).to(torch.uint8)
    return tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    total = len(episodes) if args.limit is None else min(len(episodes), args.limit)
    for idx, episode in enumerate(episodes[:total], start=1):
        if episode.frames_npz is None:
            print(f"skip_non_npz index={idx}/{total} name={episode.name}")
            continue
        out_path = args.output_dir / f"{episode.name}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"skip_existing index={idx}/{total} name={episode.name}")
            continue

        with np.load(episode.frames_npz, allow_pickle=False) as data:
            frames = np.asarray(data["frames"], dtype=np.uint8)
            actions = np.asarray(data["actions"], dtype=np.float32)
            metadata = {}
            if "metadata_json" in data:
                try:
                    metadata = json.loads(str(data["metadata_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}

        capture_initial_frame = bool(metadata.get("capture_initial_frame", False))
        if capture_initial_frame and len(frames) == len(actions) + 1:
            frame_save = resize_frames_uint8(frames, args.image_size)
            action_save = actions
        else:
            usable = min(len(frames), len(actions))
            frame_save = resize_frames_uint8(frames[:usable], args.image_size)
            action_save = actions[:usable]

        metadata.update(
            {
                "preprocessed_image_size": args.image_size,
                "preprocessed_source": str(episode.frames_npz),
                "preprocessed_frames": int(len(frame_save)),
            }
        )
        np.savez_compressed(
            out_path,
            frames=frame_save,
            actions=action_save,
            metadata_json=json.dumps(metadata),
        )
        print(
            f"precompute_episode index={idx}/{total} name={episode.name} "
            f"frames={len(frame_save)} actions={len(action_save)} output={out_path.name}"
        )
        summary.append(
            {
                "name": episode.name,
                "frames": int(len(frame_save)),
                "actions": int(len(action_save)),
                "output": str(out_path),
            }
        )

    (args.output_dir / "precompute_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
