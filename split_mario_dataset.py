from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a blocked Mario dataset into train/test episode directories.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--test-name", action="append", default=[], help="Episode name to force into the test split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_paths = sorted(path for path in args.dataset_root.glob("*.npz") if path.name != "generation_summary.json")
    if not episode_paths:
        raise FileNotFoundError(f"No .npz episodes found under {args.dataset_root}")

    by_name = {path.stem: path for path in episode_paths}
    explicit_test = []
    for name in args.test_name:
        if name not in by_name:
            raise KeyError(f"Requested test episode {name!r} was not found under {args.dataset_root}.")
        explicit_test.append(name)

    explicit_test = list(dict.fromkeys(explicit_test))
    remaining = [path.stem for path in episode_paths if path.stem not in explicit_test]

    if explicit_test:
        test_names = explicit_test
    else:
        if not 0.0 < args.test_fraction < 1.0:
            raise ValueError("--test-fraction must be between 0 and 1 when --test-name is not provided.")
        rng = random.Random(args.seed)
        shuffled = remaining[:]
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(episode_paths) * args.test_fraction)))
        test_count = min(test_count, len(episode_paths) - 1)
        test_names = sorted(shuffled[:test_count])

    train_names = sorted(path.stem for path in episode_paths if path.stem not in set(test_names))
    if not train_names or not test_names:
        raise ValueError("Split would produce an empty train or test set.")

    for out_dir in (args.train_dir, args.test_dir):
        if out_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace it.")
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in train_names:
        shutil.copy2(by_name[name], args.train_dir / by_name[name].name)
    for name in test_names:
        shutil.copy2(by_name[name], args.test_dir / by_name[name].name)

    summary = {
        "dataset_root": str(args.dataset_root),
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "seed": args.seed,
        "train_count": len(train_names),
        "test_count": len(test_names),
        "train_names": train_names,
        "test_names": test_names,
    }
    (args.train_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.test_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
