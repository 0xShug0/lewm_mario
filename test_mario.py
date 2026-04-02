from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mario_lewm.dataset import MarioTraceDataset, discover_episodes
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.planning import plan_to_goal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation and goal-conditioned planning for Mario LeWM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["offline", "plan"], default="offline")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--episode-name", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--goal-index", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> tuple[LeWorldModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LeWorldModelConfig(**checkpoint["config"])
    model = LeWorldModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def filter_action_library(action_library: torch.Tensor) -> torch.Tensor:
    if action_library.ndim != 2:
        raise ValueError("action_library must be 2D.")
    base = 8
    if action_library.shape[1] % base != 0:
        return action_library
    block_size = action_library.shape[1] // base
    rows = action_library.view(action_library.size(0), block_size, base)
    # Buttons are ordered as R L D U T S B A.
    keep = (
        (rows[:, :, 4] < 0.5).all(dim=1)
        & (rows[:, :, 5] < 0.5).all(dim=1)
        & ~((rows[:, :, 0] > 0.5) & (rows[:, :, 1] > 0.5)).any(dim=1)
        & ~((rows[:, :, 2] > 0.5) & (rows[:, :, 3] > 0.5)).any(dim=1)
    )
    filtered = action_library[keep]
    if filtered.numel() == 0:
        return action_library
    return filtered


def run_offline(args: argparse.Namespace, model: LeWorldModel, device: torch.device) -> None:
    episodes = discover_episodes(args.dataset_root)
    dataset = MarioTraceDataset(
        episodes,
        history_size=model.config.history_size,
        num_preds=model.config.num_preds,
        image_size=model.config.image_size,
        stride=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    seen = 0
    for batch_id, batch in enumerate(loader, start=1):
        pixels = batch["pixels"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        with torch.no_grad():
            output = model.compute_losses({"pixels": pixels, "action": actions})
        batch_size = pixels.size(0)
        seen += batch_size
        for key in totals:
            totals[key] += float(output[key].detach().cpu()) * batch_size
        if args.max_batches and batch_id >= args.max_batches:
            break
    metrics = {key: value / max(1, seen) for key, value in totals.items()}
    print(json.dumps({"offline_metrics": metrics}, indent=2))


def run_plan(args: argparse.Namespace, model: LeWorldModel, checkpoint: dict, device: torch.device) -> None:
    episodes = discover_episodes(args.dataset_root)
    by_name = {episode.name: episode for episode in episodes}
    episode = episodes[0] if args.episode_name is None else by_name[args.episode_name]
    history = model.config.history_size
    episode_len = len(episode.actions)
    if args.start_index + history > episode_len:
        raise ValueError("Not enough context frames for the requested start_index.")
    if args.goal_index >= episode_len:
        raise ValueError("goal_index is outside the episode.")

    init_dataset = MarioTraceDataset([episode], history_size=history, num_preds=1, image_size=model.config.image_size)
    init_sample = init_dataset[args.start_index]
    init_pixels = init_sample["pixels"][:history].unsqueeze(0).to(device)
    goal_pixels = init_dataset.get_frame_tensor(episode, args.goal_index).unsqueeze(0).unsqueeze(0).to(device)

    action_library = torch.as_tensor(checkpoint["action_library"], dtype=torch.float32, device=device)
    action_library = filter_action_library(action_library)
    actions, cost = plan_to_goal(
        model=model,
        init_pixels=init_pixels,
        goal_pixels=goal_pixels,
        action_library=action_library,
        horizon=args.horizon,
        population=args.population,
        iterations=args.iterations,
        elite_frac=args.elite_frac,
    )

    result = {
        "episode": episode.name,
        "start_index": args.start_index,
        "goal_index": args.goal_index,
        "predicted_cost": float(cost.detach().cpu()),
        "planned_actions": actions[0].detach().cpu().int().tolist(),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.mode == "offline":
        run_offline(args, model, device)
    else:
        run_plan(args, model, checkpoint, device)


if __name__ == "__main__":
    main()
