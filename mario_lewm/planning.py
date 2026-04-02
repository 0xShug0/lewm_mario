from __future__ import annotations

import math

import torch

from .model import LeWorldModel


@torch.no_grad()
def plan_to_goal(
    model: LeWorldModel,
    init_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    action_library: torch.Tensor,
    horizon: int,
    population: int,
    iterations: int,
    elite_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use CEM in latent space to find actions that approach a goal image."""
    device = next(model.parameters()).device
    init_pixels = init_pixels.to(device)
    goal_pixels = goal_pixels.to(device)
    action_library = action_library.to(device)
    if action_library.ndim != 2:
        raise ValueError("action_library must be shaped [num_actions, action_dim].")
    history = model.config.history_size
    if horizon < history:
        raise ValueError("horizon must be >= history_size.")

    num_actions = action_library.size(0)
    elite_count = max(1, int(math.ceil(population * elite_frac)))
    logits = torch.zeros(horizon, num_actions, device=device)
    goal_emb = model.encode_goal(goal_pixels)[:, -1]

    best_cost = None
    best_actions = None
    for _ in range(iterations):
        probs = logits.softmax(dim=-1)
        indices = torch.multinomial(probs, num_samples=population, replacement=True).transpose(0, 1)
        candidates = action_library[indices]
        candidates = candidates.unsqueeze(0).expand(init_pixels.size(0), -1, -1, -1)
        rollout = model.rollout(init_pixels, candidates)
        pred_last = rollout[:, :, -1]
        costs = (pred_last - goal_emb.unsqueeze(1)).pow(2).mean(dim=-1)
        flat_costs = costs[0]
        elite_ids = flat_costs.topk(k=elite_count, largest=False).indices
        elite_indices = indices[elite_ids]
        counts = torch.zeros_like(logits)
        counts.scatter_add_(
            1,
            elite_indices.transpose(0, 1),
            torch.ones_like(elite_indices.transpose(0, 1), dtype=counts.dtype),
        )
        logits = (counts + 1e-3).log()
        round_best = flat_costs[elite_ids[0]]
        if best_cost is None or round_best < best_cost:
            best_cost = round_best
            best_actions = candidates[:, elite_ids[0]]
    if best_actions is None or best_cost is None:
        raise RuntimeError("Planner did not produce an action sequence.")
    return best_actions, best_cost
