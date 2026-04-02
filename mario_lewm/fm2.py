from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

# Button order is shared across parsing, blocked-action construction, and
# conversion to the NES bitmask expected by FCEUX / gym-style helpers.
FM2_BUTTONS = ("R", "L", "D", "U", "T", "S", "B", "A")
NES_BIT_WEIGHTS = {
    "A": 0x01,
    "B": 0x02,
    "S": 0x04,
    "T": 0x08,
    "U": 0x10,
    "D": 0x20,
    "L": 0x40,
    "R": 0x80,
}


def _is_movie_line(line: str) -> bool:
    return line.startswith("|") and line.count("|") >= 3


def _parse_controller_field(field: str) -> np.ndarray:
    field = field.strip()
    if len(field) < len(FM2_BUTTONS):
        field = field.ljust(len(FM2_BUTTONS), ".")
    values = [0.0 if ch in {".", " "} else 1.0 for ch in field[: len(FM2_BUTTONS)]]
    return np.asarray(values, dtype=np.float32)


def read_fm2_header(path: str | Path) -> dict[str, str]:
    path = Path(path)
    header: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_movie_line(line):
            break
        if " " not in line:
            continue
        key, value = line.split(" ", 1)
        header[key] = value.strip()
    return header


def parse_fm2(path: str | Path) -> np.ndarray:
    path = Path(path)
    actions = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not _is_movie_line(line):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        actions.append(_parse_controller_field(parts[2]))
    if not actions:
        raise ValueError(f"No controller inputs were parsed from {path}.")
    return np.stack(actions, axis=0)


def build_action_library(sequences: Iterable[np.ndarray]) -> np.ndarray:
    """Collect unique action rows or action blocks from a dataset."""
    rows = []
    action_dim = None
    for sequence in sequences:
        if sequence.size == 0:
            continue
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError("Each action sequence must be 2D [T, action_dim].")
        if action_dim is None:
            action_dim = int(sequence.shape[1])
        elif int(sequence.shape[1]) != action_dim:
            raise ValueError(
                f"All action sequences must share the same action_dim. Got {sequence.shape[1]} and {action_dim}."
            )
        rows.append(sequence)
    if not rows:
        raise ValueError("Cannot build an action library from empty sequences.")
    merged = np.concatenate(rows, axis=0)
    unique = np.unique(merged, axis=0)
    return unique.astype(np.float32)


def block_action_sequence(actions: np.ndarray, block_size: int) -> np.ndarray:
    """Flatten consecutive per-frame controller rows into blocked actions."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(FM2_BUTTONS):
        raise ValueError(f"actions must be shaped [T, {len(FM2_BUTTONS)}].")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    usable = (len(actions) // block_size) * block_size
    if usable == 0:
        raise ValueError("Not enough actions to form a single action block.")
    blocked = actions[:usable].reshape(-1, block_size * len(FM2_BUTTONS))
    return blocked.astype(np.float32)


def unblock_action_sequence(blocked_actions: np.ndarray, block_size: int) -> np.ndarray:
    """Expand blocked actions back into per-frame controller rows."""
    blocked_actions = np.asarray(blocked_actions, dtype=np.float32)
    expected = block_size * len(FM2_BUTTONS)
    if blocked_actions.ndim != 2 or blocked_actions.shape[1] != expected:
        raise ValueError(f"blocked_actions must be shaped [T, {expected}].")
    return blocked_actions.reshape(-1, len(FM2_BUTTONS)).astype(np.float32)


def fm2_row_to_nes_action(row: np.ndarray) -> int:
    row = np.asarray(row, dtype=np.float32).reshape(len(FM2_BUTTONS))
    action = 0
    for idx, button in enumerate(FM2_BUTTONS):
        if row[idx] > 0.5:
            action |= NES_BIT_WEIGHTS[button]
    return int(action)


def fm2_rows_to_nes_actions(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return np.asarray([fm2_row_to_nes_action(row) for row in rows], dtype=np.int64)
