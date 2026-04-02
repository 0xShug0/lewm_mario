from __future__ import annotations

from dataclasses import dataclass
import json
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .fm2 import parse_fm2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class MarioEpisode:
    name: str
    directory: Path
    frame_paths: tuple[Path, ...] | None
    frames_npz: Path | None
    actions: np.ndarray
    num_frames: int | None = None
    metadata: dict[str, object] | None = None


def _find_frame_dir(episode_dir: Path) -> Path:
    for candidate in (episode_dir / "frames", episode_dir / "images", episode_dir):
        if candidate.exists() and any(path.suffix.lower() in IMAGE_EXTENSIONS for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"Could not find frames inside {episode_dir}.")


def _find_fm2(episode_dir: Path) -> Path:
    matches = sorted(episode_dir.glob("*.fm2"))
    if not matches:
        raise FileNotFoundError(f"Could not find an .fm2 movie file inside {episode_dir}.")
    return matches[0]


def discover_episodes(root: str | Path) -> list[MarioEpisode]:
    """Discover either packed .npz episodes or image-folder episodes."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    npz_files = sorted(root.glob("*.npz"))
    if npz_files:
        episodes = []
        for idx, npz_path in enumerate(npz_files, start=1):
            with np.load(npz_path, allow_pickle=False) as data:
                if "frames" not in data or "actions" not in data:
                    continue
                actions = data["actions"].astype(np.float32)
                num_frames = None
                metadata: dict[str, object] = {}
                if "metadata_json" in data:
                    try:
                        metadata = json.loads(str(data["metadata_json"]))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        metadata = {}
                    captured_frames = metadata.get("captured_frames")
                    if captured_frames is not None:
                        num_frames = int(captured_frames)
                if num_frames is None:
                    num_frames = int(data["frames"].shape[0])
                usable_steps = min(num_frames, len(actions))
                if usable_steps < 4:
                    continue
                episodes.append(
                    MarioEpisode(
                        name=npz_path.stem,
                        directory=npz_path.parent,
                        frame_paths=None,
                        frames_npz=npz_path,
                        actions=actions[:usable_steps],
                        num_frames=num_frames,
                        metadata=metadata,
                    )
                )
                print(
                    f"discover_npz_episode index={idx}/{len(npz_files)} name={npz_path.stem} "
                    f"usable_steps={usable_steps}"
                )
        if episodes:
            return episodes
    episode_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    episodes = []
    for idx, episode_dir in enumerate(episode_dirs, start=1):
        frame_dir = _find_frame_dir(episode_dir)
        frame_paths = tuple(
            sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        )
        if not frame_paths:
            continue
        actions = parse_fm2(_find_fm2(episode_dir))
        usable_steps = min(len(frame_paths), len(actions))
        if usable_steps < 4:
            continue
        episodes.append(
            MarioEpisode(
                name=episode_dir.name,
                directory=episode_dir,
                frame_paths=frame_paths[:usable_steps],
                frames_npz=None,
                actions=actions[:usable_steps],
                metadata=None,
            )
        )
        print(
            f"discover_image_episode index={idx}/{len(episode_dirs)} name={episode_dir.name} "
            f"usable_steps={usable_steps}"
        )
    if not episodes:
        raise ValueError(f"No valid episodes were found under {root}.")
    return episodes


def split_episodes(
    episodes: Sequence[MarioEpisode],
    val_fraction: float,
    seed: int,
) -> tuple[list[MarioEpisode], list[MarioEpisode]]:
    if len(episodes) == 1:
        return [episodes[0]], [episodes[0]]
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    indices = np.arange(len(episodes))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    cut = max(1, int(round(len(indices) * (1.0 - val_fraction))))
    cut = min(cut, len(indices) - 1)
    train_idx = np.sort(indices[:cut])
    val_idx = np.sort(indices[cut:])
    train = [episodes[i] for i in train_idx]
    val = [episodes[i] for i in val_idx]
    return train, val


class MarioTraceDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[MarioEpisode],
        history_size: int,
        num_preds: int,
        image_size: int,
        stride: int = 1,
        npz_load_mode: str = "lazy",
        max_cached_episodes: int = 4,
    ) -> None:
        super().__init__()
        if npz_load_mode not in {"lazy", "preload"}:
            raise ValueError("npz_load_mode must be 'lazy' or 'preload'.")
        self.episodes = list(episodes)
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds
        self.image_size = image_size
        self.stride = stride
        self.npz_load_mode = npz_load_mode
        self.max_cached_episodes = max(1, max_cached_episodes)
        self._episode_ids = {id(episode): episode_id for episode_id, episode in enumerate(self.episodes)}
        self._episode_lengths: dict[int, int] = {}
        # Keep the lazy cache at the raw uint8 frame level. Converting entire
        # episodes to float32 tensors here would waste a lot of RAM.
        self._preloaded_frames: OrderedDict[int, np.ndarray] = OrderedDict()
        for episode_id, episode in enumerate(self.episodes):
            if episode.frames_npz is not None:
                usable = min(int(episode.num_frames or len(episode.actions)), len(episode.actions))
                self._episode_lengths[episode_id] = usable
                if self.npz_load_mode == "preload":
                    self._materialize_npz_episode(episode_id)
            else:
                assert episode.frame_paths is not None
                self._episode_lengths[episode_id] = min(len(episode.frame_paths), len(episode.actions))
        self.index: list[tuple[int, int]] = []
        for episode_id, episode in enumerate(self.episodes):
            episode_len = self._episode_length(episode)
            max_start = episode_len - self.seq_len
            for start in range(0, max_start + 1, stride):
                self.index.append((episode_id, start))
        if not self.index:
            raise ValueError("No training windows were created. Check episode lengths.")
        npz_count = sum(1 for episode in self.episodes if episode.frames_npz is not None)
        print(
            f"dataset_init episodes={len(self.episodes)} npz_episodes={npz_count} "
            f"windows={len(self.index)} npz_load_mode={self.npz_load_mode} "
            f"max_cached_episodes={self.max_cached_episodes}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _episode_length(self, episode: MarioEpisode) -> int:
        episode_id = self._episode_ids[id(episode)]
        return self._episode_lengths[episode_id]

    def _load_frame(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor = TF.pil_to_tensor(image).float() / 255.0
        tensor = TF.resize(
            tensor,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return tensor

    def _load_npz_frame(self, episode: MarioEpisode, frame_idx: int) -> torch.Tensor:
        episode_id = self._episode_ids[id(episode)]
        frames = self._get_npz_frames(episode_id)
        return self._preprocess_npz_frames(frames[frame_idx : frame_idx + 1])[0]

    def _materialize_npz_episode(self, episode_id: int) -> np.ndarray:
        episode = self.episodes[episode_id]
        if episode.frames_npz is None:
            raise ValueError("Tried to materialize a non-npz episode.")
        usable = self._episode_lengths[episode_id]
        # print(
        #     f"load_npz_episode episode_id={episode_id + 1}/{len(self.episodes)} "
        #     f"name={episode.name} usable_frames={usable}"
        # )
        with np.load(episode.frames_npz, allow_pickle=False) as data:
            frames = np.asarray(data["frames"][:usable], dtype=np.uint8)
        return frames

    def _get_npz_frames(self, episode_id: int) -> np.ndarray:
        cached = self._preloaded_frames.get(episode_id)
        if cached is not None:
            self._preloaded_frames.move_to_end(episode_id)
            return cached
        frames = self._materialize_npz_episode(episode_id)
        self._preloaded_frames[episode_id] = frames
        self._preloaded_frames.move_to_end(episode_id)
        while len(self._preloaded_frames) > self.max_cached_episodes:
            self._preloaded_frames.popitem(last=False)
        return frames

    def _preprocess_npz_frames(self, frames: np.ndarray) -> torch.Tensor:
        # Apply the same normalization used for image-folder episodes after
        # converting uint8 RGB frames to float tensors.
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        if tensor.shape[-2] != self.image_size or tensor.shape[-1] != self.image_size:
            tensor = TF.resize(
                tensor,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def get_frame_tensor(self, episode: MarioEpisode, frame_idx: int) -> torch.Tensor:
        if episode.frames_npz is not None:
            return self._load_npz_frame(episode, frame_idx)
        assert episode.frame_paths is not None
        return self._load_frame(episode.frame_paths[frame_idx])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        episode_id, start = self.index[idx]
        episode = self.episodes[episode_id]
        end = start + self.seq_len
        if episode.frames_npz is not None:
            frames = self._get_npz_frames(episode_id)[start:end]
            pixels = self._preprocess_npz_frames(frames)
        else:
            assert episode.frame_paths is not None
            frame_paths = episode.frame_paths[start:end]
            pixels = torch.stack([self._load_frame(path) for path in frame_paths], dim=0)
        actions = torch.from_numpy(episode.actions[start:end]).float()
        return {
            "pixels": pixels,
            "action": actions,
            "episode_name": episode.name,
            "start_index": torch.tensor(start, dtype=torch.long),
        }
