"""Cached-feature loading, label budgets, and temporal augmentation."""

import numpy as np
import pandas as pd
import torch

from .annotations import verb_vocab


class FeatureStore:
    """Frozen features for one role (train/eval), with ego and exo aligned by row."""

    def __init__(self, feature_dir, role: str, vocab=None):
        self.role = role
        self.index = pd.read_csv(feature_dir / f"{role}_index.csv")
        self.ego = np.load(feature_dir / f"{role}_ego.npy")
        self.exo = np.load(feature_dir / f"{role}_exo.npy")

        if self.ego.shape[0] != len(self.index) or self.exo.shape[0] != len(self.index):
            raise ValueError(
                f"{role}: feature rows {self.ego.shape[0]}/{self.exo.shape[0]} "
                f"do not match index rows {len(self.index)}"
            )

        self.vocab = vocab if vocab is not None else verb_vocab()
        lookup = {}
        for position, verb in enumerate(self.vocab):
            lookup[verb] = position

        labels = []
        for verb in self.index["verb_cls"]:
            labels.append(lookup[verb])
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def dim(self) -> int:
        return self.ego.shape[-1]

    @property
    def num_frames(self) -> int:
        return self.ego.shape[1]

    def tensors(self, device):
        ego = torch.from_numpy(self.ego).to(device)
        exo = torch.from_numpy(self.exo).to(device)
        labels = torch.from_numpy(self.labels).to(device)
        return ego, exo, labels


def stratified_subset(labels: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified sample of `budget` row indices.

    Allocates as evenly as possible across classes present, then fills any
    remainder uniformly at random from what is left.
    """
    total = len(labels)
    if budget >= total:
        return np.arange(total)

    classes = np.unique(labels)
    per_class = max(1, budget // len(classes))

    chosen = []
    for label in classes:
        pool = np.where(labels == label)[0]
        take = min(per_class, len(pool))
        picked = rng.choice(pool, size=take, replace=False)
        chosen.extend(picked.tolist())

    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > budget:
        chosen = rng.choice(np.asarray(chosen), size=budget, replace=False).tolist()
    elif len(chosen) < budget:
        remaining = np.setdiff1d(np.arange(total), np.asarray(chosen))
        extra = rng.choice(remaining, size=budget - len(chosen), replace=False)
        chosen.extend(extra.tolist())

    return np.asarray(sorted(chosen))


def temporal_crop(features: torch.Tensor, generator: torch.Generator,
                  min_scale: float = 0.5) -> torch.Tensor:
    """Random contiguous temporal sub-window, resampled back to T frames.

    Used to build the second view for the `ego_ego` control, so that control
    matches the shape and cost of the cross-view objective without carrying any
    exocentric information.
    """
    batch_size, num_frames, _ = features.shape
    device = features.device

    scale = min_scale + (1.0 - min_scale) * torch.rand(
        batch_size, device=device, generator=generator
    )
    length = torch.clamp((scale * num_frames).long(), min=2)
    max_start = (num_frames - length).clamp(min=0)
    start = (torch.rand(batch_size, device=device, generator=generator) *
             (max_start + 1).float()).long()

    steps = torch.linspace(0.0, 1.0, num_frames, device=device).unsqueeze(0)
    positions = start.unsqueeze(1).float() + steps * (length - 1).unsqueeze(1).float()
    positions = positions.round().long().clamp(max=num_frames - 1)

    gather_index = positions.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return torch.gather(features, 1, gather_index)
