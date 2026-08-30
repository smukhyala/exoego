"""Assembly101 access: coarse action labels, splits, and TSM features.

Features are 2048-D per frame, stored one LMDB per camera, keyed as
    {sequence}/{view}/{view}_{frame:010d}.jpg
with frame numbers at 30 fps. Coarse labels use the same 30 fps indexing, so
labels and features share a clock and need no resampling.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np

ROOT = Path("data/assembly101")
ANN = ROOT / "annotations" / "coarse-annotations"
FEAT = ROOT / "TSM_features"

EGO_VIEW = "HMC_84346135_mono10bit"
EXO_VIEW = "C10379_rgb"


@functools.lru_cache(maxsize=1)
def action_vocab() -> dict[str, int]:
    """action_cls string -> contiguous id."""
    vocab = {}
    with open(ANN / "actions.csv") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 4:
                vocab[parts[3].strip()] = int(parts[0])
    return vocab


@functools.lru_cache(maxsize=1)
def sequence_views() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in (ANN / "coarse_seq_views.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seq, view = line.rsplit("/", 1)
        out.setdefault(seq, set()).add(view.replace(".mp4", ""))
    return out


def sequences_with(views: list[str]) -> list[str]:
    """Sequences that have every one of `views`, sorted for determinism."""
    sv = sequence_views()
    return sorted(s for s, v in sv.items() if all(w in v for w in views))


def split_sequences(split: str) -> set[str]:
    """split in {train, val, test}; unions assembly + disassembly."""
    out = set()
    for kind in ("assembly", "disassembly"):
        p = ANN / "coarse_splits" / f"{split}_coarse_{kind}.txt"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            name = line.split("\t")[0].strip()
            if name.endswith(".txt"):
                name = name[:-4]
            # label files are prefixed with assembly_/disassembly_, view dirs are not
            for pre in ("assembly_", "disassembly_"):
                if name.startswith(pre):
                    name = name[len(pre):]
                    break
            if name:
                out.add(name)
    return out


@dataclass
class Segment:
    start: int
    end: int
    action: int


def segments(seq: str) -> list[Segment]:
    """Coarse action segments for a sequence, from either label-file prefix."""
    vocab = action_vocab()
    for pre in ("assembly_", "disassembly_"):
        p = ANN / "coarse_labels" / f"{pre}{seq}.txt"
        if p.exists():
            out = []
            for line in p.read_text().splitlines():
                parts = [x for x in line.split("\t") if x.strip()]
                if len(parts) < 3:
                    continue
                a = vocab.get(parts[2].strip())
                if a is not None:
                    out.append(Segment(int(parts[0]), int(parts[1]), a))
            return out
    return []


class FeatureStore:
    """Read-only LMDB feature reader for one camera view."""

    def __init__(self, view: str):
        self.view = view
        path = FEAT / view
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — unzip TSM_features/{view}.zip into {FEAT}/"
            )
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)

    def key(self, seq: str, frame: int) -> str:
        return f"{seq}/{self.view}/{self.view}_{frame:010d}.jpg"

    def get(self, seq: str, frame: int) -> np.ndarray | None:
        with self.env.begin() as t:
            raw = t.get(self.key(seq, frame).encode())
        if raw is None:
            return None
        a = np.frombuffer(raw, dtype=np.float32)
        return a if a.shape[0] == 2048 else None

    def get_many(self, seq: str, frames) -> tuple[np.ndarray, np.ndarray]:
        """Returns (features Nx2048, the frame numbers that actually existed)."""
        feats, got = [], []
        with self.env.begin() as t:
            for f in frames:
                raw = t.get(self.key(seq, f).encode())
                if raw is None:
                    continue
                a = np.frombuffer(raw, dtype=np.float32)
                if a.shape[0] == 2048:
                    feats.append(a)
                    got.append(f)
        if not feats:
            return np.zeros((0, 2048), np.float32), np.zeros(0, np.int64)
        return np.stack(feats), np.asarray(got, dtype=np.int64)
