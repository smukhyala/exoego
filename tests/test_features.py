"""Cached feature integrity."""

import numpy as np
import pandas as pd
import pytest
import yaml

from exoego.encoders import BACKBONES
from exoego.paths import features_dir, repo_root

with open(repo_root() / "configs" / "base.yaml") as handle:
    BASE = yaml.safe_load(handle)

FEATURE_DIR = features_dir() / f"{BASE['backbone']}_T{BASE['frames']}"

# Image backbones emit one embedding per sampled frame; video backbones emit a
# fixed number of temporal tokens (VideoMAE: 8 tokens from 16 input frames).
_SPEC = BACKBONES[BASE["backbone"]]
EXPECTED_TOKENS = _SPEC["tokens"] if _SPEC.get("kind") == "video" else BASE["frames"]
pytestmark = pytest.mark.skipif(not FEATURE_DIR.exists(),
                                reason="run 04_extract_features.py first")

ROLES = ["train", "eval"]


@pytest.mark.parametrize("role", ROLES)
def test_shapes_agree_with_index(role):
    index = pd.read_csv(FEATURE_DIR / f"{role}_index.csv")
    for view_role in ["ego", "exo"]:
        features = np.load(FEATURE_DIR / f"{role}_{view_role}.npy")
        assert features.shape[0] == len(index)
        assert features.shape[1] == EXPECTED_TOKENS
        assert features.shape[2] == _SPEC["dim"]
        assert features.ndim == 3


@pytest.mark.parametrize("role", ROLES)
def test_features_are_finite(role):
    for view_role in ["ego", "exo"]:
        features = np.load(FEATURE_DIR / f"{role}_{view_role}.npy")
        assert np.isfinite(features).all()
        assert features.std() > 0


@pytest.mark.parametrize("role", ROLES)
def test_ego_and_exo_are_distinct(role):
    """A silent bug that pointed both views at the same file would otherwise
    make the cross-view objective trivially satisfiable."""
    ego = np.load(FEATURE_DIR / f"{role}_ego.npy")
    exo = np.load(FEATURE_DIR / f"{role}_exo.npy")
    assert not np.allclose(ego, exo)


@pytest.mark.parametrize("role", ROLES)
def test_index_segment_ids_unique(role):
    index = pd.read_csv(FEATURE_DIR / f"{role}_index.csv")
    assert index["segment_id"].is_unique
