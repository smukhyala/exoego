"""Manifest integrity and leakage checks."""

import pandas as pd
import pytest

from exoego.annotations import make_segment_id, verb_vocab
from exoego.paths import manifests_dir
from exoego.views import is_ego, is_exo

SEGMENTS_PATH = manifests_dir() / "segments.csv"
pytestmark = pytest.mark.skipif(not SEGMENTS_PATH.exists(),
                                reason="run 01_find_recordings.py first")


@pytest.fixture(scope="module")
def segments():
    return pd.read_csv(SEGMENTS_PATH)


def test_segment_ids_are_unique(segments):
    assert segments["segment_id"].is_unique


def test_segment_ids_match_their_fields(segments):
    for row in segments.head(200).itertuples(index=False):
        expected = make_segment_id(row.seq, row.start_frame, row.end_frame, row.action_id)
        assert row.segment_id == expected


def test_views_are_the_right_kind(segments):
    assert segments["ego_view"].map(is_ego).all()
    assert segments["exo_view"].map(is_exo).all()


def test_verbs_are_in_vocabulary(segments):
    assert set(segments["verb_cls"]).issubset(set(verb_vocab()))


def test_train_and_eval_recordings_are_disjoint(segments):
    train = set(segments[segments["role"] == "train"]["seq"])
    evaluation = set(segments[segments["role"] == "eval"]["seq"])
    assert train and evaluation
    assert train.isdisjoint(evaluation)


def test_train_and_eval_toys_are_disjoint(segments):
    """Official train/validation splits differ by toy, guarding against a model
    that memorises a specific toy rather than the verb."""
    train = set(segments[segments["role"] == "train"]["toy_id"])
    evaluation = set(segments[segments["role"] == "eval"]["toy_id"])
    assert train.isdisjoint(evaluation)


def test_frame_ranges_are_positive(segments):
    assert (segments["end_frame"] > segments["start_frame"]).all()
    assert (segments["dur_s"] > 0).all()
