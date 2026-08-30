"""Small-data checks for the paired-transfer experiment."""

import numpy as np

from src.paired_bridge import (
    mean_per_class,
    pool_segments,
    segment_rows,
    shifted_within_recording,
)


def test_segment_rows_break_on_label_or_recording_change():
    labels = np.array([1, 1, 2, 2, 2, 2])
    sequences = np.array([7, 7, 7, 8, 8, 8])
    assert segment_rows(labels, sequences).tolist() == [0, 0, 1, 2, 2, 2]


def test_pool_segments_means_all_frames_in_a_run():
    features = np.array([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]], dtype=np.float16)
    segment_ids = np.array([0, 0, 1])
    pooled = pool_segments(features, segment_ids, n_segments=2, chunk=1)
    np.testing.assert_allclose(pooled, [[2.0, 4.0], [10.0, 20.0]])


def test_shifted_control_never_moves_features_between_recordings():
    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    sequences = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shifted = shifted_within_recording(features, sequences)
    assert set(shifted[:4, 0]) == set(features[:4, 0])
    assert set(shifted[4:, 0]) == set(features[4:, 0])
    assert not np.array_equal(shifted, features)


def test_mean_per_class_does_not_reward_majority_frequency():
    target = np.array([0, 0, 0, 1])
    prediction = np.array([0, 0, 0, 0])
    assert mean_per_class(target, prediction) == 0.5
