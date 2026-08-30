"""Ego/exo synchronisation -- the assumption the whole method rests on.

Assembly101 annotates each segment once per view with identical frame numbers.
That is only usable if the ego and exo files are genuinely frame-aligned, so
this asserts it directly rather than trusting the dataset docs.
"""

import subprocess

import pandas as pd
import pytest

from exoego.paths import manifests_dir, recordings_dir

MANIFEST = manifests_dir() / "recordings.csv"


def probe(path, field):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", f"stream={field}", "-of", "csv=p=0", str(path),
    ]
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def downloaded_pairs():
    if not MANIFEST.exists():
        return []
    recordings = pd.read_csv(MANIFEST)
    pairs = []
    for row in recordings.itertuples(index=False):
        ego = recordings_dir() / row.seq / f"{row.ego_view}.mp4"
        exo = recordings_dir() / row.seq / f"{row.exo_view}.mp4"
        if ego.exists() and exo.exists():
            pairs.append(pytest.param(ego, exo, id=row.seq[:32]))
    return pairs


PAIRS = downloaded_pairs()


MAX_DRIFT_FRACTION = 0.005


@pytest.mark.skipif(not PAIRS, reason="no recordings downloaded yet")
@pytest.mark.parametrize("ego,exo", PAIRS)
def test_ego_exo_frame_drift_is_small_and_correctable(ego, exo):
    """Frame counts may differ slightly; the drift must stay small enough that
    the linear rescale in video.timeline_scale can absorb it.

    Some views drop frames uniformly through a recording (up to ~70 frames over
    28,000). That is cumulative drift, not trailing truncation, so it is
    corrected by rescaling rather than ignored. A large delta would mean the two
    files are not the same session and must not be paired.
    """
    ego_frames = int(probe(ego, "nb_frames"))
    exo_frames = int(probe(exo, "nb_frames"))
    reference = max(ego_frames, exo_frames)
    drift = abs(ego_frames - exo_frames) / reference
    assert drift < MAX_DRIFT_FRACTION, (
        f"{drift:.4%} drift ({ego_frames} vs {exo_frames}) is too large to rescale"
    )


@pytest.mark.skipif(not PAIRS, reason="no recordings downloaded yet")
@pytest.mark.parametrize("ego,exo", PAIRS)
def test_rescaled_timelines_agree_at_the_end(ego, exo):
    """After rescaling, the last annotated frame maps to within a frame or two
    of the same relative position in both views."""
    from exoego.video import timeline_scale

    ego_frames = int(probe(ego, "nb_frames"))
    exo_frames = int(probe(exo, "nb_frames"))
    reference = max(ego_frames, exo_frames)

    ego_end = (reference - 1) * timeline_scale(ego_frames, reference)
    exo_end = (reference - 1) * timeline_scale(exo_frames, reference)
    assert abs(ego_end - (ego_frames - 1)) <= 2
    assert abs(exo_end - (exo_frames - 1)) <= 2


@pytest.mark.skipif(not PAIRS, reason="no recordings downloaded yet")
@pytest.mark.parametrize("ego,exo", PAIRS)
def test_source_is_60fps(ego, exo):
    # Annotations are at 30fps; the 2x mapping in annotations.FRAME_SCALE is
    # only valid if the source really is 60fps.
    assert probe(ego, "r_frame_rate") == "60/1"
    assert probe(exo, "r_frame_rate") == "60/1"
