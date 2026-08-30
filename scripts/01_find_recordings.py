"""Select recordings that contain the actions of interest and have both views.

Writes two manifests:
  manifests/recordings.csv  -- one row per selected recording
  manifests/segments.csv    -- one row per action segment, ego and exo paired

Recordings for training come from the official `train` split and recordings for
evaluation from the official `validation` split, so the two sets are disjoint by
recording *and* by toy -- no leakage, for free.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import get_hf_file_metadata, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego import annotations as anno
from exoego.paths import clips_dir, ensure_dirs, manifests_dir
from exoego.views import EXO_VIEWS

REPO_ID = "cvml-nus/assembly101"


def rank_recordings(segments: pd.DataFrame, verbs, rank_by: str) -> pd.DataFrame:
    """Rank candidate recordings.

    `density` (segments per minute of video) is the default because download
    cost scales with video duration, not with segment count. Density ranges from
    15 to 45 segments/min across Assembly101, so preferring dense recordings
    buys the same number of training segments for roughly a third fewer bytes.
    """
    pool = segments
    if verbs:
        pool = pool[pool["verb_cls"].isin(verbs)]

    rows = []
    for seq, group in pool.groupby("seq", sort=False):
        duration_min = group["end_frame"].max() / anno.ANNOTATION_FPS / 60.0
        rows.append({
            "seq": seq,
            "split": group["split"].iloc[0],
            "n_segments": len(group),
            "n_verbs": group["verb_cls"].nunique(),
            "duration_min": round(duration_min, 2),
            "density": round(len(group) / duration_min, 1),
            "ego_view": group["ego_view"].iloc[0],
            "exo_view": group["exo_view"].iloc[0],
            "toy_ids": "|".join(sorted(group["toy_id"].astype(str).unique())),
        })
    ranked = pd.DataFrame(rows)

    if rank_by == "density":
        order = ["density", "n_verbs"]
    else:
        order = ["n_verbs", "n_segments"]
    return ranked.sort_values(order, ascending=False).reset_index(drop=True)


def smallest_exo_view(seq: str) -> tuple:
    """The cheapest exo camera for one recording.

    Exo views of the same recording range from 1.6GB to 4.0GB -- a 2.4x spread
    for footage that is scientifically interchangeable (any fixed third-person
    view serves as the exo pair), so this is free savings.
    """
    best_view = None
    best_size = None
    for view in EXO_VIEWS:
        url = hf_hub_url(repo_id=REPO_ID, filename=f"recordings/{seq}/{view}.mp4",
                         repo_type="dataset")
        try:
            size = get_hf_file_metadata(url).size
        except Exception:
            continue
        if size is None:
            continue
        if best_size is None or size < best_size:
            best_view = view
            best_size = size
    return best_view, best_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=14, help="recordings from the train split")
    parser.add_argument("--n-eval", type=int, default=6, help="recordings from the validation split")
    parser.add_argument("--verbs", nargs="*", default=None, help="restrict to these verb classes")
    parser.add_argument("--min-duration", type=float, default=0.4,
                        help="drop segments shorter than this (seconds)")
    parser.add_argument("--rank-by", choices=["density", "diversity"], default="density",
                        help="density favours recordings with more segments per GB")
    parser.add_argument("--optimize-exo-size", action="store_true",
                        help="query HF and pick the smallest exo view per recording")
    args = parser.parse_args()

    ensure_dirs()

    selected_frames = []
    recording_frames = []
    plan = [("train", "train", args.n_train), ("validation", "eval", args.n_eval)]

    for split, role, count in plan:
        print(f"scanning official {split} split ...", flush=True)
        segments = anno.segments(split, min_duration_s=args.min_duration)
        ranked = rank_recordings(segments, args.verbs, args.rank_by)
        chosen = ranked.head(count).copy()
        chosen["role"] = role

        if args.optimize_exo_size:
            print("  choosing smallest exo view per recording ...", flush=True)
            picked_views = []
            picked_sizes = []
            for seq in chosen["seq"]:
                view, size = smallest_exo_view(seq)
                picked_views.append(view)
                picked_sizes.append(size)
            chosen["exo_view"] = picked_views
            chosen["exo_bytes"] = picked_sizes

        recording_frames.append(chosen)

        exo_by_seq = dict(zip(chosen["seq"], chosen["exo_view"]))
        keep = segments[segments["seq"].isin(set(chosen["seq"]))].copy()
        keep["exo_view"] = keep["seq"].map(exo_by_seq)
        keep["role"] = role
        selected_frames.append(keep)
        print(f"  {len(ranked)} eligible recordings -> selected {len(chosen)} "
              f"({len(keep)} segments)")

    recordings = pd.concat(recording_frames, ignore_index=True)
    segments = pd.concat(selected_frames, ignore_index=True)

    clip_root = clips_dir()
    ego_paths = []
    exo_paths = []
    for row in segments.itertuples(index=False):
        stem = f"{row.start_frame:09d}_{row.end_frame:09d}"
        ego_paths.append(str(clip_root / row.seq / f"{stem}_ego.mp4"))
        exo_paths.append(str(clip_root / row.seq / f"{stem}_exo.mp4"))
    segments["ego_clip_path"] = ego_paths
    segments["exo_clip_path"] = exo_paths

    recordings_out = manifests_dir() / "recordings.csv"
    segments_out = manifests_dir() / "segments.csv"
    recordings.to_csv(recordings_out, index=False)
    segments.to_csv(segments_out, index=False)

    print()
    print(f"recordings -> {recordings_out}  ({len(recordings)} rows)")
    print(f"segments   -> {segments_out}  ({len(segments)} rows)")
    print()
    missing = set(anno.verb_vocab()) - set(segments["verb_cls"])
    if missing:
        print(f"WARNING: verbs absent from the selection: {sorted(missing)}")

    print(segments.groupby("role").agg(
        recordings=("seq", "nunique"),
        segments=("segment_id", "count"),
        verbs=("verb_cls", "nunique"),
    ).to_string())


if __name__ == "__main__":
    main()
