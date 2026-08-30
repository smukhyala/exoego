"""Cache frozen per-frame backbone features for every segment.

One sequential decode per (recording, view). A 237s 1080p recording holds ~200
segments; seeking once per segment costs hours across the dataset, while a
single ordered walk of the file takes about a minute. The frames a video needs
are held as 224x224 uint8 (a few hundred MB for the largest recording here) and
encoded in batches of `--batch-segments`.

Output per role (ego/exo) and split role (train/eval):
  features/<backbone>_T<frames>/<role>_<view_role>.npy   (N, T, D) float32
  features/<backbone>_T<frames>/<role>_index.csv         row order -> segment_id
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego import video as video_mod
from exoego.encoders import FrozenEncoder
from exoego.paths import ensure_dirs, features_dir, manifests_dir, recordings_dir


def encode_video(encoder, video_path, segments, num_frames, batch_segments,
                 scale=1.0, max_index=None):
    """Encode every segment of one video in a single decode pass.

    Returns (segment_ids, features) with features shaped (n_segments, T, dim).
    """
    plan = []
    for row in segments.itertuples(index=False):
        indices = video_mod.sample_source_indices(
            row.start_frame, row.end_frame, num_frames, scale=scale, max_index=max_index
        )
        plan.append({"segment_id": row.segment_id, "indices": indices})

    needed = set()
    for item in plan:
        for index in item["indices"]:
            needed.add(int(index))
    needed_sorted = sorted(needed)

    frames_by_index = {}
    for index, frame in video_mod.iter_needed_frames(video_path, needed_sorted):
        frames_by_index[index] = frame

    if not frames_by_index:
        raise IOError(f"decoded no frames from {video_path}")

    last_available = max(frames_by_index)
    fallback = frames_by_index[last_available]

    segment_ids = []
    features = []
    buffer_frames = []
    buffer_ids = []

    def flush():
        if not buffer_ids:
            return
        stacked = np.stack(buffer_frames)
        if encoder.kind == "video":
            encoded = encoder.encode_clips(stacked)
        else:
            flat = stacked.reshape(-1, *stacked.shape[2:])
            encoded = encoder.encode(flat)
            encoded = encoded.reshape(len(buffer_ids), num_frames, -1)
        for position in range(len(buffer_ids)):
            segment_ids.append(buffer_ids[position])
            features.append(encoded[position])
        buffer_frames.clear()
        buffer_ids.clear()

    for item in plan:
        clip = []
        for index in item["indices"]:
            key = int(index)
            if key in frames_by_index:
                clip.append(frames_by_index[key])
            else:
                clip.append(fallback)
        buffer_frames.append(np.stack(clip))
        buffer_ids.append(item["segment_id"])
        if len(buffer_ids) >= batch_segments:
            flush()
    flush()

    return segment_ids, np.stack(features).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="dinov2s")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-segments", type=int, default=8,
                        help="segments encoded per forward pass (x frames = batch size)")
    parser.add_argument("--limit-recordings", type=int, default=None)
    parser.add_argument("--views", nargs="*", default=["ego", "exo"],
                        choices=["ego", "exo"],
                        help="extract only these views (ego alone is enough for the gate)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ensure_dirs()
    segments = pd.read_csv(manifests_dir() / "segments.csv")
    encoder = FrozenEncoder(args.backbone, device=args.device)
    print(f"backbone {args.backbone} (dim {encoder.dim}) on {encoder.device}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = features_dir() / f"{args.backbone}_T{args.frames}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for role in ["train", "eval"]:
        role_segments = segments[segments["role"] == role]
        sequences = list(dict.fromkeys(role_segments["seq"]))
        if args.limit_recordings:
            sequences = sequences[: args.limit_recordings]

        collected = {"ego": {}, "exo": {}}
        for position, seq in enumerate(sequences, start=1):
            group = role_segments[role_segments["seq"] == seq]

            paths = {}
            counts = {}
            for view_role in ["ego", "exo"]:
                view = group[f"{view_role}_view"].iloc[0]
                paths[view_role] = recordings_dir() / seq / f"{view}.mp4"
            if not paths["ego"].exists() or not paths["exo"].exists():
                print(f"  MISSING video for {seq} -- skipping")
                continue
            # Both counts are always needed: the rescale factor depends on the
            # pair, even when only one view is being extracted.
            for view_role in ["ego", "exo"]:
                counts[view_role] = video_mod.frame_count(paths[view_role])

            # Annotations follow the longer (canonical) timeline; the shorter
            # view dropped frames uniformly, so scale it down to match.
            reference = max(counts["ego"], counts["exo"])
            if counts["ego"] != counts["exo"]:
                print(f"  drift: ego={counts['ego']} exo={counts['exo']} "
                      f"delta={counts['ego'] - counts['exo']} -- rescaling")

            for view_role in args.views:
                path = paths[view_role]
                scale = video_mod.timeline_scale(counts[view_role], reference)

                started = time.time()
                ids, feats = encode_video(encoder, path, group, args.frames,
                                          args.batch_segments, scale=scale,
                                          max_index=counts[view_role] - 1)
                collected[view_role][seq] = (ids, feats)
                print(f"[{role} {position}/{len(sequences)}] {view_role} {seq[:38]}... "
                      f"{len(ids)} segs in {time.time() - started:.0f}s", flush=True)

        primary = args.views[0]
        shared = [seq for seq in sequences
                  if all(seq in collected[view_role] for view_role in args.views)]

        index_ids = []
        for seq in shared:
            index_ids.extend(collected[primary][seq][0])

        for view_role in args.views:
            ordered = []
            for seq in shared:
                ids, feats = collected[view_role][seq]
                lookup = {}
                for position, segment_id in enumerate(ids):
                    lookup[segment_id] = feats[position]
                for segment_id in collected[primary][seq][0]:
                    ordered.append(lookup[segment_id])
            stacked = np.stack(ordered).astype(np.float32)
            np.save(out_dir / f"{role}_{view_role}.npy", stacked)
            print(f"  wrote {role}_{view_role}.npy {stacked.shape}")

        index = segments.set_index("segment_id").loc[index_ids].reset_index()
        index.to_csv(out_dir / f"{role}_index.csv", index=False)
        print(f"  wrote {role}_index.csv ({len(index)} rows)")


if __name__ == "__main__":
    main()
