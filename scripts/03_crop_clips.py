"""Crop synchronised ego/exo clips for annotated action segments.

Clips exist for inspection and demos. The ML path does NOT read them --
04_extract_features.py decodes each source video once instead -- so this script
defaults to a small subset. Use --all for the full set.

Two choices worth noting:
  * Output is downscaled to short-side 256 at 30fps. A 7.2s 1080p exo clip is
    24MB at source resolution; across ~4,300 segments that is ~25GB for footage
    we only ever feed to the encoder at 224px.
  * Seeking uses -ss before -i, which is fast and (with ffmpeg's default
    accurate_seek) frame-accurate when re-encoding. Both views get the same
    -ss, so any residual seek error is identical in ego and exo.
"""

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego.annotations import ANNOTATION_FPS
from exoego.paths import manifests_dir, recordings_dir
from exoego.video import frame_count, timeline_scale

SHORT_SIDE = 256
OUTPUT_FPS = 30


def build_job(row, view_role, scale=1.0):
    """One ffmpeg crop. `scale` corrects a view that dropped frames uniformly."""
    view = getattr(row, f"{view_role}_view")
    source = recordings_dir() / row.seq / f"{view}.mp4"
    target = Path(getattr(row, f"{view_role}_clip_path"))
    start_s = row.start_frame / ANNOTATION_FPS * scale
    duration_s = (row.end_frame - row.start_frame) / ANNOTATION_FPS * scale
    return {"source": source, "target": target, "start": start_s, "duration": duration_s}


def timeline_scales(segments):
    """Per-recording {seq: {"ego": scale, "exo": scale}}, from actual frame counts."""
    scales = {}
    for seq, group in segments.groupby("seq", sort=False):
        row = group.iloc[0]
        counts = {}
        for view_role in ["ego", "exo"]:
            path = recordings_dir() / seq / f"{row[f'{view_role}_view']}.mp4"
            counts[view_role] = frame_count(path) if path.exists() else 0
        reference = max(counts["ego"], counts["exo"])
        scales[seq] = {
            "ego": timeline_scale(counts["ego"], reference),
            "exo": timeline_scale(counts["exo"], reference),
        }
    return scales


def run_job(job) -> str:
    target = job["target"]
    if target.exists() and target.stat().st_size > 0:
        return "skipped"
    if not job["source"].exists():
        return "missing-source"

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{job['start']:.6f}",
        "-t", f"{job['duration']:.6f}",
        "-i", str(job["source"]),
        "-r", str(OUTPUT_FPS),
        "-vf", f"scale=-2:{SHORT_SIDE}",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-an",
        str(target),
    ]
    outcome = subprocess.run(command, capture_output=True)
    if outcome.returncode != 0:
        return "failed"
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100,
                        help="number of segment pairs to crop (ignored with --all)")
    parser.add_argument("--all", action="store_true", help="crop every segment")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--role", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    segments = pd.read_csv(manifests_dir() / "segments.csv")
    if args.role != "both":
        segments = segments[segments["role"] == args.role]
    if not args.all:
        segments = segments.head(args.limit)

    scales = timeline_scales(segments)
    jobs = []
    for row in segments.itertuples(index=False):
        for view_role in ["ego", "exo"]:
            jobs.append(build_job(row, view_role, scales[row.seq][view_role]))

    print(f"{len(segments)} segment pairs -> {len(jobs)} clips, {args.workers} workers")
    tally = {}
    with Pool(args.workers) as pool:
        for position, outcome in enumerate(pool.imap_unordered(run_job, jobs, chunksize=4), 1):
            tally[outcome] = tally.get(outcome, 0) + 1
            if position % 100 == 0:
                print(f"  {position}/{len(jobs)} {tally}", flush=True)

    print(f"done: {tally}")


if __name__ == "__main__":
    main()
