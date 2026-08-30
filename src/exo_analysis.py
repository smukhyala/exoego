"""Ego vs exo on the synchronized World Context pair — view-agnostic measures.

A first pass asked YOLO "can you see the scooter" in each view and reported that
exo saw it in 2.0% of frames against ego's 19.9%. That was an ARTIFACT, not a
finding: a COCO detector has never seen a partly-assembled scooter from directly
overhead, so it scores the work object at 0.08-0.22 confidence in exo while
scoring it 0.53-0.75 in ego. The same bias suppressed exo person counts.

The methodological point that survives: a single confidence threshold is not
comparable across two radically different viewpoints. So this module does two
things instead.

1. VIEW-AGNOSTIC measures, which need no semantics and cannot be biased by
   training distribution:
     - motion blur (variance of Laplacian)
     - camera instability (frame-to-frame difference)
   These are the mechanism. The ego camera rides a moving head; the exo camera is
   bolted to the ceiling.

2. A THRESHOLD SWEEP for person visibility, reporting the whole curve rather than
   one arbitrary operating point, so no single threshold drives the conclusion.

Usage:  python -m src.exo_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PAIRS = Path("/private/tmp/claude-501/-Users-evan-Projects-exoego/"
             "a0c6b8fb-f59d-4405-8f14-34aa52a4c40a/scratchpad/pairs")
RESULTS = Path("results/exo_analysis.json")
FPS = 2.0
OFFSET_S = 11.16
THRESHOLDS = [0.05, 0.10, 0.20, 0.35, 0.50]


def image_stats(files):
    """Blur and frame-to-frame change. No semantics, so no viewpoint bias."""
    blur, diff = [], []
    prev = None
    for p in files:
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 180))
        blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        if prev is not None:
            diff.append(float(np.abs(g.astype(np.float32) - prev).mean()))
        prev = g.astype(np.float32)
    return np.array(blur), np.array(diff)


def person_sweep(model, files, device, batch=32):
    """Person confidences per frame, so any threshold can be applied after."""
    per_frame = []
    for i in range(0, len(files), batch):
        chunk = [str(p) for p in files[i:i + batch]]
        res = model.predict(chunk, verbose=False, device=device, conf=0.02)
        for r in res:
            confs = [float(cf) for c, cf in zip(r.boxes.cls, r.boxes.conf)
                     if model.names[int(c)] == "person"]
            per_frame.append(confs)
    return per_frame


def main() -> int:
    from ultralytics import YOLO
    import torch

    ego_f = sorted((PAIRS / "ego").glob("*.jpg"))
    exo_f = sorted((PAIRS / "exo").glob("*.jpg"))
    n = min(len(ego_f), len(exo_f))
    ego_f, exo_f = ego_f[:n], exo_f[:n]
    print(f"synchronized pairs: {n} ({n/FPS/60:.1f} min @ {FPS} fps, "
          f"offset +{OFFSET_S}s)\n")

    print("computing view-agnostic image statistics...")
    eb, ed = image_stats(ego_f)
    xb, xd = image_stats(exo_f)

    print("\n" + "=" * 70)
    print("CAMERA INSTABILITY — mean frame-to-frame pixel change (0-255)")
    print("=" * 70)
    print(f"  ego  median {np.median(ed):7.2f}   p90 {np.percentile(ed,90):7.2f}")
    print(f"  exo  median {np.median(xd):7.2f}   p90 {np.percentile(xd,90):7.2f}")
    print(f"  ratio of medians (ego/exo): {np.median(ed)/max(np.median(xd),1e-9):.2f}x")
    print("  The exo camera is fixed; the ego camera rides a head. This is why")
    print("  the ego view keeps losing the workspace.")

    print("\n" + "=" * 70)
    print("MOTION BLUR — variance of Laplacian (lower = blurrier)")
    print("=" * 70)
    print(f"  ego  median {np.median(eb):8.1f}   p10 {np.percentile(eb,10):8.1f}")
    print(f"  exo  median {np.median(xb):8.1f}   p10 {np.percentile(xb,10):8.1f}")
    thr = float(np.percentile(xb, 10))
    print(f"  ego frames blurrier than exo's 10th percentile: {(eb < thr).mean():6.1%}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11n.pt")
    print(f"\nrunning person detection sweep on {device}...")
    ep = person_sweep(model, ego_f, device)
    xp = person_sweep(model, exo_f, device)

    print("\n" + "=" * 70)
    print("PERSON VISIBILITY vs CONFIDENCE THRESHOLD")
    print("(reported as a curve because one threshold is not comparable")
    print(" across viewpoints — that is exactly what broke the first pass)")
    print("=" * 70)
    print(f"  {'thresh':>7}{'ego mean':>11}{'exo mean':>11}{'exo/ego':>10}"
          f"{'exo>ego':>10}")
    print("  " + "-" * 47)
    sweep = {}
    for t in THRESHOLDS:
        e = np.array([sum(1 for c in f if c >= t) for f in ep])
        x = np.array([sum(1 for c in f if c >= t) for f in xp])
        ratio = x.mean() / max(e.mean(), 1e-9)
        sweep[str(t)] = {"ego_mean": float(e.mean()), "exo_mean": float(x.mean()),
                         "ratio": float(ratio), "exo_gt_ego": float((x > e).mean())}
        print(f"  {t:>7.2f}{e.mean():>11.2f}{x.mean():>11.2f}{ratio:>10.2f}"
              f"{(x>e).mean():>10.1%}")

    print("\n  Read this honestly: the ratio moves with threshold, so 'exo sees")
    print("  more people' is threshold-dependent and should be quoted as a range.")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_pairs": int(n), "fps": FPS, "offset_s": OFFSET_S,
        "instability_median_ego": float(np.median(ed)),
        "instability_median_exo": float(np.median(xd)),
        "instability_ratio": float(np.median(ed) / max(np.median(xd), 1e-9)),
        "blur_median_ego": float(np.median(eb)), "blur_median_exo": float(np.median(xb)),
        "ego_blurrier_than_exo_p10": float((eb < thr).mean()),
        "person_sweep": sweep,
        "note": "First-pass work-object result was a COCO domain artifact; see docstring.",
    }, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
