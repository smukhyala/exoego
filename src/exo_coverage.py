"""What does the exocentric view recover that the egocentric view loses?

Runs over the synchronized World Context ego/exo pair (offset +11.16 s, recovered
by onset cross-correlation and confirmed visually). For each synchronized frame
pair it asks both cameras the same question: can you see the work?

"The work" on this floor is two things a detector can name:
  * the WORK OBJECT  - the scooter under assembly (COCO motorcycle/bicycle)
  * the WORKERS      - how many people are in view

The headline is conditional, not marginal:

    P(exo sees the work object | ego does NOT)

Marginal rates invite "your ego camera was badly placed." The conditional
measures recovery: of the moments ego loses the task, how many does exo hold?

MediaPipe was the first choice for hands, but mediapipe 1.0.1 aborts on this
macOS build inside the Metal helper regardless of the CPU delegate flag. YOLO is
more robust here and detects the work object too, which is the better signal on
an assembly floor anyway.

Usage:  python -m src.exo_coverage [--limit N] [--batch 32]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PAIRS = Path("/private/tmp/claude-501/-Users-evan-Projects-exoego/"
             "a0c6b8fb-f59d-4405-8f14-34aa52a4c40a/scratchpad/pairs")
RESULTS = Path("results/exo_coverage.json")
FPS = 2.0
OFFSET_S = 11.16
WORK_OBJECT = {"motorcycle", "bicycle"}
CONF = 0.35


def run_view(model, files, batch, device, tag):
    n_person, n_work, work_area, blur = [], [], [], []
    t0 = time.time()
    for i in range(0, len(files), batch):
        chunk = [str(p) for p in files[i:i + batch]]
        res = model.predict(chunk, verbose=False, device=device, conf=CONF)
        for path, r in zip(chunk, res):
            names = [model.names[int(c)] for c in r.boxes.cls]
            npers = sum(1 for x in names if x == "person")
            wk = [j for j, x in enumerate(names) if x in WORK_OBJECT]
            n_person.append(npers)
            n_work.append(len(wk))
            if wk:
                xyxy = r.boxes.xyxy.cpu().numpy()
                h, w = r.orig_shape
                areas = [(xyxy[j][2] - xyxy[j][0]) * (xyxy[j][3] - xyxy[j][1]) / (w * h)
                         for j in wk]
                work_area.append(float(max(areas)))
            else:
                work_area.append(0.0)
            g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        if (i // batch) % 10 == 0:
            done = min(i + batch, len(files))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {tag} {done}/{len(files)}  {rate:.1f} fps  "
                  f"eta {(len(files)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
    return (np.array(n_person), np.array(n_work),
            np.array(work_area), np.array(blur))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    ego_f = sorted((PAIRS / "ego").glob("*.jpg"))
    exo_f = sorted((PAIRS / "exo").glob("*.jpg"))
    n = min(len(ego_f), len(exo_f))
    if args.limit:
        n = min(n, args.limit)
    ego_f, exo_f = ego_f[:n], exo_f[:n]
    print(f"synchronized pairs: {n} ({n/FPS/60:.1f} min @ {FPS} fps, offset +{OFFSET_S}s)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11n.pt")
    print(f"device={device}\n")

    ep, ew, ea, eb = run_view(model, ego_f, args.batch, device, "ego")
    xp, xw, xa, xb = run_view(model, exo_f, args.batch, device, "exo")

    ews, xws = ew > 0, xw > 0      # work object visible
    N = len(ews)

    print("\n" + "=" * 68)
    print("WORK-OBJECT VISIBILITY  (the scooter under assembly)")
    print("=" * 68)
    print(f"  frames                       : {N}")
    print(f"  ego sees work object         : {ews.mean():6.1%}")
    print(f"  exo sees work object         : {xws.mean():6.1%}")
    print(f"  both                         : {(ews & xws).mean():6.1%}")
    print(f"  neither                      : {(~ews & ~xws).mean():6.1%}")

    lost = ~ews
    rec = float(xws[lost].mean()) if lost.sum() else float("nan")
    print(f"\n  ego LOSES the work object in : {lost.mean():6.1%} "
          f"({lost.sum()} frames = {lost.sum()/FPS/60:.1f} min)")
    print(f"  ...exo still sees it in      : {rec:6.1%}   <-- RECOVERY")
    print(f"  exo-only coverage            : {(~ews & xws).mean():6.1%} of all frames")
    print(f"  ego-only coverage            : {(ews & ~xws).mean():6.1%} of all frames")

    print("\n" + "=" * 68)
    print("WORKER COVERAGE  (people visible per frame)")
    print("=" * 68)
    print(f"  ego  mean {ep.mean():5.2f}   median {np.median(ep):4.1f}   max {ep.max()}")
    print(f"  exo  mean {xp.mean():5.2f}   median {np.median(xp):4.1f}   max {xp.max()}")
    print(f"  frames where exo sees MORE people than ego: {(xp > ep).mean():6.1%}")
    print(f"  total person-detections  ego {ep.sum()}   exo {xp.sum()}  "
          f"({xp.sum()/max(ep.sum(),1):.2f}x)")

    print("\n" + "=" * 68)
    print("MOTION BLUR  (variance of Laplacian; lower = blurrier)")
    print("=" * 68)
    print(f"  ego  median {np.median(eb):8.1f}   p10 {np.percentile(eb,10):8.1f}")
    print(f"  exo  median {np.median(xb):8.1f}   p10 {np.percentile(xb,10):8.1f}")
    thr = float(np.percentile(xb, 10))
    print(f"  ego frames blurrier than exo's p10: {(eb < thr).mean():6.1%}")

    print("\n" + "=" * 68)
    print("APPARENT SIZE of work object (fraction of frame area, when seen)")
    print("=" * 68)
    print(f"  ego  median {np.median(ea[ews]) if ews.any() else float('nan'):.4f}")
    print(f"  exo  median {np.median(xa[xws]) if xws.any() else float('nan'):.4f}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_pairs": int(N), "fps": FPS, "offset_s": OFFSET_S,
        "ego_sees_work": float(ews.mean()), "exo_sees_work": float(xws.mean()),
        "ego_loses_work": float(lost.mean()), "exo_recovery_given_ego_lost": rec,
        "exo_only": float((~ews & xws).mean()), "ego_only": float((ews & ~xws).mean()),
        "persons_mean_ego": float(ep.mean()), "persons_mean_exo": float(xp.mean()),
        "exo_more_people_frac": float((xp > ep).mean()),
        "blur_median_ego": float(np.median(eb)), "blur_median_exo": float(np.median(xb)),
        "ego_blurrier_than_exo_p10": float((eb < thr).mean()),
    }, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
