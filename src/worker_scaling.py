"""Does a crowded exo frame hurt its task reading — and what does it buy?

The obvious objection to a ceiling camera is that it sees several workers at
once, so its account of "the task" should get muddier as the floor fills up. The
obvious counter is that the same crowding is the point: one exo camera covers
every worker in the bay, where ego needs one wearable per person.

Both halves are measurable on this footage.

  1. COVERAGE     how many workers a single exo camera actually holds in frame,
                  per window. Ego is 1 by construction — the wearer.
  2. ROBUSTNESS   whether exo's task labels degrade as worker count rises. If
                  exo's label entropy and its agreement with ego hold flat across
                  crowding, the dilution objection does not bite.

Writes results/worker_scaling.json for the figure.

Usage:  python -m src.worker_scaling
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

FRAMES = Path("/private/tmp/claude-501/-Users-evan-Projects-exoego/"
              "a0c6b8fb-f59d-4405-8f14-34aa52a4c40a/scratchpad/lab")
LABELS = Path("results/auto_labels.json")
OUT = Path("results/worker_scaling.json")
CONF = 0.25
WINDOW = 5.0


def entropy(c: collections.Counter) -> float:
    n = sum(c.values())
    if not n:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v)


def main() -> int:
    from ultralytics import YOLO
    import torch

    rows = json.loads(LABELS.read_text())["rows"]
    n = len(rows)
    model = YOLO("yolo11n.pt")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # middle frame of each window, both views
    counts = {}
    for view in ("ego", "exo"):
        files = []
        for w in range(n):
            f = FRAMES / view / f"{int(w * WINDOW) + 3:05d}.jpg"
            files.append(str(f) if f.exists() else None)
        got = np.zeros(n, int)
        batch = [f for f in files if f]
        idx = [i for i, f in enumerate(files) if f]
        for s in range(0, len(batch), 32):
            res = model.predict(batch[s:s + 32], verbose=False, device=dev, conf=CONF)
            for j, r in enumerate(res):
                names = [model.names[int(c)] for c in r.boxes.cls]
                got[idx[s + j]] = sum(1 for x in names if x == "person")
        counts[view] = got
        print(f"{view}: mean {got.mean():.2f} people/frame  max {got.max()}")

    exo_n = counts["exo"]

    # ---- coverage -------------------------------------------------------
    dist = collections.Counter(exo_n.tolist())
    total_worker_windows = int(exo_n.sum())
    print(f"\nexo camera holds {exo_n.mean():.2f} workers/window on average")
    print(f"one exo camera covers {total_worker_windows} worker-windows over {n} windows")
    print(f"one ego camera covers {n} worker-windows (the wearer, by construction)")
    print(f"coverage multiplier: {total_worker_windows / n:.2f}x per camera")

    # ---- robustness -----------------------------------------------------
    bins = [(1, 1), (2, 2), (3, 3), (4, 99)]
    out_bins = []
    print(f"\n{'workers':<10}{'windows':>9}{'exo entropy':>13}{'ego entropy':>13}{'verb agree':>12}")
    print("-" * 57)
    for lo, hi in bins:
        m = (exo_n >= lo) & (exo_n <= hi)
        sel = [rows[i] for i in range(n) if m[i]]
        if len(sel) < 5:
            continue
        ce = collections.Counter(r["exo"]["verb"] for r in sel if r["exo"]["verb"])
        cg = collections.Counter(r["ego"]["verb"] for r in sel if r["ego"]["verb"])
        agree = sum(r["verb_agree"] for r in sel) / len(sel)
        label = f"{lo}" if lo == hi else f"{lo}+"
        out_bins.append({"workers": label, "n": len(sel),
                         "exo_entropy": entropy(ce), "ego_entropy": entropy(cg),
                         "verb_agreement": agree})
        print(f"{label:<10}{len(sel):>9}{entropy(ce):>13.2f}{entropy(cg):>13.2f}{agree:>11.1%}")

    if len(out_bins) >= 2:
        first, last = out_bins[0], out_bins[-1]
        d = last["exo_entropy"] - first["exo_entropy"]
        print(f"\nexo entropy change from {first['workers']} to {last['workers']} "
              f"workers: {d:+.2f} bits")
        print("Flat or rising means crowding does NOT dilute exo's task reading —")
        print("the extra workers are extra coverage, not extra confusion.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "n_windows": n, "conf": CONF,
        "exo_people_mean": float(exo_n.mean()),
        "exo_people_hist": {str(k): v for k, v in sorted(dist.items())},
        "coverage_multiplier": total_worker_windows / n,
        "bins": out_bins,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
