"""Is the ceiling camera's deficit about DISTANCE, or about being exocentric?

World Context's exo camera loses to ego on task reading (1.98 vs 3.07 bits of
verb entropy, 8 vs 68 fine manipulation verbs). Two explanations fit:

  DISTANCE   it is mounted on the ceiling over a whole bay, so hands are tiny and
             several workers share the frame. Framing, fixable by moving it.
  VIEWPOINT  an over-the-shoulder perspective inherently cannot see what the
             hands are doing. Not fixable by moving it.

Cropping separates them. Cropping to the work area adds NO information the camera
did not already record — it only removes the wide field and enlarges the hands. So:

  if cropped exo recovers -> the deficit was DISTANCE/FRAMING. Mount it closer.
  if cropped exo does not -> the deficit is the VIEWPOINT itself.

The crop is taken from the original 1920x1080 stream, not from the downscaled
frames, so at the same 512px output the work genuinely occupies more pixels.

Usage:  python -m src.crop_test [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src import auto_label as al

OUT = Path("results/crop_test.json")
PRIOR = Path("results/auto_labels.json")
FINE_VERBS = ["pick up", "insert", "detach / remove", "screw / tighten"]


def entropy(c: collections.Counter) -> float:
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v) if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    prior = json.loads(PRIOR.read_text())["rows"]
    n = args.limit or len(prior)
    key = al.api_key()

    res = {}

    def work(w):
        res[w] = al.call(args.model, key, al.frames_for("exocrop", w))

    print(f"labelling cropped exo, {n} windows, model={args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, range(n)))

    ok = [w for w in range(n) if res.get(w, {}).get("verb")]
    cc = collections.Counter(res[w]["verb"] for w in ok)
    cn = collections.Counter(res[w]["noun"] for w in ok if res[w].get("noun"))

    ego = collections.Counter(prior[w]["ego"]["verb"] for w in range(n)
                              if prior[w]["ego"]["verb"])
    exo = collections.Counter(prior[w]["exo"]["verb"] for w in range(n)
                              if prior[w]["exo"]["verb"])

    def fine(c):
        return sum(c.get(v, 0) for v in FINE_VERBS)

    def top_share(c):
        n_ = sum(c.values())
        return c.most_common(1)[0][1] / n_ if n_ else 0.0

    rows = [("ego (head-worn)", ego), ("exo (ceiling, full frame)", exo),
            ("exo (cropped to work area)", cc)]

    print("\n" + "=" * 72)
    print("DOES CROPPING RECOVER THE CEILING CAMERA?")
    print("=" * 72)
    print(f"  {'view':<30}{'verb bits':>11}{'classes':>9}{'top verb':>11}{'fine':>7}")
    print("  " + "-" * 68)
    out = {}
    for name, c in rows:
        out[name] = {"entropy": entropy(c), "classes": len(c),
                     "top_share": top_share(c), "fine": fine(c),
                     "n": sum(c.values())}
        print(f"  {name:<30}{entropy(c):>11.2f}{len(c):>9}"
              f"{top_share(c):>10.0%}{fine(c):>7}")

    e_ego = entropy(ego); e_exo = entropy(exo); e_crop = entropy(cc)
    gap = e_ego - e_exo
    recovered = (e_crop - e_exo) / gap if gap else 0.0
    print(f"\n  ego-vs-exo entropy gap        : {gap:.2f} bits")
    print(f"  recovered by cropping         : {e_crop - e_exo:+.2f} bits "
          f"= {recovered:.0%} of the gap")
    print(f"  fine manipulation verbs       : ego {fine(ego)}  "
          f"exo {fine(exo)}  exo-cropped {fine(cc)}")

    if recovered > 0.5:
        verdict = ("DISTANCE. Most of the deficit was framing — the camera had the "
                   "information and the wide shot buried it. Mount exo closer.")
    elif recovered > 0.2:
        verdict = ("PARTLY DISTANCE. Cropping recovers some of the gap but not most "
                   "of it; framing helps and does not close it.")
    else:
        verdict = ("VIEWPOINT. Cropping recovers little, so the deficit is not the "
                   "wide shot — this vantage does not carry hand detail at all.")
    print(f"\n  => {verdict}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "model": args.model, "n_windows": n, "views": out,
        "entropy_gap_ego_minus_exo": gap,
        "recovered_by_crop_bits": e_crop - e_exo,
        "recovered_fraction": recovered,
        "crop": "x 0.12-0.80, y 0.15-0.80 of the original 1920x1080",
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
