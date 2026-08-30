"""Build an aligned ego/exo feature cache from the Assembly101 LMDBs.

Keeps only frames where BOTH views have a feature and a coarse action label
exists. Cross-view pretraining needs simultaneous pairs, and the probe needs
labels, so an aligned cache is the honest common denominator for every
condition — no condition gets to train on frames another one couldn't see.

Usage:
    python -m src.build_cache [--max-seqs 200] [--stride 6]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.a101 import (EGO_VIEW, EXO_VIEW, FeatureStore, segments,
                      sequences_with, split_sequences)

OUT = Path("data/cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seqs", type=int, default=250)
    ap.add_argument("--stride", type=int, default=6)  # 30fps -> 5fps
    args = ap.parse_args()

    seqs = sequences_with([EGO_VIEW, EXO_VIEW])
    print(f"sequences with both views: {len(seqs)}")

    ego, exo = FeatureStore(EGO_VIEW), FeatureStore(EXO_VIEW)
    tr, va, te = split_sequences("train"), split_sequences("val"), split_sequences("test")

    E, X, Y, S, SP = [], [], [], [], []
    kept = 0
    t0 = time.time()
    for si, seq in enumerate(seqs[: args.max_seqs]):
        segs = segments(seq)
        if not segs:
            continue
        frames, labels = [], []
        for sg in segs:
            for f in range(sg.start, sg.end, args.stride):
                frames.append(f)
                labels.append(sg.action)
        if not frames:
            continue

        fe, gote = ego.get_many(seq, frames)
        if len(gote) == 0:
            continue
        lut = {f: i for i, f in enumerate(frames)}
        fx, gotx = exo.get_many(seq, gote.tolist())
        if len(gotx) == 0:
            continue
        # intersect on frames present in both views
        idx_e = {f: i for i, f in enumerate(gote)}
        keep_e = [idx_e[f] for f in gotx]
        fe = fe[keep_e]

        split = 0 if seq in tr else (1 if seq in va else (2 if seq in te else 3))
        E.append(fe.astype(np.float16))
        X.append(fx.astype(np.float16))
        Y.append(np.array([labels[lut[f]] for f in gotx], dtype=np.int16))
        S.append(np.full(len(gotx), si, dtype=np.int16))
        SP.append(np.full(len(gotx), split, dtype=np.int8))
        kept += 1
        if kept % 10 == 0:
            n = sum(len(a) for a in Y)
            print(f"  {kept:4d} seqs  {n:7d} frames  {time.time()-t0:5.0f}s", flush=True)

    if not E:
        print("NOTHING CACHED — are the LMDBs unzipped into data/assembly101/TSM_features/?")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "aligned.npz"
    np.savez(out,
             ego=np.concatenate(E), exo=np.concatenate(X),
             y=np.concatenate(Y), seq=np.concatenate(S), split=np.concatenate(SP))
    n = sum(len(a) for a in Y)
    print(f"\nwrote {out}  seqs={kept}  frames={n}  ({out.stat().st_size/1e9:.2f} GB)")
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        m = np.concatenate(SP) == code
        print(f"  {name:6s} frames={m.sum():7d} seqs={len(set(np.concatenate(S)[m]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
