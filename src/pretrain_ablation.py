"""Does exo video during PRETRAINING make an EGO-ONLY model better?

That is the question worth asking. "Two views beat one at test time" is obvious
and useless — a robot with a head camera never gets an exo view. The useful
claim is that exo is a teacher you can throw away at deployment.

So: pretrain three ways, then evaluate all three the SAME way, on ego alone.

  ego-only   InfoNCE, positives = temporally-near frames in the ego view
  exo-only   InfoNCE, positives = temporally-near frames in the exo view
  ego+exo    InfoNCE, positives = the SIMULTANEOUS frame in the other view
             (multi-view time-contrastive, after Sermanet et al.'s TCN)

Matched budget, enforced not asserted: identical encoder architecture, identical
init seed, identical gradient steps, identical frames drawn per step. The ego+exo
condition draws half its frames from each view, so no condition sees more data
than another.

Downstream: freeze the encoder, train a linear head on coarse action classes
using N labelled sequences, N in {1,2,5,10,20,50}. Report top-1 on held-out
sequences. EGO FEATURES ONLY, for every condition.

Usage:
    python -m src.pretrain_ablation [--steps 3000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("data/cache/aligned.npz")
RESULTS = Path("results/pretrain_ablation.json")
CONDITIONS = ["none (raw)", "ego-only", "exo-only", "ego+exo"]
PROBE_SIZES = [1, 2, 5, 10, 20, 50]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Encoder(nn.Module):
    """Deliberately small. The claim is about the data, not the architecture."""

    def __init__(self, dim_in: int = 2048, dim_out: int = 1024):
        super().__init__()
        # Width matters: a first run at dim_out=256 UNDERPERFORMED raw 2048-D
        # features at every probe size, i.e. the projection destroyed more than
        # the objective added. "none (raw)" stays in the table to keep us honest.
        self.net = nn.Sequential(
            nn.Linear(dim_in, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, dim_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    logits = a @ b.T / temp
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def sample_pairs(cond, ego, exo, seq, rng, batch, window):
    """Return (anchor, positive) index arrays under the condition's rule."""
    n = len(ego)
    anchor = rng.integers(0, n, batch)
    if cond == "ego+exo":
        # positive is the SAME frame seen from the other camera
        return anchor, anchor
    # temporal positive: a nearby frame from the same sequence
    off = rng.integers(1, window + 1, batch) * rng.choice([-1, 1], batch)
    pos = np.clip(anchor + off, 0, n - 1)
    bad = seq[pos] != seq[anchor]
    pos[bad] = anchor[bad]
    return anchor, pos


def pretrain(cond, data, steps, batch, window, seed, dev):
    torch.manual_seed(seed)  # identical init across conditions
    enc = Encoder().to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ego, exo, seq = data["ego"], data["exo"], data["seq"]

    enc.train()
    for _ in range(steps):
        ai, pi = sample_pairs(cond, ego, exo, seq, rng, batch, window)
        if cond == "ego-only":
            xa, xp = ego[ai], ego[pi]
        elif cond == "exo-only":
            xa, xp = exo[ai], exo[pi]
        else:  # ego+exo: half the frames from each view, so totals match
            xa, xp = ego[ai], exo[pi]
        a = enc(torch.from_numpy(xa.astype(np.float32)).to(dev))
        p = enc(torch.from_numpy(xp.astype(np.float32)).to(dev))
        loss = info_nce(a, p)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    enc.eval()
    return enc, float(loss.item())


def segment_index(y, seq):
    """Group consecutive (seq, label) runs into coarse action segments.

    Coarse actions are ~10 s long; classifying single frames makes the task
    needlessly noisy and lets class imbalance dominate. Segment-level pooling is
    the standard setup and drops the majority-class baseline from 14.3% to 5.7%.
    """
    ch = np.empty(len(y), bool)
    ch[0] = True
    ch[1:] = (y[1:] != y[:-1]) | (seq[1:] != seq[:-1])
    return np.cumsum(ch) - 1


def pool_segments(feats, segid, nseg):
    out = np.zeros((nseg, feats.shape[1]), np.float64)
    cnt = np.zeros(nseg, np.int64)
    np.add.at(out, segid, feats.astype(np.float64))
    np.add.at(cnt, segid, 1)
    return (out / np.maximum(cnt, 1)[:, None]).astype(np.float32)


@torch.no_grad()
def embed(enc, x, dev, bs=8192):
    out = []
    for i in range(0, len(x), bs):
        out.append(enc(torch.from_numpy(x[i:i + bs].astype(np.float32)).to(dev)).cpu().numpy())
    return np.concatenate(out)


def probe(emb_tr, y_tr, seq_tr, emb_va, y_va, n_seqs, seed, dev, n_cls,
          steps: int = 600):
    """Linear head on frozen segment embeddings, trained on n_seqs sequences."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq_tr)
    if n_seqs > len(uniq):
        return None
    pick = rng.choice(uniq, n_seqs, replace=False)
    m = np.isin(seq_tr, pick)
    X, Y = emb_tr[m], y_tr[m].astype(np.int64)
    if len(X) < 10 or len(np.unique(Y)) < 2:
        return None

    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    for _ in range(steps):
        loss = F.cross_entropy(head(Xt), Yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(torch.from_numpy(emb_va).to(dev)).argmax(1).cpu().numpy()
    return float((pred == y_va).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)  # +-1s at 5fps
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if not CACHE.exists():
        print(f"missing {CACHE} — run: python -m src.build_cache")
        return 1
    z = np.load(CACHE)
    ego, exo, y, seq, split = z["ego"], z["exo"], z["y"], z["seq"], z["split"]
    n_cls = int(y.max()) + 1
    dev = device()

    # segment level: coarse actions are ~10 s, so frames are the wrong unit
    segid = segment_index(y, seq)
    nseg = int(segid[-1]) + 1
    sy = np.zeros(nseg, np.int64); sy[segid] = y
    ss = np.zeros(nseg, np.int64); ss[segid] = seq
    sp = np.zeros(nseg, np.int8);  sp[segid] = split
    str_, sva = sp == 0, sp == 1

    tr, va = split == 0, split == 1
    maj = float(np.bincount(sy[sva], minlength=n_cls).max() / max(sva.sum(), 1))
    print(f"device={dev}  frames={len(y)}  segments={nseg}  classes={n_cls}")
    print(f"train {str_.sum()} segs / {len(np.unique(seq[tr]))} seqs   "
          f"val {sva.sum()} segs / {len(np.unique(seq[va]))} seqs")
    print(f"majority-class baseline (val, segment level): {maj:.3f}\n")

    pre = {"ego": ego[tr], "exo": exo[tr], "seq": seq[tr]}
    out = {c: {n: [] for n in PROBE_SIZES} for c in CONDITIONS}

    for seed in range(args.seeds):
        for cond in CONDITIONS:
            t0 = time.time()
            if cond == "none (raw)":
                # control: no pretraining at all, raw 2048-D TSM features
                loss = float("nan")
                feats = ego
            else:
                enc, loss = pretrain(cond, pre, args.steps, args.batch,
                                     args.window, seed, dev)
                # EVERY condition is evaluated on EGO features only
                feats = embed(enc, ego, dev)

            # Pool over ALL frames with the global segment index, THEN split.
            # Train/val segments are interleaved, so remapping ids per split
            # creates gap rows and silently misaligns features from labels.
            pooled = pool_segments(feats, segid, nseg)
            ptr, ytr_s, str_s = pooled[str_], sy[str_], ss[str_]
            pva, yva_s = pooled[sva], sy[sva]

            accs = []
            for n in PROBE_SIZES:
                a = probe(ptr, ytr_s, str_s, pva, yva_s, n, seed, dev, n_cls)
                if a is not None:
                    out[cond][n].append(a)
                accs.append(a)
            shown = " ".join(f"{n}:{'--' if a is None else f'{a:.3f}'}"
                             for n, a in zip(PROBE_SIZES, accs))
            lt = "  n/a " if np.isnan(loss) else f"{loss:.3f}"
            print(f"  seed{seed} {cond:<12} loss={lt} {time.time()-t0:5.1f}s  {shown}",
                  flush=True)

    print("\n" + "=" * 78)
    print("Top-1 coarse action accuracy — EGO FEATURES ONLY at test time")
    print("=" * 78)
    print(f"{'pretraining':<14}" + "".join(f"{f'N={n}':>10}" for n in PROBE_SIZES))
    print("-" * 78)
    table = {}
    for c in CONDITIONS:
        row = [float(np.mean(out[c][n])) if out[c][n] else float("nan")
               for n in PROBE_SIZES]
        table[c] = row
        print(f"{c:<14}" + "".join(f"{v:>10.3f}" for v in row))
    print(f"{'majority':<14}" + "".join(f"{maj:>10.3f}" for _ in PROBE_SIZES))
    print("\nN = number of labelled sequences the linear head saw.")

    raw, base, best = table["none (raw)"], table["ego-only"], table["ego+exo"]
    d_pre = [b - a for a, b in zip(raw, base) if np.isfinite(a) and np.isfinite(b)]
    d_exo = [b - a for a, b in zip(base, best) if np.isfinite(a) and np.isfinite(b)]
    print(f"\n  ego-only pretraining vs raw features : {np.mean(d_pre)*100:+.1f} pts mean")
    print(f"  ego+exo vs ego-only pretraining     : {np.mean(d_exo)*100:+.1f} pts mean")
    if np.mean(d_exo) <= 0:
        print("\n  NEGATIVE RESULT: cross-view pretraining did not beat single-view.")
        print("  Report it as such.")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"probe_sizes": PROBE_SIZES, "accuracy": table, "majority_baseline": maj,
         "steps": args.steps, "seeds": args.seeds, "level": "segment"}, indent=2))
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
