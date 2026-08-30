"""Do the two viewpoints carry different KINDS of information?

The architectural hypothesis: a robot learning by watching should use

  EXO  to recognize the TASK  — what is being done ("put the pencil in the cup")
  EGO  to ground MANIPULATION — what object is in the hands right now

If that division of labour is real, it should show up as an asymmetry in the
labels themselves. Assembly101's coarse actions are verb+noun pairs ("attach
cabin", "screw chassis"), so the hypothesis makes a sharp, falsifiable
prediction:

    exo should beat ego on VERBS by more than it beats ego on NOUNS.

Equivalently: ego's relative standing should be better on nouns than on verbs.
If instead one view wins uniformly on both, the views differ in quality but not
in KIND, and the proposed split is not justified by the data.

Also tests FUSION — concatenating both views for the same segment — separately
for verbs and nouns, which says whether the two streams are complementary or
redundant.

Usage:  python -m src.verb_noun
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("data/cache/aligned.npz")
ACTIONS = Path("data/assembly101/annotations/coarse-annotations/actions.csv")
RESULTS = Path("results/verb_noun.json")
PROBE_SIZES = [10, 20, 50, 143]
SEEDS = 8


def label_maps():
    """action_id -> (verb_id, noun_id), plus vocabulary sizes."""
    verb, noun = {}, {}
    with open(ACTIONS) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 6:
                verb[int(p[0])] = int(p[1])
                noun[int(p[0])] = int(p[2])
    return verb, noun


def pool_segments(feats, segid, nseg):
    out = np.zeros((nseg, feats.shape[1]), np.float64)
    cnt = np.zeros(nseg, np.int64)
    np.add.at(out, segid, feats.astype(np.float64))
    np.add.at(cnt, segid, 1)
    return (out / np.maximum(cnt, 1)[:, None]).astype(np.float32)


def probe(Xtr, Ytr, Xva, Yva, n_cls, dev, steps=600):
    torch.manual_seed(0)
    head = nn.Linear(Xtr.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    for _ in range(steps):
        loss = F.cross_entropy(head(Xt), Yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(torch.from_numpy(Xva).to(dev)).argmax(1).cpu().numpy()
    return float((pred == Yva).mean())


def main() -> int:
    z = np.load(CACHE)
    ego, exo, y, seq, split = z["ego"], z["exo"], z["y"], z["seq"], z["split"]
    verb_of, noun_of = label_maps()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ch = np.empty(len(y), bool)
    ch[0] = True
    ch[1:] = (y[1:] != y[:-1]) | (seq[1:] != seq[:-1])
    segid = np.cumsum(ch) - 1
    nseg = int(segid[-1]) + 1

    PE = pool_segments(ego, segid, nseg)
    PX = pool_segments(exo, segid, nseg)
    PF = np.concatenate([PE, PX], axis=1)  # fusion
    sy = np.zeros(nseg, np.int64); sy[segid] = y
    ss = np.zeros(nseg, np.int64); ss[segid] = seq
    sp = np.zeros(nseg, np.int8);  sp[segid] = split
    tr, va = sp == 0, sp == 1

    sv = np.array([verb_of.get(int(a), 0) for a in sy])
    sn = np.array([noun_of.get(int(a), 0) for a in sy])
    n_verb, n_noun = int(sv.max()) + 1, int(sn.max()) + 1
    print(f"segments {nseg}   verbs {n_verb}   nouns {n_noun}")

    maj_v = float(np.bincount(sv[va], minlength=n_verb).max() / va.sum())
    maj_n = float(np.bincount(sn[va], minlength=n_noun).max() / va.sum())
    print(f"majority baseline   verb {maj_v:.3f}   noun {maj_n:.3f}   device {dev}\n")

    views = {"ego": PE, "exo": PX, "fusion": PF}
    targets = {"VERB (what task)": (sv, n_verb, maj_v),
               "NOUN (what object)": (sn, n_noun, maj_n)}
    table = {t: {v: [] for v in views} for t in targets}

    Str = ss[tr]
    for tname, (lab, ncls, _) in targets.items():
        for N in PROBE_SIZES:
            per_view = {v: [] for v in views}
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                uq = np.unique(Str)
                if N > len(uq):
                    continue
                pick = rng.choice(uq, N, replace=False)
                m = np.isin(Str, pick)
                if m.sum() < 5 or len(np.unique(lab[tr][m])) < 2:
                    continue
                for vname, P in views.items():
                    per_view[vname].append(
                        probe(P[tr][m], lab[tr][m], P[va], lab[va], ncls, dev))
            for vname in views:
                table[tname][vname].append(
                    float(np.mean(per_view[vname])) if per_view[vname] else float("nan"))
        print(f"  {tname} done")

    print("\n" + "=" * 78)
    print("VERB vs NOUN by viewpoint — top-1, train and test on the same view")
    print("=" * 78)
    for tname, (_, _, maj) in targets.items():
        print(f"\n{tname}   (majority {maj:.3f})")
        print(f"  {'view':<10}" + "".join(f"{f'N={n}':>10}" for n in PROBE_SIZES))
        print("  " + "-" * (10 + 10 * len(PROBE_SIZES)))
        for vname in views:
            print(f"  {vname:<10}" + "".join(f"{v:>10.3f}" for v in table[tname][vname]))

    print("\n" + "=" * 78)
    print("IS THE SPLIT REAL?  exo advantage over ego, verbs vs nouns")
    print("=" * 78)
    adv = {}
    for tname in targets:
        e = np.array(table[tname]["ego"], float)
        x = np.array(table[tname]["exo"], float)
        rel = (x[-1] - e[-1]) / max(e[-1], 1e-9)
        adv[tname] = float(rel)
        print(f"  {tname:<20} ego {e[-1]:.3f}   exo {x[-1]:.3f}   "
              f"exo advantage {rel:+.1%}")

    v = adv["VERB (what task)"]
    n = adv["NOUN (what object)"]
    print(f"\n  exo advantage on VERBS : {v:+.1%}")
    print(f"  exo advantage on NOUNS : {n:+.1%}")
    print(f"  asymmetry (verb - noun): {v - n:+.1%}")

    if v > n + 0.05:
        print("\n  => SPLIT SUPPORTED. Exo's edge is larger for tasks than for")
        print("     objects, which is the predicted division of labour.")
    elif n > v + 0.05:
        print("\n  => SPLIT REVERSED. Exo helps more with objects than with tasks —")
        print("     the opposite of the hypothesis.")
    else:
        print("\n  => NO ASYMMETRY. Exo wins by a similar margin on both, so the")
        print("     views differ in QUALITY, not in KIND. A verb/noun division of")
        print("     labour is not justified by this evidence.")

    print("\n" + "=" * 78)
    print("FUSION — are the two views complementary or redundant?")
    print("=" * 78)
    for tname in targets:
        e = table[tname]["ego"][-1]
        x = table[tname]["exo"][-1]
        f = table[tname]["fusion"][-1]
        best = max(e, x)
        print(f"  {tname:<20} best single {best:.3f}   fusion {f:.3f}   "
              f"{(f-best)*100:+.1f} pts")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "probe_sizes": PROBE_SIZES, "seeds": SEEDS, "accuracy": table,
        "majority": {"verb": maj_v, "noun": maj_n},
        "exo_advantage": adv, "asymmetry": float(v - n),
    }, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
