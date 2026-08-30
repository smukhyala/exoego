"""Can a representation learned from ONE viewpoint work on the OTHER?

The premise: a robot that learns by watching people only ever receives an
EXOCENTRIC view of the human. The robot's own egocentric camera is, with respect
to the human performing the task, an exocentric observer. So the deployment
distribution for "learn by watching" robotics is exo-of-human — not the
head-mounted ego video that most human-activity datasets actually contain.

If that is right, the question is not "does exo help ego." It is whether
ego-pretrained representations transfer to the exo view at all.

This measures the VIEW GAP directly, and it distinguishes the two hypotheses the
project is choosing between:

  EFFICIENCY   ego-trained and exo-trained curves are roughly parallel, exo just
               shifted up. Then exo data buys you a constant factor of labels.
  NECESSITY    the ego-trained curve PLATEAUS below the exo-trained one when
               evaluated on exo. Then no amount of ego data substitutes, and exo
               data is a precondition for deployment, not an optimization.

Four transfer directions, one linear probe, matched N:

    train ego -> test ego     within-view baseline
    train ego -> test exo     what a robot gets if you pretrain on wearables
    train exo -> test exo     within-view baseline for the robot's real view
    train exo -> test ego     the reverse gap
    train ego+exo -> test exo does adding ego to exo help the robot's view?

Assembly101 TSM features are already supervised for this task, which invalidated
the earlier self-supervised pretraining ablation. It does NOT invalidate this:
here the features are held fixed and only the probe's TRAINING VIEW changes, so
what is measured is the geometry of the feature space across viewpoints.

Usage:  python -m src.view_gap
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("data/cache/aligned.npz")
RESULTS = Path("results/view_gap.json")
PROBE_SIZES = [5, 10, 20, 50, 100, 143]
SEEDS = 8


def pool_segments(feats, segid, nseg):
    out = np.zeros((nseg, feats.shape[1]), np.float64)
    cnt = np.zeros(nseg, np.int64)
    np.add.at(out, segid, feats.astype(np.float64))
    np.add.at(cnt, segid, 1)
    return (out / np.maximum(cnt, 1)[:, None]).astype(np.float32)


def fit_probe(X, Y, n_cls, dev, steps=600, lr=1e-2):
    torch.manual_seed(0)
    head = nn.Linear(X.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    for _ in range(steps):
        loss = F.cross_entropy(head(Xt), Yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def acc(head, X, Y, dev):
    pred = head(torch.from_numpy(X).to(dev)).argmax(1).cpu().numpy()
    return float((pred == Y).mean())


def main() -> int:
    z = np.load(CACHE)
    ego, exo, y, seq, split = z["ego"], z["exo"], z["y"], z["seq"], z["split"]
    n_cls = int(y.max()) + 1
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ch = np.empty(len(y), bool)
    ch[0] = True
    ch[1:] = (y[1:] != y[:-1]) | (seq[1:] != seq[:-1])
    segid = np.cumsum(ch) - 1
    nseg = int(segid[-1]) + 1

    PE = pool_segments(ego, segid, nseg)
    PX = pool_segments(exo, segid, nseg)
    sy = np.zeros(nseg, np.int64); sy[segid] = y
    ss = np.zeros(nseg, np.int64); ss[segid] = seq
    sp = np.zeros(nseg, np.int8);  sp[segid] = split
    tr, va = sp == 0, sp == 1

    maj = float(np.bincount(sy[va], minlength=n_cls).max() / va.sum())
    print(f"segments {nseg}  train {tr.sum()}  val {va.sum()}  classes {n_cls}")
    print(f"majority baseline {maj:.3f}   device {dev}\n")

    Etr, Xtr, Ytr, Str = PE[tr], PX[tr], sy[tr], ss[tr]
    Eva, Xva, Yva = PE[va], PX[va], sy[va]

    directions = [
        ("ego  -> ego", "ego", "ego"),
        ("ego  -> exo", "ego", "exo"),
        ("exo  -> exo", "exo", "exo"),
        ("exo  -> ego", "exo", "ego"),
        ("both -> exo", "both", "exo"),
        ("both -> ego", "both", "ego"),
    ]
    table = {name: [] for name, _, _ in directions}

    for N in PROBE_SIZES:
        got = {name: [] for name, _, _ in directions}
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            uq = np.unique(Str)
            if N > len(uq):
                continue
            pick = rng.choice(uq, N, replace=False)
            m = np.isin(Str, pick)
            if m.sum() < 5 or len(np.unique(Ytr[m])) < 2:
                continue
            for name, src, dst in directions:
                if src == "ego":
                    Xs, Ys = Etr[m], Ytr[m]
                elif src == "exo":
                    Xs, Ys = Xtr[m], Ytr[m]
                else:  # both views of the SAME segments — no extra labels
                    Xs = np.concatenate([Etr[m], Xtr[m]])
                    Ys = np.concatenate([Ytr[m], Ytr[m]])
                head = fit_probe(Xs, Ys, n_cls, dev)
                Xt, Yt = (Eva, Yva) if dst == "ego" else (Xva, Yva)
                got[name].append(acc(head, Xt, Yt, dev))
        for name in got:
            table[name].append(float(np.mean(got[name])) if got[name] else float("nan"))
        print(f"  N={N:<4} " + "  ".join(
            f"{n.split('->')[0].strip()}->{n.split('->')[1].strip()}:"
            f"{table[n][-1]:.3f}" for n in table))

    print("\n" + "=" * 78)
    print("VIEW GAP — top-1 coarse action accuracy by (train view -> test view)")
    print("=" * 78)
    print(f"{'direction':<14}" + "".join(f"{f'N={n}':>10}" for n in PROBE_SIZES))
    print("-" * 78)
    for name, _, _ in directions:
        print(f"{name:<14}" + "".join(f"{v:>10.3f}" for v in table[name]))
    print(f"{'majority':<14}" + "".join(f"{maj:>10.3f}" for _ in PROBE_SIZES))

    ee = np.array(table["ego  -> ego"]); xx = np.array(table["exo  -> exo"])
    ex = np.array(table["ego  -> exo"]); bx = np.array(table["both -> exo"])
    print("\n" + "=" * 78)
    print("WHAT THIS MEANS FOR A ROBOT (its input view is exo-of-human)")
    print("=" * 78)
    print(f"  exo-trained, tested on exo : {xx[-1]:.3f}  (the right training data)")
    print(f"  ego-trained, tested on exo : {ex[-1]:.3f}  (wearable data only)")
    drop = 1 - ex[-1] / max(xx[-1], 1e-9)
    print(f"  relative drop from using the wrong view: {drop:.1%}")
    print(f"  ego-trained on exo vs majority baseline : "
          f"{ex[-1]:.3f} vs {maj:.3f} -> "
          f"{'ABOVE' if ex[-1] > maj else 'AT OR BELOW'} chance-level utility")
    print(f"\n  adding ego to exo (both -> exo): {bx[-1]:.3f} vs exo-only {xx[-1]:.3f}"
          f"  ({(bx[-1]-xx[-1])*100:+.1f} pts)")

    # Verdict must read the SLOPE, not just the endpoint gap. Efficiency means
    # parallel curves; necessity means the ego curve flattens while exo keeps
    # climbing, so extra labels cannot buy what the wrong view withheld.
    def tail_slope(a):
        a = np.asarray(a, float)
        k = min(3, len(a))
        return float(a[-1] - a[-k])

    s_ex, s_xx = tail_slope(ex), tail_slope(xx)
    print(f"\n  gain over the last {min(3,len(ex))} label sizes:")
    print(f"    ego-trained on exo : {s_ex*100:+.1f} pts")
    print(f"    exo-trained on exo : {s_xx*100:+.1f} pts")

    if ex[-1] <= maj:
        print("\n  => NECESSITY. Ego-only training is unusable on the robot's view.")
    elif s_xx > 0.01 and s_ex < 0.5 * s_xx:
        print("\n  => NECESSITY-LEANING. The ego-trained curve FLATTENS while the")
        print("     exo-trained curve keeps climbing: more wearable data does not")
        print("     close the gap. Exo is a precondition, not an optimization.")
    elif drop > 0.3:
        print("\n  => Strong view gap. Exo data is close to a precondition.")
    else:
        print("\n  => EFFICIENCY regime. Ego transfers partially; exo buys a margin.")

    # ---- LABEL EFFICIENCY -------------------------------------------------
    # How many labelled sequences does the WRONG view need to reach what the
    # RIGHT view achieves at each N? That ratio is the concrete data-efficiency
    # cost of pretraining on wearable video for a robot that sees exo.
    Ns = np.array(PROBE_SIZES, float)

    def n_needed(curve, target):
        """Smallest N (linearly interpolated) at which curve reaches target."""
        c = np.asarray(curve, float)
        for i in range(1, len(c)):
            if c[i] >= target:
                if c[i] == c[i - 1]:
                    return Ns[i]
                t = (target - c[i - 1]) / (c[i] - c[i - 1])
                return Ns[i - 1] + t * (Ns[i] - Ns[i - 1])
        return float("inf")

    print("\n" + "=" * 78)
    print("LABEL EFFICIENCY — sequences needed to hit the same accuracy on exo")
    print("=" * 78)
    print(f"  {'target acc':>11}{'exo-trained N':>16}{'ego-trained N':>16}{'ratio':>10}")
    print("  " + "-" * 53)
    ratios = []
    for i, N in enumerate(PROBE_SIZES[:-1]):
        target = xx[i]
        n_ego = n_needed(ex, target)
        if np.isfinite(n_ego) and n_ego > 0:
            r = n_ego / N
            ratios.append(r)
            print(f"  {target:>11.3f}{N:>16.0f}{n_ego:>16.1f}{r:>10.2f}x")
        else:
            print(f"  {target:>11.3f}{N:>16.0f}{'never':>16}{'>' + str(round(143/N,1)) + 'x':>10}")
    if ratios:
        print(f"\n  median label-efficiency penalty for the wrong view: "
              f"{np.median(ratios):.1f}x more labelled sequences")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "probe_sizes": PROBE_SIZES, "accuracy": table,
        "majority_baseline": maj, "seeds": SEEDS,
        "relative_drop_wrong_view": float(drop),
    }, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
