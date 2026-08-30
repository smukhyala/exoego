"""Render the label-efficiency UI from the sweep results.

Reads results/label_efficiency.csv and results/per_task.csv, computes the
data-reduction read-off, and injects everything into ui/template.html as JSON.
The page ships with real numbers only -- nothing is synthesised here.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego.paths import manifests_dir, repo_root, results_dir

BACKBONE_LABELS = {
    "dinov2s": "frozen DINOv2 ViT-S/14 (per-frame)",
    "dinov2b": "frozen DINOv2 ViT-B/14 (per-frame)",
    "clip": "frozen CLIP ViT-B/16 (per-frame)",
    "videomae": "frozen VideoMAE, Kinetics (video-native)",
}

METRIC_LABELS = {
    "mean_per_class": "mean-per-class accuracy (%)",
    "top1": "top-1 accuracy (%)",
    "retrieval_map": "retrieval mAP (%)",
}


def curve_from(frame: pd.DataFrame, metric: str) -> list:
    """Mean and sd of `metric` at each budget, ordered by budget."""
    grouped = frame.groupby("effective_budget")[metric].agg(["mean", "std"]).reset_index()
    points = []
    for row in grouped.itertuples(index=False):
        std = 0.0 if pd.isna(row.std) else float(row.std)
        points.append({
            "budget": int(row.effective_budget),
            "mean": round(float(row.mean), 5),
            "std": round(std, 5),
        })
    points.sort(key=lambda p: p["budget"])
    return points


def labels_needed(curve: list, target: float):
    """Smallest label budget at which `curve` reaches `target`.

    Interpolates in log-budget space between the two points that bracket the
    first crossing. Returns None if the curve never reaches the target, which is
    a real outcome and must not be silently rounded to the last budget.
    """
    if not curve:
        return None
    if curve[0]["mean"] >= target:
        return float(curve[0]["budget"])

    for earlier, later in zip(curve, curve[1:]):
        if later["mean"] >= target:
            span = later["mean"] - earlier["mean"]
            if span <= 0:
                return float(later["budget"])
            fraction = (target - earlier["mean"]) / span
            import math
            low = math.log10(earlier["budget"])
            high = math.log10(later["budget"])
            return float(10 ** (low + fraction * (high - low)))
    return None


def reduction_for(ego_only: list, ego_exo: list):
    """How many fewer labels ego_exo needs to match ego_only's best accuracy."""
    if not ego_only or not ego_exo:
        return None
    target = ego_only[-1]["mean"]
    full_budget = float(ego_only[-1]["budget"])
    needed = labels_needed(ego_exo, target)
    if needed is None or needed <= 0:
        return None
    return {
        "targetAcc": round(target, 5),
        "onlyLabels": full_budget,
        "exoLabels": round(needed, 1),
        "reduction": round(full_budget / needed, 3),
    }


def curves_separated(ego_only: list, ego_exo: list) -> bool:
    """Do the two curves differ by more than their own seed noise anywhere?

    Guards the headline. With overlapping curves, `reduction_for` will happily
    read a huge label saving off a chance crossing -- on this run ego_exo's noisy
    peak at 200 labels exceeds ego_only's full-budget mean, which would print as
    "20x fewer labels" from pure noise.
    """
    by_budget = {p["budget"]: p for p in ego_only}
    for point in ego_exo:
        other = by_budget.get(point["budget"])
        if other is None:
            continue
        pooled = (point["std"] ** 2 + other["std"] ** 2) ** 0.5
        if point["mean"] - other["mean"] > pooled:
            return True
    return False


def paired_stats(sweep: pd.DataFrame, metric: str, baseline: str = "ego_only") -> list:
    """Paired-by-seed differences against the baseline config.

    Seed k hands every config the identical labelled subset, so pairing removes
    between-subset variance and is far more sensitive than comparing pooled
    means with their own spreads. This is the primary statistic: the per-budget
    cells below it are exploratory, since with seven budgets roughly one cell is
    expected to clear p<0.05 by chance.
    """
    base = {}
    for row in sweep[sweep["config"] == baseline].itertuples(index=False):
        base[(row.effective_budget, row.seed)] = getattr(row, metric)

    out = []
    for config in sweep["config"].unique():
        if config == baseline:
            continue
        diffs = []
        by_budget = {}
        for row in sweep[sweep["config"] == config].itertuples(index=False):
            key = (row.effective_budget, row.seed)
            if key not in base:
                continue
            difference = getattr(row, metric) - base[key]
            diffs.append(difference)
            by_budget.setdefault(row.effective_budget, []).append(difference)
        if len(diffs) < 2:
            continue

        mean = statistics.mean(diffs)
        standard_error = statistics.stdev(diffs) / len(diffs) ** 0.5
        t_stat = abs(mean) / standard_error if standard_error > 0 else 0.0
        positive_budgets = 0
        per_budget = []
        for budget in sorted(by_budget):
            budget_mean = statistics.mean(by_budget[budget])
            if budget_mean > 0:
                positive_budgets += 1
            per_budget.append({"budget": budget, "mean": round(budget_mean, 5)})

        out.append({
            "config": config,
            "n": len(diffs),
            "mean": round(mean, 5),
            "se": round(standard_error, 5),
            "t": round(t_stat, 2),
            "wins": sum(1 for d in diffs if d > 0),
            "budgetsPositive": positive_budgets,
            "budgetCount": len(by_budget),
            "separable": bool(t_stat > 2.0),
            "perBudget": per_budget,
        })
    return out


def build_tasks(per_task: pd.DataFrame, budgets: list, low_support: int) -> list:
    smallest = min(budgets)
    tasks = []
    for verb, group in per_task.groupby("verb_cls"):
        curves = {}
        for config in ["ego_only", "ego_exo", "ego_ego"]:
            part = group[group["config"] == config]
            if part.empty:
                continue
            stats = part.groupby("budget")["accuracy"].agg(["mean", "std"]).reset_index()
            points = []
            for row in stats.itertuples(index=False):
                std = 0.0 if pd.isna(row.std) else float(row.std)
                points.append({
                    "budget": int(row.budget),
                    "mean": round(float(row.mean), 5),
                    "std": round(std, 5),
                })
            points.sort(key=lambda p: p["budget"])
            curves[config] = points

        if "ego_only" not in curves or "ego_exo" not in curves:
            continue

        low = [p for p in curves["ego_only"] if p["budget"] == smallest]
        high = [p for p in curves["ego_exo"] if p["budget"] == smallest]
        delta = (high[0]["mean"] - low[0]["mean"]) if low and high else 0.0

        info = reduction_for(curves["ego_only"], curves["ego_exo"])
        tasks.append({
            "verb": verb,
            "support": int(group["support"].iloc[0]),
            "delta": round(delta, 5),
            "reduction": info["reduction"] if info else None,
            "curves": curves,
        })

    tasks.sort(key=lambda t: (-t["delta"], -t["support"]))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="mean_per_class", choices=list(METRIC_LABELS))
    parser.add_argument("--low-support", type=int, default=15,
                        help="verbs with fewer eval clips are flagged as noisy")
    parser.add_argument("--out", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--backbone", default=None,
                        help="backbone that produced these results; defaults to base.yaml")
    args = parser.parse_args()

    backbone = args.backbone
    if backbone is None:
        import yaml
        with open(repo_root() / "configs" / "base.yaml") as handle:
            backbone = yaml.safe_load(handle)["backbone"]
    backbone_label = BACKBONE_LABELS.get(backbone, backbone)

    source = Path(args.results_dir) if args.results_dir else results_dir()
    sweep_path = source / "label_efficiency.csv"
    task_path = source / "per_task.csv"
    if not sweep_path.exists():
        raise SystemExit(f"missing {sweep_path}; run 05_train.py first")

    sweep = pd.read_csv(sweep_path)
    per_task = pd.read_csv(task_path) if task_path.exists() else pd.DataFrame()

    curves = {}
    for config in ["ego_only", "ego_exo", "ego_ego", "ego_exo_degraded"]:
        part = sweep[sweep["config"] == config]
        if not part.empty:
            curves[config] = curve_from(part, args.metric)

    budgets = sorted({p["budget"] for c in curves.values() for p in c})
    full_budget = max(budgets)
    smallest = min(budgets)

    headline = reduction_for(curves.get("ego_only", []), curves.get("ego_exo", []))
    separated = curves_separated(curves.get("ego_only", []), curves.get("ego_exo", []))

    # Majority-class baseline and the go/no-go gate.
    majority = 0.0
    segments_path = manifests_dir() / "segments.csv"
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        evaluation = segments[segments["role"] == "eval"]
        if not evaluation.empty:
            majority = float(evaluation["verb_cls"].value_counts().iloc[0] / len(evaluation))

    ego_only_full = sweep[(sweep["config"] == "ego_only") &
                          (sweep["effective_budget"] == full_budget)]["top1"].mean()
    gate_pass = bool(ego_only_full > majority * 1.25)
    inconclusive = (not gate_pass) or (not separated)
    if inconclusive:
        headline = None

    if gate_pass:
        gate_text = (
            f"<strong>ego only reaches {ego_only_full * 100:.1f}% top-1</strong> at the full "
            f"label budget, against a {majority * 100:.1f}% majority-class baseline. The "
            f"egocentric features carry real verb signal, so the differences below are "
            f"interpretable."
        )
    else:
        gate_text = (
            f"<strong>ego only reaches just {ego_only_full * 100:.1f}% top-1</strong> at the "
            f"full label budget, against a {majority * 100:.1f}% majority-class baseline. The "
            f"egocentric features are close to chance, so every difference below is noise "
            f"rather than evidence &mdash; no label-reduction figure is reported."
        )

    def at(config, budget):
        points = curves.get(config, [])
        found = [p for p in points if p["budget"] == budget]
        return found[0]["mean"] if found else None

    low_only, low_exo, low_ctl = at("ego_only", smallest), at("ego_exo", smallest), at("ego_ego", smallest)

    stats = []
    if inconclusive:
        stats.append({
            "label": "Label reduction",
            "value": "n/a",
            "note": ("not reported: the curves sit inside each other's seed noise, so any "
                     "crossing is chance"),
            "muted": True,
        })
    elif headline:
        stats.append({
            "label": "Labels to match",
            "value": f"{headline['reduction']:.1f}×",
            "note": (f"ego+exo needs ~{headline['exoLabels']:.0f} labelled clips to reach the "
                     f"{headline['targetAcc'] * 100:.1f}% that ego alone needs {full_budget:,} for"),
            "muted": headline["reduction"] < 1.05,
        })
    else:
        stats.append({
            "label": "Label reduction",
            "value": "none",
            "note": "ego+exo never reaches ego-only's best accuracy at any budget tested",
            "muted": True,
        })

    def noise_at(budget):
        spreads = []
        for config in ["ego_only", "ego_exo"]:
            found = [p for p in curves.get(config, []) if p["budget"] == budget]
            if found:
                spreads.append(found[0]["std"])
        if len(spreads) < 2:
            return None
        return (spreads[0] ** 2 + spreads[1] ** 2) ** 0.5

    if low_only is not None and low_exo is not None:
        gain = (low_exo - low_only) * 100
        spread = noise_at(smallest)
        note = "points of mean-per-class accuracy, in the scarcest-label regime"
        if spread is not None:
            note += f" (seed noise \u00b1{spread * 100:.1f})"
        stats.append({
            "label": f"Gain at {smallest} labels",
            "value": f"{gain:+.1f}",
            "note": note,
            "muted": spread is not None and abs(gain) <= spread * 100,
        })

    if low_exo is not None and low_ctl is not None:
        margin = (low_exo - low_ctl) * 100
        stats.append({
            "label": "Over the control",
            "value": f"{margin:+.1f}",
            "note": "vs. ego+ego, which has the same loss shape but no exo information",
            "muted": abs(margin) < 1.0,
        })

    tasks = build_tasks(per_task, budgets, args.low_support) if not per_task.empty else []
    if tasks:
        improved = sum(1 for t in tasks if t["delta"] > 0)
        stats.append({
            "label": "Tasks improved",
            "value": f"{improved}/{len(tasks)}",
            "note": f"verbs where ego+exo beats ego alone at {smallest} labels",
            "muted": improved <= len(tasks) / 2,
        })

    manifest_chips = []
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        manifest_chips = [
            f"{len(segments[segments['role'] == 'train']):,} train clips",
            f"{len(segments[segments['role'] == 'eval']):,} eval clips",
            f"{segments['seq'].nunique()} recordings",
        ]

    seeds = int(sweep["seed"].nunique())
    payload = {
        "meta": {
            "metricLabel": METRIC_LABELS[args.metric],
            "majority": round(majority, 5),
            "seeds": seeds,
            "gatePass": gate_pass,
            "inconclusive": inconclusive,
            "verdict": (
                "Inconclusive. The egocentric representation sits at chance, so the "
                "ego-vs-ego+exo comparison has nothing to measure yet."
                if not gate_pass else
                "No measurable effect. The egocentric representation is sound, but the "
                "exocentric gain is not separable from seed noise."
                if inconclusive else
                "Exocentric video measurably reduces the labels needed."
            ),
            "gateText": gate_text,
            "lowSupport": args.low_support,
            "diagnosis": (
                (
                    "The egocentric representation does not clear the majority-class "
                    "baseline, so there is no headroom in which an exocentric effect could "
                    "show up: all three configs are pinned together by the encoder, not by "
                    "the alignment objective. Read the gate before the curves &mdash; a "
                    "delta measured here is seed noise. The lever is the representation "
                    f"({backbone_label}), not more data or a different alignment loss."
                ) if not gate_pass else (
                    "The gate passes, so this comparison is measurable &mdash; and what it "
                    "measures is a null. Pooled over every budget and seed, ego + exo sits "
                    "within seed noise of ego alone, and the ego + ego control sits at zero, "
                    "which is what tells you the harness is not adding lift of its own. The "
                    "exocentric lean is real in direction but too small to claim at this "
                    "scale. Per-budget cells are exploratory: with seven budgets, roughly one "
                    "is expected to look significant by chance."
                )
            ) if inconclusive else "",
            "chips": manifest_chips + [
                backbone_label,
                "16 frames per clip",
                f"{seeds} seeds",
            ],
            "footnotes": [
                "Evaluation is egocentric only. The exocentric view is used during training "
                "as unlabelled paired signal and is never available at test time.",
                "Labels are restricted by budget; the unlabelled ego/exo pairs are not — that "
                "is the claim being tested. Every config takes the same number of supervised "
                "gradient steps, so no config wins by training longer.",
                "ego+ego is the control: same loss shape and parameter count as ego+exo, but "
                "its second view is another temporal crop of the same ego clip, carrying no "
                "exocentric information. A gain over ego-only that the control also shows is "
                "a contrastive-regulariser effect, not an exo effect.",
                "Train and eval recordings are disjoint by recording and by toy. Verbs with "
                "few evaluation clips are dimmed — their per-task deltas are noisy.",
            ],
        },
        "budgets": budgets,
        "fullBudget": full_budget,
        "curves": curves,
        "headline": headline,
        "paired": paired_stats(sweep, args.metric),
        "stats": stats,
        "tasks": tasks,
    }

    template = (repo_root() / "ui" / "template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))

    out_path = Path(args.out) if args.out else repo_root() / "ui" / "label_efficiency.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    print(f"wrote {out_path}  ({len(html) / 1024:.0f} KB)")
    print(f"  configs: {list(curves)}")
    print(f"  budgets: {budgets}")
    print(f"  tasks:   {len(tasks)}")
    print(f"  gate:    {'PASS' if gate_pass else 'FAIL'} "
          f"(ego_only top-1 {ego_only_full:.3f} vs majority {majority:.3f})")
    print(f"  separated: {separated} | inconclusive: {inconclusive}")
    for row in payload["paired"]:
        print(f"  paired {row['config']:18s} {row['mean']:+.4f} "
              f"se={row['se']:.4f} |t|={row['t']:.2f} wins={row['wins']}/{row['n']} "
              f"{'SEPARABLE' if row['separable'] else 'within noise'}")
    if headline:
        print(f"  headline: {headline['reduction']:.2f}x fewer labels")
    else:
        print("  headline: suppressed (not separable from seed noise)")


if __name__ == "__main__":
    main()
