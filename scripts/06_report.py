"""Aggregate the sweep into the label-efficiency curve and a summary table."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego.paths import manifests_dir, results_dir

COLOURS = {"ego_only": "#8b8b8b", "ego_exo": "#4fa3ff", "ego_ego": "#ff8c42"}
LABELS = {
    "ego_only": "ego only (baseline)",
    "ego_exo": "ego + exo (cross-view)",
    "ego_ego": "ego + ego (control)",
}


def style_axis(axis):
    axis.set_facecolor("#111214")
    axis.grid(True, color="#2a2c30", linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axis.spines[side].set_color("#3a3d42")
    axis.tick_params(colors="#b6b9c0", labelsize=9)
    axis.xaxis.label.set_color("#e6e8ec")
    axis.yaxis.label.set_color("#e6e8ec")
    axis.title.set_color("#e6e8ec")


def majority_class_rate():
    """Top-1 accuracy of always predicting the most common eval verb."""
    manifest = manifests_dir() / "segments.csv"
    if not manifest.exists():
        return None
    segments = pd.read_csv(manifest)
    evaluation = segments[segments["role"] == "eval"]
    if evaluation.empty:
        return None
    counts = evaluation["verb_cls"].value_counts()
    return float(counts.iloc[0] / len(evaluation))


def report_gate(frame: pd.DataFrame, baseline: float) -> None:
    """Go/no-go: ego features must beat the majority class before any
    ego_exo vs ego_only delta is worth interpreting."""
    baseline_config = frame[frame["config"] == "ego_only"]
    if baseline_config.empty:
        return
    full = baseline_config[baseline_config["effective_budget"] ==
                           baseline_config["effective_budget"].max()]
    achieved = full["top1"].mean()

    print("=" * 64)
    print("GO/NO-GO GATE")
    print(f"  majority-class baseline (top-1): {baseline:.3f}")
    print(f"  ego_only at full budget (top-1): {achieved:.3f}")
    if achieved > baseline * 1.25:
        print("  PASS -- ego features carry real verb signal; deltas are interpretable.")
    else:
        print("  FAIL -- ego_only is at or near the majority-class baseline.")
        print("  Any ego_exo vs ego_only difference below is noise, not evidence.")
        print("  Fix the backbone (unfreeze, or try a video-native encoder) first.")
    print("=" * 64)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=None)
    parser.add_argument("--metric", default="mean_per_class",
                        choices=["mean_per_class", "top1", "retrieval_map"])
    parser.add_argument("--baseline", type=float, default=None,
                        help="majority-class accuracy; defaults to the eval set's own")
    args = parser.parse_args()

    path = Path(args.results) if args.results else results_dir() / "label_efficiency.csv"
    frame = pd.read_csv(path)

    baseline = args.baseline
    if baseline is None:
        baseline = majority_class_rate()
    if baseline is not None:
        report_gate(frame, baseline)

    summary = frame.groupby(["config", "effective_budget"]).agg(
        mean=(args.metric, "mean"),
        std=(args.metric, "std"),
        n=(args.metric, "size"),
    ).reset_index()
    summary_path = results_dir() / f"summary_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.round(4).to_string(index=False))
    print(f"\nwrote {summary_path}")

    figure, axis = plt.subplots(figsize=(7.2, 4.6), facecolor="#0b0c0e")
    style_axis(axis)

    for config in ["ego_only", "ego_exo", "ego_ego"]:
        part = summary[summary["config"] == config].sort_values("effective_budget")
        if part.empty:
            continue
        colour = COLOURS.get(config, "#cccccc")
        axis.plot(part["effective_budget"], part["mean"], marker="o", markersize=4,
                  color=colour, linewidth=1.8, label=LABELS.get(config, config))
        axis.fill_between(part["effective_budget"], part["mean"] - part["std"].fillna(0),
                          part["mean"] + part["std"].fillna(0), color=colour, alpha=0.15,
                          linewidth=0)

    if args.baseline:
        axis.axhline(args.baseline, color="#5a5d63", linestyle="--", linewidth=1)
        axis.text(axis.get_xlim()[0], args.baseline, " majority class", fontsize=8,
                  color="#8a8d93", va="bottom")

    axis.set_xscale("log")
    axis.set_xlabel("labelled ego clips")
    axis.set_ylabel(args.metric.replace("_", " "))
    axis.set_title("Does exocentric video reduce the labels needed?", fontsize=11, pad=12)
    legend = axis.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#d6d9df")

    figure.tight_layout()
    figure_path = results_dir() / f"label_efficiency_{args.metric}.png"
    figure.savefig(figure_path, dpi=180, facecolor="#0b0c0e")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
