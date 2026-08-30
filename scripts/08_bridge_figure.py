"""Render the slide-ready paired-bridge result in the ExoEgo visual style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "paired_bridge.json"
OUT_PNG = ROOT / "figures" / "paired-bridge-task-recognition.png"
OUT_SVG = ROOT / "figures" / "paired-bridge-task-recognition.svg"

BG = "#080909"
FG = "#f2f2f0"
MUTED = "#77787d"
GRID = "#292a2d"
BAND = "#8a8b8e"


def main() -> None:
    data = json.loads(RESULT.read_text())
    raw = data["test"]["raw_exo"]
    paired = data["test"]["paired_bridge"]
    majority = 100 * data["majority_baseline"]

    x = np.array([0.0, 1.0])
    y = 100 * np.array([raw["top1"], paired["top1"]])
    low = 100 * np.array([raw["top1_ci95"][0], paired["top1_ci95"][0]])
    high = 100 * np.array([raw["top1_ci95"][1], paired["top1_ci95"][1]])

    plt.rcParams.update({
        "font.family": ["Menlo", "DejaVu Sans Mono", "monospace"],
        "font.size": 14,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "svg.fonttype": "none",
    })

    figure = plt.figure(figsize=(16, 9), dpi=120)
    axis = figure.add_axes([0.085, 0.19, 0.85, 0.61])

    # Sparse terminal-like scaffold from the reference design.
    axis.set_xlim(-0.18, 1.18)
    axis.set_ylim(0, 20.5)
    axis.set_yticks([0, 5, 10, 15, 20])
    axis.set_yticklabels(["0.0", "5.0", "10.0", "15.0", "20.0"], color=MUTED)
    axis.set_xticks(x)
    axis.set_xticklabels(["RAW EXO", "SYNCHRONIZED\nEXO → EGO BRIDGE"], color=MUTED)
    axis.tick_params(axis="both", colors=MUTED, length=8, width=1.2, pad=12)
    axis.grid(axis="y", color=GRID, linewidth=1.15)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axis.spines[side].set_color("#444549")
        axis.spines[side].set_linewidth(1.1)

    # Recording-bootstrap uncertainty, treated as one continuous visual band.
    axis.fill_between(x, low, high, color=BAND, alpha=0.19, linewidth=0, zorder=1)
    axis.plot(x, low, color="#55565a", linewidth=1.0, alpha=0.6, zorder=2)
    axis.plot(x, high, color="#55565a", linewidth=1.0, alpha=0.6, zorder=2)

    # Majority reference and the primary paired-transfer result.
    axis.axhline(majority, color="#bfc0c2", linestyle=(0, (1.5, 3.0)),
                 linewidth=1.5, zorder=2)
    axis.text(-0.16, majority + 0.34, f"MAJORITY CLASS  {majority:.1f}%",
              color="#bfc0c2", fontsize=13, ha="left", va="bottom")

    axis.plot(x, y, color=FG, linewidth=3.2, zorder=4)
    axis.scatter([x[0]], [y[0]], s=150, facecolor=BG, edgecolor="#a7a8ab",
                 linewidth=3.0, zorder=5)
    axis.scatter([x[1]], [y[1]], s=170, facecolor=FG, edgecolor=BG,
                 linewidth=2.0, zorder=5)
    axis.vlines(x, low, high, colors="#c5c6c8", linewidth=1.4, zorder=3)
    axis.hlines(low, x - 0.018, x + 0.018, colors="#c5c6c8", linewidth=1.4, zorder=3)
    axis.hlines(high, x - 0.018, x + 0.018, colors="#c5c6c8", linewidth=1.4, zorder=3)

    axis.text(x[0] - 0.04, y[0] + 0.7, f"{y[0]:.1f}%", color="#b9babd",
              fontsize=25, ha="right", va="bottom")
    axis.text(x[1] + 0.05, y[1] + 0.7, f"{y[1]:.1f}%", color=FG,
              fontsize=29, ha="left", va="bottom")

    figure.text(0.03, 0.92, "EGO-TRAINED HEAD  /  TASK RECOGNITION TOP-1 (%)",
                color=MUTED, fontsize=15, ha="left", va="center")
    figure.text(0.03, 0.865,
                "SYNCHRONIZED PAIRS TRANSFER THIRD-PERSON TASK SIGNAL",
                color=FG, fontsize=25, ha="left", va="center")
    figure.text(0.03, 0.105,
                "ASSEMBLY101  /  OFFICIAL RECORDING-DISJOINT TEST SPLIT  /  "
                "875 COARSE-ACTION SEGMENTS  /  68 RECORDINGS",
                color=MUTED, fontsize=13, ha="left", va="center")
    figure.text(0.03, 0.06,
                "SAME FROZEN EGO-TRAINED HEAD  ·  BRIDGE TRAINED WITHOUT EXO ACTION LABELS",
                color="#55565a", fontsize=11, ha="left", va="center")
    figure.text(0.935, 0.105, "BAND = 95% RECORDING BOOTSTRAP",
                color="#55565a", fontsize=10, ha="right", va="center")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PNG, dpi=120)
    figure.savefig(OUT_SVG)
    plt.close(figure)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
