#!/usr/bin/env python
"""Draw the Deliverable 2 modelling architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "architecture.png"


def box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#2c5f7c",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.4, color="#173b51", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color="#4d7e98",
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top row: data preparation.
    box(ax, 0.02, 0.73, 0.20, 0.18,
        "Raw sources\nOpenAQ, weather,\nstation and calendar files", "#e8f1f5")
    box(ax, 0.27, 0.73, 0.20, 0.18,
        "Phase 2 cleaning\nand feature builder\n(09_build_features.py)", "#e8f1f5")
    box(ax, 0.52, 0.73, 0.20, 0.18,
        "Shared feature table\nfeatures_daily.csv\n2,113 days × 63 columns", "#dcebf2")
    box(ax, 0.77, 0.73, 0.20, 0.18,
        "Shared loader\nchronological split,\nlabel merge, scorable mask", "#dcebf2")

    arrow(ax, 0.22, 0.82, 0.27, 0.82)
    arrow(ax, 0.47, 0.82, 0.52, 0.82)
    arrow(ax, 0.72, 0.82, 0.77, 0.82)

    # Middle row: two modelling tracks.
    box(ax, 0.11, 0.44, 0.30, 0.17,
        "Classification track\nMajority · persistence\nlogistic regression · HGB · MLP",
        "#f4f8fa")
    box(ax, 0.58, 0.44, 0.30, 0.17,
        "Regression track\nPersistence · Ridge\nhistogram gradient boosting",
        "#f4f8fa")
    arrow(ax, 0.86, 0.73, 0.73, 0.61)
    arrow(ax, 0.86, 0.73, 0.27, 0.61)

    # Bottom row: evaluation and delivery.
    box(ax, 0.03, 0.14, 0.21, 0.17,
        "Shared metrics\nmacro F1 · AUC\nRMSE · MAE · R²", "#e8f1f5")
    box(ax, 0.29, 0.14, 0.21, 0.17,
        "Bias / error analysis\nsubgroups, signed error,\nsmall-support warnings", "#e8f1f5")
    box(ax, 0.55, 0.14, 0.20, 0.17,
        "Frozen outputs\nmetrics JSON,\npredictions, charts", "#dcebf2")
    box(ax, 0.79, 0.14, 0.18, 0.17,
        "Deliverable 2 report\n(future app layer would\nconsume forecasts)", "#dcebf2")

    arrow(ax, 0.27, 0.44, 0.14, 0.31)
    arrow(ax, 0.73, 0.44, 0.39, 0.31)
    arrow(ax, 0.73, 0.44, 0.14, 0.31)
    arrow(ax, 0.24, 0.22, 0.29, 0.22)
    arrow(ax, 0.50, 0.22, 0.55, 0.22)
    arrow(ax, 0.75, 0.22, 0.79, 0.22)

    ax.text(0.5, 0.965, "SmogShield Deliverable 2 — modelling and evaluation architecture",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#173b51")
    ax.text(0.5, 0.035,
            "Test is touched once per final model. Preprocessing is fitted on train only. "
            "Every method is scored on the same scorable rows.",
            ha="center", va="bottom", fontsize=7.8, color="#526875")

    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
