#!/usr/bin/env python3
"""Create the manuscript figure for the two new real-stream evaluations."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_RESULTS = ROOT / "reproducibility" / "results" / "real_streams"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 600,
            "figure.dpi": 120,
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.86", linestyle=":", linewidth=0.6, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    configure_style()
    actual_root = REPOSITORY_RESULTS / "getusppe" / "results"
    ppe_root = REPOSITORY_RESULTS / "ppe_match" / "results"
    optimizer = pd.read_csv(actual_root / "optimizer_preset_comparison.csv").set_index("metric")
    actual = pd.read_csv(actual_root / "test_summary.csv")
    ppe = pd.read_csv(ppe_root / "test_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.45))

    # (a) Data-adaptive calibration effect on historical-assignment metrics.
    ax = axes[0, 0]
    optimizer_metrics = [
        ("allocation_overlap", "Allocation\noverlap"),
        ("priority_overlap", "Priority\noverlap"),
        ("recipient_average_precision", "Recipient\nAP"),
        ("ndcg_all", "Full-list\nNDCG"),
    ]
    changes = [float(optimizer.loc[key, "relative_change_percent"]) for key, _ in optimizer_metrics]
    xpos = np.arange(len(changes))
    ax.bar(xpos, changes, color=["0.78", "0.58", "0.38", "0.18"], width=0.68, zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.7)
    for position, value in zip(xpos, changes):
        ax.text(position, max(value, 0.0) + 0.28, f"{value:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([label for _, label in optimizer_metrics])
    ax.set_ylabel("Change from preset (%)")
    ax.set_ylim(-1.5, 13.5)
    clean_axis(ax)
    panel_label(ax, "a")

    # (b) Agreement with historical GetUsPPE recipient assignments.
    ax = axes[0, 1]
    models = ["ACM-4 (coupled)", "Calibrated scalar", "Demand proportional", "Equal allocation"]
    model_labels = ["ACM-4", "Scalar", "Demand", "Equal"]
    metric_specs = [
        ("allocation_overlap", "Allocation overlap", "o", "0.10"),
        ("recipient_average_precision", "Recipient AP", "s", "0.42"),
        ("ndcg_all", "Full-list NDCG", "^", "0.70"),
    ]
    offsets = [-0.18, 0.0, 0.18]
    for offset, (metric, label, marker, color) in zip(offsets, metric_specs):
        values = actual.loc[actual["metric"].eq(metric)].set_index("model").loc[models]
        x = np.arange(len(models)) + offset
        mean = values["mean"].to_numpy(float)
        low = values["ci_low"].to_numpy(float)
        high = values["ci_high"].to_numpy(float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            fmt=marker,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            capsize=2.2,
            label=label,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Test performance")
    ax.set_ylim(0.0, 0.56)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    panel_label(ax, "b")

    # (c) Prespecified operational score on PPE-Match.
    ax = axes[1, 0]
    ppe_models = ["ACM-4 (coupled)", "Proximity", "Demand proportional", "Equal allocation"]
    ppe_labels = ["ACM-4", "Proximity", "Demand", "Equal"]
    score = ppe.loc[ppe["metric"].eq("operational_score")].set_index("model").loc[ppe_models]
    mean = score["mean"].to_numpy(float)
    low = score["ci_low"].to_numpy(float)
    high = score["ci_high"].to_numpy(float)
    x = np.arange(len(ppe_models))
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color="0.15",
        markerfacecolor="white",
        markeredgecolor="0.15",
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ppe_labels)
    ax.set_ylabel("Composite operational score")
    ax.set_ylim(0.13, 0.215)
    clean_axis(ax)
    panel_label(ax, "c")

    # (d) Coverage-distance trade-off on PPE-Match.
    ax = axes[1, 1]
    coverage = ppe.loc[ppe["metric"].eq("recipient_coverage")].set_index("model")
    distance = ppe.loc[ppe["metric"].eq("unit_miles")].set_index("model")
    markers = ["o", "s", "^", "D"]
    grays = ["0.10", "0.35", "0.60", "0.82"]
    for model, label, marker, color in zip(ppe_models, ppe_labels, markers, grays):
        xv = float(coverage.loc[model, "mean"])
        yv = float(distance.loc[model, "mean"])
        ax.scatter(
            xv,
            yv,
            marker=marker,
            s=35,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
        offsets_text = {
            "ACM-4": (4, 4),
            "Proximity": (-48, 5),
            "Demand": (5, -11),
            "Equal": (4, 4),
        }
        ax.annotate(label, (xv, yv), xytext=offsets_text[label], textcoords="offset points", fontsize=6.8)
    ax.set_yscale("log")
    ax.set_xlabel("Recipient coverage")
    ax.set_ylabel("Unit-miles (log scale)")
    ax.set_xlim(-0.004, 0.086)
    ax.set_ylim(8.0, 1500.0)
    clean_axis(ax)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.97, wspace=0.34, hspace=0.38)
    output = ROOT / "reproducibility" / "figures" / "fig_real10_real_stream_validation.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
