from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .real_datasets import RealPanel
from .real_experiments import DirectValidationBundle, SemiEmpiricalBundle, real_features
from .style import MODEL_STYLES, apply_nature_style, clean_axis, panel_label, save_figure


RANK_METRIC_LABELS = {
    "spearman_rho": "Spearman rank correlation",
    "ndcg_at_3": "NDCG@3",
    "top3_recall": "Top-3 recall",
}
ALLOCATION_METRIC_LABELS = {
    "service_level": "Service level",
    "unmet_demand_rate": "Unmet-demand rate",
    "average_lead_time_days": "Average lead time (days)",
    "gini_fill": "Gini coefficient",
    "jain_fairness": "Jain fairness",
    "max_min_fairness": "Max-min fairness",
    "geographic_equity": "Geographic equity",
    "priority_weighted_equity": "Priority-weighted equity",
}


def _style(model: str) -> dict[str, Any]:
    return MODEL_STYLES.get(model, dict(color="0.45", linestyle="-", marker="o"))


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _panel_description(panel: RealPanel) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(panel.region_names):
        mask = panel.regions == region_index
        for k, resource in enumerate(panel.resource_names):
            demand = panel.demand[:, mask, k].ravel()
            capacity = panel.capacity[:, mask, k].ravel()
            confidence = panel.reporting_confidence[:, mask, k].ravel()
            rows.append(
                {
                    "source": panel.source,
                    "region": region,
                    "resource": resource,
                    "n_facilities": int(mask.sum()),
                    "n_time_steps": panel.n_days,
                    "start_date": panel.dates.min().date(),
                    "end_date": panel.dates.max().date(),
                    "mean_observed_demand": demand.mean(),
                    "p95_observed_demand": np.quantile(demand, 0.95),
                    "mean_capacity": capacity.mean(),
                    "mean_capacity_stress": np.mean(demand / np.maximum(capacity, 1.0)),
                    "mean_reporting_confidence": confidence.mean(),
                }
            )
    return pd.DataFrame(rows)


def _source_table(nhs: RealPanel, hhs: RealPanel) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_role": "Primary real-data calibration and holdout validation",
                "source": nhs.source,
                "geography": "London and Midlands, England",
                "cadence": "Daily",
                "period": f"{nhs.dates.min().date()} to {nhs.dates.max().date()}",
                "facilities": nhs.n_nodes,
                "observed_outcomes": "; ".join(nhs.resource_names),
                "selection_rule": "Six trusts per region by reconstructed bed capacity after >=95% completeness screening",
                "source_url": nhs.source_url,
            },
            {
                "analysis_role": "External geographic and reporting-system validation",
                "source": hhs.source,
                "geography": "California and New York, United States",
                "cadence": "Weekly",
                "period": f"{hhs.dates.min().date()} to {hhs.dates.max().date()}",
                "facilities": hhs.n_nodes,
                "observed_outcomes": "; ".join(hhs.resource_names),
                "selection_rule": "Six short-term hospitals per state by staffed adult ICU capacity on prespecified screening weeks",
                "source_url": hhs.source_url,
            },
        ]
    )


def _mapping_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Observed occupancy/capacity", "Capacity stress", "T and F", "Observed; clipped ratio", "Direct validation"],
            ["Confirmed/suspected admissions", "Demand pressure", "T and F", "Observed; robustly scaled", "Direct validation"],
            ["Positive short-run stress change", "Lead/temporal risk proxy", "T, PT and PF", "Observed trend", "Direct validation"],
            ["Source coverage and natural missingness", "Reporting confidence", "PT and PF ambiguity", "Observed metadata", "Both"],
            ["Hospital/DC inventories", "Inventory risk", "T and F", "Simulated, empirically anchored", "Semi-empirical allocation only"],
            ["Transport and supplier availability", "Lead/supplier risk", "T, F, PT and PF", "Simulated; explicitly labelled", "Semi-empirical allocation only"],
        ],
        columns=["input", "encoded_feature", "ACM_channels", "provenance", "analysis_scope"],
    )


def _figure_observed_dynamics(panel: RealPanel, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.3), sharex=True)
    region_styles = [dict(color="black", linestyle="-"), dict(color="0.45", linestyle="--")]
    for region_index, region in enumerate(panel.region_names):
        mask = panel.regions == region_index
        admissions = panel.admissions_confirmed[:, mask].sum(axis=1)
        axes[0].plot(panel.dates, admissions, label=region, **region_styles[region_index])
        for k in range(min(2, panel.n_resources)):
            style = dict(region_styles[region_index])
            style["linestyle"] = "-" if k == 0 else ":"
            axes[1].plot(
                panel.dates,
                panel.demand[:, mask, k].sum(axis=1),
                label=f"{region}: {panel.resource_names[k]}",
                **style,
            )
        k = panel.n_resources - 1
        axes[2].plot(panel.dates, panel.demand[:, mask, k].sum(axis=1), label=region, **region_styles[region_index])
    axes[0].set_ylabel("Daily admissions")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].set_ylabel("Occupied critical beds")
    axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7.2, borderaxespad=0.1)
    axes[2].set_ylabel(f"Occupied {panel.resource_names[-1]}")
    axes[2].set_xlabel("Date")
    axes[2].legend(ncol=2, loc="upper left")
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, "fig_real01_nhs_observed_dynamics", ("png",), dpi)


def _figure_stress_heatmaps(panel: RealPanel, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(panel.n_resources, 1, figsize=(7.2, 2.35 * panel.n_resources), sharex=True)
    axes = np.atleast_1d(axes)
    stress = panel.demand / np.maximum(panel.capacity, 1.0)
    for k, ax in enumerate(axes):
        image = ax.imshow(stress[:, :, k].T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(panel.n_nodes))
        ax.set_yticklabels([f"H{i+1}" for i in range(panel.n_nodes)], fontsize=7)
        ax.set_ylabel(panel.resource_names[k])
        panel_label(ax, chr(97 + k))
        ax.tick_params(which="both", direction="in")
    tick_positions = np.linspace(0, panel.n_days - 1, 6, dtype=int)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([panel.dates[i].strftime("%b %d") for i in tick_positions])
    axes[-1].set_xlabel("Date")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
    colorbar.set_label("Observed capacity stress")
    save_figure(fig, output, "fig_real02_nhs_capacity_stress_heatmaps", ("png",), dpi)


def _figure_evidence(bundle: DirectValidationBundle, output: Path, dpi: int) -> None:
    evidence = bundle.evidence_table.copy()
    example = evidence.loc[evidence["priority"].idxmax()]
    subset = evidence[evidence["resource"].eq(example["resource"])].nlargest(6, "priority")
    features = real_features(bundle.panel, int(np.where(bundle.panel.dates == pd.Timestamp(example["date"]))[0][0]))
    hospital_index = bundle.panel.node_ids.index(str(example["node_id"]))
    resource_index = bundle.panel.resource_names.index(str(example["resource"]))
    signals = [
        features["demand_pressure"][hospital_index, resource_index],
        features["backlog_pressure"][hospital_index, resource_index],
        features["inventory_risk"][hospital_index, resource_index],
        features["lead_risk"][hospital_index, resource_index],
        features["capacity_risk"][hospital_index, resource_index],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    channels = ["T", "F", "PT", "PF"]
    x = np.arange(4)
    axes[0].bar(x - 0.17, [example[f"evidence_{c}"] for c in channels], 0.34, color="white", edgecolor="black", hatch="//", label="Encoded evidence")
    axes[0].bar(x + 0.17, [example[f"state_{c}"] for c in channels], 0.34, color="0.55", edgecolor="black", label="Converged state")
    axes[0].set_xticks(x, channels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Membership degree")
    axes[0].legend(fontsize=8)
    signal_labels = ["Demand", "Backlog", "Inventory", "Trend", "Capacity"]
    axes[1].barh(np.arange(5), signals, color="0.55", edgecolor="black")
    axes[1].set_yticks(np.arange(5), signal_labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Normalized input")
    axes[2].barh(np.arange(len(subset)), subset["priority"], color="0.65", edgecolor="black")
    axes[2].set_yticks(np.arange(len(subset)), [f"H{bundle.panel.node_ids.index(str(v))+1}" for v in subset["node_id"]])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("ACM allocation priority")
    axes[2].set_title(f"{pd.Timestamp(example['date']).date()}, {example['resource']}")
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, "fig_real03_evidence_channels_and_priority", ("png",), dpi)


def _ranking_figure(bundle: DirectValidationBundle, output: Path, stem: str, dpi: int) -> None:
    models = ["ACM-4 (coupled)", "FCM", "Robust-LP", "Current stress", "Admissions"]
    summary = bundle.summary[bundle.summary["model"].isin(models)]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25))
    for ax, metric, label in zip(axes, RANK_METRIC_LABELS, "abc"):
        data = summary[summary["metric"].eq(metric)].set_index("model").reindex(models).dropna()
        x = np.arange(len(data))
        y = data["mean"].to_numpy()
        err = np.vstack([y - data["ci95_low"].to_numpy(), data["ci95_high"].to_numpy() - y])
        bars = ax.bar(x, y, color=[_style(m)["color"] for m in data.index], edgecolor="black")
        for bar, model in zip(bars, data.index):
            if model == "ACM-4 (coupled)":
                bar.set_hatch("//")
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        ax.set_xticks(x, [m.replace("ACM-4 ", "ACM ").replace("Current ", "Current\n") for m in data.index], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(RANK_METRIC_LABELS[metric])
        ax.set_ylim((-0.2, 1.05) if metric == "spearman_rho" else (0.45, 1.02))
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, stem, ("png",), dpi)


def _allocation_performance_figure(bundle: SemiEmpiricalBundle, output: Path, dpi: int) -> None:
    models = ["ACM-4 (coupled)", "FCM", "Robust-LP", "Equal allocation"]
    metrics = ["service_level", "unmet_demand_rate", "average_lead_time_days", "gini_fill"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
    for ax, metric, label in zip(axes.ravel(), metrics, "abcd"):
        data = bundle.summary[(bundle.summary["metric"] == metric) & bundle.summary["model"].isin(models)].set_index("model").reindex(models)
        x = np.arange(len(models))
        y = data["mean"].to_numpy()
        err = np.vstack([y - data["ci95_low"].to_numpy(), data["ci95_high"].to_numpy() - y])
        bars = ax.bar(x, y, color=[_style(m)["color"] for m in models], edgecolor="black")
        bars[0].set_hatch("//")
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        ax.set_xticks(x, ["ACM-4", "FCM", "Robust-LP", "Equal"], rotation=25, ha="right")
        ax.set_ylabel(ALLOCATION_METRIC_LABELS[metric])
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, "fig_real06_semi_empirical_performance", ("png",), dpi)


def _missingness_figure(bundle: SemiEmpiricalBundle, output: Path, dpi: int) -> None:
    summary = bundle.missingness.groupby(["missing_probability", "model"], as_index=False).agg(
        service_level=("service_level", "mean"),
        service_se=("service_level", "sem"),
        gini_fill=("gini_fill", "mean"),
        gini_se=("gini_fill", "sem"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for model in ["ACM-4 (coupled)", "FCM", "Robust-LP"]:
        data = summary[summary["model"].eq(model)]
        style = _style(model)
        axes[0].errorbar(data["missing_probability"], data["service_level"], yerr=1.96 * data["service_se"], label=model, capsize=2, **style)
        axes[1].errorbar(data["missing_probability"], data["gini_fill"], yerr=1.96 * data["gini_se"], label=model, capsize=2, **style)
    axes[0].set_ylabel("Service level")
    axes[1].set_ylabel("Gini coefficient")
    for label, ax in zip("ab", axes):
        ax.set_xlabel("Artificial reporting-missing probability")
        panel_label(ax, label)
        clean_axis(ax)
    axes[0].legend(fontsize=8)
    save_figure(fig, output, "fig_real07_missingness_robustness", ("png",), dpi)


def _ablation_figure(bundle: SemiEmpiricalBundle, output: Path, dpi: int) -> None:
    reference = "ACM-4 (coupled)"
    models = ["ACM-2", "ACM-3", "ACM-4 (independent)"]
    metrics = ["service_level", "gini_fill"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    labels = ["2 channels", "3 channels", "4 channels\nindependent"]
    for ax, metric, label in zip(axes, metrics, "ab"):
        wide = bundle.run_metrics.pivot(index="seed", columns="model", values=metric)
        diffs = np.column_stack([(wide[m] - wide[reference]).dropna().to_numpy() for m in models])
        means = diffs.mean(axis=0)
        ses = diffs.std(axis=0, ddof=1) / np.sqrt(diffs.shape[0])
        y = np.arange(len(models))
        ax.errorbar(means, y, xerr=1.96 * ses, fmt="o", color="black", ecolor="0.45", capsize=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel(rf"Paired difference from coupled ACM-4")
        ax.set_title(ALLOCATION_METRIC_LABELS[metric])
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, "fig_real08_channel_and_coupling_ablation", ("png",), dpi)


def _fairness_figure(bundle: SemiEmpiricalBundle, output: Path, dpi: int) -> None:
    models = ["ACM-4 (coupled)", "FCM", "Robust-LP", "Equal allocation"]
    metrics = ["jain_fairness", "max_min_fairness", "geographic_equity", "priority_weighted_equity"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))
    for ax, metric, label in zip(axes.ravel(), metrics, "abcd"):
        data = bundle.summary[(bundle.summary["metric"] == metric) & bundle.summary["model"].isin(models)].set_index("model").reindex(models)
        y_pos = np.arange(len(models))
        y = data["mean"].to_numpy()
        err = np.vstack([y - data["ci95_low"].to_numpy(), data["ci95_high"].to_numpy() - y])
        ax.errorbar(y, y_pos, xerr=err, fmt="o", color="black", ecolor="0.45", capsize=3)
        ax.set_yticks(y_pos, ["ACM-4", "FCM", "Robust-LP", "Equal"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(ALLOCATION_METRIC_LABELS[metric])
        ax.set_xlim(max(0.0, np.nanmin(data["ci95_low"]) - 0.02), min(1.01, np.nanmax(data["ci95_high"]) + 0.02))
        panel_label(ax, label)
        clean_axis(ax)
    save_figure(fig, output, "fig_real09_multidimensional_fairness", ("png",), dpi)


def _effect_figure(bundle: SemiEmpiricalBundle, output: Path, dpi: int) -> None:
    metrics = ["service_level", "unmet_demand_rate", "average_lead_time_days", "gini_fill"]
    tests = bundle.statistical_tests[
        bundle.statistical_tests["metric"].isin(metrics)
        & bundle.statistical_tests["comparison"].isin(["FCM", "Robust-LP", "Equal allocation"])
    ].copy()
    tests["label"] = tests["metric"].map(ALLOCATION_METRIC_LABELS) + " | " + tests["comparison"]
    tests = tests.sort_values(["metric", "comparison"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(tests))
    colors = np.where(tests["significant_0_05"], "0.20", "0.72")
    ax.barh(y, tests["mean_improvement_favorable"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, tests["label"], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired difference (positive favors ACM)")
    clean_axis(ax)
    save_figure(fig, output, "fig_real10_paired_effects_and_significance", ("png",), dpi)


def _claims_table(nhs_direct: DirectValidationBundle, hhs_direct: DirectValidationBundle, semi: SemiEmpiricalBundle) -> pd.DataFrame:
    def rank_value(bundle: DirectValidationBundle, model: str, metric: str) -> float:
        row = bundle.summary[(bundle.summary.model == model) & (bundle.summary.metric == metric)]
        return float(row.iloc[0]["mean"])

    main = semi.summary.pivot(index="model", columns="metric", values="mean")
    best_service = str(main["service_level"].idxmax())
    best_gini = str(main["gini_fill"].idxmin())
    rows = [
        {
            "candidate_claim": "The real-data analysis validates that the ACM priority is associated with subsequent observed capacity stress.",
            "status": "Supported descriptively",
            "evidence": f"NHS holdout NDCG@3={rank_value(nhs_direct, 'ACM-4 (coupled)', 'ndcg_at_3'):.3f}; HHS external NDCG@3={rank_value(hhs_direct, 'ACM-4 (coupled)', 'ndcg_at_3'):.3f}.",
            "required_wording": "Use association/ranking language; do not claim causal effect.",
        },
        {
            "candidate_claim": "ACM is the best predictor of future capacity stress.",
            "status": "Not supported",
            "evidence": f"Current-stress/Robust-LP comparators achieved NHS NDCG@3={max(rank_value(nhs_direct, 'Current stress', 'ndcg_at_3'), rank_value(nhs_direct, 'Robust-LP', 'ndcg_at_3')):.3f}.",
            "required_wording": "Report the benchmark result transparently and avoid superiority language.",
        },
        {
            "candidate_claim": "ACM produced the highest semi-empirical service level.",
            "status": "Supported" if best_service == "ACM-4 (coupled)" else "Not supported",
            "evidence": f"Highest mean service level was produced by {best_service}; ACM mean={main.loc['ACM-4 (coupled)', 'service_level']:.3f}.",
            "required_wording": "Name the actual best model and describe trade-offs.",
        },
        {
            "candidate_claim": "Four coupled channels are necessary.",
            "status": "Not established by these data",
            "evidence": "The 2-, 3-, 4-channel and independent-channel ablations are reported with confidence intervals; performance differences are empirical and metric-dependent.",
            "required_wording": "State that the ablation tests incremental benefit, not logical necessity.",
        },
        {
            "candidate_claim": "The semi-empirical experiment is fully observational.",
            "status": "Not supported",
            "evidence": f"Observed NHS occupancy drives demand, but transport, inventories and supplier availability are simulated. Lowest mean Gini was produced by {best_gini}.",
            "required_wording": "Always label allocation results as semi-empirical.",
        },
    ]
    return pd.DataFrame(rows)


def _write_results_note(
    path: Path,
    nhs_direct: DirectValidationBundle,
    hhs_direct: DirectValidationBundle,
    semi: SemiEmpiricalBundle,
) -> None:
    rank = nhs_direct.summary.pivot(index="model", columns="metric", values="mean")
    external = hhs_direct.summary.pivot(index="model", columns="metric", values="mean")
    allocation = semi.summary.pivot(index="model", columns="metric", values="mean")
    tests = semi.statistical_tests[
        (semi.statistical_tests.metric == "service_level")
        & (semi.statistical_tests.comparison.isin(["FCM", "Robust-LP", "Equal allocation"]))
    ][["comparison", "mean_improvement_favorable", "p_holm"]]
    lines = [
        "# Real-data and semi-empirical result guardrails",
        "",
        "## Study design",
        "",
        f"The primary panel contains {nhs_direct.panel.n_nodes} NHS trusts observed daily from {nhs_direct.panel.dates.min().date()} to {nhs_direct.panel.dates.max().date()}. The first half was used only to select the partial-evidence coefficient (gamma={nhs_direct.selected_partial_coefficient:.2f}); all reported NHS ranking metrics use the untouched second half.",
        f"External transportability was evaluated without recalibration in {hhs_direct.panel.n_nodes} HHS hospitals from California and New York.",
        "",
        "## Direct validation",
        "",
        f"On the NHS holdout, ACM-4 achieved NDCG@3={rank.loc['ACM-4 (coupled)', 'ndcg_at_3']:.3f}, Spearman rho={rank.loc['ACM-4 (coupled)', 'spearman_rho']:.3f}, and top-3 recall={rank.loc['ACM-4 (coupled)', 'top3_recall']:.3f} for subsequent observed capacity stress.",
        f"On the external HHS panel, corresponding values were NDCG@3={external.loc['ACM-4 (coupled)', 'ndcg_at_3']:.3f}, rho={external.loc['ACM-4 (coupled)', 'spearman_rho']:.3f}, and recall={external.loc['ACM-4 (coupled)', 'top3_recall']:.3f}.",
        f"A simple current-stress comparator scored NHS NDCG@3={rank.loc['Current stress', 'ndcg_at_3']:.3f}; the manuscript must therefore not claim that ACM is the strongest short-horizon predictor.",
        "",
        "## Semi-empirical allocation",
        "",
        f"Across 30 paired runs, ACM-4 mean service level was {allocation.loc['ACM-4 (coupled)', 'service_level']:.3f}, unmet-demand rate {allocation.loc['ACM-4 (coupled)', 'unmet_demand_rate']:.3f}, lead time {allocation.loc['ACM-4 (coupled)', 'average_lead_time_days']:.3f} days, and Gini {allocation.loc['ACM-4 (coupled)', 'gini_fill']:.3f}.",
        "Observed NHS demand and capacity anchor these runs, but inventories, transport lead times, and supplier availability are simulated. These results must always be called semi-empirical, not real-world allocation outcomes.",
        "",
        "Paired service-level comparisons (positive favors ACM):",
        "",
    ]
    for _, row in tests.iterrows():
        lines.append(f"- vs {row['comparison']}: difference={row['mean_improvement_favorable']:.4f}, Holm-adjusted p={row['p_holm']:.4g}.")
    lines.extend(
        [
            "",
            "## Claim rules",
            "",
            "- Report associations and out-of-sample ranking accuracy; do not claim causal clinical benefit.",
            "- Report negative and null comparator results exactly as generated.",
            "- Do not reuse any numerical claim from the old manuscript unless it appears in the new result tables.",
            "- Cite NHS England for the primary panel, HHS for the original US data, and CMU Delphi for the retrieval mirror.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def create_real_outputs(
    nhs_direct: DirectValidationBundle,
    hhs_direct: DirectValidationBundle,
    semi: SemiEmpiricalBundle,
    output_root: Path,
    dpi: int = 600,
) -> dict[str, Path]:
    apply_nature_style(dpi)
    figures = output_root / "figures"
    tables = output_root / "tables"
    raw = output_root / "raw"
    supplement = output_root / "supplement"
    for directory in [figures, tables, raw, supplement]:
        directory.mkdir(parents=True, exist_ok=True)
    for stale in output_root.rglob(".*.tmp.*"):
        stale.unlink(missing_ok=True)

    sources = _source_table(nhs_direct.panel, hhs_direct.panel)
    descriptions = pd.concat([_panel_description(nhs_direct.panel), _panel_description(hhs_direct.panel)], ignore_index=True)
    claims = _claims_table(nhs_direct, hhs_direct, semi)
    fairness = semi.summary[semi.summary.metric.isin(["gini_fill", "jain_fairness", "max_min_fairness", "geographic_equity", "priority_weighted_equity"])].copy()
    ablation = semi.summary[semi.summary.model.isin(["ACM-2", "ACM-3", "ACM-4 (coupled)", "ACM-4 (independent)"])].copy()
    table_map = {
        "table_real_data_sources.csv": sources,
        "table_signal_channel_provenance.csv": _mapping_table(),
        "table_nhs_selected_trusts.csv": nhs_direct.panel.selection_table,
        "table_hhs_selected_hospitals.csv": hhs_direct.panel.selection_table,
        "table_real_data_descriptive_statistics.csv": descriptions,
        "table_partial_coefficient_calibration_real.csv": nhs_direct.calibration,
        "table_nhs_holdout_ranking_validation.csv": nhs_direct.summary,
        "table_nhs_holdout_paired_tests.csv": nhs_direct.statistical_tests,
        "table_hhs_external_ranking_validation.csv": hhs_direct.summary,
        "table_hhs_external_paired_tests.csv": hhs_direct.statistical_tests,
        "table_semi_empirical_allocation_summary.csv": semi.summary,
        "table_semi_empirical_paired_tests.csv": semi.statistical_tests,
        "table_multidimensional_fairness.csv": fairness,
        "table_channel_coupling_ablation.csv": ablation,
        "table_missingness_robustness.csv": semi.missingness,
        "table_claim_guardrails.csv": claims,
    }
    for name, frame in table_map.items():
        _save_csv(frame, tables / name)
    _save_csv(nhs_direct.daily_metrics, raw / "nhs_daily_ranking_metrics.csv")
    _save_csv(hhs_direct.daily_metrics, raw / "hhs_weekly_ranking_metrics.csv")
    _save_csv(semi.run_metrics, raw / "semi_empirical_per_run_metrics.csv")
    _save_csv(nhs_direct.evidence_table, raw / "nhs_example_evidence_decomposition.csv")
    _save_csv(nhs_direct.panel.long_data, raw / "nhs_selected_panel_processed.csv")
    _save_csv(hhs_direct.panel.long_data, raw / "hhs_selected_panel_processed.csv")
    semi.graph.metadata.to_csv(supplement / "real_concept_metadata.csv", index=False)
    np.savetxt(supplement / "real_adjacency_matrix.csv", semi.graph.adjacency, delimiter=",")
    np.savetxt(supplement / "real_sign_matrix.csv", semi.graph.sign, delimiter=",")
    np.savetxt(supplement / "real_edge_strength_matrix.csv", semi.graph.edge_strength, delimiter=",")

    _figure_observed_dynamics(nhs_direct.panel, figures, dpi)
    _figure_stress_heatmaps(nhs_direct.panel, figures, dpi)
    _figure_evidence(nhs_direct, figures, dpi)
    _ranking_figure(nhs_direct, figures, "fig_real04_nhs_holdout_ranking_validation", dpi)
    _ranking_figure(hhs_direct, figures, "fig_real05_hhs_external_ranking_validation", dpi)
    _allocation_performance_figure(semi, figures, dpi)
    _missingness_figure(semi, figures, dpi)
    _ablation_figure(semi, figures, dpi)
    _fairness_figure(semi, figures, dpi)
    _effect_figure(semi, figures, dpi)

    _write_results_note(output_root / "REAL_RESULTS_INTERPRETATION.md", nhs_direct, hhs_direct, semi)
    validation_rows: list[dict[str, Any]] = []
    for image_path in sorted(path for path in figures.glob("*.png") if not path.name.startswith(".")):
        with Image.open(image_path) as image:
            info_dpi = image.info.get("dpi", (0, 0))
            validation_rows.append(
                {
                    "figure": image_path.name,
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi_x": float(info_dpi[0]),
                    "dpi_y": float(info_dpi[1]),
                    "mode": image.mode,
                    "passed": min(info_dpi) >= 590 and image.width >= 1800,
                }
            )
    pdf_files = [str(p.relative_to(output_root)) for p in output_root.rglob("*.pdf")]
    validation = {
        "passed": all(row["passed"] for row in validation_rows) and not pdf_files and len(validation_rows) == 10,
        "figures": validation_rows,
        "pdf_files": pdf_files,
        "expected_figure_count": 10,
        "actual_figure_count": len(validation_rows),
    }
    (output_root / "VALIDATION_REPORT.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_root / "RUN_SUMMARY.json").write_text(
        json.dumps(
            {
                "primary_source": nhs_direct.panel.source,
                "external_source": hhs_direct.panel.source,
                "selected_partial_coefficient": nhs_direct.selected_partial_coefficient,
                "nhs_holdout_start": str(nhs_direct.panel.dates[nhs_direct.calibration_end].date()),
                "paired_allocation_runs": int(semi.run_metrics.seed.nunique()),
                "figure_count": len(validation_rows),
                "figure_formats": ["png"],
                "pdf_generated": False,
                "validation_passed": validation["passed"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "MANIFEST.json").write_text(json.dumps(_manifest(output_root), indent=2), encoding="utf-8")
    return {"tables": tables, "figures": figures, "raw": raw, "validation": output_root / "VALIDATION_REPORT.json"}
