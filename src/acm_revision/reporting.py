from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .core import ConceptGraph, RESOURCE_LABELS, SimulationResult, scaled_tanh
from .experiments import ExperimentBundle, FAIRNESS_METRICS, LOWER_IS_BETTER, PRIMARY_METRICS
from .learning import FEATURE_NAMES, LearningBundle
from .style import MODEL_STYLES, apply_nature_style, clean_axis, panel_label, save_figure


METRIC_LABELS = {
    "service_level": "Service level",
    "unmet_demand_rate": "Unmet demand",
    "average_lead_time_days": "Lead time (days)",
    "gini_fill": "Gini of fill rates",
    "jain_fairness": "Jain fairness",
    "max_min_fairness": "Max-min fairness",
    "geographic_equity": "Geographic equity",
    "priority_weighted_equity": "Priority-weighted equity",
    "mean_convergence_iterations": "Mean convergence iterations",
    "final_convergence_gap": "Final convergence gap",
}


def _style_for(model: str) -> dict[str, Any]:
    return MODEL_STYLES.get(model, dict(color="0.5", linestyle="-", marker="o"))


def _representative(bundle: ExperimentBundle, model: str, seed: int | None = None) -> SimulationResult:
    candidates = [r for r in bundle.results if r.model == model]
    if seed is not None:
        exact = [r for r in candidates if r.seed == seed]
        if exact:
            return exact[0]
    return candidates[0]


def _aggregate_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.30 * frame["service_level"]
        + 0.30 * (1.0 - frame["unmet_demand_rate"])
        + 0.20 * (1.0 - np.clip(frame["average_lead_time_days"] / 7.0, 0.0, 1.0))
        + 0.20 * (1.0 - frame["gini_fill"])
    )


def export_tables(
    bundle: ExperimentBundle,
    learning: LearningBundle,
    profile: dict,
    model_cfg: dict,
    output_root: Path,
) -> dict[str, Path]:
    table_dir = output_root / "tables"
    raw_dir = output_root / "raw"
    supplement_dir = output_root / "supplement"
    for d in (table_dir, raw_dir, supplement_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    bundle.run_metrics.to_csv(raw_dir / "per_run_metrics.csv", index=False)
    bundle.summary.to_csv(table_dir / "table_performance_summary_long.csv", index=False)
    bundle.statistical_tests.to_csv(table_dir / "table_paired_statistical_tests.csv", index=False)
    bundle.omnibus_tests.to_csv(table_dir / "table_friedman_tests.csv", index=False)
    bundle.calibration.to_csv(table_dir / "table_partial_coefficient_calibration.csv", index=False)
    bundle.sensitivity.to_csv(raw_dir / "sensitivity_all_runs.csv", index=False)
    bundle.sensitivity_grid.to_csv(raw_dir / "sensitivity_alpha_mix_grid.csv", index=False)
    learning.training_history.to_csv(raw_dir / "ppo_training_history.csv", index=False)
    learning.calibration.to_csv(table_dir / "table_learning_baseline_calibration.csv", index=False)

    summary_primary = bundle.summary[bundle.summary["metric"].isin(PRIMARY_METRICS)].copy()
    summary_primary["mean_sd"] = summary_primary.apply(lambda r: f"{r['mean']:.4f} +/- {r['sd']:.4f}", axis=1)
    summary_primary["ci95"] = summary_primary.apply(lambda r: f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", axis=1)
    performance_wide = summary_primary.pivot(index="model", columns="metric", values="mean_sd").reset_index()
    performance_wide.to_csv(table_dir / "table_main_performance_mean_sd.csv", index=False)

    ablation_models = ["ACM-4 (coupled)", "ACM-4 (independent)", "ACM-3", "ACM-2", "FCM"]
    performance_wide[performance_wide["model"].isin(ablation_models)].to_csv(table_dir / "table_ablation.csv", index=False)

    fairness = bundle.summary[bundle.summary["metric"].isin(FAIRNESS_METRICS)].copy()
    fairness.to_csv(table_dir / "table_fairness_metrics.csv", index=False)

    convergence = bundle.summary[bundle.summary["metric"].isin(["mean_convergence_iterations", "final_convergence_gap"])].copy()
    stability = pd.DataFrame(
        [
            {
                "variant": "ACM-4 coupled",
                "operator_2_norm": bundle.graph.operator_norm_coupled,
                "contraction_bound": bundle.graph.contraction_bound_coupled,
                "condition_satisfied": bundle.graph.contraction_bound_coupled < 1.0,
            },
            {
                "variant": "ACM-4 independent",
                "operator_2_norm": bundle.graph.operator_norm_independent,
                "contraction_bound": bundle.graph.contraction_bound_independent,
                "condition_satisfied": bundle.graph.contraction_bound_independent < 1.0,
            },
        ]
    )
    convergence.to_csv(table_dir / "table_convergence_empirical.csv", index=False)
    stability.to_csv(table_dir / "table_stability_theoretical.csv", index=False)

    sensitivity_summary = bundle.sensitivity.groupby(["factor", "value"], as_index=False).agg(
        service_level_mean=("service_level", "mean"),
        service_level_sd=("service_level", "std"),
        unmet_demand_mean=("unmet_demand_rate", "mean"),
        lead_time_mean=("average_lead_time_days", "mean"),
        gini_mean=("gini_fill", "mean"),
        jain_mean=("jain_fairness", "mean"),
        max_min_mean=("max_min_fairness", "mean"),
        geographic_equity_mean=("geographic_equity", "mean"),
        priority_weighted_equity_mean=("priority_weighted_equity", "mean"),
        contraction_bound=("contraction_bound", "mean"),
    )
    temp = bundle.sensitivity.copy()
    temp["aggregate_score"] = _aggregate_score(temp)
    agg = temp.groupby(["factor", "value"], as_index=False)["aggregate_score"].mean()
    sensitivity_summary = sensitivity_summary.merge(agg, on=["factor", "value"])
    baseline_scores = sensitivity_summary.loc[
        ((sensitivity_summary["factor"] == "transfer_alpha") & np.isclose(sensitivity_summary["value"], model_cfg["transfer_alpha"]))
        | ((sensitivity_summary["factor"] == "physical_evidence_mix") & np.isclose(sensitivity_summary["value"], model_cfg["physical_evidence_mix"]))
        | ((sensitivity_summary["factor"] == "partial_truth_coefficient") & np.isclose(sensitivity_summary["value"], bundle.selected_partial_coefficient))
        | ((sensitivity_summary["factor"].isin(["demand_scale", "lead_time_scale", "capacity_scale", "rbf_sigma_scale"])) & np.isclose(sensitivity_summary["value"], 1.0))
    ].groupby("factor")["aggregate_score"].first()
    sensitivity_summary["relative_change_percent"] = sensitivity_summary.apply(
        lambda r: 100.0 * (r["aggregate_score"] - baseline_scores.get(r["factor"], r["aggregate_score"])) / max(abs(baseline_scores.get(r["factor"], 1.0)), 1e-12), axis=1
    )
    sensitivity_summary.to_csv(table_dir / "table_sensitivity_summary.csv", index=False)

    parameters = [
        ("Simulation horizon", profile["days"], "days"),
        ("Hospitals", profile["n_hospitals"], "count"),
        ("Distribution centers", profile["n_dcs"], "count"),
        ("Suppliers", profile["n_suppliers"], "count"),
        ("Evaluation runs", profile["n_runs"], "paired seeds"),
        ("Transfer alpha", model_cfg["transfer_alpha"], "dimensionless"),
        ("Physical evidence mix", model_cfg["physical_evidence_mix"], "dimensionless"),
        ("Selected partial coefficient", bundle.selected_partial_coefficient, "cross-validated"),
        ("Priority objective weight", model_cfg["priority_objective_weight"], "LP coefficient"),
        ("Convergence tolerance", model_cfg["convergence_tolerance"], "L-infinity"),
        ("Backlog rule", model_cfg["backlog_mode"], "policy"),
        ("PPO interactions", profile["ppo_interactions"], "training decisions"),
        ("Bootstrap samples", profile["bootstrap_samples"], "resamples"),
    ]
    parameter_table = pd.DataFrame(parameters, columns=["parameter", "value", "unit_or_note"])
    parameter_table.to_csv(table_dir / "table_simulation_parameters.csv", index=False)

    signal_map = pd.DataFrame(
        [
            ("Hospital demand pressure", "Observed demand / recent demand scale", "Hospital shortage-risk T and PT when high", "Reporting confidence"),
            ("Hospital backlog pressure", "Backlog / (demand + backlog)", "Hospital shortage-risk T and PT when high", "Operational record"),
            ("Hospital inventory risk", "1 - available stock / projected need", "Hospital shortage-risk T and PT when high", "Inventory accuracy"),
            ("Hospital lead-time risk", "Normalized realized route delay", "Hospital shortage-risk T and PT when high", "Route reliability"),
            ("Regional capacity risk", "1 - DC stock / regional projected need", "Hospital shortage-risk T and PT when high", "Supply visibility"),
            ("DC supply availability", "Mean of stock, spare-load, capacity, and route adequacy", "DC availability T and PT when high", "Operational and route records"),
            ("Supplier availability", "Mean of supplier and regional capacity adequacy", "Supplier availability T and PT when high", "Supplier and capacity records"),
        ],
        columns=["signal", "operational_definition", "channel_effect", "confidence_source"],
    )
    signal_map.to_csv(table_dir / "table_signal_to_channel_mapping.csv", index=False)

    baseline_settings = pd.DataFrame(
        [
            ("ACM-4 (coupled)", "4-channel coupled operator + daily physical evidence", "Same allocation LP"),
            ("FCM", "Single state, same transfer and physical evidence", "Same allocation LP"),
            ("ACM-2", "True/false channels", "Same allocation LP"),
            ("ACM-3", "True/false/undirected uncertainty", "Same allocation LP"),
            ("ACM-4 (independent)", "Four channels; diagonal channel coupling", "Same allocation LP"),
            ("SGC-GNN", "Two graph-smoothing passes + 64/32 MLP", "Same allocation LP"),
            ("Bayesian network", "Discretized causal risk posterior", "Same allocation LP"),
            ("PPO", f"Clipped policy optimization; {profile['ppo_interactions']} interactions", "Same allocation LP"),
            ("Robust-LP", "95% upper-demand uncertainty score", "Robust priorities + same feasibility constraints"),
        ],
        columns=["model", "configuration", "allocation_mechanism"],
    )
    baseline_settings.to_csv(table_dir / "table_baseline_settings.csv", index=False)

    representative = _representative(bundle, "ACM-4 (coupled)")
    representative.daily.to_csv(raw_dir / "representative_acm_daily.csv", index=False)
    representative.hospital_resource.to_csv(table_dir / "table_hospital_resource_outcomes.csv", index=False)
    representative.allocations.to_csv(raw_dir / "representative_acm_allocations.csv", index=False)
    checkpoint = representative.allocations[representative.allocations["day"].isin([0, 10, 20, 30, 39])]
    if not checkpoint.empty:
        allocation_schedule = checkpoint.groupby(["hospital", "day"], as_index=False)["quantity"].sum().pivot(index="hospital", columns="day", values="quantity").fillna(0).reset_index()
    else:
        allocation_schedule = pd.DataFrame()
    allocation_schedule.to_csv(table_dir / "table_allocation_checkpoints.csv", index=False)

    meta = bundle.graph.metadata
    meta.to_csv(supplement_dir / "concept_metadata.csv", index=False)
    pd.DataFrame(bundle.graph.adjacency).to_csv(supplement_dir / "adjacency_matrix.csv", index=False)
    pd.DataFrame(bundle.graph.sign).to_csv(supplement_dir / "sign_matrix.csv", index=False)
    pd.DataFrame(bundle.graph.edge_strength).to_csv(supplement_dir / "edge_strength_matrix.csv", index=False)
    for c, name in enumerate(("true", "false", "partially_true", "partially_false")):
        pd.DataFrame(bundle.graph.channel_weights[c]).to_csv(supplement_dir / f"weight_matrix_{name}.csv", index=False)
    pd.DataFrame(bundle.graph.coupled_matrix).to_csv(supplement_dir / "channel_coupling_matrix.csv", index=False)

    formulas = pd.DataFrame(
        [
            ("Aggregate performance score", "0.30*Service + 0.30*(1-Unmet) + 0.20*(1-min(Lead/7,1)) + 0.20*(1-Gini)", "All components scaled to [0,1]; higher is better"),
            ("Relative change", "100*(Score_variant-Score_baseline)/abs(Score_baseline)", "Computed within each sensitivity factor"),
            ("Service level", "Total fulfilled original demand / total original demand", "Backlogged demand is not double counted"),
            ("Average lead time", "Sum(received quantity*realized route delay)/sum(received quantity)", "Only received shipments within horizon"),
        ],
        columns=["quantity", "formula", "interpretation"],
    )
    formulas.to_csv(table_dir / "table_metric_formulas.csv", index=False)

    workbook = table_dir / "all_manuscript_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        performance_wide.to_excel(writer, sheet_name="Main performance", index=False)
        bundle.statistical_tests.to_excel(writer, sheet_name="Paired tests", index=False)
        bundle.omnibus_tests.to_excel(writer, sheet_name="Friedman tests", index=False)
        fairness.to_excel(writer, sheet_name="Fairness", index=False)
        sensitivity_summary.to_excel(writer, sheet_name="Sensitivity", index=False)
        bundle.calibration.to_excel(writer, sheet_name="Projection calibration", index=False)
        stability.to_excel(writer, sheet_name="Stability", index=False)
        baseline_settings.to_excel(writer, sheet_name="Baselines", index=False)
        signal_map.to_excel(writer, sheet_name="Signal mapping", index=False)
        parameter_table.to_excel(writer, sheet_name="Parameters", index=False)
        formulas.to_excel(writer, sheet_name="Metric formulas", index=False)
        representative.hospital_resource.to_excel(writer, sheet_name="Hospital outcomes", index=False)
    paths["workbook"] = workbook
    return paths


def plot_workflow(output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.12, 0.72, "Daily physical inputs\nD, I, Q, L, C"),
        (0.37, 0.72, "Four-channel\nevidence encoding"),
        (0.62, 0.72, "Coupled ACM update\nand projection"),
        (0.87, 0.72, "Priority score\nand normalization"),
        (0.62, 0.28, "Allocation LP\n(shared constraints)"),
        (0.37, 0.28, "Delayed arrivals and\ninventory balances"),
        (0.12, 0.28, "KPIs, fairness and\nstatistical analysis"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x - 0.105, y - 0.09), 0.21, 0.18, facecolor="white", edgecolor="black", linewidth=0.8))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5)
    arrows = [(0.225, 0.72, 0.265, 0.72), (0.475, 0.72, 0.515, 0.72), (0.725, 0.72, 0.765, 0.72),
              (0.87, 0.63, 0.69, 0.37), (0.515, 0.28, 0.475, 0.28), (0.265, 0.28, 0.225, 0.28)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    ax.annotate("", xy=(0.12, 0.63), xytext=(0.37, 0.37),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9, connectionstyle="arc3,rad=-0.28"))
    ax.text(0.26, 0.48, "Next-day physical state", ha="center", va="center", fontsize=7.5)
    ax.text(0.12, 0.09, "Evaluation output", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, output_dir, "fig01_decision_support_workflow", formats, dpi)


def plot_graph_structure(graph: ConceptGraph, output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax = axes[0, 0]
    ax.axis("off")
    levels = [(0.10, "Suppliers\n(2 x 4 concepts)"), (0.50, "Distribution centers\n(2 x 4 concepts)"), (0.90, "Hospitals\n(12 x 4 concepts)")]
    box_specs = [
        (0.50, 0.82, "Supplier availability\n2 x 4 concepts", "0.86"),
        (0.50, 0.52, "DC availability\n2 x 4 concepts", "0.72"),
        (0.50, 0.18, "Hospital shortage risk\n12 x 4 concepts", "0.58"),
    ]
    for x, y, label, gray in box_specs:
        ax.add_patch(plt.Rectangle((x - 0.21, y - 0.08), 0.42, 0.16, facecolor=gray, edgecolor="black", lw=0.8))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.74), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.54, 0.67, "availability (+)", ha="left", va="center", fontsize=6.5)
    ax.annotate("", xy=(0.44, 0.26), xytext=(0.44, 0.44), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.03, 0.35, "DC availability\nreduces risk (-)", ha="left", va="center", fontsize=6.3)
    ax.annotate("", xy=(0.56, 0.44), xytext=(0.56, 0.26), arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="--"))
    ax.text(0.61, 0.35, "Hospital pressure\ndepletes DC (-)", ha="left", va="center", fontsize=6.3)
    ax.text(0.50, 0.02, "Within-region spillover (+); cross-resource risk links (+)", ha="center", fontsize=6.1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Case-specific concept structure")
    panel_label(ax, "a")

    matrices = [graph.adjacency, graph.sign, np.sum(graph.channel_weights, axis=0) * graph.edge_strength]
    titles = ["Adjacency matrix", "Signed influence matrix", "Normalized edge magnitude"]
    for p, (matrix, title) in enumerate(zip(matrices, titles), start=1):
        ax = axes.ravel()[p]
        if p == 1:
            vmin, vmax = 0.0, 1.0
        elif p == 2:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(matrix, cmap="Greys", interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Source concept")
        ax.set_ylabel("Target concept")
        for boundary in (47.5, 55.5):
            ax.axvline(boundary, color="0.35", lw=0.55)
            ax.axhline(boundary, color="0.35", lw=0.55)
        ax.set_xticks([23.5, 51.5, 59.5], ["Hosp.", "DC", "Sup."], fontsize=6.5)
        ax.set_yticks([23.5, 51.5, 59.5], ["Hosp.", "DC", "Sup."], fontsize=6.5)
        ax.tick_params(direction="in", top=True, right=True, labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        panel_label(ax, chr(ord("a") + p))
    save_figure(fig, output_dir, "fig02_case_graph_and_matrices", formats, dpi)


def plot_transfer_and_stability(bundle: ExperimentBundle, model_cfg: dict, output_dir: Path, formats: list[str], dpi: int) -> None:
    alpha = float(model_cfg["transfer_alpha"])
    x = np.linspace(-4, 4, 500)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    ax.plot(x, scaled_tanh(x, alpha), color="black")
    ax.axhline(0, color="0.6", lw=0.6); ax.axhline(1, color="0.6", lw=0.6)
    ax.set(xlabel="Input", ylabel=r"$\psi_\alpha(x)$", title=r"Bounded transfer: $\mathbb{R}\to(0,1)$")
    clean_axis(ax); panel_label(ax, "a")
    ax = axes[0, 1]
    derivative = 0.5 * alpha / np.cosh(alpha * x) ** 2
    ax.plot(x, derivative, color="black", linestyle="--")
    ax.axhline(alpha / 2, color="0.5", linestyle=":", label=r"$L_\psi=\alpha/2$")
    ax.set(xlabel="Input", ylabel="Derivative", title="Global Lipschitz bound")
    ax.legend(loc="upper right"); clean_axis(ax); panel_label(ax, "b")
    ax = axes[1, 0]
    norms = np.linspace(0.0, 4.0, 300)
    dynamic_mix = 1.0 - float(model_cfg["physical_evidence_mix"])
    bounds_curve = dynamic_mix * (alpha / 2.0) * norms
    ax.plot(norms, bounds_curve, color="black")
    ax.axhline(1.0, color="0.45", linestyle="--")
    ax.axvline(bundle.graph.operator_norm_coupled, color="0.25", linestyle=":")
    ax.plot(bundle.graph.operator_norm_coupled, bundle.graph.contraction_bound_coupled, "o", color="black")
    ax.set(xlabel=r"Block-operator norm $\|B_C\|_2$", ylabel="Sufficient bound", title="Configured norm and stability margin")
    ax.text(3.92, 1.02, "threshold = 1", ha="right", va="bottom", fontsize=7)
    ax.text(bundle.graph.operator_norm_coupled + 0.06, 0.70, "configured norm = 1.35", rotation=90, ha="left", va="center", fontsize=7)
    clean_axis(ax); panel_label(ax, "c")
    ax = axes[1, 1]
    names = ["Coupled", "Independent"]
    bounds = [bundle.graph.contraction_bound_coupled, bundle.graph.contraction_bound_independent]
    ax.bar(names, bounds, color=["0.25", "0.72"], edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set(ylabel="Sufficient contraction bound", title="Configured operator condition")
    clean_axis(ax); panel_label(ax, "d")
    save_figure(fig, output_dir, "fig03_transfer_and_stability", formats, dpi)


def plot_scenario_dynamics(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    result = _representative(bundle, "ACM-4 (coupled)")
    daily = result.daily
    days = daily["day"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    axes[0].plot(days, result.fulfilled_history.sum(axis=(1, 2)), color="black", linestyle="--", label="Fulfilled")
    actual_demand = result.demand_history.sum(axis=(1, 2))
    axes[0].plot(days, actual_demand, color="0.55", linestyle="-", label="Original demand")
    axes[0].set_ylabel("Units/day"); axes[0].legend(ncol=2, loc="upper right"); panel_label(axes[0], "a")
    axes[1].plot(days, daily["hospital_inventory"], color="black", label="Hospital")
    axes[1].plot(days, daily["dc_inventory"], color="0.55", linestyle="--", label="DC")
    axes[1].set_ylabel("Inventory units"); axes[1].legend(ncol=2, loc="upper right"); panel_label(axes[1], "b")
    axes[2].plot(days, daily["backlog_units"], color="black")
    axes[2].set(xlabel="Day", ylabel="Backlog units"); panel_label(axes[2], "c")
    for ax in axes: clean_axis(ax)
    save_figure(fig, output_dir, "fig04_epidemic_scenario_dynamics", formats, dpi)


def plot_convergence(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for model in ["ACM-4 (coupled)", "FCM"]:
        result = _representative(bundle, model)
        gaps = result.convergence_history[min(20, len(result.convergence_history) - 1)]
        axes[0, 0].semilogy(np.arange(1, len(gaps) + 1), gaps, label=model, **{k: v for k, v in _style_for(model).items() if k != "marker"})
    axes[0, 0].set(xlabel="Inner iteration", ylabel=r"$\|x^{m+1}-x^m\|_\infty$", title="Representative convergence")
    axes[0, 0].legend(loc="upper right"); clean_axis(axes[0, 0]); panel_label(axes[0, 0], "a")

    models = ["ACM-4 (coupled)", "FCM", "ACM-4 (independent)"]
    for model in models:
        result = _representative(bundle, model)
        axes[0, 1].plot(np.arange(len(result.iteration_history)), result.iteration_history, label=model, **_style_for(model))
    axes[0, 1].set(xlabel="Day", ylabel="Iterations", title="Daily convergence effort")
    axes[0, 1].legend(loc="upper right", fontsize=7); clean_axis(axes[0, 1]); panel_label(axes[0, 1], "b")

    iterative = ["ACM-2", "ACM-3", "ACM-4 (coupled)", "ACM-4 (independent)", "FCM"]
    conv = bundle.summary[(bundle.summary["metric"] == "mean_convergence_iterations") & bundle.summary["model"].isin(iterative)].copy()
    conv["order"] = conv["model"].map({m: i for i, m in enumerate(iterative)})
    conv = conv.sort_values("order")
    names = conv["model"].tolist()
    y = np.arange(len(names))
    axes[1, 0].errorbar(conv["mean"], y, xerr=[conv["mean"] - conv["ci95_low"], conv["ci95_high"] - conv["mean"]], fmt="o", color="black", ecolor="0.45", capsize=2)
    axes[1, 0].set_yticks(y, names, fontsize=7)
    axes[1, 0].set(xlabel="Mean iterations (95% CI)", title="Across-run convergence")
    clean_axis(axes[1, 0]); panel_label(axes[1, 0], "c")

    result = _representative(bundle, "ACM-4 (coupled)")
    score = result.score_history[:, :, 0]
    for h in [0, 3, 6, 10]:
        axes[1, 1].plot(np.arange(score.shape[0]), score[:, h], label=f"H{h+1}", **dict(color=str(0.1 + 0.18 * (h % 4)), linestyle=["-", "--", ":", "-."][h % 4]))
    axes[1, 1].set(xlabel="Day", ylabel="Ventilator decision score", title="Adaptive hospital scores")
    axes[1, 1].legend(ncol=2, fontsize=7); clean_axis(axes[1, 1]); panel_label(axes[1, 1], "d")
    save_figure(fig, output_dir, "fig05_convergence_and_dynamic_scores", formats, dpi)


def plot_channels_and_interpretability(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    result = _representative(bundle, "ACM-4 (coupled)")
    snap = result.evidence_snapshot
    day = min(20, result.priority_history.shape[0] - 1)
    h, k = np.unravel_index(np.argmax(result.priority_history[day]), result.priority_history[day].shape)
    state = np.asarray(snap.get("hospital_state"))
    detail = snap.get("evidence_detail", {})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    if state.size:
        channels = ["T", "F", "PT", "PF"]
        vals = state[h, k]
        axes[0, 0].bar(channels, vals, color=["0.15", "0.45", "0.68", "0.85"], edgecolor="black", linewidth=0.7)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set(ylabel="Membership degree", title=f"H{h+1}-{RESOURCE_LABELS[k]} channel state")
    clean_axis(axes[0, 0]); panel_label(axes[0, 0], "a")
    sources = np.asarray(detail.get("source_scores"))
    if sources.size:
        source_vals = sources[h, k]
        source_names = [str(v) for v in np.asarray(detail.get("source_names", FEATURE_NAMES[: len(source_vals)]))]
        axes[0, 1].barh(np.arange(len(source_vals)), source_vals, color="0.55", edgecolor="black", linewidth=0.6)
        axes[0, 1].set_yticks(np.arange(len(source_vals)), source_names, fontsize=7)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set(xlim=(0, 1), xlabel="Normalized risk evidence", title="Operational evidence behind priority")
    clean_axis(axes[0, 1]); panel_label(axes[0, 1], "b")
    axes[1, 0].plot(np.arange(result.priority_history.shape[0]), result.priority_history[:, h, k], color="black")
    axes[1, 0].axvline(day, color="0.5", linestyle="--")
    axes[1, 0].set(xlabel="Day", ylabel="Normalized priority", title="Priority trajectory")
    clean_axis(axes[1, 0]); panel_label(axes[1, 0], "c")
    gamma = bundle.selected_partial_coefficient
    terms = np.array([state[h, k, 0], -state[h, k, 1], gamma * state[h, k, 2], -gamma * state[h, k, 3]]) if state.size else np.zeros(4)
    axes[1, 1].bar([r"$+T$", r"$-F$", r"$+\gamma PT$", r"$-\gamma PF$"], terms, color=["0.15", "0.45", "0.68", "0.85"], edgecolor="black", linewidth=0.6)
    axes[1, 1].axhline(0, color="black", lw=0.7)
    axes[1, 1].tick_params(axis="x", labelrotation=25)
    axes[1, 1].set(ylabel="Score contribution", title=rf"Decision projection ($\gamma={gamma:.2f}$)")
    clean_axis(axes[1, 1]); panel_label(axes[1, 1], "d")
    save_figure(fig, output_dir, "fig06_channels_and_interpretability", formats, dpi)


def _plot_metric_panels(bundle: ExperimentBundle, models: list[str], metrics: list[str], stem: str, output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for p, metric in enumerate(metrics):
        ax = axes.ravel()[p]
        sub = bundle.summary[(bundle.summary["metric"] == metric) & bundle.summary["model"].isin(models)].copy()
        sub["order"] = sub["model"].map({m: i for i, m in enumerate(models)})
        sub = sub.sort_values("order")
        x = np.arange(len(sub))
        means = sub["mean"].to_numpy()
        lo = means - sub["ci95_low"].to_numpy()
        hi = sub["ci95_high"].to_numpy() - means
        ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color="black", ecolor="0.45", capsize=2)
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in sub["model"]], rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(METRIC_LABELS[metric])
        better = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        ax.set_title(better)
        clean_axis(ax); panel_label(ax, chr(ord("a") + p))
    save_figure(fig, output_dir, stem, formats, dpi)


def plot_performance(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    models = list(bundle.run_metrics["model"].unique())
    _plot_metric_panels(bundle, models, list(PRIMARY_METRICS), "fig07_model_performance_95ci", output_dir, formats, dpi)


def plot_robustness(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    metrics = list(PRIMARY_METRICS)
    models = list(bundle.run_metrics["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for p, metric in enumerate(metrics):
        ax = axes.ravel()[p]
        data = [bundle.run_metrics.loc[bundle.run_metrics["model"] == m, metric].to_numpy() for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color="black", linewidth=1.2))
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(str(0.92 - 0.06 * (i % 6)))
            box.set_edgecolor("black")
        ax.set_xticks(np.arange(1, len(models) + 1), [m.replace(" ", "\n", 1) for m in models], rotation=35, ha="right", fontsize=6.3)
        ax.set_ylabel(METRIC_LABELS[metric])
        clean_axis(ax); panel_label(ax, chr(ord("a") + p))
    save_figure(fig, output_dir, "fig08_robustness_30_paired_runs", formats, dpi)


def plot_ablation(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    models = ["ACM-4 (coupled)", "ACM-4 (independent)", "ACM-3", "ACM-2", "FCM"]
    _plot_metric_panels(bundle, models, list(PRIMARY_METRICS), "fig09_ablation_channels_and_coupling", output_dir, formats, dpi)


def plot_fairness(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    metrics = list(FAIRNESS_METRICS)
    models = list(bundle.run_metrics["model"].unique())
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4))
    for p, metric in enumerate(metrics):
        ax = axes.ravel()[p]
        sub = bundle.summary[(bundle.summary["metric"] == metric)].set_index("model").reindex(models).reset_index()
        x = np.arange(len(models))
        ax.plot(x, sub["mean"], color="black", linestyle="-", marker="o")
        ax.fill_between(x, sub["ci95_low"], sub["ci95_high"], color="0.85", alpha=0.65)
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in models], rotation=35, ha="right", fontsize=5.8)
        ax.set_ylabel(METRIC_LABELS[metric])
        clean_axis(ax); panel_label(ax, chr(ord("a") + p))
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.05, 0.75, "Directions", fontsize=10)
    axes.ravel()[-1].text(0.05, 0.56, "Gini: lower is fairer\nAll other indices: higher is fairer", fontsize=9, va="top")
    save_figure(fig, output_dir, "fig10_multidimensional_fairness", formats, dpi)


def plot_sensitivity(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    factors = ["transfer_alpha", "physical_evidence_mix", "partial_truth_coefficient", "demand_scale", "lead_time_scale", "capacity_scale"]
    labels = ["Transfer alpha", "Physical evidence mix", "Partial coefficient", "Demand scale", "Lead-time scale", "Capacity scale"]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0))
    for p, (factor, label) in enumerate(zip(factors, labels)):
        ax = axes.ravel()[p]
        sub = bundle.sensitivity[bundle.sensitivity["factor"] == factor].groupby("value", as_index=False).agg(
            service=("service_level", "mean"), service_se=("service_level", "sem"),
            gini=("gini_fill", "mean"), gini_se=("gini_fill", "sem"))
        ax.errorbar(sub["value"], sub["service"], yerr=1.96 * sub["service_se"], color="black", linestyle="-", marker="o", capsize=2, label="Service")
        ax.errorbar(sub["value"], 1 - sub["gini"], yerr=1.96 * sub["gini_se"], color="0.65", linestyle=":", marker="^", capsize=2, label=r"$1-$Gini")
        baseline = bundle.selected_partial_coefficient if factor == "partial_truth_coefficient" else {"transfer_alpha": 1.0, "physical_evidence_mix": 0.45}.get(factor, 1.0)
        ax.axvline(baseline, color="0.45", linestyle="--", linewidth=0.7)
        ax.set(xlabel=label, ylabel="Favorable metric value")
        if p == 0: ax.legend(loc="lower right", fontsize=7)
        clean_axis(ax); panel_label(ax, chr(ord("a") + p))
    save_figure(fig, output_dir, "fig11_wide_parameter_sensitivity", formats, dpi)


def plot_sensitivity_grid(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    grouped = bundle.sensitivity_grid.groupby(["transfer_alpha", "physical_evidence_mix"], as_index=False).mean(numeric_only=True)
    alphas = np.sort(grouped["transfer_alpha"].unique())
    mixes = np.sort(grouped["physical_evidence_mix"].unique())
    service = grouped.pivot(index="physical_evidence_mix", columns="transfer_alpha", values="service_level").reindex(index=mixes, columns=alphas).to_numpy()
    bound = grouped.pivot(index="physical_evidence_mix", columns="transfer_alpha", values="contraction_bound").reindex(index=mixes, columns=alphas).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for p, (matrix, title) in enumerate([(service, "Mean service level"), (bound, "Sufficient contraction bound")]):
        im = axes[p].imshow(matrix, origin="lower", cmap="Greys", aspect="auto")
        axes[p].set_xticks(np.arange(len(alphas)), [f"{x:.1f}" for x in alphas])
        axes[p].set_yticks(np.arange(len(mixes)), [f"{x:.2f}" for x in mixes])
        axes[p].set(xlabel="Transfer alpha", ylabel="Physical evidence mix", title=title)
        fig.colorbar(im, ax=axes[p], fraction=0.046, pad=0.03)
        panel_label(axes[p], chr(ord("a") + p))
    save_figure(fig, output_dir, "fig12_sensitivity_grid_and_stability", formats, dpi)


def plot_training_and_calibration(bundle: ExperimentBundle, learning: LearningBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    hist = learning.training_history
    axes[0, 0].plot(hist["interaction"], hist["mean_reward"], color="black")
    axes[0, 0].set(xlabel="PPO interactions", ylabel="Mean reward", title="PPO training diagnostic")
    clean_axis(axes[0, 0]); panel_label(axes[0, 0], "a")
    axes[0, 1].plot(hist["interaction"], hist["policy_target_correlation"], color="black", linestyle="--")
    axes[0, 1].set(xlabel="PPO interactions", ylabel="Target correlation", title="Policy calibration")
    clean_axis(axes[0, 1]); panel_label(axes[0, 1], "b")
    cal = bundle.calibration.groupby("partial_coefficient", as_index=False).agg(mean=("composite_score", "mean"), sd=("composite_score", "std"))
    axes[1, 0].errorbar(cal["partial_coefficient"], cal["mean"], yerr=cal["sd"].fillna(0), color="black", marker="o", capsize=2)
    axes[1, 0].axvline(bundle.selected_partial_coefficient, color="0.5", linestyle="--", label=f"Selected = {bundle.selected_partial_coefficient:.2f}")
    axes[1, 0].set(xlabel="Partial-truth coefficient", ylabel="Validation score", title="Decision projection calibration")
    axes[1, 0].legend(loc="lower right", fontsize=7); clean_axis(axes[1, 0]); panel_label(axes[1, 0], "c")
    lc = learning.calibration.copy()
    axes[1, 1].bar(lc["model"], lc["target_correlation"].fillna(0), color=["0.25", "0.65", "0.45"], edgecolor="black", linewidth=0.7)
    axes[1, 1].tick_params(axis="x", labelrotation=25)
    axes[1, 1].set(ylabel="Training target correlation", title="Learning-baseline diagnostics")
    clean_axis(axes[1, 1]); panel_label(axes[1, 1], "d")
    save_figure(fig, output_dir, "fig13_training_and_projection_calibration", formats, dpi)


def plot_effect_sizes(bundle: ExperimentBundle, output_dir: Path, formats: list[str], dpi: int) -> None:
    tests = bundle.statistical_tests[bundle.statistical_tests["metric"].isin(PRIMARY_METRICS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for p, metric in enumerate(PRIMARY_METRICS):
        ax = axes.ravel()[p]
        sub = tests[tests["metric"] == metric].sort_values("paired_cohen_dz")
        y = np.arange(len(sub))
        ax.scatter(sub["paired_cohen_dz"], y, color="black", marker="o")
        ax.axvline(0, color="0.5", linestyle="--")
        ax.set_yticks(y, sub["comparison"], fontsize=7)
        ax.set(xlabel="Paired Cohen dz (ACM - comparator)", title=METRIC_LABELS[metric])
        clean_axis(ax); panel_label(ax, chr(ord("a") + p))
    save_figure(fig, output_dir, "fig14_paired_effect_sizes", formats, dpi)


def export_reproducibility_metadata(profile: dict, model_cfg: dict, bundle: ExperimentBundle, output_root: Path) -> None:
    metadata = {
        "profile": profile,
        "model": model_cfg,
        "selected_partial_truth_coefficient": bundle.selected_partial_coefficient,
        "operator_global_scale": bundle.graph.operator_scale,
        "effective_self_memory": 0.42 * bundle.graph.operator_scale,
        "operator_norm_coupled": bundle.graph.operator_norm_coupled,
        "operator_norm_independent": bundle.graph.operator_norm_independent,
        "contraction_bound_coupled": bundle.graph.contraction_bound_coupled,
        "contraction_bound_independent": bundle.graph.contraction_bound_independent,
        "figure_output_policy": "raster-only; no PDF",
        "primary_metrics": list(PRIMARY_METRICS),
        "fairness_metrics": list(FAIRNESS_METRICS),
    }
    path = output_root / "supplement" / "reproducibility_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def export_interpretation_guardrails(bundle: ExperimentBundle, output_root: Path) -> None:
    reference = "ACM-4 (coupled)"
    means = bundle.run_metrics.groupby("model", as_index=True).mean(numeric_only=True)
    tests = bundle.statistical_tests
    lines = [
        "# Results interpretation guardrails",
        "",
        "This file is generated from the paper-profile outputs and records the interpretation supported by the computed estimates and statistical tests.",
        "",
        "## ACM versus FCM",
        "",
    ]
    for metric in PRIMARY_METRICS:
        row = tests[(tests["metric"] == metric) & (tests["comparison"] == "FCM")].iloc[0]
        direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        lines.append(
            f"- {METRIC_LABELS[metric]} ({direction}): ACM={means.loc[reference, metric]:.4f}, "
            f"FCM={means.loc['FCM', metric]:.4f}, Holm-adjusted p={row['p_holm']:.4g}."
        )
    lines.extend(["", "## Mean-ranking checks", ""])
    for metric in PRIMARY_METRICS:
        ranking = means[metric].sort_values(ascending=metric in LOWER_IS_BETTER)
        lines.append(f"- Best mean {METRIC_LABELS[metric]}: {ranking.index[0]} ({ranking.iloc[0]:.4f}).")
    significant = tests[tests["significant_0_05"]]
    lines.extend(["", "## Statistically supported ACM advantages", ""])
    if significant.empty:
        lines.append("- No paired comparison remains significant after Holm correction.")
    else:
        for row in significant.itertuples(index=False):
            lines.append(f"- {METRIC_LABELS.get(row.metric, row.metric)} versus {row.comparison}: adjusted p={row.p_holm:.4g}.")
    lines.extend(
        [
            "",
            "## Mandatory writing constraints",
            "",
            "- Do not reuse the old 94.8%, 5.2%, 2.1-day or 58.1% improvement claims.",
            "- Do not claim that four channels are necessary unless the ablation table and paired tests support that statement.",
            "- Describe non-significant differences as comparable performance, not superiority.",
            "- External validity cannot be claimed until a real-world CSV is supplied and fig15/table_external_validation are generated.",
            "- Keep the robust-LP formulation and the SGC/PPO training definitions exactly aligned with the code when writing Methods.",
            "",
        ]
    )
    (output_root / "RESULTS_INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")


def create_all_outputs(bundle: ExperimentBundle, learning: LearningBundle, profile: dict, model_cfg: dict, output_root: Path) -> None:
    apply_nature_style(int(profile["dpi"]))
    formats = list(profile["output_formats"])
    dpi = int(profile["dpi"])
    fig_dir = output_root / "figures"
    export_tables(bundle, learning, profile, model_cfg, output_root)
    export_reproducibility_metadata(profile, model_cfg, bundle, output_root)
    export_interpretation_guardrails(bundle, output_root)
    plot_workflow(fig_dir, formats, dpi)
    plot_graph_structure(bundle.graph, fig_dir, formats, dpi)
    plot_transfer_and_stability(bundle, model_cfg, fig_dir, formats, dpi)
    plot_scenario_dynamics(bundle, fig_dir, formats, dpi)
    plot_convergence(bundle, fig_dir, formats, dpi)
    plot_channels_and_interpretability(bundle, fig_dir, formats, dpi)
    plot_performance(bundle, fig_dir, formats, dpi)
    plot_robustness(bundle, fig_dir, formats, dpi)
    plot_ablation(bundle, fig_dir, formats, dpi)
    plot_fairness(bundle, fig_dir, formats, dpi)
    plot_sensitivity(bundle, fig_dir, formats, dpi)
    plot_sensitivity_grid(bundle, fig_dir, formats, dpi)
    plot_training_and_calibration(bundle, learning, fig_dir, formats, dpi)
    plot_effect_sizes(bundle, fig_dir, formats, dpi)
