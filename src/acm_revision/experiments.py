from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    ACMPolicy,
    ConceptGraph,
    FCMPolicy,
    HeuristicPolicy,
    PriorityPolicy,
    SimulationResult,
    build_concept_graph,
    generate_scenario,
    run_simulation,
)
from .learning import LearningBundle


PRIMARY_METRICS = (
    "service_level",
    "unmet_demand_rate",
    "average_lead_time_days",
    "gini_fill",
)
FAIRNESS_METRICS = (
    "gini_fill",
    "jain_fairness",
    "max_min_fairness",
    "geographic_equity",
    "priority_weighted_equity",
)
LOWER_IS_BETTER = {"unmet_demand_rate", "unmet_demand_units", "average_lead_time_days", "gini_fill", "final_convergence_gap"}


@dataclass
class ExperimentBundle:
    results: list[SimulationResult]
    run_metrics: pd.DataFrame
    summary: pd.DataFrame
    statistical_tests: pd.DataFrame
    omnibus_tests: pd.DataFrame
    calibration: pd.DataFrame
    sensitivity: pd.DataFrame
    sensitivity_grid: pd.DataFrame
    graph: ConceptGraph
    selected_partial_coefficient: float


def make_policy_factories(model_cfg: dict, learning: LearningBundle, selected_gamma: float) -> dict[str, Callable[[], PriorityPolicy]]:
    common = dict(
        alpha=float(model_cfg["transfer_alpha"]),
        physical_mix=float(model_cfg["physical_evidence_mix"]),
        max_iter=int(model_cfg["inner_max_iter"]),
        tolerance=float(model_cfg["convergence_tolerance"]),
        temperature=float(model_cfg["priority_temperature"]),
    )
    return {
        "ACM-4 (coupled)": lambda: ACMPolicy("ACM-4 (coupled)", 4, True, partial_coefficient=selected_gamma, **common),
        "FCM": lambda: FCMPolicy(**common),
        "ACM-2": lambda: ACMPolicy("ACM-2", 2, True, partial_coefficient=0.0, **common),
        "ACM-3": lambda: ACMPolicy("ACM-3", 3, True, partial_coefficient=0.0, **common),
        "ACM-4 (independent)": lambda: ACMPolicy("ACM-4 (independent)", 4, False, partial_coefficient=selected_gamma, **common),
        "SGC-GNN": lambda: learning.sgc_policy,
        "Bayesian network": lambda: learning.bayesian_policy,
        "PPO": lambda: learning.ppo_policy,
        "Robust-LP": lambda: HeuristicPolicy("Robust-LP", robust_z=1.64, temperature=float(model_cfg["priority_temperature"])),
    }


def _scenario_kwargs(profile: dict, model_cfg: dict) -> dict:
    return dict(
        days=int(profile["days"]),
        n_hospitals=int(profile["n_hospitals"]),
        demand_scale=float(model_cfg.get("demand_scale", 1.0)),
        lead_time_scale=float(model_cfg.get("lead_time_scale", 1.0)),
        capacity_scale=float(model_cfg.get("capacity_scale", 1.0)),
        reporting_noise=float(model_cfg["reporting_noise"]),
        missing_probability=float(model_cfg["reporting_missing_probability"]),
    )


def calibrate_partial_coefficient(profile: dict, model_cfg: dict, graph: ConceptGraph) -> tuple[float, pd.DataFrame]:
    candidates = [0.25, 0.50, 0.75, 1.00]
    runs = min(8, max(4, int(profile["sensitivity_runs"])))
    rows: list[dict[str, float]] = []
    for gamma in candidates:
        for rep in range(runs):
            seed = int(profile["master_seed"]) + 7000 + rep
            scenario = generate_scenario(seed, **_scenario_kwargs(profile, model_cfg))
            policy = ACMPolicy(
                "ACM-4 (coupled)",
                channels=4,
                coupled=True,
                alpha=float(model_cfg["transfer_alpha"]),
                physical_mix=float(model_cfg["physical_evidence_mix"]),
                max_iter=int(model_cfg["inner_max_iter"]),
                tolerance=float(model_cfg["convergence_tolerance"]),
                partial_coefficient=gamma,
                temperature=float(model_cfg["priority_temperature"]),
            )
            result = run_simulation(
                scenario,
                graph,
                policy,
                backlog_mode=str(model_cfg["backlog_mode"]),
                fairness_weight=float(model_cfg["fairness_objective_weight"]),
                lead_penalty=float(model_cfg["lead_time_penalty"]),
                priority_objective_weight=float(model_cfg["priority_objective_weight"]),
            )
            composite = (
                result.metrics["service_level"]
                - result.metrics["unmet_demand_rate"]
                - 0.04 * result.metrics["average_lead_time_days"]
                - 0.20 * result.metrics["gini_fill"]
            )
            rows.append({"partial_coefficient": gamma, "replicate": rep, "composite_score": composite, **{m: result.metrics[m] for m in PRIMARY_METRICS}})
    frame = pd.DataFrame(rows)
    means = frame.groupby("partial_coefficient")["composite_score"].mean()
    selected = float(means.idxmax())
    frame["selected"] = frame["partial_coefficient"].eq(selected)
    return selected, frame


def run_main_experiment(
    profile: dict,
    model_cfg: dict,
    graph: ConceptGraph,
    learning: LearningBundle,
    selected_gamma: float,
) -> list[SimulationResult]:
    factories = make_policy_factories(model_cfg, learning, selected_gamma)
    results: list[SimulationResult] = []
    for run in range(int(profile["n_runs"])):
        seed = int(profile["master_seed"]) + run
        scenario = generate_scenario(seed, **_scenario_kwargs(profile, model_cfg))
        for _, factory in factories.items():
            result = run_simulation(
                scenario,
                graph,
                factory(),
                backlog_mode=str(model_cfg["backlog_mode"]),
                fairness_weight=float(model_cfg["fairness_objective_weight"]),
                lead_penalty=float(model_cfg["lead_time_penalty"]),
                priority_objective_weight=float(model_cfg["priority_objective_weight"]),
            )
            results.append(result)
    return results


def metrics_frame(results: list[SimulationResult]) -> pd.DataFrame:
    return pd.DataFrame([{"model": r.model, "seed": r.seed, **r.metrics} for r in results])


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_metrics(run_metrics: pd.DataFrame, bootstrap_samples: int, master_seed: int) -> pd.DataFrame:
    metric_cols = [c for c in run_metrics.columns if c not in {"model", "seed"}]
    rows: list[dict[str, float | str]] = []
    for model, group in run_metrics.groupby("model", sort=False):
        for metric in metric_cols:
            values = group[metric].to_numpy(float)
            lo, hi = bootstrap_mean_ci(values, bootstrap_samples, master_seed + abs(hash((model, metric))) % 100000)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "median": np.median(values),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_tests(run_metrics: pd.DataFrame, reference: str = "ACM-4 (coupled)") -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    models = [m for m in run_metrics["model"].unique() if m != reference]
    for metric in PRIMARY_METRICS + FAIRNESS_METRICS[1:]:
        pivot = run_metrics.pivot(index="seed", columns="model", values=metric).dropna()
        ref = pivot[reference].to_numpy()
        for model in models:
            other = pivot[model].to_numpy()
            diff = ref - other
            if metric in LOWER_IS_BETTER:
                improvement = other - ref
            else:
                improvement = ref - other
            if np.allclose(diff, 0.0):
                stat, p = 0.0, 1.0
            else:
                try:
                    stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    stat, p = 0.0, 1.0
                if not np.isfinite(p):
                    stat, p = 0.0, 1.0
            dz = float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) > 0 else 0.0
            rows.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparison": model,
                    "n_pairs": len(diff),
                    "reference_mean": ref.mean(),
                    "comparison_mean": other.mean(),
                    "mean_improvement_favorable": improvement.mean(),
                    "wilcoxon_statistic": stat,
                    "p_raw": p,
                    "paired_cohen_dz": dz,
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = _holm_adjust(out.loc[idx, "p_raw"].to_numpy())
    out["significant_0_05"] = out["p_holm"] < 0.05
    return out


def friedman_tests(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in PRIMARY_METRICS:
        pivot = run_metrics.pivot(index="seed", columns="model", values=metric).dropna()
        arrays = [pivot[c].to_numpy() for c in pivot.columns]
        stat, p = stats.friedmanchisquare(*arrays)
        rows.append({"metric": metric, "n_runs": len(pivot), "n_models": len(arrays), "friedman_chi_square": stat, "p_value": p})
    return pd.DataFrame(rows)


def run_sensitivity(profile: dict, model_cfg: dict, base_graph: ConceptGraph, selected_gamma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = int(profile["master_seed"])
    runs = int(profile["sensitivity_runs"])
    factors = {
        "transfer_alpha": [0.60, 0.80, 1.00, 1.20, 1.40],
        "physical_evidence_mix": [0.25, 0.35, 0.45, 0.55, 0.65],
        "partial_truth_coefficient": [0.25, 0.50, 0.75, 1.00],
        "demand_scale": [0.70, 0.85, 1.00, 1.15, 1.30],
        "lead_time_scale": [0.70, 0.85, 1.00, 1.15, 1.30],
        "capacity_scale": [0.80, 0.90, 1.00, 1.10, 1.20],
        "rbf_sigma_scale": [0.70, 1.00, 1.30],
    }
    rows: list[dict[str, float | str]] = []
    for factor, values in factors.items():
        for value in values:
            for rep in range(runs):
                cfg = dict(model_cfg)
                cfg[factor] = value
                graph = base_graph
                if factor in {"transfer_alpha", "physical_evidence_mix", "rbf_sigma_scale"}:
                    sigma_scale = float(value) if factor == "rbf_sigma_scale" else 1.0
                    graph = build_concept_graph(
                        int(profile["n_hospitals"]),
                        int(profile["n_dcs"]),
                        int(profile["n_suppliers"]),
                        np.repeat(np.arange(2), int(profile["n_hospitals"]) // 2),
                        alpha=float(cfg["transfer_alpha"]),
                        physical_evidence_mix=float(cfg["physical_evidence_mix"]),
                        operator_target_norm=float(cfg["operator_target_norm"]),
                        gaussian_mu=cfg["gaussian_mu"],
                        gaussian_sigma=np.asarray(cfg["gaussian_sigma"]) * sigma_scale,
                        gaussian_beta=cfg["gaussian_beta"],
                    )
                gamma = float(value) if factor == "partial_truth_coefficient" else selected_gamma
                seed = master + 30000 + rep
                scenario = generate_scenario(seed, **_scenario_kwargs(profile, cfg))
                policy = ACMPolicy(
                    "ACM-4 (coupled)", 4, True,
                    alpha=float(cfg["transfer_alpha"]),
                    physical_mix=float(cfg["physical_evidence_mix"]),
                    max_iter=int(cfg["inner_max_iter"]),
                    tolerance=float(cfg["convergence_tolerance"]),
                    partial_coefficient=gamma,
                    temperature=float(cfg["priority_temperature"]),
                )
                result = run_simulation(scenario, graph, policy, str(cfg["backlog_mode"]), float(cfg["fairness_objective_weight"]), float(cfg["lead_time_penalty"]), float(cfg["priority_objective_weight"]))
                rows.append({"factor": factor, "value": value, "replicate": rep, **{m: result.metrics[m] for m in PRIMARY_METRICS + FAIRNESS_METRICS[1:]}, "contraction_bound": graph.contraction_bound_coupled})

    grid_rows: list[dict[str, float]] = []
    for alpha in [0.60, 0.80, 1.00, 1.20, 1.40]:
        for mix in [0.25, 0.35, 0.45, 0.55, 0.65]:
            graph = build_concept_graph(
                int(profile["n_hospitals"]), int(profile["n_dcs"]), int(profile["n_suppliers"]),
                np.repeat(np.arange(2), int(profile["n_hospitals"]) // 2),
                alpha=alpha, physical_evidence_mix=mix, operator_target_norm=float(model_cfg["operator_target_norm"]),
                gaussian_mu=model_cfg["gaussian_mu"], gaussian_sigma=model_cfg["gaussian_sigma"], gaussian_beta=model_cfg["gaussian_beta"],
            )
            for rep in range(runs):
                seed = master + 40000 + rep
                scenario = generate_scenario(seed, **_scenario_kwargs(profile, model_cfg))
                policy = ACMPolicy("ACM-4 (coupled)", 4, True, alpha, mix, int(model_cfg["inner_max_iter"]), float(model_cfg["convergence_tolerance"]), selected_gamma, float(model_cfg["priority_temperature"]))
                result = run_simulation(scenario, graph, policy, str(model_cfg["backlog_mode"]), float(model_cfg["fairness_objective_weight"]), float(model_cfg["lead_time_penalty"]), float(model_cfg["priority_objective_weight"]))
                grid_rows.append({"transfer_alpha": alpha, "physical_evidence_mix": mix, "replicate": rep, **{m: result.metrics[m] for m in PRIMARY_METRICS}, "contraction_bound": graph.contraction_bound_coupled})
    return pd.DataFrame(rows), pd.DataFrame(grid_rows)


def build_experiment_bundle(profile: dict, model_cfg: dict, graph: ConceptGraph, learning: LearningBundle) -> ExperimentBundle:
    selected_gamma, calibration = calibrate_partial_coefficient(profile, model_cfg, graph)
    results = run_main_experiment(profile, model_cfg, graph, learning, selected_gamma)
    run_metrics = metrics_frame(results)
    summary = summarize_metrics(run_metrics, int(profile["bootstrap_samples"]), int(profile["master_seed"]))
    tests = paired_tests(run_metrics)
    omnibus = friedman_tests(run_metrics)
    sensitivity, sensitivity_grid = run_sensitivity(profile, model_cfg, graph, selected_gamma)
    return ExperimentBundle(results, run_metrics, summary, tests, omnibus, calibration, sensitivity, sensitivity_grid, graph, selected_gamma)
