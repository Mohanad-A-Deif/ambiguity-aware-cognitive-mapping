from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    ACMPolicy,
    ConceptGraph,
    FCMPolicy,
    HeuristicPolicy,
    PriorityPolicy,
    Scenario,
    SimulationResult,
    build_concept_graph,
    run_simulation,
)
from .experiments import metrics_frame, paired_tests, summarize_metrics
from .real_datasets import RealPanel


@dataclass
class DirectValidationBundle:
    panel: RealPanel
    scores: dict[str, np.ndarray]
    targets: np.ndarray
    daily_metrics: pd.DataFrame
    summary: pd.DataFrame
    statistical_tests: pd.DataFrame
    calibration: pd.DataFrame
    selected_partial_coefficient: float
    calibration_end: int
    horizon_steps: int
    evidence_table: pd.DataFrame


@dataclass
class SemiEmpiricalBundle:
    panel: RealPanel
    results: list[SimulationResult]
    run_metrics: pd.DataFrame
    summary: pd.DataFrame
    statistical_tests: pd.DataFrame
    missingness: pd.DataFrame
    graph: ConceptGraph
    selected_partial_coefficient: float
    holdout_start: int


class EqualAllocationPolicy(PriorityPolicy):
    display_name = "Equal allocation"

    def priority(self, features: dict[str, np.ndarray], day: int) -> tuple[np.ndarray, dict[str, Any]]:
        h, r = features["observed_demand"].shape
        priority = np.full((h, r), 1.0 / h)
        return priority, {"score": np.zeros((h, r)), "iterations": 1, "convergence_gaps": np.array([0.0])}


def _graph(panel: RealPanel, model_cfg: dict) -> ConceptGraph:
    return build_concept_graph(
        n_hospitals=panel.n_nodes,
        n_dcs=len(panel.region_names),
        n_suppliers=2,
        regions=panel.regions,
        alpha=float(model_cfg["transfer_alpha"]),
        physical_evidence_mix=float(model_cfg["physical_evidence_mix"]),
        operator_target_norm=float(model_cfg["operator_target_norm"]),
        gaussian_mu=model_cfg["gaussian_mu"],
        gaussian_sigma=model_cfg["gaussian_sigma"],
        gaussian_beta=model_cfg["gaussian_beta"],
        resources=panel.resource_names,
    )


def _policy_factories(model_cfg: dict, gamma: float, include_equal: bool = False) -> dict[str, Callable[[], PriorityPolicy]]:
    common = dict(
        alpha=float(model_cfg["transfer_alpha"]),
        physical_mix=float(model_cfg["physical_evidence_mix"]),
        max_iter=int(model_cfg["inner_max_iter"]),
        tolerance=float(model_cfg["convergence_tolerance"]),
        temperature=float(model_cfg["priority_temperature"]),
    )
    factories: dict[str, Callable[[], PriorityPolicy]] = {
        "ACM-4 (coupled)": lambda: ACMPolicy(
            "ACM-4 (coupled)", 4, True, partial_coefficient=gamma, **common
        ),
        "FCM": lambda: FCMPolicy(**common),
        "Robust-LP": lambda: HeuristicPolicy(
            "Robust-LP", robust_z=1.64, temperature=float(model_cfg["priority_temperature"])
        ),
        "ACM-2": lambda: ACMPolicy("ACM-2", 2, True, partial_coefficient=0.0, **common),
        "ACM-3": lambda: ACMPolicy("ACM-3", 3, True, partial_coefficient=0.0, **common),
        "ACM-4 (independent)": lambda: ACMPolicy(
            "ACM-4 (independent)", 4, False, partial_coefficient=gamma, **common
        ),
    }
    if include_equal:
        factories["Equal allocation"] = EqualAllocationPolicy
    return factories


def _dummy_scenario(panel: RealPanel) -> Scenario:
    daily_total = np.maximum(panel.demand.mean(axis=0).sum(axis=0), 1.0)
    return Scenario(
        seed=0,
        demand=panel.demand,
        observed_demand=panel.demand,
        reporting_confidence=panel.reporting_confidence,
        supplier_capacity=np.tile((0.5 * daily_total)[None, None, :], (panel.n_days, 2, 1)),
        lead_dh=np.ones((len(panel.region_names), panel.n_nodes, panel.n_resources), dtype=int),
        lead_sd=np.ones((2, len(panel.region_names), panel.n_resources), dtype=int),
        initial_hospital_inventory=np.maximum(panel.capacity[0] - panel.demand[0], 0.0),
        initial_dc_inventory=np.tile(daily_total[None, :], (len(panel.region_names), 1)),
        regions=panel.regions,
        hospital_size=np.maximum(panel.capacity.mean(axis=(0, 2)), 1.0),
        resource_names=panel.resource_names,
    )


def real_features(panel: RealPanel, day: int) -> dict[str, np.ndarray]:
    demand = panel.demand[day]
    capacity = np.maximum(panel.capacity[day], 1.0)
    stress = np.clip(demand / capacity, 0.0, 1.5)
    previous = panel.demand[max(day - 2, 0) : day + 1] / np.maximum(
        panel.capacity[max(day - 2, 0) : day + 1], 1.0
    )
    trend = np.maximum(stress - previous.mean(axis=0), 0.0)
    admissions = panel.admissions_confirmed[day] + 0.5 * panel.admissions_suspected[day]
    admission_scale = max(float(np.nanpercentile(panel.admissions_confirmed, 95)), 1.0)
    admission_pressure = np.clip(admissions / admission_scale, 0.0, 1.0)[:, None]
    demand_pressure = np.clip(0.70 * stress / 1.05 + 0.30 * admission_pressure, 0.0, 1.0)
    backlog_pressure = np.clip((stress - 0.65) / 0.55, 0.0, 1.0)
    inventory_risk = np.clip(stress / 1.05, 0.0, 1.0)
    lead_risk = np.clip(4.0 * trend + (1.0 - panel.reporting_confidence[day]) * 0.35, 0.0, 1.0)
    capacity_risk = np.clip(stress / 1.10, 0.0, 1.0)
    spare = np.maximum(capacity - demand, 0.0)
    n_regions = len(panel.region_names)
    dc_stock_risk = np.zeros((n_regions, panel.n_resources))
    dc_load_risk = np.zeros_like(dc_stock_risk)
    for region in range(n_regions):
        mask = panel.regions == region
        dc_stock_risk[region] = capacity_risk[mask].mean(axis=0)
        dc_load_risk[region] = demand_pressure[mask].mean(axis=0)
    supplier_risk = np.tile(capacity_risk.mean(axis=0)[None, :], (2, 1))
    return {
        "observed_demand": demand,
        "demand_pressure": demand_pressure,
        "backlog_pressure": backlog_pressure,
        "inventory_risk": inventory_risk,
        "lead_risk": lead_risk,
        "capacity_risk": capacity_risk,
        "reporting_confidence": panel.reporting_confidence[day],
        "hospital_inventory": spare,
        "dc_inventory": np.maximum(1.0 - dc_stock_risk, 0.0),
        "dc_stock_risk": dc_stock_risk,
        "dc_load_risk": dc_load_risk,
        "supplier_risk": supplier_risk,
    }


def _future_targets(panel: RealPanel, horizon_steps: int) -> np.ndarray:
    stress = panel.demand / np.maximum(panel.capacity, 1.0)
    targets = np.full_like(stress, np.nan)
    for day in range(panel.n_days - horizon_steps):
        future = stress[day + 1 : day + horizon_steps + 1]
        targets[day] = 0.75 * future.max(axis=0) + 0.25 * future.mean(axis=0)
    return targets


def _score_policies(
    panel: RealPanel, graph: ConceptGraph, model_cfg: dict, gamma: float
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    scenario = _dummy_scenario(panel)
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for name, factory in _policy_factories(model_cfg, gamma).items():
        policy = factory()
        policy.reset(scenario, graph)
        history = np.zeros_like(panel.demand)
        diag_history: list[dict[str, Any]] = []
        for day in range(panel.n_days):
            priority, diag = policy.priority(real_features(panel, day), day)
            history[day] = priority
            diag_history.append(diag)
        scores[name] = history
        diagnostics[name] = diag_history
    stress = panel.demand / np.maximum(panel.capacity, 1.0)
    scores["Current stress"] = stress
    admissions = panel.admissions_confirmed + 0.5 * panel.admissions_suspected
    scores["Admissions"] = np.repeat(admissions[:, :, None], panel.n_resources, axis=2)

    example_day = int(np.nanargmax(panel.demand.sum(axis=(1, 2))))
    diag = diagnostics["ACM-4 (coupled)"][example_day]
    hospital_state = np.asarray(diag["hospital_state"])
    evidence = np.asarray(diag["evidence_detail"]["hospital_evidence"])
    rows: list[dict[str, Any]] = []
    for h, node in enumerate(panel.node_ids):
        for k, resource in enumerate(panel.resource_names):
            row: dict[str, Any] = {
                "date": panel.dates[example_day],
                "node_id": node,
                "node_name": panel.node_names[h],
                "resource": resource,
                "priority": scores["ACM-4 (coupled)"][example_day, h, k],
                "stress": stress[example_day, h, k],
            }
            for c, label in enumerate(("T", "F", "PT", "PF")):
                row[f"state_{label}"] = hospital_state[h, k, c]
                row[f"evidence_{label}"] = evidence[h, k, c]
            rows.append(row)
    return scores, pd.DataFrame(rows)


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    ideal = np.argsort(-y_true)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.maximum(y_true, 0.0)
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / idcg if idcg > 0 else 1.0


def _ranking_rows(
    panel: RealPanel,
    scores: dict[str, np.ndarray],
    targets: np.ndarray,
    start: int,
    stop: int,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, history in scores.items():
        for day in range(start, stop):
            for k, resource in enumerate(panel.resource_names):
                y = targets[day, :, k]
                s = history[day, :, k]
                if not np.all(np.isfinite(y)):
                    continue
                rho = stats.spearmanr(y, s).statistic
                rho = 0.0 if not np.isfinite(rho) else float(rho)
                top_true = set(np.argsort(-y)[:3].tolist())
                top_score = set(np.argsort(-s)[:3].tolist())
                rows.append(
                    {
                        "source": panel.source,
                        "split": split,
                        "date": panel.dates[day],
                        "step": day,
                        "block": day // 7 if (panel.dates[1] - panel.dates[0]).days == 1 else day // 4,
                        "resource": resource,
                        "model": model,
                        "spearman_rho": rho,
                        "ndcg_at_3": _ndcg_at_k(y, s, 3),
                        "top3_recall": len(top_true & top_score) / 3.0,
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_ci(x: np.ndarray, seed: int, n_boot: int = 3000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _ranking_summary(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    block = (
        rows.groupby(["source", "split", "model", "resource", "block"], as_index=False)[
            ["spearman_rho", "ndcg_at_3", "top3_recall"]
        ].mean()
    )
    out: list[dict[str, Any]] = []
    for (source, split, model), group in block.groupby(["source", "split", "model"], sort=False):
        for metric in ["spearman_rho", "ndcg_at_3", "top3_recall"]:
            values = group[metric].to_numpy(float)
            lo, hi = _bootstrap_ci(values, seed + abs(hash((source, split, model, metric))) % 100000)
            out.append(
                {
                    "source": source,
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "n_blocks": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(out)


def _holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted


def _direct_tests(rows: pd.DataFrame, reference: str = "ACM-4 (coupled)") -> pd.DataFrame:
    block = rows.groupby(["block", "resource", "model"], as_index=False)[
        ["spearman_rho", "ndcg_at_3", "top3_recall"]
    ].mean()
    records: list[dict[str, Any]] = []
    for metric in ["spearman_rho", "ndcg_at_3", "top3_recall"]:
        pivot = block.pivot(index=["block", "resource"], columns="model", values=metric).dropna()
        for model in pivot.columns:
            if model == reference:
                continue
            difference = pivot[reference].to_numpy() - pivot[model].to_numpy()
            if np.allclose(difference, 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                try:
                    statistic, pvalue = stats.wilcoxon(difference, zero_method="wilcox")
                except ValueError:
                    statistic, pvalue = 0.0, 1.0
                if not np.isfinite(pvalue):
                    statistic, pvalue = 0.0, 1.0
            favorable = float(np.mean(difference))
            records.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparison": model,
                    "n_pairs": len(difference),
                    "mean_difference": favorable,
                    "wilcoxon_statistic": statistic,
                    "p_raw": pvalue,
                    "rank_biserial": float(np.mean(difference > 0) - np.mean(difference < 0)),
                }
            )
    out = pd.DataFrame(records)
    out["p_holm"] = np.nan
    for _, indices in out.groupby("metric").groups.items():
        out.loc[indices, "p_holm"] = _holm(out.loc[indices, "p_raw"].to_numpy(float))
    out["significant_0_05"] = out["p_holm"] < 0.05
    return out


def run_direct_validation(
    panel: RealPanel,
    model_cfg: dict,
    master_seed: int,
    selected_gamma: float | None = None,
) -> DirectValidationBundle:
    graph = _graph(panel, model_cfg)
    horizon = 7 if (panel.dates[1] - panel.dates[0]).days == 1 else 1
    calibration_end = panel.n_days // 2 if selected_gamma is None else 0
    targets = _future_targets(panel, horizon)
    calibration_rows: list[dict[str, Any]] = []
    if selected_gamma is None:
        for gamma in [0.0, 0.25, 0.50, 0.75, 1.0]:
            scores, _ = _score_policies(panel, graph, model_cfg, gamma)
            rows = _ranking_rows(panel, {"ACM-4 (coupled)": scores["ACM-4 (coupled)"]}, targets, 2, calibration_end, "calibration")
            objective = float(rows["ndcg_at_3"].mean())
            calibration_rows.append({"partial_coefficient": gamma, "mean_ndcg_at_3": objective})
        calibration = pd.DataFrame(calibration_rows)
        best = calibration["mean_ndcg_at_3"].max()
        selected_gamma = float(calibration.loc[calibration["mean_ndcg_at_3"].eq(best), "partial_coefficient"].min())
        calibration["selected"] = calibration["partial_coefficient"].eq(selected_gamma)
    else:
        calibration = pd.DataFrame(
            [{"partial_coefficient": selected_gamma, "mean_ndcg_at_3": np.nan, "selected": True}]
        )
    scores, evidence = _score_policies(panel, graph, model_cfg, float(selected_gamma))
    if calibration_end:
        cal = _ranking_rows(panel, scores, targets, 2, calibration_end, "calibration")
        test = _ranking_rows(panel, scores, targets, calibration_end, panel.n_days - horizon, "holdout")
        daily = pd.concat([cal, test], ignore_index=True)
        evaluation = test
    else:
        daily = _ranking_rows(panel, scores, targets, 2, panel.n_days - horizon, "external")
        evaluation = daily
    summary = _ranking_summary(evaluation, master_seed)
    tests = _direct_tests(evaluation)
    return DirectValidationBundle(
        panel, scores, targets, daily, summary, tests, calibration, float(selected_gamma), calibration_end, horizon, evidence
    )


def make_semi_empirical_scenario(
    panel: RealPanel,
    start: int,
    days: int,
    seed: int,
    missing_probability: float,
) -> Scenario:
    rng = np.random.default_rng(seed)
    demand = panel.demand[start : start + days].copy()
    empirical_confidence = panel.reporting_confidence[start : start + days].copy()
    observed = demand * np.clip(1.0 + rng.normal(0.0, 0.05, demand.shape), 0.75, 1.25)
    missing = rng.random(demand.shape) < missing_probability
    confidence = empirical_confidence.copy()
    confidence[missing] = np.minimum(confidence[missing], 0.35)
    for day, h, k in zip(*np.where(missing)):
        observed[day, h, k] = observed[day - 1, h, k] if day > 0 else demand[day, h, k]
    n_regions = len(panel.region_names)
    r_count = panel.n_resources
    lead_dh = np.zeros((n_regions, panel.n_nodes, r_count), dtype=int)
    for dc in range(n_regions):
        for h in range(panel.n_nodes):
            local = dc == panel.regions[h]
            lead_dh[dc, h] = rng.integers(1, 3, r_count) if local else rng.integers(3, 5, r_count)
    lead_sd = rng.integers(1, 4, size=(2, n_regions, r_count))
    smoothed_total = (
        pd.DataFrame(demand.sum(axis=1)).rolling(5, min_periods=1, center=True).mean().to_numpy()
    )
    supplier_capacity = np.zeros((days, 2, r_count))
    disruption_center = rng.integers(max(5, days // 3), max(6, 2 * days // 3 + 1))
    time = np.arange(days)
    disruption = 1.0 - 0.18 * np.exp(-0.5 * ((time - disruption_center) / 4.0) ** 2)
    for day in range(days):
        total_supply = 0.88 * smoothed_total[day] * disruption[day]
        split = rng.uniform(0.46, 0.54, r_count)
        supplier_capacity[day, 0] = total_supply * split
        supplier_capacity[day, 1] = total_supply * (1.0 - split)
    mean_hr = np.maximum(demand[: min(7, days)].mean(axis=0), 0.1)
    mean_total = mean_hr.sum(axis=0)
    initial_hospital = mean_hr * rng.uniform(0.20, 0.45, size=mean_hr.shape)
    initial_dc = np.tile(0.75 * mean_total[None, :], (n_regions, 1))
    initial_dc *= rng.uniform(0.85, 1.15, size=initial_dc.shape)
    capacity_mean = panel.capacity[start : start + days].mean(axis=(0, 2))
    return Scenario(
        seed=seed,
        demand=demand,
        observed_demand=observed,
        reporting_confidence=confidence,
        supplier_capacity=supplier_capacity,
        lead_dh=lead_dh,
        lead_sd=lead_sd,
        initial_hospital_inventory=initial_hospital,
        initial_dc_inventory=initial_dc,
        regions=panel.regions,
        hospital_size=np.maximum(capacity_mean, 1.0),
        resource_names=panel.resource_names,
    )


def run_semi_empirical_experiment(
    panel: RealPanel,
    model_cfg: dict,
    selected_gamma: float,
    master_seed: int,
    n_runs: int = 30,
) -> SemiEmpiricalBundle:
    graph = _graph(panel, model_cfg)
    holdout_start = panel.n_days // 2
    days = min(40, panel.n_days - holdout_start)
    max_start = max(1, panel.n_days - days - holdout_start + 1)
    factories = _policy_factories(model_cfg, selected_gamma, include_equal=True)
    results: list[SimulationResult] = []
    clinical_weights = np.linspace(1.0, 0.70, panel.n_resources)
    for run in range(n_runs):
        start = holdout_start + (run % max_start)
        seed = master_seed + 5000 + run
        scenario = make_semi_empirical_scenario(panel, start, days, seed, missing_probability=0.10)
        for factory in factories.values():
            results.append(
                run_simulation(
                    scenario,
                    graph,
                    factory(),
                    backlog_mode="backlog",
                    fairness_weight=float(model_cfg["fairness_objective_weight"]),
                    lead_penalty=float(model_cfg["lead_time_penalty"]),
                    priority_objective_weight=float(model_cfg["priority_objective_weight"]),
                    clinical_weights=clinical_weights,
                )
            )
    run_frame = metrics_frame(results)
    summary = summarize_metrics(run_frame, 3000, master_seed)
    tests = paired_tests(run_frame)

    missing_rows: list[dict[str, Any]] = []
    sensitivity_factories = {
        key: value
        for key, value in factories.items()
        if key in {"ACM-4 (coupled)", "FCM", "Robust-LP"}
    }
    for probability in [0.0, 0.10, 0.20, 0.30, 0.40]:
        for rep in range(6):
            start = holdout_start + (rep % max_start)
            seed = master_seed + 9000 + rep
            scenario = make_semi_empirical_scenario(panel, start, days, seed, missing_probability=probability)
            for factory in sensitivity_factories.values():
                result = run_simulation(
                    scenario,
                    graph,
                    factory(),
                    backlog_mode="backlog",
                    fairness_weight=float(model_cfg["fairness_objective_weight"]),
                    lead_penalty=float(model_cfg["lead_time_penalty"]),
                    priority_objective_weight=float(model_cfg["priority_objective_weight"]),
                    clinical_weights=clinical_weights,
                )
                missing_rows.append(
                    {
                        "missing_probability": probability,
                        "replicate": rep,
                        "model": result.model,
                        **result.metrics,
                    }
                )
    return SemiEmpiricalBundle(
        panel,
        results,
        run_frame,
        summary,
        tests,
        pd.DataFrame(missing_rows),
        graph,
        selected_gamma,
        holdout_start,
    )
