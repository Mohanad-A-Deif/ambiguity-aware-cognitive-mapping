from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import ACMPolicy, ConceptGraph, FCMPolicy, RESOURCE_LABELS, Scenario, run_simulation
from .style import clean_axis, panel_label, save_figure


REQUIRED_COLUMNS = {"date", "hospital", "resource", "demand", "initial_inventory", "lead_time_days", "capacity", "region"}


def load_real_scenario(path: Path, master_seed: int) -> Scenario:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Real-world CSV is missing required columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "hospital", "resource"])
    hospitals = list(data["hospital"].drop_duplicates())
    resources = list(RESOURCE_LABELS)
    dates = list(data["date"].drop_duplicates())
    if len(hospitals) != 12:
        raise ValueError("External validation currently requires exactly 12 hospitals to match the case graph.")
    if not set(resources).issubset(set(data["resource"])):
        raise ValueError(f"External validation requires resources {resources}.")
    h_map = {h: i for i, h in enumerate(hospitals)}
    r_map = {r: i for i, r in enumerate(resources)}
    d_map = {d: i for i, d in enumerate(dates)}
    demand = np.zeros((len(dates), len(hospitals), len(resources)))
    confidence = np.ones_like(demand)
    initial = np.zeros((len(hospitals), len(resources)))
    lead = np.ones((2, len(hospitals), len(resources)), dtype=int)
    regions = np.zeros(len(hospitals), dtype=int)
    for row in data.itertuples(index=False):
        if row.resource not in r_map:
            continue
        d, h, r = d_map[row.date], h_map[row.hospital], r_map[row.resource]
        demand[d, h, r] = max(float(row.demand), 0.0)
        if hasattr(row, "reporting_confidence") and pd.notna(row.reporting_confidence):
            confidence[d, h, r] = float(np.clip(row.reporting_confidence, 0.0, 1.0))
        if d == 0:
            initial[h, r] = max(float(row.initial_inventory), 0.0)
        regions[h] = int(row.region) - 1 if int(row.region) in (1, 2) else int(row.region)
        base_lead = int(np.clip(round(float(row.lead_time_days)), 1, 7))
        lead[regions[h], h, r] = base_lead
        lead[1 - regions[h], h, r] = min(base_lead + 1, 7)
    capacity_daily = data.groupby(["date", "resource"], as_index=False)["capacity"].sum()
    supplier_capacity = np.zeros((len(dates), 2, len(resources)))
    for row in capacity_daily.itertuples(index=False):
        supplier_capacity[d_map[row.date], :, r_map[row.resource]] = max(float(row.capacity), 0.0) / 2.0
    mean_daily = np.maximum(demand.mean(axis=0).sum(axis=0), 1.0)
    initial_dc = np.tile(1.35 * mean_daily[None, :], (2, 1))
    lead_sd = np.full((2, 2, len(resources)), 2, dtype=int)
    return Scenario(
        seed=master_seed,
        demand=demand,
        observed_demand=demand.copy(),
        reporting_confidence=confidence,
        supplier_capacity=supplier_capacity,
        lead_dh=lead,
        lead_sd=lead_sd,
        initial_hospital_inventory=initial,
        initial_dc_inventory=initial_dc,
        regions=regions,
        hospital_size=np.ones(len(hospitals)),
    )


def run_real_world_validation(
    csv_path: Path,
    graph: ConceptGraph,
    model_cfg: dict,
    selected_gamma: float,
    output_root: Path,
    formats: list[str],
    dpi: int,
    master_seed: int,
) -> None:
    scenario = load_real_scenario(csv_path, master_seed)
    common = dict(
        alpha=float(model_cfg["transfer_alpha"]),
        physical_mix=float(model_cfg["physical_evidence_mix"]),
        max_iter=int(model_cfg["inner_max_iter"]),
        tolerance=float(model_cfg["convergence_tolerance"]),
        temperature=float(model_cfg["priority_temperature"]),
    )
    policies = [ACMPolicy("ACM-4 (coupled)", 4, True, partial_coefficient=selected_gamma, **common), FCMPolicy(**common)]
    results = [
        run_simulation(scenario, graph, p, str(model_cfg["backlog_mode"]), float(model_cfg["fairness_objective_weight"]), float(model_cfg["lead_time_penalty"]), float(model_cfg["priority_objective_weight"]))
        for p in policies
    ]
    table = pd.DataFrame([{"model": r.model, **r.metrics} for r in results])
    table.to_csv(output_root / "tables" / "table_external_validation.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for result, ls, gray in zip(results, ["-", "--"], ["black", "0.55"]):
        axes[0].plot(result.daily["day"], result.daily["service_level"], color=gray, linestyle=ls, label=result.model)
        axes[1].plot(result.daily["day"], result.daily["backlog_units"], color=gray, linestyle=ls, label=result.model)
    axes[0].set_ylabel("Daily service level"); axes[0].legend(loc="lower right"); panel_label(axes[0], "a")
    axes[1].set(xlabel="Day", ylabel="Backlog units"); panel_label(axes[1], "b")
    for ax in axes: clean_axis(ax)
    save_figure(fig, output_root / "figures", "fig15_external_validation", formats, dpi)
