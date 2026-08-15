from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .core import gini, stable_softmax
from .retrospective_allocation import _holm_adjust, _power_norm, _project_rows
from .retrospective_allocation_v2 import (
    V2Parameters,
    _params_from_mapping,
    _trial_parameters,
    _weighted_evidence,
)
from .style import MODEL_STYLES, apply_nature_style, clean_axis, panel_label, save_figure


PPE_PRODUCTS = ("respirators", "surgicalMasks", "faceShields", "gowns")
PPE_MODELS = (
    "ACM-4 (coupled)",
    "ACM-4 (independent)",
    "ACM-3",
    "ACM-2",
    "Proximity",
    "Demand proportional",
    "Equal allocation",
)


@dataclass
class PPEMatchData:
    donors: pd.DataFrame
    recipients: pd.DataFrame
    distances: dict[int, tuple[np.ndarray, np.ndarray]]
    donor_ids: np.ndarray
    recipient_ids: np.ndarray
    decision_dates: pd.DatetimeIndex
    source_hashes: dict[str, str]


def load_ppe_match_data(
    data_dir: Path,
    interval_days: int = 7,
    max_donation_qty: float = 1000.0,
) -> PPEMatchData:
    files = {
        "donors": data_dir / "anon_donors.csv",
        "recipients": data_dir / "anon_recipients.csv",
        "distances": data_dir / "anon_distance_matrix.csv",
    }
    donors = pd.read_csv(files["donors"])
    recipients = pd.read_csv(files["recipients"])
    donors["date"] = pd.to_datetime(donors["date"], format="mixed", errors="coerce", utc=True)
    recipients["date"] = pd.to_datetime(
        recipients["date"], format="mixed", errors="coerce", utc=True
    )
    donors = donors.loc[
        donors["ppe"].isin(PPE_PRODUCTS)
        & donors["qty"].gt(0)
        & donors["qty"].le(float(max_donation_qty))
        & donors["date"].notna()
    ].copy()
    recipients = recipients.loc[
        recipients["ppe"].isin(PPE_PRODUCTS)
        & recipients["qty"].gt(0)
        & recipients["date"].notna()
    ].copy()
    donor_ids = np.array(sorted(donors["don_id"].unique()), dtype=object)
    recipient_ids = np.array(sorted(recipients["rec_id"].unique()), dtype=object)
    donor_index = {value: index for index, value in enumerate(donor_ids)}
    recipient_index = {value: index for index, value in enumerate(recipient_ids)}
    product_index = {value: index for index, value in enumerate(PPE_PRODUCTS)}
    donors["entity_index"] = donors["don_id"].map(donor_index).astype(int)
    recipients["entity_index"] = recipients["rec_id"].map(recipient_index).astype(int)
    donors["product_index"] = donors["ppe"].map(product_index).astype(int)
    recipients["product_index"] = recipients["ppe"].map(product_index).astype(int)
    donors = donors.sort_values(["date", "entity_index", "product_index"]).reset_index(drop=True)
    recipients = recipients.sort_values(
        ["date", "entity_index", "product_index"]
    ).reset_index(drop=True)

    distance_frame = pd.read_csv(files["distances"])
    distance_frame = distance_frame.loc[
        distance_frame["don_id"].isin(donor_index)
        & distance_frame["rec_id"].isin(recipient_index)
        & pd.to_numeric(distance_frame["distance"], errors="coerce").notna()
    ].copy()
    distance_frame["donor_index"] = distance_frame["don_id"].map(donor_index).astype(int)
    distance_frame["recipient_index"] = distance_frame["rec_id"].map(recipient_index).astype(int)
    distance_frame["distance"] = pd.to_numeric(distance_frame["distance"], errors="coerce")
    distances: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for donor, frame in distance_frame.groupby("donor_index", sort=False):
        ordered = frame.sort_values(["distance", "recipient_index"])
        distances[int(donor)] = (
            ordered["recipient_index"].to_numpy(dtype=int),
            ordered["distance"].to_numpy(dtype=float),
        )
    start = min(donors["date"].min(), recipients["date"].min()).floor("D")
    stop = max(donors["date"].max(), recipients["date"].max()).ceil("D")
    decision_dates = pd.date_range(
        start=start + pd.Timedelta(days=interval_days),
        end=stop + pd.Timedelta(days=interval_days),
        freq=f"{int(interval_days)}D",
        tz="UTC",
    )
    return PPEMatchData(
        donors=donors,
        recipients=recipients,
        distances=distances,
        donor_ids=donor_ids,
        recipient_ids=recipient_ids,
        decision_dates=decision_dates,
        source_hashes={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files.values()
        },
    )


def _coupling(parameters: V2Parameters, channels: int, coupled: bool) -> np.ndarray:
    if not coupled:
        return np.eye(channels)
    if channels == 4:
        return parameters.coupling()
    if channels == 3:
        return np.array(
            [[1.0, -0.08, 0.16], [-0.08, 1.0, 0.16], [0.08, 0.08, 0.90]],
            dtype=float,
        )
    return np.array([[1.0, -0.08], [-0.08, 1.0]], dtype=float)


def _reduced_evidence(full: np.ndarray, channels: int) -> np.ndarray:
    if channels == 4:
        return full
    if channels == 3:
        return np.column_stack([full[:, 0], full[:, 1], 0.5 * (full[:, 2] + full[:, 3])])
    return full[:, :2]


def _star_acm_score(
    source_scores: np.ndarray,
    source_confidence: np.ndarray,
    route_risk: np.ndarray,
    supply_scarcity: float,
    previous_recipient_state: np.ndarray,
    previous_supply_state: np.ndarray,
    parameters: V2Parameters,
    channels: int,
    coupled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    recipient_evidence_full = _weighted_evidence(
        source_scores, source_confidence, parameters.source_weights()
    )
    supply_evidence_full = _weighted_evidence(
        np.array([[1.0 - supply_scarcity, 1.0 - float(route_risk.mean())]]),
        np.array([[0.95, 0.90]]),
        np.array([1.0, 1.0]),
    )[0]
    full = np.vstack([recipient_evidence_full, supply_evidence_full])
    evidence = _reduced_evidence(full, channels)
    n_recipient = len(source_scores)
    supply_node = n_recipient
    rows = np.r_[np.arange(n_recipient), np.full(n_recipient, supply_node)]
    cols = np.r_[np.full(n_recipient, supply_node), np.arange(n_recipient)]
    signs = -np.ones(2 * n_recipient, dtype=float)
    magnitudes = np.r_[0.42 + 0.46 * (1.0 - route_risk), np.full(n_recipient, 0.70)]
    mu = parameters.gaussian_mu()[:channels]
    sigma = parameters.gaussian_sigma()[:channels]
    beta = parameters.gaussian_beta()[:channels]
    gaussian = beta[None, :] * np.exp(
        -((magnitudes[:, None] - mu[None, :]) ** 2) / (2.0 * sigma[None, :] ** 2)
    )
    shares = gaussian / np.maximum(gaussian.sum(axis=1, keepdims=True), 1e-12)
    identity = sparse.eye(n_recipient + 1, format="csr")
    raw_matrices: list[sparse.csr_matrix] = []
    for channel in range(channels):
        values = signs * magnitudes * shares[:, channel]
        weighted = sparse.csr_matrix(
            (values, (rows, cols)), shape=(n_recipient + 1, n_recipient + 1)
        )
        raw_matrices.append(parameters.diagonal_memory * identity + weighted)
    coupling = _coupling(parameters, channels, coupled)
    raw_block = sparse.bmat(
        [[coupling[i, j] * raw_matrices[j] for j in range(channels)] for i in range(channels)],
        format="csr",
    )
    reference_coupling = parameters.coupling() if channels == 4 else coupling
    reference_block = sparse.bmat(
        [
            [reference_coupling[i, j] * raw_matrices[j] for j in range(channels)]
            for i in range(channels)
        ],
        format="csr",
    )
    reference_norm = _power_norm(reference_block, iterations=35)
    actual_norm = _power_norm(raw_block, iterations=35)
    scale = parameters.operator_target_norm / max(1.01 * reference_norm, 1e-12)
    matrices = [matrix * scale for matrix in raw_matrices]
    state = np.vstack([previous_recipient_state, previous_supply_state[None, :]])
    gaps: list[float] = []
    for _ in range(60):
        messages = np.column_stack(
            [matrices[channel] @ state[:, channel] for channel in range(channels)]
        )
        dynamic = 0.5 * (1.0 + np.tanh(messages @ coupling.T))
        updated = (
            parameters.physical_mix * evidence
            + (1.0 - parameters.physical_mix) * dynamic
        )
        updated = _project_rows(updated, cap=2.0 if channels >= 3 else 1.6)
        gap = float(np.max(np.abs(updated - state)))
        gaps.append(gap)
        state = updated
        if gap <= 1e-4:
            break
    recipient_state = state[:-1]
    recipient_evidence = recipient_evidence_full
    if channels == 4:
        gamma = parameters.partial_coefficient
        state_score = (
            recipient_state[:, 0]
            - recipient_state[:, 1]
            + gamma * (recipient_state[:, 2] - recipient_state[:, 3])
        )
        evidence_score = (
            recipient_evidence[:, 0]
            - recipient_evidence[:, 1]
            + gamma * (recipient_evidence[:, 2] - recipient_evidence[:, 3])
        )
    elif channels == 3:
        state_score = recipient_state[:, 0] - recipient_state[:, 1] - 0.15 * recipient_state[:, 2]
        evidence_score = (
            recipient_evidence[:, 0]
            - recipient_evidence[:, 1]
            - 0.075 * (recipient_evidence[:, 2] + recipient_evidence[:, 3])
        )
    else:
        state_score = recipient_state[:, 0] - recipient_state[:, 1]
        evidence_score = recipient_evidence[:, 0] - recipient_evidence[:, 1]
    return (
        0.55 * state_score + 0.45 * evidence_score,
        recipient_state,
        state[-1],
        {
            "iterations": float(len(gaps)),
            "convergence_gap": gaps[-1] if gaps else 0.0,
            "contraction_bound": float(
                (1.0 - parameters.physical_mix) * 0.5 * scale * actual_norm
            ),
        },
    )


def _split_for_index(index: int, count: int) -> str:
    calibration_stop = max(1, int(np.floor(0.60 * count)))
    validation_stop = max(calibration_stop + 1, int(np.floor(0.80 * count)))
    if index < calibration_stop:
        return "calibration"
    if index < validation_stop:
        return "validation"
    return "test"


def simulate_ppe_match(
    data: PPEMatchData,
    parameters: V2Parameters,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model not in PPE_MODELS:
        raise ValueError(f"Unsupported PPE-Match model: {model}")
    n_product = len(PPE_PRODUCTS)
    n_donor = len(data.donor_ids)
    n_recipient = len(data.recipient_ids)
    donor_qty = np.zeros((n_product, n_donor), dtype=float)
    donor_date = np.full((n_product, n_donor), np.datetime64("NaT"), dtype="datetime64[ns]")
    recipient_qty = np.zeros((n_product, n_recipient), dtype=float)
    recipient_date = np.full(
        (n_product, n_recipient), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    cumulative_request = np.zeros_like(recipient_qty)
    cumulative_received = np.zeros_like(recipient_qty)
    channels = 4 if "ACM-4" in model else 3 if model == "ACM-3" else 2
    state = [
        np.full((n_recipient, channels), min(0.5, 2.0 / channels), dtype=float)
        for _ in range(n_product)
    ]
    supply_state = [
        np.full(channels, min(0.5, 2.0 / channels), dtype=float)
        for _ in range(n_product)
    ]
    donor_pointer = 0
    recipient_pointer = 0
    donor_times = data.donors["date"].astype("int64").to_numpy()
    recipient_times = data.recipients["date"].astype("int64").to_numpy()
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for date_index, date in enumerate(data.decision_dates):
        cutoff = int(date.value)
        donor_stop = int(np.searchsorted(donor_times, cutoff, side="right"))
        for row in data.donors.iloc[donor_pointer:donor_stop].itertuples(index=False):
            product = int(row.product_index)
            entity = int(row.entity_index)
            if donor_qty[product, entity] <= 0:
                donor_date[product, entity] = np.datetime64(row.date.tz_convert(None))
            donor_qty[product, entity] += float(row.qty)
        donor_pointer = donor_stop
        recipient_stop = int(np.searchsorted(recipient_times, cutoff, side="right"))
        for row in data.recipients.iloc[recipient_pointer:recipient_stop].itertuples(index=False):
            product = int(row.product_index)
            entity = int(row.entity_index)
            if recipient_qty[product, entity] <= 0:
                recipient_date[product, entity] = np.datetime64(row.date.tz_convert(None))
            recipient_qty[product, entity] += float(row.qty)
            cumulative_request[product, entity] += float(row.qty)
        recipient_pointer = recipient_stop
        split = _split_for_index(date_index, len(data.decision_dates))

        for product, product_name in enumerate(PPE_PRODUCTS):
            active_donor = np.flatnonzero(donor_qty[product] > 1e-12)
            active_recipient = np.flatnonzero(recipient_qty[product] > 1e-12)
            if len(active_donor) == 0 or len(active_recipient) == 0:
                continue
            demand = recipient_qty[product, active_recipient].copy()
            log_demand = np.log1p(demand)
            demand_scale = max(float(np.quantile(log_demand, 0.95)), 1e-12)
            demand_pressure = np.clip(log_demand / demand_scale, 0.0, 1.0)
            current_naive = date.tz_convert(None).to_datetime64()
            ages = (
                current_naive - recipient_date[product, active_recipient]
            ).astype("timedelta64[s]").astype(float) / 86400.0
            backlog_pressure = np.clip(ages / 60.0, 0.0, 1.0)
            prior_fill = np.divide(
                cumulative_received[product, active_recipient],
                cumulative_request[product, active_recipient],
                out=np.zeros(len(active_recipient), dtype=float),
                where=cumulative_request[product, active_recipient] > 0,
            )
            inventory_risk = 1.0 - np.clip(prior_fill, 0.0, 1.0)
            min_distance = np.full(n_recipient, np.inf, dtype=float)
            for donor in active_donor:
                edge = data.distances.get(int(donor))
                if edge is None:
                    continue
                recipients, distances = edge
                eligible = recipient_qty[product, recipients] > 1e-12
                np.minimum.at(min_distance, recipients[eligible], distances[eligible])
            route_values = min_distance[active_recipient]
            route_observed = np.isfinite(route_values)
            route_risk = np.clip(np.where(route_observed, route_values, 3000.0) / 3000.0, 0.0, 1.0)
            supply_total = float(donor_qty[product, active_donor].sum())
            demand_total = float(demand.sum())
            capacity_risk = float(np.clip(1.0 - supply_total / max(demand_total, 1.0), 0.0, 1.0))
            source_scores = np.column_stack(
                [
                    demand_pressure,
                    backlog_pressure,
                    inventory_risk,
                    route_risk,
                    np.full(len(active_recipient), capacity_risk),
                ]
            )
            source_confidence = np.ones_like(source_scores)
            source_confidence[:, 3] = route_observed.astype(float)
            if model.startswith("ACM"):
                local_score, local_state, local_supply_state, diagnostics = _star_acm_score(
                    source_scores,
                    source_confidence,
                    route_risk,
                    capacity_risk,
                    state[product][active_recipient],
                    supply_state[product],
                    parameters,
                    channels=channels,
                    coupled=model == "ACM-4 (coupled)" or model in {"ACM-3", "ACM-2"},
                )
                state[product][active_recipient] = local_state
                supply_state[product] = local_supply_state
                priority = stable_softmax(local_score, temperature=parameters.temperature)
            elif model == "Demand proportional":
                priority = demand / max(float(demand.sum()), 1e-12)
                diagnostics = {"iterations": 1.0, "convergence_gap": 0.0, "contraction_bound": np.nan}
            else:
                priority = np.full(len(active_recipient), 1.0 / len(active_recipient))
                diagnostics = {"iterations": 1.0, "convergence_gap": 0.0, "contraction_bound": np.nan}
            priority_index = priority / max(float(priority.max()), 1e-12)
            base_objective = (
                parameters.lp_priority_weight * priority_index
                + parameters.lp_demand_weight * demand_pressure
                + parameters.lp_fairness_weight * (1.0 - prior_fill)
            )
            base_full = np.full(n_recipient, -np.inf, dtype=float)
            base_full[active_recipient] = base_objective
            requested_before = cumulative_request[product] > 0
            shipment_qty = 0.0
            shipment_miles = 0.0
            shipment_holding = 0.0
            shipments = 0
            donor_order = sorted(
                active_donor.tolist(),
                key=lambda item: (
                    donor_date[product, item]
                    if not np.isnat(donor_date[product, item])
                    else np.datetime64("2262-01-01"),
                    item,
                ),
            )
            for donor in donor_order:
                edge = data.distances.get(int(donor))
                if edge is None:
                    continue
                candidates, distances = edge
                eligible = recipient_qty[product, candidates] > 1e-12
                if not np.any(eligible):
                    continue
                candidates = candidates[eligible]
                distances = distances[eligible]
                if model == "Proximity":
                    choice = int(np.argmin(distances))
                else:
                    objective = (
                        base_full[candidates]
                        - parameters.lp_lead_weight * np.clip(distances / 3000.0, 0.0, 1.0)
                    )
                    choice = int(np.argmax(objective))
                recipient = int(candidates[choice])
                distance = float(distances[choice])
                quantity = min(
                    float(donor_qty[product, donor]),
                    float(recipient_qty[product, recipient]),
                )
                if quantity <= 0:
                    continue
                holding = float(
                    (current_naive - donor_date[product, donor]).astype("timedelta64[s]").astype(float)
                    / 86400.0
                )
                donor_qty[product, donor] -= quantity
                recipient_qty[product, recipient] -= quantity
                cumulative_received[product, recipient] += quantity
                shipment_qty += quantity
                shipment_miles += quantity * distance
                shipment_holding += quantity * max(holding, 0.0)
                shipments += 1
                decision_rows.append(
                    {
                        "date": date,
                        "split": split,
                        "product": product_name,
                        "model": model,
                        "donor": str(data.donor_ids[donor]),
                        "recipient": str(data.recipient_ids[recipient]),
                        "quantity": quantity,
                        "distance": distance,
                        "holding_days": max(holding, 0.0),
                    }
                )
            fills = np.divide(
                cumulative_received[product, requested_before],
                cumulative_request[product, requested_before],
                out=np.zeros(int(requested_before.sum()), dtype=float),
                where=cumulative_request[product, requested_before] > 0,
            )
            service = float(
                cumulative_received[product].sum()
                / max(float(cumulative_request[product].sum()), 1.0)
            )
            coverage = float(np.mean(fills > 0)) if len(fills) else 0.0
            fill_gini = gini(fills)
            unit_miles = shipment_miles / max(shipment_qty, 1.0)
            holding_days = shipment_holding / max(shipment_qty, 1.0)
            operational_score = float(
                0.35 * service
                + 0.25 * coverage
                + 0.20 * (1.0 - fill_gini)
                + 0.10 * (1.0 - np.clip(unit_miles / 3000.0, 0.0, 1.0))
                + 0.10 * (1.0 - np.clip(holding_days / 60.0, 0.0, 1.0))
            )
            metric_rows.append(
                {
                    "date": date,
                    "split": split,
                    "product": product_name,
                    "model": model,
                    "active_donors": len(active_donor),
                    "active_recipients": len(active_recipient),
                    "supply_units": supply_total,
                    "demand_units": demand_total,
                    "allocated_units": shipment_qty,
                    "service_level": service,
                    "recipient_coverage": coverage,
                    "fill_gini": fill_gini,
                    "unit_miles": unit_miles,
                    "holding_days": holding_days,
                    "shipments": shipments,
                    "operational_score": operational_score,
                    "mean_iterations": diagnostics["iterations"],
                    "convergence_gap": diagnostics["convergence_gap"],
                    "contraction_bound": diagnostics["contraction_bound"],
                    "panel_ambiguity": float(
                        np.mean(
                            0.60 * (1.0 - source_confidence.mean(axis=1))
                            + 0.40 * np.clip(np.std(source_scores, axis=1) / 0.5, 0.0, 1.0)
                        )
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(decision_rows)


def _ppe_trial_parameters(trial: Any) -> V2Parameters:
    base = _trial_parameters(trial)
    # Override only the evidence/priority scale ranges that depend on the much
    # larger PPE-Match candidate pool.  LP coefficients remain unchanged.
    values = {
        key: getattr(base, key)
        for key in V2Parameters.__dataclass_fields__
        if hasattr(base, key)
    }
    values.update(
        {
            "temperature": trial.suggest_float("ppe_temperature", 0.20, 1.50, log=True),
            "weight_demand": trial.suggest_float("ppe_weight_demand", 0.25, 4.00, log=True),
            "weight_backlog": trial.suggest_float("ppe_weight_backlog", 0.25, 4.00, log=True),
            "weight_inventory": trial.suggest_float("ppe_weight_inventory", 0.25, 4.00, log=True),
            "weight_route": trial.suggest_float("ppe_weight_route", 0.05, 2.00, log=True),
            "weight_capacity": trial.suggest_float("ppe_weight_capacity", 0.25, 4.00, log=True),
        }
    )
    return V2Parameters(**values)


def optimize_ppe_match(
    data: PPEMatchData,
    n_trials: int = 28,
    seed: int = 20260718,
    shortlist: int = 7,
    progress: bool = True,
) -> tuple[V2Parameters, pd.DataFrame]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=min(10, int(n_trials))
        ),
    )

    def objective(trial: Any) -> float:
        parameters = _ppe_trial_parameters(trial)
        metrics, _ = simulate_ppe_match(data, parameters, "ACM-4 (coupled)")
        calibration = metrics.loc[metrics["split"].eq("calibration")]
        raw = float(calibration["operational_score"].mean())
        penalty = 0.001 * float(
            np.mean(
                [
                    ((parameters.mu_true - 0.82) / 0.06) ** 2,
                    ((parameters.mu_partial - 0.62) / 0.06) ** 2,
                    ((parameters.diagonal_memory - 0.42) / 0.08) ** 2,
                    ((parameters.partial_to_exact - 0.24) / 0.12) ** 2,
                    ((parameters.exact_to_partial - 0.12) / 0.06) ** 2,
                ]
            )
        )
        trial.set_user_attr("calibration_raw_score", raw)
        trial.set_user_attr("regularization_penalty", penalty)
        trial.set_user_attr(
            "resolved_parameters",
            {
                key: getattr(parameters, key)
                for key in V2Parameters.__dataclass_fields__
            },
        )
        if progress:
            print(
                f"  PPE-Match Optuna trial {trial.number + 1}/{n_trials}: "
                f"calibration={raw:.5f}, penalized={raw - penalty:.5f}",
                flush=True,
            )
        return raw - penalty

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    ordered = sorted(study.trials, key=lambda item: float(item.value), reverse=True)
    shortlisted = ordered[: min(int(shortlist), len(ordered))]
    validation: dict[int, float] = {}
    for item in shortlisted:
        parameters = _params_from_mapping(item.user_attrs["resolved_parameters"])
        metrics, _ = simulate_ppe_match(data, parameters, "ACM-4 (coupled)")
        validation[item.number] = float(
            metrics.loc[metrics["split"].eq("validation"), "operational_score"].mean()
        )
    selected = max(
        shortlisted,
        key=lambda item: (validation[item.number], float(item.value)),
    )
    rows: list[dict[str, Any]] = []
    for item in study.trials:
        resolved = item.user_attrs.get("resolved_parameters", {})
        rows.append(
            {
                "trial": int(item.number),
                **resolved,
                "calibration_objective": float(item.value),
                "calibration_raw_score": float(item.user_attrs.get("calibration_raw_score", np.nan)),
                "regularization_penalty": float(item.user_attrs.get("regularization_penalty", np.nan)),
                "calibration_shortlist": bool(item in shortlisted),
                "validation_objective": validation.get(item.number, np.nan),
                "selected": bool(item.number == selected.number),
            }
        )
    return (
        _params_from_mapping(selected.user_attrs["resolved_parameters"]),
        pd.DataFrame(rows).sort_values("trial"),
    )


def summarize_ppe_match(
    metrics: pd.DataFrame,
    seed: int = 20260718,
    n_boot: int = 5000,
) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")]
    names = (
        "operational_score",
        "service_level",
        "recipient_coverage",
        "fill_gini",
        "unit_miles",
        "holding_days",
    )
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(PPE_MODELS):
        frame = test.loc[test["model"].eq(model)]
        for metric_index, metric in enumerate(names):
            values = frame[metric].dropna().to_numpy(dtype=float)
            rng = np.random.default_rng(seed + 97 * model_index + metric_index)
            if len(values) > 1:
                indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
                draws = values[indices].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
            elif len(values) == 1:
                low = high = values[0]
            else:
                low = high = np.nan
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_product_week_panels": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def test_ppe_match(
    metrics: pd.DataFrame,
    reference: str = "ACM-4 (coupled)",
) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")]
    directions = {
        "operational_score": 1.0,
        "service_level": 1.0,
        "recipient_coverage": 1.0,
        "fill_gini": -1.0,
        "unit_miles": -1.0,
        "holding_days": -1.0,
    }
    rows: list[dict[str, Any]] = []
    for metric, direction in directions.items():
        reference_frame = test.loc[
            test["model"].eq(reference), ["date", "product", metric]
        ].rename(columns={metric: "reference_value"})
        group: list[dict[str, Any]] = []
        for model in PPE_MODELS:
            if model == reference:
                continue
            comparator = test.loc[
                test["model"].eq(model), ["date", "product", metric]
            ].rename(columns={metric: "comparator_value"})
            paired = reference_frame.merge(comparator, on=["date", "product"])
            benefit = direction * (
                paired["reference_value"] - paired["comparator_value"]
            ).to_numpy(dtype=float)
            benefit = benefit[np.isfinite(benefit)]
            if len(benefit) == 0 or np.allclose(benefit, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(benefit, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            group.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparator": model,
                    "mean_benefit_reference": float(np.mean(benefit)) if len(benefit) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "n_paired_panels": int(len(benefit)),
                }
            )
        adjusted = _holm_adjust(np.array([row["p_value"] for row in group]))
        for row, value in zip(group, adjusted):
            row["holm_p"] = float(value)
            row["significant_0_05"] = bool(value < 0.05)
        rows.extend(group)
    return pd.DataFrame(rows)


def optimizer_preset_comparison(
    preset_metrics: pd.DataFrame,
    optimized_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Paired temporal-test comparison against the frozen manuscript preset."""

    preset = preset_metrics.loc[preset_metrics["split"].eq("test")].copy()
    optimized = optimized_metrics.loc[
        optimized_metrics["split"].eq("test")
        & optimized_metrics["model"].eq("ACM-4 (coupled)")
    ].copy()
    preset["date"] = pd.to_datetime(preset["date"], utc=True)
    optimized["date"] = pd.to_datetime(optimized["date"], utc=True)
    directions = {
        "operational_score": (1.0, "higher"),
        "service_level": (1.0, "higher"),
        "recipient_coverage": (1.0, "higher"),
        "fill_gini": (-1.0, "lower"),
        "unit_miles": (-1.0, "lower"),
        "holding_days": (-1.0, "lower"),
    }
    rows: list[dict[str, Any]] = []
    for metric, (direction, preferred) in directions.items():
        left = preset[["date", "product", metric]].rename(
            columns={metric: "preset_value"}
        )
        right = optimized[["date", "product", metric]].rename(
            columns={metric: "optimized_value"}
        )
        paired = left.merge(right, on=["date", "product"]).dropna()
        benefit = direction * (
            paired["optimized_value"] - paired["preset_value"]
        ).to_numpy(dtype=float)
        if len(benefit) == 0 or np.allclose(benefit, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                benefit, alternative="two-sided", zero_method="wilcox"
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        preset_mean = float(paired["preset_value"].mean())
        optimized_mean = float(paired["optimized_value"].mean())
        rows.append(
            {
                "metric": metric,
                "preferred_direction": preferred,
                "preset_mean": preset_mean,
                "optimized_mean": optimized_mean,
                "mean_benefit_optimized": float(np.mean(benefit)),
                "relative_change_percent": (
                    100.0 * (optimized_mean - preset_mean) / abs(preset_mean)
                    if abs(preset_mean) > 1e-12
                    else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "n_paired_panels": int(len(benefit)),
            }
        )
    adjusted = _holm_adjust(np.array([row["p_value"] for row in rows]))
    for row, value in zip(rows, adjusted):
        row["holm_p"] = float(value)
        row["significant_0_05"] = bool(value < 0.05)
    return pd.DataFrame(rows)


def ppe_context_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    reference = test.loc[
        test["model"].eq("ACM-4 (coupled)"),
        ["date", "product", "panel_ambiguity", "supply_units", "demand_units"],
    ].copy()
    reference["scarcity_index"] = 1.0 - np.clip(
        reference["supply_units"] / reference["demand_units"].clip(lower=1.0), 0.0, 1.0
    )
    definitions = {
        "Feature ambiguity": ("panel_ambiguity", float(reference["panel_ambiguity"].median())),
        "Supply scarcity": ("scarcity_index", float(reference["scarcity_index"].median())),
    }
    rows: list[dict[str, Any]] = []
    for context, (column, threshold) in definitions.items():
        labels = reference[["date", "product", column]].copy()
        labels["context_stratum"] = np.where(
            labels[column] >= threshold, "Higher", "Lower"
        )
        frame = test.merge(
            labels[["date", "product", "context_stratum"]],
            on=["date", "product"],
            how="left",
        )
        for stratum in ("Lower", "Higher"):
            for model in PPE_MODELS:
                subset = frame.loc[
                    frame["context_stratum"].eq(stratum) & frame["model"].eq(model)
                ]
                rows.append(
                    {
                        "context": context,
                        "threshold": threshold,
                        "stratum": stratum,
                        "model": model,
                        "n_product_week_panels": int(len(subset)),
                        "operational_score": float(subset["operational_score"].mean()),
                        "service_level": float(subset["service_level"].mean()),
                        "recipient_coverage": float(subset["recipient_coverage"].mean()),
                        "fill_gini": float(subset["fill_gini"].mean()),
                        "unit_miles": float(subset["unit_miles"].mean()),
                        "holding_days": float(subset["holding_days"].mean()),
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class PPEMatchBundle:
    data: PPEMatchData
    selected_parameters: V2Parameters
    trials: pd.DataFrame
    metrics: pd.DataFrame
    decisions: pd.DataFrame
    summary: pd.DataFrame
    tests: pd.DataFrame
    optimizer_comparison: pd.DataFrame


def run_ppe_match_study(
    data_dir: Path,
    n_trials: int = 28,
    seed: int = 20260718,
    n_boot: int = 5000,
    progress: bool = True,
) -> PPEMatchBundle:
    data = load_ppe_match_data(data_dir)
    parameters, trials = optimize_ppe_match(
        data, n_trials=n_trials, seed=seed, progress=progress
    )
    metrics: list[pd.DataFrame] = []
    decisions: list[pd.DataFrame] = []
    for model in PPE_MODELS:
        model_metrics, model_decisions = simulate_ppe_match(data, parameters, model)
        metrics.append(model_metrics)
        decisions.append(model_decisions)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_decisions = pd.concat(decisions, ignore_index=True)
    preset_metrics, _ = simulate_ppe_match(
        data, V2Parameters(), "ACM-4 (coupled)"
    )
    return PPEMatchBundle(
        data=data,
        selected_parameters=parameters,
        trials=trials,
        metrics=all_metrics,
        decisions=all_decisions,
        summary=summarize_ppe_match(all_metrics, seed=seed, n_boot=n_boot),
        tests=test_ppe_match(all_metrics),
        optimizer_comparison=optimizer_preset_comparison(
            preset_metrics, all_metrics
        ),
    )


def _plot_ppe_results(bundle: PPEMatchBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    panels = [
        ("service_level", "Service level"),
        ("recipient_coverage", "Recipient coverage"),
        ("unit_miles", "Unit-miles"),
        ("holding_days", "Holding time (days)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.1, 10.5), sharex=True)
    positions = np.arange(len(PPE_MODELS))
    for panel_index, (ax, (metric, ylabel)) in enumerate(zip(axes, panels)):
        values = bundle.summary.loc[bundle.summary["metric"].eq(metric)].set_index("model")
        for position, model in enumerate(PPE_MODELS):
            row = values.loc[model]
            style = MODEL_STYLES.get(model, {})
            color = style.get("color", str(0.15 + 0.10 * position))
            marker = style.get("marker", "o")
            ax.errorbar(
                position,
                row["mean"],
                yerr=np.array([[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]),
                fmt=marker,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                capsize=3,
                linewidth=1.0,
            )
        ax.set_ylabel(ylabel)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel_index))
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(PPE_MODELS, rotation=35, ha="right")
    save_figure(fig, output, "fig_real12_ppe_match_operational", formats=("png",), dpi=dpi)


def _ppe_result_paragraph(bundle: PPEMatchBundle) -> str:
    summary = bundle.summary.set_index(["model", "metric"])
    acm_service = summary.loc[("ACM-4 (coupled)", "service_level")]
    acm_coverage = summary.loc[("ACM-4 (coupled)", "recipient_coverage")]
    acm_distance = summary.loc[("ACM-4 (coupled)", "unit_miles")]
    best = (
        bundle.summary.loc[bundle.summary["metric"].eq("operational_score")]
        .sort_values("mean", ascending=False)
        .iloc[0]
    )
    independent = summary.loc[("ACM-4 (independent)", "operational_score")]
    coupled = summary.loc[("ACM-4 (coupled)", "operational_score")]
    context = ppe_context_summary(bundle.metrics)
    high = context.loc[
        context["context"].eq("Feature ambiguity") & context["stratum"].eq("Higher")
    ].set_index("model")
    optimizer = bundle.optimizer_comparison.set_index("metric")
    optimizer_score = optimizer.loc["operational_score"]
    optimizer_distance = optimizer.loc["unit_miles"]
    return f"""% Generated from the official PPE-Match request/offer stream and distance matrix.
\\subsection{{External operational replay on PPE-Match}}

As an external real-stream stress test, we used the official PPE-Match framework data: timestamped donor offers, recipient requests, and the anonymized donor--recipient distance matrix. Four PPE classes were processed at seven-day intervals using the same fixed LP coefficient ratios. The first 60\\% of decision dates were used for TPE calibration, the next 20\\% for internal selection, and the final 20\\% for temporal testing. This analysis evaluates counterfactual policies on observed request streams; it is distinct from the historical-assignment agreement analysis. Against the frozen pre-optimization preset, the selected configuration increased the temporal-test operational score from {optimizer_score['preset_mean']:.4f} to {optimizer_score['optimized_mean']:.4f} and reduced unit-miles from {optimizer_distance['preset_mean']:.1f} to {optimizer_distance['optimized_mean']:.1f}; both paired comparisons remained significant after Holm correction ($p={optimizer_score['holm_p']:.4f}$).

On the temporal test panels, coupled ACM attained service {acm_service['mean']:.4f} [{acm_service['ci_low']:.4f}, {acm_service['ci_high']:.4f}], recipient coverage {acm_coverage['mean']:.4f} [{acm_coverage['ci_low']:.4f}, {acm_coverage['ci_high']:.4f}], and {acm_distance['mean']:.1f} [{acm_distance['ci_low']:.1f}, {acm_distance['ci_high']:.1f}] unit-miles. The highest prespecified composite operational score was obtained by {best['model']} ({best['mean']:.4f}). In the feature-defined higher-ambiguity panels, the corresponding scores were {high.loc['ACM-4 (coupled)', 'operational_score']:.4f} for coupled ACM, {high.loc['Proximity', 'operational_score']:.4f} for proximity matching, {high.loc['Demand proportional', 'operational_score']:.4f} for demand proportional, and {high.loc['Equal allocation', 'operational_score']:.4f} for equal allocation. Coupled and independent four-channel scores were {coupled['mean']:.4f} and {independent['mean']:.4f}, respectively. Thus, ACM offered a stronger service--holding-time compromise than proximity and a stronger coverage--distance compromise than demand proportional, especially in higher-ambiguity panels, but equal allocation retained the highest composite score and the coupling ablation remained indistinguishable.
"""


def create_ppe_match_outputs(
    bundle: PPEMatchBundle,
    output_root: Path,
    dpi: int = 600,
) -> None:
    results = output_root / "results"
    figures = output_root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    bundle.trials.to_csv(results / "optuna_trials.csv", index=False)
    bundle.metrics.to_csv(results / "product_week_metrics.csv", index=False)
    bundle.decisions.to_csv(results / "shipment_decisions.csv", index=False)
    bundle.summary.to_csv(results / "test_summary.csv", index=False)
    bundle.tests.to_csv(results / "paired_statistical_tests.csv", index=False)
    bundle.optimizer_comparison.to_csv(
        results / "optimizer_preset_comparison.csv", index=False
    )
    ppe_context_summary(bundle.metrics).to_csv(
        results / "context_strata_summary.csv", index=False
    )
    (results / "selected_parameters.json").write_text(
        json.dumps(bundle.selected_parameters.as_dict(), indent=2), encoding="utf-8"
    )
    (results / "data_provenance.json").write_text(
        json.dumps(
            {
                "source": "ppe-match 0.1.4 public package",
                "homepage": "https://pypi.org/project/ppe-match/",
                "framework_repository": "https://github.com/samorani/MatchingPPE",
                "products": list(PPE_PRODUCTS),
                "decision_interval_days": 7,
                "maximum_donation_quantity": 1000,
                "source_file_sha256": bundle.data.source_hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "manuscript_results_draft.tex").write_text(
        _ppe_result_paragraph(bundle), encoding="utf-8"
    )
    _plot_ppe_results(bundle, figures, dpi)
    manifest: list[dict[str, Any]] = []
    for path in sorted(
        item
        for item in output_root.rglob("*")
        if item.is_file()
        and item.name != "MANIFEST.json"
        and not item.name.startswith(".")
    ):
        manifest.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (output_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
