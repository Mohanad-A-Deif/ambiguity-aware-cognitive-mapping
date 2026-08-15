from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import ConceptGraph, stable_softmax
from .retrospective_allocation import (
    ACMParameters,
    MODEL_ORDER,
    EventRecord,
    ProductSeries,
    RetrospectiveBundle,
    _aggregated_metrics,
    _cluster_bootstrap_ci,
    _evidence_from_sources,
    _holm_adjust,
    _power_norm,
    _project_rows,
    _scalar_coefficient_table,
    _scalar_features,
    ambiguity_analysis,
    build_sparse_product_graph,
    describe_splits,
    load_getusppe_series,
    paired_model_tests,
    summarize_test_metrics,
)
from .style import MODEL_STYLES, apply_nature_style, clean_axis, panel_label, save_figure


@dataclass(frozen=True)
class V2Parameters(ACMParameters):
    """ACM parameters plus calibrated evidence-source weights.

    LP coefficients remain fixed at the manuscript values.  The source weights
    alter only the evidence encoder and are normalized to mean one, making the
    absolute scale unidentified by construction.
    """

    weight_demand: float = 1.0
    weight_backlog: float = 1.0
    weight_inventory: float = 1.0
    weight_route: float = 1.0
    weight_capacity: float = 1.0

    def source_weights(self) -> np.ndarray:
        values = np.array(
            [
                self.weight_demand,
                self.weight_backlog,
                self.weight_inventory,
                self.weight_route,
                self.weight_capacity,
            ],
            dtype=float,
        )
        return values / max(float(values.mean()), 1e-12)

    def as_dict(self) -> dict[str, float]:
        values = super().as_dict()
        values.update(
            {
                "weight_demand": self.weight_demand,
                "weight_backlog": self.weight_backlog,
                "weight_inventory": self.weight_inventory,
                "weight_route": self.weight_route,
                "weight_capacity": self.weight_capacity,
            }
        )
        return values


@dataclass
class RetrospectiveV2Bundle(RetrospectiveBundle):
    """V2 outputs, including the frozen preset-versus-optimized comparison."""

    optimizer_comparison: pd.DataFrame


def prepare_v2_series(
    data_dir: Path,
    lookback_days: int = 90,
) -> tuple[list[ProductSeries], pd.DataFrame]:
    """Load the actual-allocation panels and remove outcome-derived evidence.

    The replay remains conditional on each logged batch quantity so that it
    evaluates recipient assignment rather than donation-volume forecasting.
    Evidence scarcity, however, is reconstructed exclusively from offers
    timestamped by the decision and commitments timestamped before it.
    """

    series, audit = load_getusppe_series(data_dir, lookback_days=lookback_days)
    event_rows: list[dict[str, Any]] = []
    for product in series:
        for event in product.records:
            idx = np.flatnonzero(event.active)
            if len(idx) == 0:
                continue
            demand_total = float(event.request_cap[idx].sum())
            old_scarcity = float(
                np.clip(
                    1.0 - event.observed_allocation.sum() / max(demand_total, 1.0),
                    0.0,
                    1.0,
                )
            )
            # Recover the policy component from the legacy 50/50 mixture, then
            # replace only the outcome-derived scarcity term.
            policy_risk = np.clip(2.0 * event.source_scores[idx, 4] - old_scarcity, 0.0, 1.0)
            if event.supply_confidence > 0:
                predecision_scarcity = float(
                    np.clip(1.0 - event.predecision_supply / max(demand_total, 1.0), 0.0, 1.0)
                )
            else:
                predecision_scarcity = 0.50

            event.source_scores[idx, 3] = event.route_risk[idx]
            event.source_scores[idx, 4] = np.clip(
                0.5 * policy_risk + 0.5 * predecision_scarcity, 0.0, 1.0
            )
            policy_observed = np.clip(2.0 * event.source_confidence[idx, 4] - 1.0, 0.0, 1.0)
            event.source_confidence[idx, 3] = event.route_confidence
            event.source_confidence[idx, 4] = 0.5 * (
                policy_observed + event.supply_confidence
            )
            event.dc_stock_risk[:, 0] = predecision_scarcity
            event.supplier_risk[0, 0] = predecision_scarcity

            scores = event.source_scores[idx]
            confidence = event.source_confidence[idx]
            disagreement = np.clip(np.std(scores, axis=1) / 0.5, 0.0, 1.0)
            missingness = 1.0 - confidence.mean(axis=1)
            event.ambiguity[idx] = np.clip(
                0.60 * missingness + 0.40 * disagreement, 0.0, 1.0
            )
            event_rows.append(
                {
                    "product": product.product,
                    "timestamp": event.timestamp,
                    "split": event.split,
                    "donor_key": event.donor_key,
                    "predecision_supply": event.predecision_supply,
                    "logged_batch_quantity": float(event.observed_allocation.sum()),
                    "supply_recorded": bool(event.supply_confidence),
                    "supply_covers_logged_batch": bool(
                        event.predecision_supply + 1e-9 >= event.observed_allocation.sum()
                    ),
                    "route_recorded": bool(event.route_confidence),
                }
            )

    event_audit = pd.DataFrame(event_rows)
    audit_extra = pd.DataFrame(
        [
            {
                "item": "V2 event batches with a timestamped donor offer",
                "value": int(event_audit["supply_recorded"].sum()),
            },
            {
                "item": "V2 share of event batches with a timestamped donor offer",
                "value": float(event_audit["supply_recorded"].mean()),
            },
            {
                "item": "V2 timestamped offers covering the logged batch",
                "value": float(
                    event_audit.loc[event_audit["supply_recorded"], "supply_covers_logged_batch"].mean()
                ),
            },
            {
                "item": "V2 event batches with donor-state route information",
                "value": float(event_audit["route_recorded"].mean()),
            },
            {
                "item": "V2 evaluation budget",
                "value": "logged batch quantity; evidence excludes that quantity",
            },
        ]
    )
    return series, pd.concat([audit, audit_extra], ignore_index=True)


def _weighted_evidence(
    scores: np.ndarray,
    confidence: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    c = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    w = w / max(float(w.sum()), 1e-12)
    affirmative = np.clip(2.0 * (p - 0.5), 0.0, 1.0)
    negative = np.clip(2.0 * (0.5 - p), 0.0, 1.0)
    ambiguity = np.clip((1.0 - c) + 0.15 * (1.0 - np.abs(2.0 * p - 1.0)), 0.0, 1.0)
    evidence = np.column_stack(
        [
            np.sum(w * c * affirmative, axis=1),
            np.sum(w * c * negative, axis=1),
            np.sum(w * ambiguity * p, axis=1),
            np.sum(w * ambiguity * (1.0 - p), axis=1),
        ]
    )
    return _project_rows(evidence, cap=2.0)


def _event_evidence_v2(
    event: EventRecord,
    graph: ConceptGraph,
    parameters: V2Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.zeros((len(graph.metadata), 4), dtype=float)
    hospital_evidence = _weighted_evidence(
        event.source_scores,
        event.source_confidence,
        parameters.source_weights(),
    )
    evidence[graph.hospital_indices[:, 0]] = hospital_evidence
    for region in range(4):
        scores = np.array(
            [
                1.0 - event.dc_stock_risk[region, 0],
                1.0 - event.dc_load_risk[region, 0],
                1.0 - event.source_scores[event.active, 4].mean() if event.active.any() else 0.5,
                1.0 - event.source_scores[event.active, 3].mean() if event.active.any() else 0.5,
            ],
            dtype=float,
        )[None, :]
        confidence = np.array([[0.95, 0.90, 0.80, 0.80]], dtype=float)
        evidence[graph.dc_indices[region, 0]] = _evidence_from_sources(scores, confidence)[0]
    supplier_scores = np.array(
        [[1.0 - event.supplier_risk[0, 0], 1.0 - event.dc_stock_risk[:, 0].mean()]],
        dtype=float,
    )
    evidence[graph.supplier_indices[0, 0]] = _evidence_from_sources(
        supplier_scores, np.array([[0.95, 0.90]], dtype=float)
    )[0]
    return _project_rows(evidence, cap=2.0), hospital_evidence


class EventLocalACMPolicy:
    """Stateful ACM whose operator is restricted to the event-active graph."""

    def __init__(
        self,
        graph: ConceptGraph,
        parameters: V2Parameters,
        channels: int = 4,
        coupled: bool = True,
        tolerance: float = 1e-4,
        max_iter: int = 60,
    ) -> None:
        self.graph = graph
        self.parameters = parameters
        self.channels = int(channels)
        self.coupled = bool(coupled)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.state = np.full(
            (len(graph.metadata), self.channels), min(0.5, 2.0 / self.channels), dtype=float
        )
        self.raw_transitions = [
            (matrix / max(float(graph.operator_scale), 1e-12)).tocsr()
            for matrix in graph.transition_matrices
        ]

    def _reduced_evidence(self, full: np.ndarray) -> np.ndarray:
        if self.channels == 4:
            return full
        if self.channels == 3:
            return np.column_stack([full[:, 0], full[:, 1], 0.5 * (full[:, 2] + full[:, 3])])
        if self.channels == 2:
            return full[:, :2]
        raise ValueError("Only two, three, or four channels are supported.")

    def _coupling(self) -> np.ndarray:
        if not self.coupled:
            return np.eye(self.channels)
        if self.channels == 4:
            return self.graph.coupled_matrix
        if self.channels == 3:
            return np.array(
                [[1.0, -0.08, 0.16], [-0.08, 1.0, 0.16], [0.08, 0.08, 0.90]],
                dtype=float,
            )
        return np.array([[1.0, -0.08], [-0.08, 1.0]], dtype=float)

    def score(self, event: EventRecord) -> tuple[np.ndarray, dict[str, Any]]:
        full_evidence, hospital_evidence = _event_evidence_v2(
            event, self.graph, self.parameters
        )
        active_hospital = self.graph.hospital_indices[event.active, 0]
        node_indices = np.concatenate(
            [active_hospital, self.graph.dc_indices[:, 0], self.graph.supplier_indices[:, 0]]
        ).astype(int)
        evidence = self._reduced_evidence(full_evidence[node_indices])
        coupling = self._coupling()
        raw_matrices = [
            matrix[node_indices][:, node_indices].tocsr()
            for matrix in self.raw_transitions[: self.channels]
        ]
        raw_block = sparse.bmat(
            [[coupling[i, j] * raw_matrices[j] for j in range(self.channels)] for i in range(self.channels)],
            format="csr",
        )
        # Use the coupled reference to set the common channel-matrix scale;
        # otherwise the independent ablation would be silently renormalized to
        # the same norm and would no longer match the manuscript definition.
        reference_coupling = (
            self.graph.coupled_matrix if self.channels == 4 else coupling
        )
        reference_block = sparse.bmat(
            [
                [reference_coupling[i, j] * raw_matrices[j] for j in range(self.channels)]
                for i in range(self.channels)
            ],
            format="csr",
        )
        reference_norm = _power_norm(reference_block, iterations=35)
        actual_norm = _power_norm(raw_block, iterations=35)
        scale = self.parameters.operator_target_norm / max(1.01 * reference_norm, 1e-12)
        matrices = [(matrix * scale).tocsr() for matrix in raw_matrices]

        state = self.state[node_indices].copy()
        gaps: list[float] = []
        for _ in range(self.max_iter):
            messages = np.column_stack(
                [matrices[channel] @ state[:, channel] for channel in range(self.channels)]
            )
            dynamic = 0.5 * (1.0 + np.tanh(messages @ coupling.T))
            updated = (
                self.parameters.physical_mix * evidence
                + (1.0 - self.parameters.physical_mix) * dynamic
            )
            updated = _project_rows(updated, cap=2.0 if self.channels >= 3 else 1.6)
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = updated
            if gap <= self.tolerance:
                break
        self.state[node_indices] = state

        local_hospital_state = state[: len(active_hospital)]
        local_hospital_evidence = hospital_evidence[event.active]
        if self.channels == 4:
            gamma = self.parameters.partial_coefficient
            state_score = (
                local_hospital_state[:, 0]
                - local_hospital_state[:, 1]
                + gamma * (local_hospital_state[:, 2] - local_hospital_state[:, 3])
            )
            evidence_score = (
                local_hospital_evidence[:, 0]
                - local_hospital_evidence[:, 1]
                + gamma * (local_hospital_evidence[:, 2] - local_hospital_evidence[:, 3])
            )
        elif self.channels == 3:
            state_score = (
                local_hospital_state[:, 0]
                - local_hospital_state[:, 1]
                - 0.15 * local_hospital_state[:, 2]
            )
            evidence_score = (
                local_hospital_evidence[:, 0]
                - local_hospital_evidence[:, 1]
                - 0.075 * (local_hospital_evidence[:, 2] + local_hospital_evidence[:, 3])
            )
        else:
            state_score = local_hospital_state[:, 0] - local_hospital_state[:, 1]
            evidence_score = local_hospital_evidence[:, 0] - local_hospital_evidence[:, 1]
        local_score = 0.55 * state_score + 0.45 * evidence_score
        score = np.zeros(len(event.active), dtype=float)
        score[event.active] = local_score
        coupled_norm = float(scale * actual_norm)
        return score, {
            "iterations": len(gaps),
            "convergence_gap": gaps[-1] if gaps else 0.0,
            "active_nodes": len(node_indices),
            "operator_scale": float(scale),
            "operator_norm": coupled_norm,
            "contraction_bound": float(
                (1.0 - self.parameters.physical_mix) * 0.5 * coupled_norm
            ),
        }


class EventLocalFCMPolicy:
    def __init__(self, graph: ConceptGraph, parameters: V2Parameters) -> None:
        self.graph = graph
        self.parameters = parameters
        self.state = np.full(len(graph.metadata), 0.5, dtype=float)
        signed = graph.adjacency.multiply(graph.sign).multiply(graph.edge_strength).tocsr()
        self.raw = parameters.diagonal_memory * sparse.eye(len(graph.metadata), format="csr") + signed

    def score(self, event: EventRecord) -> tuple[np.ndarray, dict[str, Any]]:
        evidence, _ = _event_evidence_v2(event, self.graph, self.parameters)
        scalar_evidence = np.clip(0.5 + 0.5 * (evidence[:, 0] - evidence[:, 1]), 0.0, 1.0)
        active_hospital = self.graph.hospital_indices[event.active, 0]
        node_indices = np.concatenate(
            [active_hospital, self.graph.dc_indices[:, 0], self.graph.supplier_indices[:, 0]]
        ).astype(int)
        raw = self.raw[node_indices][:, node_indices].tocsr()
        raw_norm = _power_norm(raw, iterations=35)
        scale = self.parameters.operator_target_norm / max(1.01 * raw_norm, 1e-12)
        transition = raw * scale
        state = self.state[node_indices].copy()
        gaps: list[float] = []
        for _ in range(60):
            dynamic = 0.5 * (1.0 + np.tanh(transition @ state))
            updated = (
                self.parameters.physical_mix * scalar_evidence[node_indices]
                + (1.0 - self.parameters.physical_mix) * dynamic
            )
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = np.clip(updated, 0.0, 1.0)
            if gap <= 1e-4:
                break
        self.state[node_indices] = state
        score = np.zeros(len(event.active), dtype=float)
        score[event.active] = state[: len(active_hospital)]
        norm = float(scale * raw_norm)
        return score, {
            "iterations": len(gaps),
            "convergence_gap": gaps[-1] if gaps else 0.0,
            "active_nodes": len(node_indices),
            "operator_scale": float(scale),
            "operator_norm": norm,
            "contraction_bound": float(
                (1.0 - self.parameters.physical_mix) * 0.5 * norm
            ),
        }


def _allocate_v2(
    event: EventRecord,
    score: np.ndarray,
    parameters: V2Parameters,
    priority_mode: str = "softmax",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_idx = np.flatnonzero(event.active & (event.request_cap > 0))
    allocation = np.zeros_like(event.request_cap)
    priority = np.zeros_like(event.request_cap)
    objective = np.full_like(event.request_cap, -np.inf)
    if len(active_idx) == 0:
        return allocation, priority, objective
    if priority_mode == "equal":
        priority[active_idx] = 1.0 / len(active_idx)
    elif priority_mode == "demand":
        demand = event.request_cap[active_idx]
        priority[active_idx] = demand / max(float(demand.sum()), 1e-12)
    else:
        priority[active_idx] = stable_softmax(
            np.asarray(score, dtype=float)[active_idx],
            axis=0,
            temperature=parameters.temperature,
        )
    # Softmax mass is O(1/H), so applying fixed LP coefficients directly would
    # make the priority term vanish when the retrospective candidate pool has
    # thousands of facilities rather than the 12 hospitals in the case model.
    # Max-normalization preserves ordering, maps the term to [0, 1], and is
    # fixed independently of allocation outcomes.
    priority_index = priority[active_idx] / max(float(priority[active_idx].max()), 1e-12)
    objective[active_idx] = (
        parameters.lp_priority_weight * priority_index
        + parameters.lp_demand_weight * event.normalized_request[active_idx]
        + parameters.lp_fairness_weight * (1.0 - event.prior_fill[active_idx])
        - parameters.lp_lead_weight * event.route_risk[active_idx]
    )
    order = active_idx[np.lexsort((active_idx, -objective[active_idx]))]
    # Conditional matching replay: the dispatch total is fixed to the logged
    # batch, while no logged quantity is used by the evidence encoder.
    remaining = float(event.observed_allocation.sum())
    for position in order:
        if remaining <= 1e-12:
            break
        quantity = min(float(event.request_cap[position]), remaining)
        allocation[position] = quantity
        remaining -= quantity
    return allocation, priority, objective


def run_v2_model(
    series: list[ProductSeries],
    parameters: V2Parameters,
    model: str,
    estimator: Any | None = None,
    allowed_splits: set[str] | None = None,
    build_diagnostics: bool = True,
    collect_allocations: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    stateful = model in {
        "ACM-4 (coupled)",
        "ACM-4 (independent)",
        "ACM-3",
        "ACM-2",
        "FCM",
    }
    for product in series:
        monthly: dict[tuple[pd.Timestamp, str], dict[str, Any]] = {}
        graph: ConceptGraph | None = None
        policy: Any | None = None
        if stateful:
            graph = build_sparse_product_graph(product, parameters)
            if model == "ACM-4 (coupled)":
                policy = EventLocalACMPolicy(graph, parameters, channels=4, coupled=True)
            elif model == "ACM-4 (independent)":
                policy = EventLocalACMPolicy(graph, parameters, channels=4, coupled=False)
            elif model == "ACM-3":
                policy = EventLocalACMPolicy(graph, parameters, channels=3, coupled=True)
            elif model == "ACM-2":
                policy = EventLocalACMPolicy(graph, parameters, channels=2, coupled=True)
            else:
                policy = EventLocalFCMPolicy(graph, parameters)

        for event in product.records:
            diagnostics: dict[str, Any] = {
                "iterations": 1,
                "convergence_gap": 0.0,
                "active_nodes": int(event.active.sum()),
                "operator_scale": np.nan,
                "operator_norm": np.nan,
                "contraction_bound": np.nan,
            }
            if stateful:
                score, diagnostics = policy.score(event)
                priority_mode = "softmax"
            elif model == "Calibrated scalar":
                if estimator is None:
                    raise ValueError("Calibrated scalar requires a fitted estimator.")
                score = estimator.decision_function(_scalar_features(event))
                priority_mode = "softmax"
            elif model == "Demand proportional":
                score = np.log1p(event.request_cap)
                priority_mode = "demand"
            elif model == "Robust priority":
                score = (
                    event.source_scores[:, 0]
                    + 1.2 * event.source_scores[:, 1]
                    + 0.9 * event.source_scores[:, 2]
                    + 0.35 * event.source_scores[:, 3]
                    + 0.45 * event.source_scores[:, 4]
                )
                priority_mode = "softmax"
            elif model == "Equal allocation":
                score = np.zeros_like(event.request_cap)
                priority_mode = "equal"
            else:
                raise ValueError(f"Unsupported model: {model}")

            if build_diagnostics and stateful:
                diagnostic_rows.append(
                    {
                        "product": product.product,
                        "timestamp": event.timestamp,
                        "split": event.split,
                        "model": model,
                        **diagnostics,
                    }
                )
            if not event.evaluable or (
                allowed_splits is not None and event.split not in allowed_splits
            ):
                continue
            predicted, priority, objective = _allocate_v2(
                event, score, parameters, priority_mode=priority_mode
            )
            month = event.timestamp.tz_convert(None).to_period("M").start_time
            key = (month, event.split)
            if key not in monthly:
                monthly[key] = {
                    "observed": np.zeros_like(event.observed_allocation),
                    "predicted": np.zeros_like(event.observed_allocation),
                    "priority_prediction": np.zeros_like(event.observed_allocation),
                    "request": np.zeros_like(event.request_cap),
                    "ambiguity_numerator": np.zeros_like(event.request_cap),
                    "n_events": 0,
                }
            monthly[key]["observed"] += event.observed_allocation
            monthly[key]["predicted"] += predicted
            monthly[key]["priority_prediction"] += event.observed_allocation.sum() * priority
            monthly[key]["request"] += event.request_cap
            monthly[key]["ambiguity_numerator"] += event.ambiguity * event.request_cap
            monthly[key]["n_events"] += 1
            if collect_allocations and event.split == "test":
                for position in np.flatnonzero(event.active & (event.request_cap > 0)):
                    allocation_rows.append(
                        {
                            "product": product.product,
                            "timestamp": event.timestamp,
                            "split": event.split,
                            "model": model,
                            "facilityKey": str(product.facility_ids[position]),
                            "state": str(product.states[position]),
                            "county": str(product.counties[position]),
                            "request_cap": float(event.request_cap[position]),
                            "observed_allocation": float(event.observed_allocation[position]),
                            "predicted_allocation": float(predicted[position]),
                            "priority": float(priority[position]),
                            "objective_coefficient": float(objective[position]),
                            "ambiguity": float(event.ambiguity[position]),
                            "predecision_supply": float(event.predecision_supply),
                        }
                    )
        for (month, split), accumulated in monthly.items():
            values = _aggregated_metrics(
                accumulated["observed"],
                accumulated["predicted"],
                accumulated["priority_prediction"],
                accumulated["request"],
                accumulated["ambiguity_numerator"],
            )
            metric_rows.append(
                {
                    "product": product.product,
                    "timestamp": month,
                    "month": month,
                    "split": split,
                    "model": model,
                    "n_events": int(accumulated["n_events"]),
                    **values,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(allocation_rows), pd.DataFrame(diagnostic_rows)


def fit_scalar_v2(
    series: list[ProductSeries],
    parameters: V2Parameters,
    c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 20260718,
) -> tuple[Any, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for product in series:
        for event in product.records:
            if event.split != "calibration" or not event.evaluable:
                continue
            idx = np.flatnonzero(event.active & (event.request_cap > 0))
            x_rows.append(_scalar_features(event)[idx])
            y_rows.append((event.observed_allocation[idx] > 0).astype(int))
    x_train = np.vstack(x_rows)
    y_train = np.concatenate(y_rows)
    rows: list[dict[str, Any]] = []
    fitted: dict[float, Any] = {}
    for value in c_grid:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(value),
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
        estimator.fit(x_train, y_train)
        fitted[float(value)] = estimator
        validation = run_v2_model(
            series,
            parameters,
            "Calibrated scalar",
            estimator=estimator,
            allowed_splits={"validation"},
            build_diagnostics=False,
        )[0]
        rows.append(
            {
                "C": float(value),
                "validation_overlap": float(validation["allocation_overlap"].mean()),
                "validation_priority_overlap": float(validation["priority_overlap"].mean()),
                "validation_ndcg_all": float(validation["ndcg_all"].mean()),
            }
        )
    calibration = pd.DataFrame(rows).sort_values(
        ["validation_overlap", "validation_priority_overlap", "validation_ndcg_all"],
        ascending=False,
    )
    selected = float(calibration.iloc[0]["C"])
    calibration["selected"] = calibration["C"].eq(selected)
    return fitted[selected], calibration.sort_values("C").reset_index(drop=True)


def _selection_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return -np.inf
    return float(
        0.55 * frame["allocation_overlap"].mean()
        + 0.20 * frame["priority_overlap"].mean()
        + 0.15 * frame["recipient_average_precision"].mean()
        + 0.10 * frame["recipient_recall_at_observed_k"].mean()
    )


def _optimizer_preset_comparison(
    preset_metrics: pd.DataFrame,
    optimized_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the frozen selected configuration with the prior preset on test."""

    preset = preset_metrics.loc[preset_metrics["split"].eq("test")].copy()
    optimized = optimized_metrics.loc[
        optimized_metrics["split"].eq("test")
        & optimized_metrics["model"].eq("ACM-4 (coupled)")
    ].copy()
    preset["month"] = pd.to_datetime(preset["month"], utc=True)
    optimized["month"] = pd.to_datetime(optimized["month"], utc=True)
    metric_names = (
        "allocation_overlap",
        "priority_overlap",
        "recipient_average_precision",
        "recipient_recall_at_observed_k",
        "ndcg_all",
    )
    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        left = preset[["product", "month", metric]].rename(
            columns={metric: "preset_value"}
        )
        right = optimized[["product", "month", metric]].rename(
            columns={metric: "optimized_value"}
        )
        paired = left.merge(right, on=["product", "month"]).dropna()
        benefit = (
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
                "preferred_direction": "higher",
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


def _trial_parameters(trial: Any) -> V2Parameters:
    return V2Parameters(
        mu_true=trial.suggest_float("mu_true", 0.76, 0.88),
        mu_partial=trial.suggest_float("mu_partial", 0.56, 0.68),
        sigma_exact=trial.suggest_float("sigma_exact", 0.18, 0.28),
        sigma_partial=trial.suggest_float("sigma_partial", 0.20, 0.30),
        beta_partial=trial.suggest_float("beta_partial", 0.70, 0.95),
        diagonal_memory=trial.suggest_float("diagonal_memory", 0.34, 0.50),
        partial_to_exact=trial.suggest_float("partial_to_exact", 0.12, 0.36),
        exact_to_partial=trial.suggest_float("exact_to_partial", 0.06, 0.18),
        partial_self=trial.suggest_float("partial_self", 0.80, 0.94),
        physical_mix=trial.suggest_float("physical_mix", 0.40, 0.55),
        partial_coefficient=trial.suggest_categorical(
            "partial_coefficient", [0.25, 0.50, 0.75, 1.00]
        ),
        temperature=trial.suggest_float("temperature", 0.20, 0.40),
        weight_demand=trial.suggest_float("weight_demand", 0.50, 2.00, log=True),
        weight_backlog=trial.suggest_float("weight_backlog", 0.50, 2.00, log=True),
        weight_inventory=trial.suggest_float("weight_inventory", 0.50, 2.00, log=True),
        weight_route=trial.suggest_float("weight_route", 0.50, 2.00, log=True),
        weight_capacity=trial.suggest_float("weight_capacity", 0.50, 2.00, log=True),
    )


def _params_from_mapping(values: dict[str, Any]) -> V2Parameters:
    allowed = set(V2Parameters.__dataclass_fields__)
    return V2Parameters(**{key: value for key, value in values.items() if key in allowed})


def optimize_v2_parameters(
    series: list[ProductSeries],
    n_trials: int = 36,
    seed: int = 20260718,
    shortlist: int = 8,
    progress: bool = True,
) -> tuple[V2Parameters, pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised by user environments
        raise RuntimeError("Optuna is required for the V2 calibration pipeline.") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(12, n_trials))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    default = V2Parameters()
    enqueue = {
        "mu_true": default.mu_true,
        "mu_partial": default.mu_partial,
        "sigma_exact": default.sigma_exact,
        "sigma_partial": default.sigma_partial,
        "beta_partial": default.beta_partial,
        "diagonal_memory": default.diagonal_memory,
        "partial_to_exact": default.partial_to_exact,
        "exact_to_partial": default.exact_to_partial,
        "partial_self": default.partial_self,
        "physical_mix": default.physical_mix,
        "partial_coefficient": default.partial_coefficient,
        "temperature": default.temperature,
        "weight_demand": default.weight_demand,
        "weight_backlog": default.weight_backlog,
        "weight_inventory": default.weight_inventory,
        "weight_route": default.weight_route,
        "weight_capacity": default.weight_capacity,
    }
    study.enqueue_trial(enqueue)

    def objective(trial: Any) -> float:
        parameters = _trial_parameters(trial)
        metrics = run_v2_model(
            series,
            parameters,
            "ACM-4 (coupled)",
            allowed_splits={"calibration"},
            build_diagnostics=False,
        )[0]
        score = _selection_score(metrics)
        # Weak shrinkage toward the transparent manuscript configuration.
        normalized = np.array(
            [
                (parameters.mu_true - 0.82) / 0.06,
                (parameters.mu_partial - 0.62) / 0.06,
                (parameters.sigma_exact - 0.22) / 0.05,
                (parameters.sigma_partial - 0.25) / 0.05,
                (parameters.diagonal_memory - 0.42) / 0.08,
                (parameters.partial_to_exact - 0.24) / 0.12,
                (parameters.exact_to_partial - 0.12) / 0.06,
                (parameters.partial_self - 0.88) / 0.07,
            ]
        )
        penalty = 0.0025 * float(np.mean(normalized**2))
        trial.set_user_attr("calibration_raw_score", score)
        trial.set_user_attr("regularization_penalty", penalty)
        if progress:
            print(
                f"  Optuna trial {trial.number + 1}/{n_trials}: "
                f"calibration={score:.5f}, penalized={score - penalty:.5f}",
                flush=True,
            )
        return score - penalty

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    ordered = sorted(study.trials, key=lambda item: float(item.value), reverse=True)
    shortlisted = ordered[: min(int(shortlist), len(ordered))]
    validation_rows: dict[int, dict[str, float]] = {}
    for item in shortlisted:
        parameters = _params_from_mapping(item.params)
        metrics = run_v2_model(
            series,
            parameters,
            "ACM-4 (coupled)",
            allowed_splits={"validation"},
            build_diagnostics=False,
        )[0]
        validation_rows[item.number] = {
            "validation_objective": _selection_score(metrics),
            "validation_overlap": float(metrics["allocation_overlap"].mean()),
            "validation_priority_overlap": float(metrics["priority_overlap"].mean()),
            "validation_ndcg_all": float(metrics["ndcg_all"].mean()),
        }
    selected_trial = max(
        shortlisted,
        key=lambda item: (
            validation_rows[item.number]["validation_objective"],
            float(item.value),
        ),
    )
    rows: list[dict[str, Any]] = []
    for item in study.trials:
        row: dict[str, Any] = {
            "trial": int(item.number),
            **item.params,
            "calibration_objective": float(item.value),
            "calibration_raw_score": float(item.user_attrs.get("calibration_raw_score", np.nan)),
            "regularization_penalty": float(item.user_attrs.get("regularization_penalty", np.nan)),
            "calibration_shortlist": bool(item in shortlisted),
            "selected": bool(item.number == selected_trial.number),
            "validation_objective": np.nan,
            "validation_overlap": np.nan,
            "validation_priority_overlap": np.nan,
            "validation_ndcg_all": np.nan,
        }
        row.update(validation_rows.get(item.number, {}))
        rows.append(row)
    return _params_from_mapping(selected_trial.params), pd.DataFrame(rows).sort_values("trial")


def run_v2_study(
    data_dir: Path,
    n_trials: int = 36,
    seed: int = 20260718,
    n_boot: int = 5000,
    progress: bool = True,
) -> RetrospectiveV2Bundle:
    series, audit = prepare_v2_series(data_dir, lookback_days=90)
    selected, trials = optimize_v2_parameters(
        series,
        n_trials=n_trials,
        seed=seed,
        progress=progress,
    )
    scalar, scalar_calibration = fit_scalar_v2(series, selected, seed=seed)
    metrics_frames: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    allocations = pd.DataFrame()
    for model in MODEL_ORDER:
        estimator = scalar if model == "Calibrated scalar" else None
        metrics, model_allocations, diagnostics = run_v2_model(
            series,
            selected,
            model,
            estimator=estimator,
            build_diagnostics=True,
            collect_allocations=model == "ACM-4 (coupled)",
        )
        metrics_frames.append(metrics)
        if not diagnostics.empty:
            diagnostics_frames.append(diagnostics)
        if not model_allocations.empty:
            allocations = model_allocations
    metrics = pd.concat(metrics_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    preset_metrics, _, _ = run_v2_model(
        series,
        V2Parameters(),
        "ACM-4 (coupled)",
        build_diagnostics=False,
    )
    optimizer_comparison = _optimizer_preset_comparison(preset_metrics, metrics)
    summary = summarize_test_metrics(metrics, seed=seed, n_boot=n_boot)
    tests = paired_model_tests(metrics, split="test", seed=seed, n_boot=n_boot)
    ambiguity_summary, ambiguity_tests = ambiguity_analysis(
        metrics, seed=seed, n_boot=n_boot
    )
    graph_diagnostics = (
        diagnostics.groupby(["product", "model"], as_index=False)
        .agg(
            mean_iterations=("iterations", "mean"),
            max_convergence_gap=("convergence_gap", "max"),
            max_operator_norm=("operator_norm", "max"),
            max_contraction_bound=("contraction_bound", "max"),
            median_active_nodes=("active_nodes", "median"),
        )
    )
    return RetrospectiveV2Bundle(
        audit=audit,
        selected_parameters=selected,
        calibration_trials=trials,
        scalar_calibration=scalar_calibration,
        scalar_coefficients=_scalar_coefficient_table(scalar),
        metrics=metrics,
        allocations=allocations,
        diagnostics=diagnostics,
        summary=summary,
        statistical_tests=tests,
        ambiguity_summary=ambiguity_summary,
        ambiguity_tests=ambiguity_tests,
        graph_diagnostics=graph_diagnostics,
        split_description=describe_splits(metrics),
        lookback_sensitivity=pd.DataFrame(
            [{"lookback_days": 90, "model": "ACM-4 (coupled)", "analysis": "V2 primary"}]
        ),
        optimizer_comparison=optimizer_comparison,
    )


def _model_style(model: str) -> dict[str, Any]:
    style = MODEL_STYLES.get(model, {})
    return {
        "color": style.get("color", "0.4"),
        "marker": style.get("marker", "o"),
        "linestyle": style.get("linestyle", "-"),
    }


def _plot_primary(bundle: RetrospectiveBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    metrics = [
        ("allocation_overlap", "Allocation overlap"),
        ("recipient_average_precision", "Recipient AP"),
        ("ndcg_all", "Full-list NDCG"),
    ]
    models = list(MODEL_ORDER)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 9.0), sharex=True)
    positions = np.arange(len(models))
    for panel, (ax, (metric, label)) in enumerate(zip(axes, metrics)):
        values = bundle.summary.loc[bundle.summary["metric"].eq(metric)].set_index("model")
        for position, model in enumerate(models):
            row = values.loc[model]
            style = _model_style(model)
            ax.errorbar(
                position,
                row["mean"],
                yerr=np.array([[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]),
                fmt=style["marker"],
                color=style["color"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                capsize=3,
                linewidth=1.0,
            )
        ax.set_ylabel(label)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel))
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(models, rotation=35, ha="right")
    save_figure(fig, output, "fig_real10_getusppe_v2_validation", formats=("png",), dpi=dpi)


def _plot_calibration(bundle: RetrospectiveBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    frame = bundle.calibration_trials.copy()
    selected = frame.loc[frame["selected"]].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.8))
    axes[0].plot(
        frame["trial"], frame["calibration_raw_score"],
        color="0.35", linestyle=":", marker="o", markersize=3,
    )
    axes[0].axvline(selected["trial"], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Calibration score")
    clean_axis(axes[0])
    panel_label(axes[0], "a")
    shortlisted = frame.loc[frame["calibration_shortlist"]].dropna(
        subset=["validation_objective"]
    )
    axes[1].scatter(
        shortlisted["calibration_raw_score"],
        shortlisted["validation_objective"],
        facecolors="white",
        edgecolors="0.35",
        marker="o",
    )
    axes[1].scatter(
        [selected["calibration_raw_score"]],
        [selected["validation_objective"]],
        color="black",
        marker="D",
        label="Selected",
    )
    axes[1].set_xlabel("Calibration score")
    axes[1].set_ylabel("Internal-validation score")
    axes[1].legend(loc="best")
    clean_axis(axes[1])
    panel_label(axes[1], "b")
    save_figure(fig, output, "fig_real11_getusppe_v2_calibration", formats=("png",), dpi=dpi)


def _result_paragraph(bundle: RetrospectiveBundle) -> str:
    summary = bundle.summary.set_index(["model", "metric"])
    coupled = summary.loc[("ACM-4 (coupled)", "allocation_overlap")]
    ranking = summary.loc[("ACM-4 (coupled)", "recipient_average_precision")]
    baseline = (
        bundle.summary.loc[
            bundle.summary["metric"].eq("allocation_overlap")
            & ~bundle.summary["model"].eq("ACM-4 (coupled)")
        ]
        .sort_values("mean", ascending=False)
        .iloc[0]
    )
    test = bundle.statistical_tests.loc[
        bundle.statistical_tests["metric"].eq("allocation_overlap")
        & bundle.statistical_tests["comparator"].eq(baseline["model"])
    ].iloc[0]
    conclusion = (
        "The Holm-adjusted paired comparison was significant."
        if bool(test["significant_0_05"])
        else "The Holm-adjusted paired comparison was not significant."
    )
    optimizer = bundle.optimizer_comparison.set_index("metric")
    preset_ap = optimizer.loc["recipient_average_precision"]
    preset_ndcg = optimizer.loc["ndcg_all"]
    preset_overlap = optimizer.loc["allocation_overlap"]
    return f"""% Generated by run_retrospective_allocation_v2.py; verify cross-references before typesetting.
\\subsection{{Timestamp-controlled retrospective allocation replay}}

Allocation validity was evaluated by replaying successfully delivered GetUsPPE batches using only acute-care requests timestamped before each decision. Donor availability was reconstructed from offers timestamped by the decision minus commitments timestamped earlier; the logged batch quantity was used only as a conditional dispatch budget and was excluded from evidence encoding. Lead risk used donor-to-recipient route strata rather than request age, and the cognitive operator was restricted and renormalized to the event-active graph while carrying each concept's previous converged state forward. Because the candidate count varied from tens to thousands, the softmax priority was divided by its active-set maximum before entering the LP; this outcome-independent normalization preserves its ordering and prevents the priority coefficient from vanishing as the candidate pool grows. The evidence encoder was calibrated with seeded TPE optimization on the calibration period, shortlisted configurations were selected on internal validation, the LP coefficient ratios remained fixed at the manuscript values, and test outcomes were not supplied to the optimizer. Relative to the frozen pre-optimization preset, tuning increased test recipient average precision from {preset_ap['preset_mean']:.3f} to {preset_ap['optimized_mean']:.3f} and full-list NDCG from {preset_ndcg['preset_mean']:.3f} to {preset_ndcg['optimized_mean']:.3f} (Holm-adjusted $p={preset_ndcg['holm_p']:.4f}$ for NDCG), while allocation overlap remained {preset_overlap['optimized_mean']:.3f}.

On the temporal test panels, coupled ACM obtained allocation overlap {coupled['mean']:.3f} [{coupled['ci_low']:.3f}, {coupled['ci_high']:.3f}] and recipient average precision {ranking['mean']:.3f} [{ranking['ci_low']:.3f}, {ranking['ci_high']:.3f}]. The strongest comparator on allocation overlap was {baseline['model']} at {baseline['mean']:.3f} [{baseline['ci_low']:.3f}, {baseline['ci_high']:.3f}]. {conclusion} These estimates measure agreement with historical recipient assignments conditional on batch size, not causal clinical benefit; donor preferences and shipment constraints absent from the public archive remain potential sources of disagreement.
"""


def create_v2_outputs(
    bundle: RetrospectiveBundle,
    output_root: Path,
    data_dir: Path,
    dpi: int = 600,
) -> None:
    results = output_root / "results"
    figures = output_root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frames = {
        "data_audit.csv": bundle.audit,
        "split_description.csv": bundle.split_description,
        "optuna_trials.csv": bundle.calibration_trials,
        "scalar_calibration.csv": bundle.scalar_calibration,
        "scalar_coefficients.csv": bundle.scalar_coefficients,
        "monthly_model_metrics.csv": bundle.metrics,
        "test_summary.csv": bundle.summary,
        "paired_statistical_tests.csv": bundle.statistical_tests,
        "ambiguity_summary.csv": bundle.ambiguity_summary,
        "ambiguity_statistical_tests.csv": bundle.ambiguity_tests,
        "graph_diagnostics.csv": bundle.graph_diagnostics,
        "coupled_acm_test_allocations.csv": bundle.allocations,
        "convergence_diagnostics.csv": bundle.diagnostics,
        "optimizer_preset_comparison.csv": bundle.optimizer_comparison,
    }
    for name, frame in frames.items():
        frame.to_csv(results / name, index=False)
    (results / "selected_parameters.json").write_text(
        json.dumps(bundle.selected_parameters.as_dict(), indent=2), encoding="utf-8"
    )
    (output_root / "manuscript_results_draft.tex").write_text(
        _result_paragraph(bundle), encoding="utf-8"
    )
    hashes = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("all_requests.csv", "all_offers.csv", "all_matches.csv")
    }
    (results / "data_provenance.json").write_text(
        json.dumps(
            {
                "dataset_repository": "https://github.com/GetUsPPE/ppe_needs_retrospective",
                "dataset_article": "https://doi.org/10.1002/puh2.65",
                "optimization": "Optuna TPE; calibration objective, validation selection",
                "lp_parameters": "fixed manuscript coefficients",
                "source_file_sha256": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_primary(bundle, figures, dpi)
    _plot_calibration(bundle, figures, dpi)
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
