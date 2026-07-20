from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog


RESOURCE_LABELS = ("VNT", "ICU", "PPE", "AVD")
CLINICAL_WEIGHTS = np.array([0.90, 1.00, 0.45, 0.72], dtype=float)


def scaled_tanh(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Membership-preserving transfer function with codomain [0, 1]."""
    return 0.5 * (1.0 + np.tanh(alpha * x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def stable_softmax(x: np.ndarray, axis: int = 0, temperature: float = 1.0) -> np.ndarray:
    z = x / max(float(temperature), 1e-8)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def project_capped_simplex_rows(x: np.ndarray, cap: float = 2.0) -> np.ndarray:
    """Euclidean projection onto [0,1]^K with a row-sum upper bound."""
    out = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    for i in range(out.shape[0]):
        row = out[i]
        if row.sum() <= cap:
            continue
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            candidate = np.clip(row - mid, 0.0, 1.0)
            if candidate.sum() > cap:
                lo = mid
            else:
                hi = mid
        out[i] = np.clip(row - hi, 0.0, 1.0)
    return out


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.clip(x, 0.0, None)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1.0) / n)


def jain_index(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).ravel()
    denom = x.size * np.sum(x * x)
    return float((x.sum() ** 2) / denom) if denom > 0 else 1.0


def weighted_equity(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if w.sum() <= 0:
        return 1.0
    w = w / w.sum()
    mean = np.sum(w * x)
    mad = np.sum(w * np.abs(x - mean))
    return float(np.clip(1.0 - mad / max(mean, 1e-8), 0.0, 1.0))


@dataclass
class Scenario:
    seed: int
    demand: np.ndarray
    observed_demand: np.ndarray
    reporting_confidence: np.ndarray
    supplier_capacity: np.ndarray
    lead_dh: np.ndarray
    lead_sd: np.ndarray
    initial_hospital_inventory: np.ndarray
    initial_dc_inventory: np.ndarray
    regions: np.ndarray
    hospital_size: np.ndarray
    resource_names: tuple[str, ...] = RESOURCE_LABELS

    @property
    def days(self) -> int:
        return int(self.demand.shape[0])

    @property
    def n_hospitals(self) -> int:
        return int(self.demand.shape[1])

    @property
    def n_resources(self) -> int:
        return int(self.demand.shape[2])


def generate_scenario(
    seed: int,
    days: int = 40,
    n_hospitals: int = 12,
    demand_scale: float = 1.0,
    lead_time_scale: float = 1.0,
    capacity_scale: float = 1.0,
    reporting_noise: float = 0.08,
    missing_probability: float = 0.06,
) -> Scenario:
    rng = np.random.default_rng(seed)
    r = len(RESOURCE_LABELS)
    regions = np.repeat(np.arange(2), n_hospitals // 2)
    if regions.size < n_hospitals:
        regions = np.r_[regions, np.ones(n_hospitals - regions.size, dtype=int)]
    hospital_size = np.linspace(0.75, 1.30, n_hospitals)
    rng.shuffle(hospital_size)

    base = np.array([2.8, 3.5, 72.0, 9.0], dtype=float)
    resource_phase = np.array([0.0, 1.5, -1.0, 2.0])
    t = np.arange(days, dtype=float)
    demand = np.zeros((days, n_hospitals, r), dtype=float)
    latent_rate = np.zeros_like(demand)
    for h in range(n_hospitals):
        region_peak = 14.0 + 7.0 * regions[h] + rng.normal(0, 1.5)
        severity = 0.85 + 0.50 * rng.random()
        for k in range(r):
            surge = 1.0 + severity * 1.35 * np.exp(-0.5 * ((t - region_peak - resource_phase[k]) / 5.5) ** 2)
            secondary = 0.35 * np.exp(-0.5 * ((t - 31.0 + 2.0 * regions[h]) / 4.0) ** 2)
            weekly = 1.0 + 0.06 * np.sin(2 * np.pi * (t + h) / 7.0)
            rate = base[k] * hospital_size[h] * surge * (1.0 + secondary) * weekly * demand_scale
            latent_rate[:, h, k] = np.clip(rate, 0.05, None)
            demand[:, h, k] = rng.poisson(latent_rate[:, h, k])

    reporting_confidence = np.ones_like(demand)
    observed = demand * np.clip(1.0 + rng.normal(0.0, reporting_noise, demand.shape), 0.60, 1.40)
    missing = rng.random(demand.shape) < missing_probability
    reporting_confidence[missing] = 0.35
    for day, h, k in zip(*np.where(missing)):
        if day > 0:
            observed[day, h, k] = observed[day - 1, h, k]
        else:
            observed[day, h, k] = latent_rate[day, h, k]

    lead_base = np.array([2.0, 2.3, 1.5, 1.9])
    lead_dh = np.zeros((2, n_hospitals, r), dtype=int)
    for d in range(2):
        for h in range(n_hospitals):
            region_penalty = 0.65 if d != regions[h] else 0.0
            for k in range(r):
                mu = np.log(max(lead_base[k] + region_penalty, 1.0))
                val = rng.lognormal(mu, 0.22) * lead_time_scale
                lead_dh[d, h, k] = int(np.clip(np.rint(val), 1, 7))
    lead_sd = np.clip(np.rint(rng.lognormal(np.log(2.2), 0.18, size=(2, 2, r)) * lead_time_scale), 1, 6).astype(int)

    mean_daily = latent_rate.mean(axis=0).sum(axis=0)
    supplier_capacity = np.zeros((days, 2, r), dtype=float)
    disruption = 1.0 - 0.20 * np.exp(-0.5 * ((t - 18.0) / 4.5) ** 2)
    for day in range(days):
        variability = rng.uniform(0.90, 1.10, size=(2, r))
        supplier_capacity[day] = 0.49 * mean_daily[None, :] * disruption[day] * variability * capacity_scale

    initial_hospital_inventory = latent_rate[:3].mean(axis=0) * rng.uniform(0.45, 0.85, size=(n_hospitals, r))
    initial_dc_inventory = np.tile(mean_daily[None, :] * 1.05, (2, 1)) * rng.uniform(0.90, 1.10, size=(2, r))
    return Scenario(
        seed=seed,
        demand=demand,
        observed_demand=observed,
        reporting_confidence=reporting_confidence,
        supplier_capacity=supplier_capacity,
        lead_dh=lead_dh,
        lead_sd=lead_sd,
        initial_hospital_inventory=initial_hospital_inventory,
        initial_dc_inventory=initial_dc_inventory,
        regions=regions,
        hospital_size=hospital_size,
    )


@dataclass
class ConceptGraph:
    metadata: pd.DataFrame
    adjacency: np.ndarray
    sign: np.ndarray
    edge_strength: np.ndarray
    channel_weights: np.ndarray
    transition_matrices: list[np.ndarray]
    coupled_matrix: np.ndarray
    independent_matrix: np.ndarray
    operator_norm_coupled: float
    operator_norm_independent: float
    contraction_bound_coupled: float
    contraction_bound_independent: float
    operator_scale: float
    hospital_indices: np.ndarray
    dc_indices: np.ndarray
    supplier_indices: np.ndarray


def build_concept_graph(
    n_hospitals: int,
    n_dcs: int,
    n_suppliers: int,
    regions: np.ndarray,
    alpha: float,
    physical_evidence_mix: float,
    operator_target_norm: float,
    gaussian_mu: Iterable[float],
    gaussian_sigma: Iterable[float],
    gaussian_beta: Iterable[float],
    resources: Iterable[str] = RESOURCE_LABELS,
) -> ConceptGraph:
    resources = tuple(resources)
    rows: list[dict[str, Any]] = []
    hospital_idx = np.zeros((n_hospitals, len(resources)), dtype=int)
    dc_idx = np.zeros((n_dcs, len(resources)), dtype=int)
    supplier_idx = np.zeros((n_suppliers, len(resources)), dtype=int)
    idx = 0
    for h in range(n_hospitals):
        for k, resource in enumerate(resources):
            hospital_idx[h, k] = idx
            rows.append(
                dict(
                    index=idx,
                    kind="hospital",
                    entity=f"H{h+1}",
                    resource=resource,
                    region=int(regions[h]),
                    concept_semantic="shortage_risk",
                )
            )
            idx += 1
    for d in range(n_dcs):
        for k, resource in enumerate(resources):
            dc_idx[d, k] = idx
            rows.append(
                dict(
                    index=idx,
                    kind="dc",
                    entity=f"DC{d+1}",
                    resource=resource,
                    region=d,
                    concept_semantic="supply_availability",
                )
            )
            idx += 1
    for s in range(n_suppliers):
        for k, resource in enumerate(resources):
            supplier_idx[s, k] = idx
            rows.append(
                dict(
                    index=idx,
                    kind="supplier",
                    entity=f"S{s+1}",
                    resource=resource,
                    region=s,
                    concept_semantic="supply_availability",
                )
            )
            idx += 1
    n = idx
    adjacency = np.zeros((n, n), dtype=float)
    sign = np.zeros((n, n), dtype=float)
    strength = np.zeros((n, n), dtype=float)

    def edge(source: int, target: int, polarity: float, magnitude: float) -> None:
        adjacency[target, source] = 1.0
        sign[target, source] = np.sign(polarity)
        strength[target, source] = float(np.clip(magnitude, 0.0, 1.0))

    for k in range(len(resources)):
        for s in range(n_suppliers):
            for d in range(n_dcs):
                edge(supplier_idx[s, k], dc_idx[d, k], +1, 0.90 if s == d else 0.62)
        for d in range(n_dcs):
            for h in range(n_hospitals):
                # Greater DC availability reduces hospital shortage risk.
                edge(dc_idx[d, k], hospital_idx[h, k], -1, 0.88 if d == regions[h] else 0.42)
                # Greater hospital pressure depletes DC availability.
                edge(hospital_idx[h, k], dc_idx[d, k], -1, 0.70 if d == regions[h] else 0.28)
        for h in range(n_hospitals):
            for h2 in range(n_hospitals):
                if h != h2 and regions[h] == regions[h2] and abs(h - h2) <= 2:
                    edge(hospital_idx[h, k], hospital_idx[h2, k], +1, 0.20)

    # Cross-resource relations are optional and only instantiated when the
    # corresponding resource positions exist.  This keeps the original
    # four-resource synthetic case unchanged while allowing external datasets
    # with fewer observed resource classes.
    candidate_cross_pairs = [(0, 1, 0.42), (1, 0, 0.42), (1, 3, 0.24), (3, 1, 0.24), (2, 3, 0.18)]
    cross_pairs = [pair for pair in candidate_cross_pairs if pair[0] < len(resources) and pair[1] < len(resources)]
    for h in range(n_hospitals):
        for source_k, target_k, mag in cross_pairs:
            edge(hospital_idx[h, source_k], hospital_idx[h, target_k], +1, mag)

    mu = np.asarray(list(gaussian_mu), dtype=float)
    sigma = np.asarray(list(gaussian_sigma), dtype=float)
    beta = np.asarray(list(gaussian_beta), dtype=float)
    channel_weights = np.zeros((4, n, n), dtype=float)
    edge_positions = np.argwhere(adjacency > 0)
    for i, j in edge_positions:
        z = strength[i, j]
        scores = beta * np.exp(-((z - mu) ** 2) / (2.0 * sigma**2))
        scores = scores / max(scores.sum(), 1e-12)
        channel_weights[:, i, j] = scores

    transition: list[np.ndarray] = []
    for c in range(4):
        w = adjacency * sign * strength * channel_weights[c]
        transition.append(0.42 * np.eye(n) + w)

    coupling = np.array(
        [
            [1.00, 0.00, 0.24, 0.00],
            [0.00, 1.00, 0.00, 0.24],
            [0.12, 0.00, 0.88, 0.00],
            [0.00, 0.12, 0.00, 0.88],
        ],
        dtype=float,
    )
    independent = np.eye(4)

    def block_operator(cmat: np.ndarray, mats: list[np.ndarray]) -> np.ndarray:
        blocks = [[cmat[i, j] * mats[j] for j in range(4)] for i in range(4)]
        return np.block(blocks)

    raw_block = block_operator(coupling, transition)
    raw_norm = np.linalg.norm(raw_block, 2)
    scale = operator_target_norm / max(raw_norm, 1e-12)
    transition = [m * scale for m in transition]
    coupled_block = block_operator(coupling, transition)
    independent_block = block_operator(independent, transition)
    norm_c = float(np.linalg.norm(coupled_block, 2))
    norm_i = float(np.linalg.norm(independent_block, 2))
    dynamic_mix = 1.0 - physical_evidence_mix
    bound_c = dynamic_mix * (alpha / 2.0) * norm_c
    bound_i = dynamic_mix * (alpha / 2.0) * norm_i

    return ConceptGraph(
        metadata=pd.DataFrame(rows),
        adjacency=adjacency,
        sign=sign,
        edge_strength=strength,
        channel_weights=channel_weights,
        transition_matrices=transition,
        coupled_matrix=coupling,
        independent_matrix=independent,
        operator_norm_coupled=norm_c,
        operator_norm_independent=norm_i,
        contraction_bound_coupled=bound_c,
        contraction_bound_independent=bound_i,
        operator_scale=float(scale),
        hospital_indices=hospital_idx,
        dc_indices=dc_idx,
        supplier_indices=supplier_idx,
    )


def _evidence_from_sources(source_scores: np.ndarray, source_confidence: np.ndarray) -> np.ndarray:
    p = np.clip(source_scores, 0.0, 1.0)
    c = np.clip(source_confidence, 0.0, 1.0)
    affirmative = np.clip(2.0 * (p - 0.5), 0.0, 1.0)
    negative = np.clip(2.0 * (0.5 - p), 0.0, 1.0)
    # Partial channels are reserved primarily for low-confidence or genuinely
    # borderline evidence; they are not a duplicate of the exact T/F channels.
    ambiguity = np.clip((1.0 - c) + 0.15 * (1.0 - np.abs(2.0 * p - 1.0)), 0.0, 1.0)
    t = np.average(c * affirmative, axis=-1)
    f = np.average(c * negative, axis=-1)
    pt = np.average(ambiguity * p, axis=-1)
    pf = np.average(ambiguity * (1.0 - p), axis=-1)
    return project_capped_simplex_rows(np.stack([t, f, pt, pf], axis=-1).reshape(-1, 4)).reshape(*p.shape[:-1], 4)


def encode_physical_evidence(features: dict[str, np.ndarray], graph: ConceptGraph) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    demand_pressure = np.clip(features["demand_pressure"], 0.0, 1.0)
    backlog_pressure = np.clip(features["backlog_pressure"], 0.0, 1.0)
    inventory_risk = np.clip(features["inventory_risk"], 0.0, 1.0)
    lead_risk = np.clip(features["lead_risk"], 0.0, 1.0)
    capacity_risk = np.clip(features["capacity_risk"], 0.0, 1.0)
    reporting = np.clip(features["reporting_confidence"], 0.0, 1.0)
    h, r = demand_pressure.shape
    sources = np.stack([demand_pressure, backlog_pressure, inventory_risk, lead_risk, capacity_risk], axis=-1)
    confidences = np.stack([reporting, np.ones((h, r)), np.full((h, r), 0.95), np.full((h, r), 0.85), np.full((h, r), 0.90)], axis=-1)
    hospital_evidence = _evidence_from_sources(sources, confidences)

    n = len(graph.metadata)
    evidence = np.full((n, 4), 0.25, dtype=float)
    for i in range(h):
        for k in range(r):
            evidence[graph.hospital_indices[i, k]] = hospital_evidence[i, k]

    dc_stock_risk = np.clip(features["dc_stock_risk"], 0.0, 1.0)
    dc_load_risk = np.clip(features["dc_load_risk"], 0.0, 1.0)
    for d in range(dc_stock_risk.shape[0]):
        for k in range(r):
            # DC concepts represent supply availability, not risk.  The
            # evidence is therefore expressed in favorable direction so that
            # the causal signs have one unambiguous interpretation.
            p = np.array(
                [
                    1.0 - dc_stock_risk[d, k],
                    1.0 - dc_load_risk[d, k],
                    1.0 - capacity_risk[:, k].mean(),
                    1.0 - lead_risk[:, k].mean(),
                ]
            )
            c = np.array([0.95, 0.90, 0.85, 0.85])
            evidence[graph.dc_indices[d, k]] = _evidence_from_sources(p[None, :], c[None, :])[0]

    supplier_risk = np.clip(features["supplier_risk"], 0.0, 1.0)
    for s in range(supplier_risk.shape[0]):
        for k in range(r):
            # Supplier concepts also represent availability.  Demand pressure
            # reaches the DC through the hospital-to-DC causal edge and is not
            # duplicated as supplier evidence.
            p = np.array([1.0 - supplier_risk[s, k], 1.0 - capacity_risk[:, k].mean()])
            c = np.array([0.95, 0.90])
            evidence[graph.supplier_indices[s, k]] = _evidence_from_sources(p[None, :], c[None, :])[0]

    details = {
        "source_scores": sources,
        "source_confidence": confidences,
        "hospital_evidence": hospital_evidence,
        "source_names": np.array(["Demand pressure", "Backlog pressure", "Inventory risk", "Lead-time risk", "Capacity risk"]),
    }
    return project_capped_simplex_rows(evidence), details


class PriorityPolicy:
    display_name = "policy"

    def reset(self, scenario: Scenario, graph: ConceptGraph) -> None:
        self.scenario = scenario
        self.graph = graph

    def priority(self, features: dict[str, np.ndarray], day: int) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class ACMPolicy(PriorityPolicy):
    def __init__(
        self,
        display_name: str = "ACM-4 (coupled)",
        channels: int = 4,
        coupled: bool = True,
        alpha: float = 1.0,
        physical_mix: float = 0.45,
        max_iter: int = 60,
        tolerance: float = 1e-4,
        partial_coefficient: float = 0.5,
        temperature: float = 0.30,
    ) -> None:
        self.display_name = display_name
        self.channels = channels
        self.coupled = coupled
        self.alpha = alpha
        self.physical_mix = physical_mix
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.partial_coefficient = partial_coefficient
        self.temperature = temperature

    def reset(self, scenario: Scenario, graph: ConceptGraph) -> None:
        super().reset(scenario, graph)
        self.state = np.full((len(graph.metadata), self.channels), min(0.5, 2.0 / self.channels), dtype=float)

    def _reduced_evidence(self, full: np.ndarray) -> np.ndarray:
        if self.channels == 4:
            return full
        if self.channels == 3:
            return np.c_[full[:, 0], full[:, 1], 0.5 * (full[:, 2] + full[:, 3])]
        if self.channels == 2:
            return full[:, :2]
        raise ValueError("ACM ablations support 2, 3 or 4 channels.")

    def _coupling(self) -> np.ndarray:
        if self.channels == 4:
            return self.graph.coupled_matrix if self.coupled else self.graph.independent_matrix
        if self.channels == 3:
            return np.array([[1.0, -0.08, 0.16], [-0.08, 1.0, 0.16], [0.08, 0.08, 0.90]]) if self.coupled else np.eye(3)
        return np.array([[1.0, -0.08], [-0.08, 1.0]]) if self.coupled else np.eye(2)

    def priority(self, features: dict[str, np.ndarray], day: int) -> tuple[np.ndarray, dict[str, Any]]:
        full_evidence, evidence_detail = encode_physical_evidence(features, self.graph)
        evidence = self._reduced_evidence(full_evidence)
        coupling = self._coupling()
        mats = self.graph.transition_matrices[: self.channels]
        gaps: list[float] = []
        state = self.state.copy()
        for _ in range(self.max_iter):
            messages = np.column_stack([mats[c] @ state[:, c] for c in range(self.channels)])
            dynamic = scaled_tanh(messages @ coupling.T, self.alpha)
            updated = self.physical_mix * evidence + (1.0 - self.physical_mix) * dynamic
            cap = 2.0 if self.channels >= 3 else 1.6
            updated = project_capped_simplex_rows(updated, cap=cap)
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = updated
            if gap <= self.tolerance:
                break
        self.state = state
        hr = state[self.graph.hospital_indices.ravel()].reshape(self.graph.hospital_indices.shape + (self.channels,))
        evidence_hr = full_evidence[self.graph.hospital_indices.ravel()].reshape(self.graph.hospital_indices.shape + (4,))
        if self.channels == 4:
            state_score = hr[..., 0] - hr[..., 1] + self.partial_coefficient * (hr[..., 2] - hr[..., 3])
            evidence_score = evidence_hr[..., 0] - evidence_hr[..., 1] + self.partial_coefficient * (evidence_hr[..., 2] - evidence_hr[..., 3])
            score = 0.55 * state_score + 0.45 * evidence_score
        elif self.channels == 3:
            state_score = hr[..., 0] - hr[..., 1] - 0.15 * hr[..., 2]
            evidence_score = evidence_hr[..., 0] - evidence_hr[..., 1] - 0.15 * 0.5 * (evidence_hr[..., 2] + evidence_hr[..., 3])
            score = 0.55 * state_score + 0.45 * evidence_score
        else:
            state_score = hr[..., 0] - hr[..., 1]
            evidence_score = evidence_hr[..., 0] - evidence_hr[..., 1]
            score = 0.55 * state_score + 0.45 * evidence_score
        priority = stable_softmax(score, axis=0, temperature=self.temperature)
        return priority, {
            "score": score,
            "state": state.copy(),
            "hospital_state": hr.copy(),
            "evidence": full_evidence,
            "evidence_detail": evidence_detail,
            "convergence_gaps": np.asarray(gaps),
            "iterations": len(gaps),
        }


class FCMPolicy(PriorityPolicy):
    display_name = "FCM"

    def __init__(self, alpha: float = 1.0, physical_mix: float = 0.45, max_iter: int = 60, tolerance: float = 1e-4, temperature: float = 0.30):
        self.alpha = alpha
        self.physical_mix = physical_mix
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.temperature = temperature

    def reset(self, scenario: Scenario, graph: ConceptGraph) -> None:
        super().reset(scenario, graph)
        self.state = np.full(len(graph.metadata), 0.5)
        signed = graph.adjacency * graph.sign * graph.edge_strength
        m = 0.42 * np.eye(len(graph.metadata)) + signed
        target = min(graph.operator_norm_coupled, 1.35)
        self.transition = m * target / max(np.linalg.norm(m, 2), 1e-12)

    def priority(self, features: dict[str, np.ndarray], day: int) -> tuple[np.ndarray, dict[str, Any]]:
        evidence, detail = encode_physical_evidence(features, self.graph)
        scalar_evidence = np.clip(0.5 + 0.5 * (evidence[:, 0] - evidence[:, 1]), 0.0, 1.0)
        state = self.state.copy()
        gaps: list[float] = []
        for _ in range(self.max_iter):
            dynamic = scaled_tanh(self.transition @ state, self.alpha)
            updated = self.physical_mix * scalar_evidence + (1.0 - self.physical_mix) * dynamic
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = np.clip(updated, 0.0, 1.0)
            if gap <= self.tolerance:
                break
        self.state = state
        score = state[self.graph.hospital_indices]
        return stable_softmax(score, axis=0, temperature=self.temperature), {
            "score": score,
            "state": state.copy(),
            "evidence": evidence,
            "evidence_detail": detail,
            "convergence_gaps": np.asarray(gaps),
            "iterations": len(gaps),
        }


class HeuristicPolicy(PriorityPolicy):
    def __init__(self, display_name: str = "Robust-LP", robust_z: float = 1.64, temperature: float = 0.30):
        self.display_name = display_name
        self.robust_z = robust_z
        self.temperature = temperature

    def priority(self, features: dict[str, np.ndarray], day: int) -> tuple[np.ndarray, dict[str, Any]]:
        demand = np.maximum(features["observed_demand"], 0.0)
        upper = demand + self.robust_z * np.sqrt(demand + 1.0)
        score = (
            upper / np.maximum(features["hospital_inventory"] + 1.0, 1.0)
            + 1.2 * features["backlog_pressure"]
            + 0.35 * features["lead_risk"]
            + 0.45 * features["capacity_risk"]
        )
        return stable_softmax(score, axis=0, temperature=self.temperature), {"score": score, "iterations": 1, "convergence_gaps": np.array([0.0])}


@dataclass
class SimulationResult:
    model: str
    seed: int
    metrics: dict[str, float]
    daily: pd.DataFrame
    hospital_resource: pd.DataFrame
    allocations: pd.DataFrame
    priority_history: np.ndarray
    score_history: np.ndarray
    demand_history: np.ndarray
    fulfilled_history: np.ndarray
    backlog_history: np.ndarray
    inventory_history: np.ndarray
    convergence_history: list[np.ndarray]
    iteration_history: np.ndarray
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)


def _build_features(
    scenario: Scenario,
    day: int,
    hospital_inventory: np.ndarray,
    dc_inventory: np.ndarray,
    backlog: np.ndarray,
    expected_inbound: np.ndarray,
) -> dict[str, np.ndarray]:
    observed = scenario.observed_demand[day]
    recent_scale = np.maximum(np.mean(scenario.observed_demand[max(0, day - 3) : day + 1], axis=0), 1.0)
    demand_pressure = np.clip(observed / (1.6 * recent_scale), 0.0, 1.0)
    backlog_pressure = np.clip(backlog / np.maximum(observed + backlog, 1.0), 0.0, 1.0)
    forecast = scenario.observed_demand[min(day + 1, scenario.days - 1)]
    inventory_risk = 1.0 - np.clip((hospital_inventory + expected_inbound) / np.maximum(forecast + backlog, 1.0), 0.0, 1.0)
    min_lead = scenario.lead_dh.min(axis=0)
    lead_risk = np.clip((min_lead - 1.0) / 6.0, 0.0, 1.0)
    region_dc = scenario.regions
    assigned_stock = dc_inventory[region_dc[:, None], np.arange(scenario.n_resources)[None, :]]
    regional_need = np.zeros_like(assigned_stock)
    for h in range(scenario.n_hospitals):
        hs = scenario.regions[h]
        regional_need[h] = (forecast[scenario.regions == hs] + backlog[scenario.regions == hs]).sum(axis=0)
    capacity_risk = 1.0 - np.clip(assigned_stock / np.maximum(regional_need, 1.0), 0.0, 1.0)
    dc_stock_risk = np.zeros_like(dc_inventory)
    dc_load_risk = np.zeros_like(dc_inventory)
    for d in range(dc_inventory.shape[0]):
        mask = scenario.regions == d
        expected = (forecast[mask] + backlog[mask]).sum(axis=0)
        dc_stock_risk[d] = 1.0 - np.clip(dc_inventory[d] / np.maximum(expected, 1.0), 0.0, 1.0)
        dc_load_risk[d] = np.clip(expected / np.maximum(dc_inventory[d] + 1.0, 1.0) / 2.0, 0.0, 1.0)
    total_future = forecast.sum(axis=0)
    supplier_risk = np.zeros((scenario.supplier_capacity.shape[1], scenario.n_resources))
    for s in range(supplier_risk.shape[0]):
        supplier_risk[s] = 1.0 - np.clip(scenario.supplier_capacity[day, s] / np.maximum(0.5 * total_future, 1.0), 0.0, 1.0)
    return {
        "observed_demand": observed,
        "demand_pressure": demand_pressure,
        "backlog_pressure": backlog_pressure,
        "inventory_risk": inventory_risk,
        "lead_risk": lead_risk,
        "capacity_risk": capacity_risk,
        "reporting_confidence": scenario.reporting_confidence[day],
        "hospital_inventory": hospital_inventory,
        "dc_inventory": dc_inventory,
        "dc_stock_risk": dc_stock_risk,
        "dc_load_risk": dc_load_risk,
        "supplier_risk": supplier_risk,
    }


def solve_allocation_lp(
    priority: np.ndarray,
    request: np.ndarray,
    dc_inventory: np.ndarray,
    lead_dh: np.ndarray,
    historical_fill: np.ndarray,
    priority_objective_weight: float,
    fairness_weight: float,
    lead_penalty: float,
    clinical_weights: np.ndarray | None = None,
) -> np.ndarray:
    d_count, h_count, r_count = lead_dh.shape
    nvar = d_count * h_count * r_count

    def vidx(d: int, h: int, k: int) -> int:
        return (d * h_count + h) * r_count + k

    req_norm = request / np.maximum(request.max(axis=0, keepdims=True), 1.0)
    if clinical_weights is None:
        clinical_weights = CLINICAL_WEIGHTS[:r_count]
    clinical_weights = np.asarray(clinical_weights, dtype=float)
    if clinical_weights.shape != (r_count,):
        raise ValueError(f"clinical_weights must have shape ({r_count},), got {clinical_weights.shape}.")
    c = np.zeros(nvar)
    for d in range(d_count):
        for h in range(h_count):
            for k in range(r_count):
                fairness_bonus = 1.0 - np.clip(historical_fill[h, k], 0.0, 1.0)
                utility = (
                    priority_objective_weight * clinical_weights[k] * priority[h, k]
                    + 0.20 * req_norm[h, k]
                    + fairness_weight * fairness_bonus
                    - lead_penalty * (lead_dh[d, h, k] - 1.0) / 6.0
                )
                c[vidx(d, h, k)] = -utility

    aub: list[np.ndarray] = []
    bub: list[float] = []
    for d in range(d_count):
        for k in range(r_count):
            row = np.zeros(nvar)
            for h in range(h_count):
                row[vidx(d, h, k)] = 1.0
            aub.append(row)
            bub.append(max(dc_inventory[d, k], 0.0))
    for h in range(h_count):
        for k in range(r_count):
            row = np.zeros(nvar)
            for d in range(d_count):
                row[vidx(d, h, k)] = 1.0
            aub.append(row)
            bub.append(max(request[h, k], 0.0))
    for d in range(d_count):
        row = np.zeros(nvar)
        for h in range(h_count):
            for k in range(r_count):
                row[vidx(d, h, k)] = 1.0
        aub.append(row)
        bub.append(max(0.85 * dc_inventory[d].sum(), 0.0))

    result = linprog(c, A_ub=np.asarray(aub), b_ub=np.asarray(bub), bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Allocation LP failed: {result.message}")
    return result.x.reshape(d_count, h_count, r_count)


def run_simulation(
    scenario: Scenario,
    graph: ConceptGraph,
    policy: PriorityPolicy,
    backlog_mode: str = "backlog",
    fairness_weight: float = 0.16,
    lead_penalty: float = 0.08,
    priority_objective_weight: float = 1.80,
    clinical_weights: np.ndarray | None = None,
) -> SimulationResult:
    if backlog_mode not in {"backlog", "lost_sales"}:
        raise ValueError("backlog_mode must be 'backlog' or 'lost_sales'.")
    policy.reset(scenario, graph)
    t_count, h_count, r_count = scenario.demand.shape
    d_count = scenario.initial_dc_inventory.shape[0]
    max_lead = int(max(scenario.lead_dh.max(), scenario.lead_sd.max()))
    horizon = t_count + max_lead + 2

    hospital_inventory = scenario.initial_hospital_inventory.copy()
    dc_inventory = scenario.initial_dc_inventory.copy()
    backlog = np.zeros((h_count, r_count))
    hospital_arrivals = np.zeros((horizon, h_count, r_count))
    hospital_arrival_lead = np.zeros_like(hospital_arrivals)
    dc_arrivals = np.zeros((horizon, d_count, r_count))

    fulfilled_hist = np.zeros((t_count, h_count, r_count))
    backlog_hist = np.zeros_like(fulfilled_hist)
    inventory_hist = np.zeros_like(fulfilled_hist)
    priority_hist = np.zeros_like(fulfilled_hist)
    score_hist = np.zeros_like(fulfilled_hist)
    allocation_hist = np.zeros((t_count, d_count, h_count, r_count))
    iteration_hist = np.zeros(t_count, dtype=int)
    convergence_hist: list[np.ndarray] = []
    daily_rows: list[dict[str, float]] = []
    allocation_rows: list[dict[str, Any]] = []
    cumulative_demand = np.zeros((h_count, r_count))
    cumulative_fulfilled = np.zeros((h_count, r_count))
    total_received = 0.0
    weighted_received_lead = 0.0
    evidence_snapshot: dict[str, Any] = {}

    for day in range(t_count):
        received = hospital_arrivals[day]
        hospital_inventory += received
        total_received += float(received.sum())
        weighted_received_lead += float(hospital_arrival_lead[day].sum())
        dc_inventory += dc_arrivals[day]

        new_demand = scenario.demand[day]
        cumulative_demand += new_demand
        need = new_demand + backlog if backlog_mode == "backlog" else new_demand
        fulfilled = np.minimum(hospital_inventory, need)
        hospital_inventory -= fulfilled
        cumulative_fulfilled += fulfilled
        residual = need - fulfilled
        backlog = residual if backlog_mode == "backlog" else np.zeros_like(residual)

        expected_inbound = hospital_arrivals[day + 1 : min(horizon, day + max_lead + 1)].sum(axis=0)
        features = _build_features(scenario, day, hospital_inventory, dc_inventory, backlog, expected_inbound)
        priority, diagnostics = policy.priority(features, day)
        priority_hist[day] = priority
        score = np.asarray(diagnostics.get("score", priority))
        if score.shape == (h_count, r_count):
            score_hist[day] = score
        iteration_hist[day] = int(diagnostics.get("iterations", 1))
        convergence_hist.append(np.asarray(diagnostics.get("convergence_gaps", [0.0]), dtype=float))
        if day == min(20, t_count - 1):
            evidence_snapshot = diagnostics

        forecast_window = scenario.observed_demand[day + 1 : min(t_count, day + 4)]
        forecast = forecast_window.sum(axis=0) if forecast_window.size else scenario.observed_demand[day]
        request = np.maximum(backlog + forecast - hospital_inventory - expected_inbound, 0.0)
        historical_fill = cumulative_fulfilled / np.maximum(cumulative_demand, 1.0)
        q = solve_allocation_lp(
            priority,
            request,
            dc_inventory,
            scenario.lead_dh,
            historical_fill,
            priority_objective_weight,
            fairness_weight,
            lead_penalty,
            clinical_weights=clinical_weights,
        )
        allocation_hist[day] = q
        dc_inventory -= q.sum(axis=1)
        dc_inventory = np.maximum(dc_inventory, 0.0)
        for d in range(d_count):
            for h in range(h_count):
                for k in range(r_count):
                    qty = q[d, h, k]
                    if qty <= 1e-10:
                        continue
                    arrival_day = day + int(scenario.lead_dh[d, h, k])
                    if arrival_day < horizon:
                        hospital_arrivals[arrival_day, h, k] += qty
                        hospital_arrival_lead[arrival_day, h, k] += qty * scenario.lead_dh[d, h, k]
                    allocation_rows.append(
                        {
                            "day": day,
                            "dc": f"DC{d+1}",
                            "hospital": f"H{h+1}",
                            "resource": scenario.resource_names[k],
                            "quantity": qty,
                            "lead_time_days": int(scenario.lead_dh[d, h, k]),
                            "arrival_day": arrival_day,
                            "priority": priority[h, k],
                        }
                    )

        regional_need = np.zeros((d_count, r_count))
        for d in range(d_count):
            mask = scenario.regions == d
            regional_need[d] = (backlog[mask] + scenario.observed_demand[min(day + 1, t_count - 1), mask]).sum(axis=0)
        for s in range(scenario.supplier_capacity.shape[1]):
            for k in range(r_count):
                capacity = scenario.supplier_capacity[day, s, k]
                weights = regional_need[:, k] / max(regional_need[:, k].sum(), 1.0)
                for d in range(d_count):
                    qty = capacity * weights[d]
                    arrival_day = day + int(scenario.lead_sd[s, d, k])
                    if arrival_day < horizon:
                        dc_arrivals[arrival_day, d, k] += qty

        fulfilled_hist[day] = fulfilled
        backlog_hist[day] = backlog
        inventory_hist[day] = hospital_inventory
        fill_day = float(fulfilled.sum() / max(new_demand.sum(), 1.0))
        service_ratios = cumulative_fulfilled.sum(axis=1) / np.maximum(cumulative_demand.sum(axis=1), 1.0)
        daily_rows.append(
            {
                "day": day,
                "service_level": fill_day,
                "unmet_units": float(np.maximum(cumulative_demand - cumulative_fulfilled, 0.0).sum()),
                "backlog_units": float(backlog.sum()),
                "hospital_inventory": float(hospital_inventory.sum()),
                "dc_inventory": float(dc_inventory.sum()),
                "gini_fill": gini(service_ratios),
                "jain_fill": jain_index(service_ratios),
                "mean_iterations": float(iteration_hist[day]),
                "final_convergence_gap": float(convergence_hist[-1][-1]) if convergence_hist[-1].size else 0.0,
            }
        )

    total_demand = float(cumulative_demand.sum())
    total_fulfilled = float(cumulative_fulfilled.sum())
    service_level = total_fulfilled / max(total_demand, 1.0)
    unmet_units = max(total_demand - total_fulfilled, 0.0)
    unmet_rate = unmet_units / max(total_demand, 1.0)
    avg_lead = weighted_received_lead / max(total_received, 1e-12)
    fill_hr = cumulative_fulfilled / np.maximum(cumulative_demand, 1.0)
    fill_h = cumulative_fulfilled.sum(axis=1) / np.maximum(cumulative_demand.sum(axis=1), 1.0)
    region_fill = []
    for region in np.unique(scenario.regions):
        mask = scenario.regions == region
        region_fill.append(cumulative_fulfilled[mask].sum() / max(cumulative_demand[mask].sum(), 1.0))
    geographic_equity = 1.0 - abs(float(region_fill[0] - region_fill[1])) if len(region_fill) == 2 else 1.0
    if clinical_weights is None:
        clinical_weights = CLINICAL_WEIGHTS[:r_count]
    clinical_weights = np.asarray(clinical_weights, dtype=float)
    priority_weights = cumulative_demand * clinical_weights[None, :]
    metrics = {
        "service_level": service_level,
        "unmet_demand_rate": unmet_rate,
        "unmet_demand_units": unmet_units,
        "average_lead_time_days": avg_lead,
        "gini_fill": gini(fill_h),
        "jain_fairness": jain_index(fill_h),
        "max_min_fairness": float(np.min(fill_h)),
        "geographic_equity": float(np.clip(geographic_equity, 0.0, 1.0)),
        "priority_weighted_equity": weighted_equity(fill_hr, priority_weights),
        "mean_convergence_iterations": float(np.mean(iteration_hist)),
        "median_convergence_iterations": float(np.median(iteration_hist)),
        "final_convergence_gap": float(np.median([g[-1] for g in convergence_hist if g.size])),
        "total_demand_units": total_demand,
        "total_fulfilled_units": total_fulfilled,
    }

    hr_rows: list[dict[str, Any]] = []
    for h in range(h_count):
        for k in range(r_count):
            hr_rows.append(
                {
                    "hospital": f"H{h+1}",
                    "resource": scenario.resource_names[k],
                    "region": int(scenario.regions[h]) + 1,
                    "demand": cumulative_demand[h, k],
                    "fulfilled": cumulative_fulfilled[h, k],
                    "fill_rate": fill_hr[h, k],
                    "final_backlog": backlog[h, k],
                    "mean_priority": priority_hist[:, h, k].mean(),
                }
            )

    return SimulationResult(
        model=policy.display_name,
        seed=scenario.seed,
        metrics=metrics,
        daily=pd.DataFrame(daily_rows),
        hospital_resource=pd.DataFrame(hr_rows),
        allocations=pd.DataFrame(allocation_rows),
        priority_history=priority_hist,
        score_history=score_hist,
        demand_history=scenario.demand.copy(),
        fulfilled_history=fulfilled_hist,
        backlog_history=backlog_hist,
        inventory_history=inventory_hist,
        convergence_history=convergence_hist,
        iteration_history=iteration_hist,
        evidence_snapshot=evidence_snapshot,
    )
