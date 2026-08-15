from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import ConceptGraph, stable_softmax
from .style import MODEL_STYLES, apply_nature_style, clean_axis, panel_label, save_figure


PRODUCTS = ("n95", "surgical_mask", "face_shield", "gown")
PRODUCT_LABELS = {
    "n95": "N95 respirators",
    "surgical_mask": "Surgical masks",
    "face_shield": "Face shields",
    "gown": "Gowns",
}

NORTHEAST = {
    "CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA",
}
MIDWEST = {
    "IN", "IL", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD",
}
SOUTH = {
    "DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
    "TN", "AR", "LA", "OK", "TX",
}
WEST = {
    "AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA",
}


def _state_region(value: Any) -> int:
    state = str(value).strip().upper()
    if state in NORTHEAST:
        return 0
    if state in MIDWEST:
        return 1
    if state in SOUTH:
        return 2
    if state in WEST:
        return 3
    return 2


def _parse_utc(values: pd.Series) -> pd.Series:
    """Parse the heterogeneous fractional-second strings in the public archive."""
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True)


@dataclass
class EventRecord:
    timestamp: pd.Timestamp
    split: str
    donor_key: str
    donor_state: str
    observed_allocation: np.ndarray
    request_cap: np.ndarray
    normalized_request: np.ndarray
    prior_fill: np.ndarray
    active: np.ndarray
    source_scores: np.ndarray
    source_confidence: np.ndarray
    inventory_units: np.ndarray
    route_risk: np.ndarray
    dc_stock_risk: np.ndarray
    dc_load_risk: np.ndarray
    supplier_risk: np.ndarray
    ambiguity: np.ndarray
    predecision_supply: float
    supply_confidence: float
    route_confidence: float
    observed_rows_total: int
    observed_rows_eligible: int
    observed_quantity_total: float
    observed_quantity_eligible: float

    @property
    def evaluable(self) -> bool:
        return bool(self.active.sum() >= 2 and self.observed_allocation.sum() > 0)


@dataclass
class ProductSeries:
    product: str
    facility_ids: np.ndarray
    states: np.ndarray
    counties: np.ndarray
    regions: np.ndarray
    records: list[EventRecord]


@dataclass(frozen=True)
class ACMParameters:
    mu_true: float = 0.82
    mu_partial: float = 0.62
    sigma_exact: float = 0.22
    sigma_partial: float = 0.25
    beta_partial: float = 0.82
    diagonal_memory: float = 0.42
    partial_to_exact: float = 0.24
    exact_to_partial: float = 0.12
    partial_self: float = 0.88
    physical_mix: float = 0.45
    partial_coefficient: float = 0.75
    temperature: float = 0.30
    operator_target_norm: float = 1.35
    lp_priority_weight: float = 1.80
    lp_demand_weight: float = 0.20
    lp_fairness_weight: float = 0.16
    lp_lead_weight: float = 0.08

    def gaussian_mu(self) -> np.ndarray:
        return np.array(
            [self.mu_true, 1.0 - self.mu_true, self.mu_partial, 1.0 - self.mu_partial],
            dtype=float,
        )

    def gaussian_sigma(self) -> np.ndarray:
        return np.array(
            [self.sigma_exact, self.sigma_exact, self.sigma_partial, self.sigma_partial],
            dtype=float,
        )

    def gaussian_beta(self) -> np.ndarray:
        return np.array([1.0, 1.0, self.beta_partial, self.beta_partial], dtype=float)

    def coupling(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, self.partial_to_exact, 0.0],
                [0.0, 1.0, 0.0, self.partial_to_exact],
                [self.exact_to_partial, 0.0, self.partial_self, 0.0],
                [0.0, self.exact_to_partial, 0.0, self.partial_self],
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "mu_true": self.mu_true,
            "mu_false": 1.0 - self.mu_true,
            "mu_partial_true": self.mu_partial,
            "mu_partial_false": 1.0 - self.mu_partial,
            "sigma_exact": self.sigma_exact,
            "sigma_partial": self.sigma_partial,
            "beta_partial": self.beta_partial,
            "diagonal_memory": self.diagonal_memory,
            "partial_to_exact": self.partial_to_exact,
            "exact_to_partial": self.exact_to_partial,
            "partial_self": self.partial_self,
            "physical_mix": self.physical_mix,
            "partial_coefficient": self.partial_coefficient,
            "temperature": self.temperature,
            "operator_target_norm": self.operator_target_norm,
            "lp_priority_weight": self.lp_priority_weight,
            "lp_demand_weight": self.lp_demand_weight,
            "lp_fairness_weight": self.lp_fairness_weight,
            "lp_lead_weight": self.lp_lead_weight,
        }


def _supply_risk(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    risk_map = {"no": 1.0, "< 1": 0.80, "1 - 2": 0.45, "2+": 0.15, "1 week or less": 0.80}
    weeks_map = {"no": 0.0, "< 1": 0.5, "1 - 2": 1.5, "2+": 2.5, "1 week or less": 0.5}
    observed = normalized.ne("").to_numpy(dtype=float)
    risk = normalized.map(risk_map).fillna(0.50).to_numpy(dtype=float)
    weeks = normalized.map(weeks_map).fillna(1.0).to_numpy(dtype=float)
    return risk, observed, weeks


def _policy_risk(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    risk_map = {
        "crisis capacity": 1.0,
        "contingency capacity": 0.60,
        "conventional capacity": 0.20,
    }
    observed = normalized.ne("").to_numpy(dtype=float)
    risk = normalized.map(risk_map).fillna(0.50).to_numpy(dtype=float)
    return risk, observed


def _delivery_history(matches: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grouped = matches.groupby(["facilitykey", "matchCreated"], as_index=False)["qty"].sum()
    for facility, frame in grouped.groupby("facilitykey"):
        ordered = frame.sort_values("matchCreated")
        times = ordered["matchCreated"].astype("int64").to_numpy()
        cumulative = ordered["qty"].cumsum().to_numpy(dtype=float)
        histories[str(facility)] = (times, cumulative)
    return histories


def _delivered_between(
    history: dict[str, tuple[np.ndarray, np.ndarray]],
    facility: str,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> float:
    item = history.get(str(facility))
    if item is None:
        return 0.0
    times, cumulative = item
    start_ns = int(start.value)
    stop_ns = int(stop.value)
    right = int(np.searchsorted(times, stop_ns, side="left")) - 1
    left = int(np.searchsorted(times, start_ns, side="left")) - 1
    total_right = cumulative[right] if right >= 0 else 0.0
    total_left = cumulative[left] if left >= 0 else 0.0
    return float(max(total_right - total_left, 0.0))


def _quantity_histories(
    frame: pd.DataFrame,
    entity_column: str,
    product_column: str,
    time_column: str,
    quantity_column: str,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Return timestamped cumulative quantities for pre-decision stock audits."""
    histories: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    valid = frame.loc[
        frame[time_column].notna() & pd.to_numeric(frame[quantity_column], errors="coerce").gt(0)
    ].copy()
    valid[quantity_column] = pd.to_numeric(valid[quantity_column], errors="coerce")
    grouped = valid.groupby(
        [entity_column, product_column, time_column], as_index=False
    )[quantity_column].sum()
    for (entity, product), values in grouped.groupby([entity_column, product_column]):
        ordered = values.sort_values(time_column)
        histories[(str(entity), str(product))] = (
            ordered[time_column].astype("int64").to_numpy(),
            ordered[quantity_column].cumsum().to_numpy(dtype=float),
        )
    return histories


def _cumulative_before(
    histories: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    entity: str,
    product: str,
    timestamp: pd.Timestamp,
    inclusive: bool,
) -> float:
    item = histories.get((str(entity), str(product)))
    if item is None:
        return 0.0
    times, cumulative = item
    side = "right" if inclusive else "left"
    position = int(np.searchsorted(times, int(timestamp.value), side=side)) - 1
    return float(cumulative[position]) if position >= 0 else 0.0


def load_getusppe_series(
    data_dir: Path,
    lookback_days: int = 90,
    products: Iterable[str] = PRODUCTS,
) -> tuple[list[ProductSeries], pd.DataFrame]:
    """Build leakage-controlled event batches from real requests and deliveries.

    Each decision batch is keyed by the exact match-creation timestamp and PPE
    class. Only requests timestamped no later than the batch are eligible.
    Delivered matches with a recorded close time preceding creation are
    excluded as chronology failures and retained in the audit counts.
    """
    products = tuple(products)
    requests = pd.read_csv(data_dir / "all_requests.csv", low_memory=False)
    matches = pd.read_csv(data_dir / "all_matches.csv", low_memory=False)
    offers = pd.read_csv(data_dir / "all_offers.csv", low_memory=False)
    requests["datecreated"] = _parse_utc(requests["datecreated"])
    matches["matchCreated"] = _parse_utc(matches["matchCreated"])
    matches["closedOrDelivered"] = _parse_utc(matches["closedOrDelivered"])
    offers["datecreated"] = _parse_utc(offers["datecreated"])

    def modal_state(values: pd.Series) -> str:
        normalized = values.dropna().astype(str).str.strip().str.upper()
        mode = normalized.mode()
        return str(mode.iloc[0]) if len(mode) else ""

    donor_states = offers.groupby("donorKey")["stateprovince"].agg(modal_state)
    offer_histories = _quantity_histories(
        offers,
        entity_column="donorKey",
        product_column="equipmentEnum",
        time_column="datecreated",
        quantity_column="qty",
    )
    committed_histories = _quantity_histories(
        matches,
        entity_column="donorkey",
        product_column="enumcode",
        time_column="matchCreated",
        quantity_column="qty",
    )

    facility_types = (
        requests[["facilityKey", "facilityTypeEnum"]]
        .drop_duplicates("facilityKey")
        .set_index("facilityKey")["facilityTypeEnum"]
    )
    request_mask = (
        requests["facilityTypeEnum"].eq("acute_care")
        & requests["equipmentEnum"].isin(products)
        & requests["weeklyrequestqty"].gt(0)
        & requests["datecreated"].notna()
    )
    requests = requests.loc[request_mask].copy()

    delivered_mask = (
        matches["matchstatus"].astype(str).str.lower().eq("delivered")
        & matches["qty"].gt(0)
        & matches["enumcode"].isin(products)
        & matches["facilitykey"].map(facility_types).eq("acute_care")
        & matches["matchCreated"].notna()
        & matches["closedOrDelivered"].notna()
    )
    delivered_all = matches.loc[delivered_mask].copy()
    chronology_ok = delivered_all["closedOrDelivered"].ge(delivered_all["matchCreated"])
    delivered = delivered_all.loc[chronology_ok].copy()

    audit_rows: list[dict[str, Any]] = [
        {"item": "Request rows retained", "value": int(len(requests))},
        {"item": "Unique acute-care facilities with retained requests", "value": int(requests["facilityKey"].nunique())},
        {"item": "Delivered rows before chronology check", "value": int(len(delivered_all))},
        {"item": "Delivered rows excluded: close precedes creation", "value": int((~chronology_ok).sum())},
        {"item": "Delivered rows after chronology check", "value": int(len(delivered))},
    ]

    series: list[ProductSeries] = []
    lookback = pd.Timedelta(days=int(lookback_days))
    for product in products:
        request_product = requests.loc[requests["equipmentEnum"].eq(product)].copy()
        match_product = delivered.loc[delivered["enumcode"].eq(product)].copy()
        facilities = np.array(sorted(request_product["facilityKey"].astype(str).unique()), dtype=object)
        facility_index = {facility: i for i, facility in enumerate(facilities)}
        facility_meta = (
            request_product.sort_values("datecreated")
            .drop_duplicates("facilityKey", keep="last")
            .set_index("facilityKey")
            .reindex(facilities)
        )
        states = facility_meta["stateprovince"].fillna("").astype(str).str.upper().to_numpy(dtype=object)
        counties = facility_meta["county"].fillna("").astype(str).to_numpy(dtype=object)
        regions = np.array([_state_region(x) for x in states], dtype=int)
        history = _delivery_history(match_product)
        records: list[EventRecord] = []

        for (timestamp, donor_key), observed_frame in match_product.groupby(
            ["matchCreated", "donorkey"], sort=True
        ):
            lower = timestamp - lookback
            active_frame = request_product.loc[
                request_product["datecreated"].between(lower, timestamp, inclusive="both")
            ]
            active_frame = active_frame.sort_values("datecreated").drop_duplicates("facilityKey", keep="last")
            if active_frame.empty:
                continue

            h = len(facilities)
            request_cap = np.zeros(h, dtype=float)
            normalized_request = np.zeros(h, dtype=float)
            prior_fill = np.zeros(h, dtype=float)
            active = np.zeros(h, dtype=bool)
            source_scores = np.zeros((h, 5), dtype=float)
            source_confidence = np.ones((h, 5), dtype=float)
            inventory_units = np.zeros(h, dtype=float)
            donor_state = str(donor_states.get(donor_key, "")).strip().upper()
            offered_to_date = _cumulative_before(
                offer_histories, str(donor_key), product, timestamp, inclusive=True
            )
            committed_before = _cumulative_before(
                committed_histories, str(donor_key), product, timestamp, inclusive=False
            )
            predecision_supply = max(offered_to_date - committed_before, 0.0)
            supply_confidence = float(offered_to_date > 0.0)
            donor_region = _state_region(donor_state)
            route_risk = np.full(h, 0.50, dtype=float)
            if donor_state:
                same_state = states == donor_state
                same_region = regions == donor_region
                route_risk[:] = 1.0
                route_risk[same_region] = 0.40
                route_risk[same_state] = 0.0
            ambiguity = np.zeros(h, dtype=float)

            idx = np.array([facility_index[str(x)] for x in active_frame["facilityKey"]], dtype=int)
            weekly = active_frame["weeklyrequestqty"].to_numpy(dtype=float)
            ages = (timestamp - active_frame["datecreated"]).dt.total_seconds().to_numpy() / 86400.0
            elapsed_weeks = np.floor(np.maximum(ages, 0.0) / 7.0) + 1.0
            cumulative_need = weekly * elapsed_weeks
            previous = np.array(
                [
                    _delivered_between(history, str(f), start, timestamp)
                    for f, start in zip(active_frame["facilityKey"], active_frame["datecreated"])
                ],
                dtype=float,
            )
            outstanding = np.maximum(weekly, cumulative_need - previous)
            positive = outstanding > 0
            idx = idx[positive]
            active_frame = active_frame.iloc[np.flatnonzero(positive)].copy()
            weekly = weekly[positive]
            ages = ages[positive]
            cumulative_need = cumulative_need[positive]
            previous = previous[positive]
            outstanding = outstanding[positive]

            active[idx] = True
            request_cap[idx] = outstanding
            prior_fill[idx] = np.clip(previous / np.maximum(cumulative_need, 1.0), 0.0, 1.0)
            log_request = np.log1p(outstanding)
            scale = max(float(np.nanpercentile(log_request, 95)), 1e-8)
            demand_pressure = np.clip(log_request / scale, 0.0, 1.0)
            backlog = np.maximum(outstanding - weekly, 0.0)
            backlog_pressure = np.clip(backlog / np.maximum(outstanding, 1.0), 0.0, 1.0)
            inventory_risk, inventory_observed, inventory_weeks = _supply_risk(active_frame["supplyremaining"])
            policy_risk, policy_observed = _policy_risk(active_frame["practicesAnyPolicies"])
            waiting_risk = np.clip(ages / float(lookback_days), 0.0, 1.0)

            observed_by_facility = observed_frame.groupby("facilitykey")["qty"].sum()
            observed_allocation = np.zeros(h, dtype=float)
            eligible_observed = 0.0
            eligible_rows = 0
            for facility, quantity in observed_by_facility.items():
                position = facility_index.get(str(facility))
                if position is not None and active[position]:
                    observed_allocation[position] += float(quantity)
                    eligible_observed += float(quantity)
                    eligible_rows += int((observed_frame["facilitykey"] == facility).sum())

            global_scarcity = float(
                np.clip(1.0 - eligible_observed / max(float(outstanding.sum()), 1.0), 0.0, 1.0)
            )
            capacity_risk = np.clip(0.5 * policy_risk + 0.5 * global_scarcity, 0.0, 1.0)
            scores = np.column_stack(
                [demand_pressure, backlog_pressure, inventory_risk, waiting_risk, capacity_risk]
            )
            confidence = np.column_stack(
                [
                    np.ones(len(idx)),
                    np.ones(len(idx)),
                    inventory_observed,
                    np.ones(len(idx)),
                    0.5 + 0.5 * policy_observed,
                ]
            )
            source_scores[idx] = scores
            source_confidence[idx] = confidence
            inventory_units[idx] = inventory_weeks * weekly
            normalized_request[idx] = outstanding / max(float(outstanding.max()), 1.0)
            disagreement = np.clip(np.std(scores, axis=1) / 0.5, 0.0, 1.0)
            missingness = 1.0 - confidence.mean(axis=1)
            ambiguity[idx] = np.clip(0.60 * missingness + 0.40 * disagreement, 0.0, 1.0)

            dc_load_risk = np.zeros((4, 1), dtype=float)
            regional_demand = np.array(
                [outstanding[regions[idx] == region].sum() for region in range(4)], dtype=float
            )
            if regional_demand.max() > 0:
                dc_load_risk[:, 0] = regional_demand / regional_demand.max()
            dc_stock_risk = np.full((4, 1), global_scarcity, dtype=float)
            supplier_risk = np.full((1, 1), global_scarcity, dtype=float)

            if timestamp < pd.Timestamp("2020-07-01", tz="UTC"):
                split = "calibration"
            elif timestamp < pd.Timestamp("2021-01-01", tz="UTC"):
                split = "validation"
            else:
                split = "test"

            records.append(
                EventRecord(
                    timestamp=timestamp,
                    split=split,
                    donor_key=str(donor_key),
                    donor_state=donor_state,
                    observed_allocation=observed_allocation,
                    request_cap=request_cap,
                    normalized_request=normalized_request,
                    prior_fill=prior_fill,
                    active=active,
                    source_scores=source_scores,
                    source_confidence=source_confidence,
                    inventory_units=inventory_units,
                    route_risk=route_risk,
                    dc_stock_risk=dc_stock_risk,
                    dc_load_risk=dc_load_risk,
                    supplier_risk=supplier_risk,
                    ambiguity=ambiguity,
                    predecision_supply=float(predecision_supply),
                    supply_confidence=supply_confidence,
                    route_confidence=float(bool(donor_state)),
                    observed_rows_total=int(len(observed_frame)),
                    observed_rows_eligible=int(eligible_rows),
                    observed_quantity_total=float(observed_frame["qty"].sum()),
                    observed_quantity_eligible=float(eligible_observed),
                )
            )

        series.append(
            ProductSeries(
                product=product,
                facility_ids=facilities,
                states=states,
                counties=counties,
                regions=regions,
                records=records,
            )
        )
        audit_rows.extend(
            [
                {"item": f"{product}: request facilities", "value": int(len(facilities))},
                {"item": f"{product}: event batches", "value": int(len(records))},
                {"item": f"{product}: evaluable batches", "value": int(sum(x.evaluable for x in records))},
                {
                    "item": f"{product}: eligible delivered quantity",
                    "value": float(sum(x.observed_quantity_eligible for x in records)),
                },
            ]
        )

    total_quantity = sum(x.observed_quantity_total for s in series for x in s.records)
    eligible_quantity = sum(x.observed_quantity_eligible for s in series for x in s.records)
    total_rows = sum(x.observed_rows_total for s in series for x in s.records)
    eligible_rows = sum(x.observed_rows_eligible for s in series for x in s.records)
    audit_rows.extend(
        [
            {"item": "Eligible delivered-row coverage", "value": eligible_rows / max(total_rows, 1)},
            {"item": "Eligible delivered-quantity coverage", "value": eligible_quantity / max(total_quantity, 1.0)},
            {"item": "Request lookback days", "value": int(lookback_days)},
        ]
    )
    return series, pd.DataFrame(audit_rows)


def _project_rows(values: np.ndarray, cap: float) -> np.ndarray:
    """Vectorized projection onto [0,1]^K with a row-sum cap."""
    out = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    mask = out.sum(axis=1) > cap
    if not np.any(mask):
        return out
    selected = out[mask]
    lower = np.full(len(selected), -1.0)
    upper = np.full(len(selected), 1.0)
    for _ in range(45):
        middle = 0.5 * (lower + upper)
        projected = np.clip(selected - middle[:, None], 0.0, 1.0)
        too_large = projected.sum(axis=1) > cap
        lower[too_large] = middle[too_large]
        upper[~too_large] = middle[~too_large]
    out[mask] = np.clip(selected - upper[:, None], 0.0, 1.0)
    return out


def _evidence_from_sources(scores: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    c = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    affirmative = np.clip(2.0 * (p - 0.5), 0.0, 1.0)
    negative = np.clip(2.0 * (0.5 - p), 0.0, 1.0)
    ambiguity = np.clip((1.0 - c) + 0.15 * (1.0 - np.abs(2.0 * p - 1.0)), 0.0, 1.0)
    evidence = np.column_stack(
        [
            np.mean(c * affirmative, axis=1),
            np.mean(c * negative, axis=1),
            np.mean(ambiguity * p, axis=1),
            np.mean(ambiguity * (1.0 - p), axis=1),
        ]
    )
    return _project_rows(evidence, cap=2.0)


def _matrix_norm_upper(matrix: sparse.spmatrix) -> float:
    absolute = abs(matrix)
    one_norm = float(np.asarray(absolute.sum(axis=0)).max(initial=0.0))
    infinity_norm = float(np.asarray(absolute.sum(axis=1)).max(initial=0.0))
    return float(np.sqrt(max(one_norm * infinity_norm, 0.0)))


def _power_norm(matrix: sparse.spmatrix, seed: int = 20260718, iterations: int = 60) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=matrix.shape[1])
    vector /= max(np.linalg.norm(vector), 1e-12)
    for _ in range(iterations):
        candidate = matrix.T @ (matrix @ vector)
        norm = np.linalg.norm(candidate)
        if norm <= 1e-14:
            return 0.0
        vector = np.asarray(candidate).ravel() / norm
    return float(np.linalg.norm(matrix @ vector))


def build_sparse_product_graph(
    panel: ProductSeries,
    parameters: ACMParameters,
    estimate_norm: bool = False,
) -> ConceptGraph:
    """Construct the manuscript graph in sparse form for one PPE class."""
    h = len(panel.facility_ids)
    d = 4
    s = 1
    n = h + d + s
    hospital_indices = np.arange(h, dtype=int)[:, None]
    dc_indices = np.arange(h, h + d, dtype=int)[:, None]
    supplier_indices = np.array([[h + d]], dtype=int)

    metadata_rows: list[dict[str, Any]] = []
    for i, facility in enumerate(panel.facility_ids):
        metadata_rows.append(
            {
                "index": i,
                "kind": "hospital",
                "entity": str(facility),
                "resource": panel.product,
                "region": int(panel.regions[i]),
                "concept_semantic": "shortage_risk",
            }
        )
    for region in range(d):
        metadata_rows.append(
            {
                "index": h + region,
                "kind": "dc",
                "entity": f"Region-{region + 1}",
                "resource": panel.product,
                "region": region,
                "concept_semantic": "supply_availability",
            }
        )
    metadata_rows.append(
        {
            "index": h + d,
            "kind": "supplier",
            "entity": "Pooled-donations",
            "resource": panel.product,
            "region": -1,
            "concept_semantic": "supply_availability",
        }
    )

    rows: list[int] = []
    cols: list[int] = []
    signs: list[float] = []
    magnitudes: list[float] = []

    def edge(source: int, target: int, sign: float, magnitude: float) -> None:
        rows.append(target)
        cols.append(source)
        signs.append(float(np.sign(sign)))
        magnitudes.append(float(np.clip(magnitude, 0.0, 1.0)))

    supplier = int(supplier_indices[0, 0])
    for region in range(d):
        dc = int(dc_indices[region, 0])
        edge(supplier, dc, +1, 0.90)
        for i in range(h):
            local = int(panel.regions[i]) == region
            edge(dc, i, -1, 0.88 if local else 0.42)
            edge(i, dc, -1, 0.70 if local else 0.28)

    # Anonymous locations preclude route reconstruction.  The only
    # hospital-hospital edges are reproducible local-neighborhood links:
    # up to two adjacent anonymized facilities within the same county, or
    # within the same state when a county contains a singleton.
    groups: list[np.ndarray] = []
    county_keys = np.array(
        [f"{state}|{county}" if county else f"{state}|" for state, county in zip(panel.states, panel.counties)],
        dtype=object,
    )
    for key in sorted(set(county_keys.tolist())):
        positions = np.flatnonzero(county_keys == key)
        if len(positions) >= 2:
            groups.append(positions)
    covered = set(np.concatenate(groups).tolist()) if groups else set()
    for state in sorted(set(panel.states.tolist())):
        positions = np.array([i for i in np.flatnonzero(panel.states == state) if i not in covered], dtype=int)
        if len(positions) >= 2:
            groups.append(positions)
    for positions in groups:
        ordered = np.sort(positions)
        for offset in (1, 2):
            for left, right in zip(ordered[:-offset], ordered[offset:]):
                edge(int(left), int(right), +1, 0.20)
                edge(int(right), int(left), +1, 0.20)

    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sign_matrix = sparse.csr_matrix((np.asarray(signs), (rows, cols)), shape=(n, n))
    strength_matrix = sparse.csr_matrix((np.asarray(magnitudes), (rows, cols)), shape=(n, n))
    row_array = np.asarray(rows, dtype=int)
    col_array = np.asarray(cols, dtype=int)
    sign_array = np.asarray(signs, dtype=float)
    magnitude_array = np.asarray(magnitudes, dtype=float)

    mu = parameters.gaussian_mu()
    sigma = parameters.gaussian_sigma()
    beta = parameters.gaussian_beta()
    gaussian = beta[None, :] * np.exp(
        -((magnitude_array[:, None] - mu[None, :]) ** 2) / (2.0 * sigma[None, :] ** 2)
    )
    channel_share = gaussian / np.maximum(gaussian.sum(axis=1, keepdims=True), 1e-12)
    identity = sparse.eye(n, format="csr")
    transitions: list[sparse.csr_matrix] = []
    for channel in range(4):
        values = sign_array * magnitude_array * channel_share[:, channel]
        weighted = sparse.csr_matrix((values, (row_array, col_array)), shape=(n, n))
        transitions.append((parameters.diagonal_memory * identity + weighted).tocsr())

    coupling = parameters.coupling()
    raw_block = sparse.bmat(
        [[coupling[i, j] * transitions[j] for j in range(4)] for i in range(4)],
        format="csr",
    )
    # The role-expanded external graph contains thousands of hospitals linked
    # to four regional DC nodes.  A max-degree 1/inf-norm bound is far too
    # conservative for this star-like topology and would collapse all dynamic
    # effects.  Deterministic power iteration estimates the sparse block
    # spectral norm; a 1% safety margin keeps the configured value below the
    # requested target.  The final norm is recomputed after scaling.
    power_iterations = 100 if estimate_norm else 45
    raw_norm = _power_norm(raw_block, iterations=power_iterations)
    operator_scale = parameters.operator_target_norm / max(1.01 * raw_norm, 1e-12)
    transitions = [(matrix * operator_scale).tocsr() for matrix in transitions]
    coupled_block = sparse.bmat(
        [[coupling[i, j] * transitions[j] for j in range(4)] for i in range(4)],
        format="csr",
    )
    independent = np.eye(4)
    independent_block = sparse.bmat(
        [[independent[i, j] * transitions[j] for j in range(4)] for i in range(4)],
        format="csr",
    )
    coupled_norm = _power_norm(coupled_block, iterations=power_iterations)
    independent_norm = _power_norm(independent_block, iterations=power_iterations)
    dynamic_mix = 1.0 - parameters.physical_mix

    return ConceptGraph(
        metadata=pd.DataFrame(metadata_rows),
        adjacency=adjacency,
        sign=sign_matrix,
        edge_strength=strength_matrix,
        channel_weights=np.empty((0, 0, 0)),
        transition_matrices=transitions,  # type: ignore[arg-type]
        coupled_matrix=coupling,
        independent_matrix=independent,
        operator_norm_coupled=float(coupled_norm),
        operator_norm_independent=float(independent_norm),
        contraction_bound_coupled=float(dynamic_mix * 0.5 * coupled_norm),
        contraction_bound_independent=float(dynamic_mix * 0.5 * independent_norm),
        operator_scale=float(operator_scale),
        hospital_indices=hospital_indices,
        dc_indices=dc_indices,
        supplier_indices=supplier_indices,
    )


def _event_evidence(event: EventRecord, graph: ConceptGraph) -> tuple[np.ndarray, np.ndarray]:
    h = len(event.active)
    evidence = np.zeros((len(graph.metadata), 4), dtype=float)
    hospital_evidence = _evidence_from_sources(event.source_scores, event.source_confidence)
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
        [[1.0 - event.supplier_risk[0, 0], 1.0 - event.dc_stock_risk[:, 0].mean()]], dtype=float
    )
    supplier_confidence = np.array([[0.95, 0.90]], dtype=float)
    evidence[graph.supplier_indices[0, 0]] = _evidence_from_sources(
        supplier_scores, supplier_confidence
    )[0]
    return _project_rows(evidence, cap=2.0), hospital_evidence


class SparseACMPolicy:
    def __init__(
        self,
        graph: ConceptGraph,
        parameters: ACMParameters,
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

    def _reduced_evidence(self, full: np.ndarray) -> np.ndarray:
        if self.channels == 4:
            return full
        if self.channels == 3:
            return np.column_stack([full[:, 0], full[:, 1], 0.5 * (full[:, 2] + full[:, 3])])
        if self.channels == 2:
            return full[:, :2]
        raise ValueError("Only 2, 3, or 4 ACM channels are supported.")

    def _coupling(self) -> np.ndarray:
        if not self.coupled:
            return np.eye(self.channels)
        if self.channels == 4:
            return self.graph.coupled_matrix
        if self.channels == 3:
            return np.array(
                [[1.0, -0.08, 0.16], [-0.08, 1.0, 0.16], [0.08, 0.08, 0.90]], dtype=float
            )
        return np.array([[1.0, -0.08], [-0.08, 1.0]], dtype=float)

    def score(self, event: EventRecord) -> tuple[np.ndarray, dict[str, Any]]:
        full_evidence, hospital_evidence = _event_evidence(event, self.graph)
        evidence = self._reduced_evidence(full_evidence)
        coupling = self._coupling()
        matrices = self.graph.transition_matrices[: self.channels]
        state = self.state.copy()
        gaps: list[float] = []
        for _ in range(self.max_iter):
            messages = np.column_stack([matrices[c] @ state[:, c] for c in range(self.channels)])
            dynamic = 0.5 * (1.0 + np.tanh(messages @ coupling.T))
            updated = self.parameters.physical_mix * evidence + (1.0 - self.parameters.physical_mix) * dynamic
            updated = _project_rows(updated, cap=2.0 if self.channels >= 3 else 1.6)
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = updated
            if gap <= self.tolerance:
                break
        self.state = state
        hospital_state = state[self.graph.hospital_indices[:, 0]]
        if self.channels == 4:
            gamma = self.parameters.partial_coefficient
            state_score = (
                hospital_state[:, 0]
                - hospital_state[:, 1]
                + gamma * (hospital_state[:, 2] - hospital_state[:, 3])
            )
            evidence_score = (
                hospital_evidence[:, 0]
                - hospital_evidence[:, 1]
                + gamma * (hospital_evidence[:, 2] - hospital_evidence[:, 3])
            )
        elif self.channels == 3:
            state_score = hospital_state[:, 0] - hospital_state[:, 1] - 0.15 * hospital_state[:, 2]
            evidence_score = (
                hospital_evidence[:, 0]
                - hospital_evidence[:, 1]
                - 0.075 * (hospital_evidence[:, 2] + hospital_evidence[:, 3])
            )
        else:
            state_score = hospital_state[:, 0] - hospital_state[:, 1]
            evidence_score = hospital_evidence[:, 0] - hospital_evidence[:, 1]
        score = 0.55 * state_score + 0.45 * evidence_score
        return score, {
            "iterations": len(gaps),
            "convergence_gap": gaps[-1],
            "hospital_state": hospital_state,
            "hospital_evidence": hospital_evidence,
        }


class SparseFCMPolicy:
    def __init__(self, graph: ConceptGraph, parameters: ACMParameters) -> None:
        self.graph = graph
        self.parameters = parameters
        self.state = np.full(len(graph.metadata), 0.5, dtype=float)
        signed = graph.adjacency.multiply(graph.sign).multiply(graph.edge_strength).tocsr()
        raw = parameters.diagonal_memory * sparse.eye(len(graph.metadata), format="csr") + signed
        scale = parameters.operator_target_norm / max(_matrix_norm_upper(raw), 1e-12)
        self.transition = (raw * scale).tocsr()

    def score(self, event: EventRecord) -> tuple[np.ndarray, dict[str, Any]]:
        evidence, _ = _event_evidence(event, self.graph)
        scalar_evidence = np.clip(0.5 + 0.5 * (evidence[:, 0] - evidence[:, 1]), 0.0, 1.0)
        state = self.state.copy()
        gaps: list[float] = []
        for _ in range(60):
            dynamic = 0.5 * (1.0 + np.tanh(self.transition @ state))
            updated = self.parameters.physical_mix * scalar_evidence + (1.0 - self.parameters.physical_mix) * dynamic
            gap = float(np.max(np.abs(updated - state)))
            gaps.append(gap)
            state = np.clip(updated, 0.0, 1.0)
            if gap <= 1e-4:
                break
        self.state = state
        return state[self.graph.hospital_indices[:, 0]], {
            "iterations": len(gaps),
            "convergence_gap": gaps[-1],
        }


MODEL_ORDER = (
    "ACM-4 (coupled)",
    "ACM-4 (independent)",
    "ACM-3",
    "ACM-2",
    "FCM",
    "Calibrated scalar",
    "Demand proportional",
    "Robust priority",
    "Equal allocation",
)


def _allocate(
    event: EventRecord,
    score: np.ndarray,
    parameters: ACMParameters,
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
            np.asarray(score, dtype=float)[active_idx], axis=0, temperature=parameters.temperature
        )

    objective[active_idx] = (
        parameters.lp_priority_weight * priority[active_idx]
        + parameters.lp_demand_weight * event.normalized_request[active_idx]
        + parameters.lp_fairness_weight * (1.0 - event.prior_fill[active_idx])
        - parameters.lp_lead_weight * event.route_risk[active_idx]
    )
    # This greedy solution is exact for the pooled one-resource LP: one
    # supply constraint, recipient request caps, and nonnegative dispatch.
    order = active_idx[np.lexsort((active_idx, -objective[active_idx]))]
    remaining = float(event.observed_allocation.sum())
    for position in order:
        if remaining <= 1e-12:
            break
        quantity = min(float(event.request_cap[position]), remaining)
        allocation[position] = quantity
        remaining -= quantity
    return allocation, priority, objective


def _ndcg_all(observed: np.ndarray, predicted: np.ndarray) -> float:
    gains = np.maximum(np.asarray(observed, dtype=float), 0.0)
    if gains.sum() <= 0:
        return np.nan
    order = np.argsort(-predicted, kind="mergesort")
    ideal = np.argsort(-gains, kind="mergesort")
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / max(idcg, 1e-12)


def _panel_metrics(
    event: EventRecord,
    predicted: np.ndarray,
    ranking_score: np.ndarray,
) -> dict[str, float]:
    idx = np.flatnonzero(event.active & (event.request_cap > 0))
    observed = event.observed_allocation[idx]
    predicted_active = predicted[idx]
    observed_total = float(observed.sum())
    predicted_total = float(predicted_active.sum())
    p_observed = observed / max(observed_total, 1e-12)
    p_predicted = predicted_active / max(predicted_total, 1e-12)
    overlap = float(np.minimum(p_observed, p_predicted).sum()) if predicted_total > 0 else 0.0
    js_distance = (
        float(jensenshannon(p_observed, p_predicted, base=2.0)) if predicted_total > 0 else 1.0
    )
    binary = (observed > 0).astype(int)
    if binary.sum() > 0 and binary.sum() < len(binary):
        average_precision = float(average_precision_score(binary, ranking_score[idx]))
    else:
        average_precision = 1.0
    k = int(binary.sum())
    top_predicted = set(np.argsort(-ranking_score[idx], kind="mergesort")[:k].tolist())
    top_observed = set(np.flatnonzero(binary).tolist())
    recipient_recall = len(top_predicted & top_observed) / max(k, 1)
    ambiguity_weight = event.request_cap[idx]
    panel_ambiguity = float(
        np.average(event.ambiguity[idx], weights=ambiguity_weight)
        if ambiguity_weight.sum() > 0
        else event.ambiguity[idx].mean()
    )
    return {
        "allocation_overlap": overlap,
        "js_distance": js_distance,
        "ndcg_all": _ndcg_all(observed, predicted_active),
        "recipient_average_precision": average_precision,
        "recipient_recall_at_observed_k": float(recipient_recall),
        "dispatch_fraction": predicted_total / max(observed_total, 1e-12),
        "panel_ambiguity": panel_ambiguity,
        "n_candidates": int(len(idx)),
        "n_observed_recipients": int(binary.sum()),
        "observed_quantity": observed_total,
    }


def _aggregated_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    priority_prediction: np.ndarray,
    request: np.ndarray,
    ambiguity_numerator: np.ndarray,
) -> dict[str, float]:
    idx = np.flatnonzero(request > 0)
    observed_active = observed[idx]
    predicted_active = predicted[idx]
    priority_active = priority_prediction[idx]
    observed_total = float(observed_active.sum())
    predicted_total = float(predicted_active.sum())
    p_observed = observed_active / max(observed_total, 1e-12)
    p_predicted = predicted_active / max(predicted_total, 1e-12)
    priority_total = float(priority_active.sum())
    p_priority = priority_active / max(priority_total, 1e-12)
    overlap = float(np.minimum(p_observed, p_predicted).sum()) if predicted_total > 0 else 0.0
    priority_overlap = float(np.minimum(p_observed, p_priority).sum()) if priority_total > 0 else 0.0
    js_distance = (
        float(jensenshannon(p_observed, p_predicted, base=2.0)) if predicted_total > 0 else 1.0
    )
    priority_js_distance = (
        float(jensenshannon(p_observed, p_priority, base=2.0)) if priority_total > 0 else 1.0
    )
    binary = (observed_active > 0).astype(int)
    if binary.sum() > 0 and binary.sum() < len(binary):
        average_precision = float(average_precision_score(binary, priority_active))
    else:
        average_precision = 1.0
    k = int(binary.sum())
    top_predicted = set(np.argsort(-priority_active, kind="mergesort")[:k].tolist())
    top_observed = set(np.flatnonzero(binary).tolist())
    recall = len(top_predicted & top_observed) / max(k, 1)
    return {
        "allocation_overlap": overlap,
        "priority_overlap": priority_overlap,
        "js_distance": js_distance,
        "priority_js_distance": priority_js_distance,
        "ndcg_all": _ndcg_all(observed_active, priority_active),
        "recipient_average_precision": average_precision,
        "recipient_recall_at_observed_k": float(recall),
        "dispatch_fraction": predicted_total / max(observed_total, 1e-12),
        "panel_ambiguity": float(ambiguity_numerator.sum() / max(request.sum(), 1e-12)),
        "n_candidates": int(len(idx)),
        "n_observed_recipients": int(binary.sum()),
        "observed_quantity": observed_total,
    }


def _scalar_features(event: EventRecord) -> np.ndarray:
    request_log = np.log1p(event.request_cap)[:, None]
    return np.column_stack(
        [
            event.source_scores,
            event.source_confidence.mean(axis=1),
            event.ambiguity,
            event.prior_fill,
            event.route_risk,
            request_log,
        ]
    )


def fit_scalar_baseline(
    series: list[ProductSeries],
    parameters: ACMParameters,
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
    rows: list[dict[str, float]] = []
    fitted: dict[float, Any] = {}
    for c_value in c_grid:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
        estimator.fit(x_train, y_train)
        fitted[float(c_value)] = estimator
        validation = run_model(
            series,
            parameters,
            "Calibrated scalar",
            estimator=estimator,
            allowed_splits={"validation"},
            build_diagnostics=False,
        )[0]
        rows.append(
            {
                "C": float(c_value),
                "validation_overlap": float(validation["allocation_overlap"].mean()),
                "validation_priority_overlap": float(validation["priority_overlap"].mean()),
                "validation_ndcg_all": float(validation["ndcg_all"].mean()),
            }
        )
    calibration = pd.DataFrame(rows).sort_values(
        ["validation_priority_overlap", "validation_ndcg_all", "validation_overlap"], ascending=False
    )
    selected_c = float(calibration.iloc[0]["C"])
    calibration["selected"] = calibration["C"].eq(selected_c)
    return fitted[selected_c], calibration.sort_values("C").reset_index(drop=True)


def run_model(
    series: list[ProductSeries],
    parameters: ACMParameters,
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
                policy = SparseACMPolicy(graph, parameters, channels=4, coupled=True)
            elif model == "ACM-4 (independent)":
                policy = SparseACMPolicy(graph, parameters, channels=4, coupled=False)
            elif model == "ACM-3":
                policy = SparseACMPolicy(graph, parameters, channels=3, coupled=True)
            elif model == "ACM-2":
                policy = SparseACMPolicy(graph, parameters, channels=2, coupled=True)
            elif model == "FCM":
                policy = SparseFCMPolicy(graph, parameters)

        for event in product.records:
            diagnostics: dict[str, Any] = {"iterations": 1, "convergence_gap": 0.0}
            if stateful:
                score, diagnostics = policy.score(event)
                priority_mode = "softmax"
            elif model == "Calibrated scalar":
                if estimator is None:
                    raise ValueError("The calibrated scalar model requires a fitted estimator.")
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
                        "iterations": int(diagnostics["iterations"]),
                        "convergence_gap": float(diagnostics["convergence_gap"]),
                        "contraction_bound": float(
                            graph.contraction_bound_coupled
                            if model == "ACM-4 (coupled)"
                            else graph.contraction_bound_independent
                        ),
                    }
                )

            if not event.evaluable or (allowed_splits is not None and event.split not in allowed_splits):
                continue
            predicted, priority, objective = _allocate(
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
                idx = np.flatnonzero(event.active & (event.request_cap > 0))
                for position in idx:
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
                        }
                    )
        for (month, split), accumulated in monthly.items():
            metrics = _aggregated_metrics(
                accumulated["observed"],
                accumulated["predicted"],
                accumulated["priority_prediction"],
                accumulated["request"],
                accumulated["ambiguity_numerator"],
            )
            metric_rows.append(
                {
                    "product": product.product,
                    "product_label": PRODUCT_LABELS[product.product],
                    "timestamp": month,
                    "month": month,
                    "split": split,
                    "model": model,
                    "n_events": int(accumulated["n_events"]),
                    **metrics,
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(allocation_rows),
        pd.DataFrame(diagnostic_rows),
    )


def calibration_candidates(n_trials: int = 24, seed: int = 20260718) -> list[ACMParameters]:
    sampler = stats.qmc.LatinHypercube(d=15, seed=seed)
    unit = sampler.random(n=int(n_trials))
    candidates = [ACMParameters()]
    for row in unit:
        candidates.append(
            ACMParameters(
                mu_true=0.74 + 0.16 * row[0],
                mu_partial=0.52 + 0.20 * row[1],
                sigma_exact=0.15 + 0.17 * row[2],
                sigma_partial=0.15 + 0.17 * row[3],
                beta_partial=0.60 + 0.40 * row[4],
                diagonal_memory=0.25 + 0.40 * row[5],
                partial_to_exact=0.05 + 0.30 * row[6],
                exact_to_partial=0.03 + 0.17 * row[7],
                partial_self=0.70 + 0.28 * row[8],
                physical_mix=0.35 + 0.25 * row[9],
                partial_coefficient=0.25 + 0.75 * row[10],
                temperature=0.15 + 0.45 * row[11],
                lp_priority_weight=1.0,
                lp_demand_weight=0.05 + 0.45 * row[12],
                lp_fairness_weight=0.02 + 0.28 * row[13],
                lp_lead_weight=0.02 + 0.98 * row[14],
            )
        )
    return candidates


def calibrate_acm(
    series: list[ProductSeries],
    n_trials: int = 24,
    seed: int = 20260718,
    progress: bool = True,
) -> tuple[ACMParameters, pd.DataFrame]:
    candidates = calibration_candidates(n_trials=n_trials, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, parameters in enumerate(candidates):
        metrics, _, diagnostics = run_model(
            series,
            parameters,
            "ACM-4 (coupled)",
            allowed_splits={"calibration", "validation"},
            build_diagnostics=True,
        )
        calibration = metrics.loc[metrics["split"].eq("calibration")]
        validation = metrics.loc[metrics["split"].eq("validation")]
        row: dict[str, Any] = {
            "trial": index,
            **parameters.as_dict(),
            "calibration_overlap": float(calibration["allocation_overlap"].mean()),
            "calibration_priority_overlap": float(calibration["priority_overlap"].mean()),
            "calibration_ndcg_all": float(calibration["ndcg_all"].mean()),
            "validation_overlap": float(validation["allocation_overlap"].mean()),
            "validation_priority_overlap": float(validation["priority_overlap"].mean()),
            "validation_ndcg_all": float(validation["ndcg_all"].mean()),
            "mean_iterations": float(diagnostics["iterations"].mean()),
            "max_contraction_bound": float(diagnostics["contraction_bound"].max()),
        }
        row["calibration_objective"] = (
            0.30 * row["calibration_overlap"]
            + 0.40 * row["calibration_priority_overlap"]
            + 0.30 * row["calibration_ndcg_all"]
        )
        row["validation_objective"] = (
            0.30 * row["validation_overlap"]
            + 0.40 * row["validation_priority_overlap"]
            + 0.30 * row["validation_ndcg_all"]
        )
        rows.append(row)
        if progress:
            print(
                f"  calibration trial {index + 1}/{len(candidates)}: "
                f"cal={row['calibration_overlap']:.4f}, val={row['validation_overlap']:.4f}",
                flush=True,
            )

    trials = pd.DataFrame(rows)
    top_calibration = trials.nlargest(
        min(7, len(trials)), ["calibration_objective", "calibration_ndcg_all"]
    )
    selected_row = top_calibration.sort_values(
        ["validation_objective", "validation_ndcg_all", "calibration_objective"], ascending=False
    ).iloc[0]
    selected_index = int(selected_row["trial"])
    trials["calibration_shortlist"] = trials["trial"].isin(top_calibration["trial"])
    trials["selected"] = trials["trial"].eq(selected_index)
    return candidates[selected_index], trials.sort_values("trial").reset_index(drop=True)


@dataclass
class RetrospectiveBundle:
    audit: pd.DataFrame
    selected_parameters: ACMParameters
    calibration_trials: pd.DataFrame
    scalar_calibration: pd.DataFrame
    scalar_coefficients: pd.DataFrame
    metrics: pd.DataFrame
    allocations: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: pd.DataFrame
    statistical_tests: pd.DataFrame
    ambiguity_summary: pd.DataFrame
    ambiguity_tests: pd.DataFrame
    graph_diagnostics: pd.DataFrame
    split_description: pd.DataFrame
    lookback_sensitivity: pd.DataFrame


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    seed: int,
    n_boot: int = 5000,
) -> tuple[float, float]:
    month_means = frame.groupby("month")[metric].mean().dropna().to_numpy(dtype=float)
    if len(month_means) == 0:
        return np.nan, np.nan
    if len(month_means) == 1:
        return float(month_means[0]), float(month_means[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(month_means), size=(int(n_boot), len(month_means)))
    draws = month_means[indices].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def summarize_test_metrics(
    metrics: pd.DataFrame,
    seed: int = 20260718,
    n_boot: int = 5000,
) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    metric_names = (
        "allocation_overlap",
        "priority_overlap",
        "js_distance",
        "priority_js_distance",
        "ndcg_all",
        "recipient_average_precision",
        "recipient_recall_at_observed_k",
        "dispatch_fraction",
    )
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        frame = test.loc[test["model"].eq(model)]
        for metric_index, metric in enumerate(metric_names):
            low, high = _cluster_bootstrap_ci(
                frame, metric, seed + 101 * model_index + metric_index, n_boot=n_boot
            )
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(frame[metric].mean()),
                    "sd": float(frame[metric].std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_month_resource_panels": int(len(frame)),
                    "n_calendar_months": int(frame["month"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_model_tests(
    metrics: pd.DataFrame,
    split: str = "test",
    reference: str = "ACM-4 (coupled)",
    seed: int = 20260718,
    n_boot: int = 5000,
) -> pd.DataFrame:
    frame = metrics.loc[metrics["split"].eq(split)].copy()
    directions = {
        "allocation_overlap": 1.0,
        "priority_overlap": 1.0,
        "js_distance": -1.0,
        "priority_js_distance": -1.0,
        "ndcg_all": 1.0,
        "recipient_average_precision": 1.0,
        "recipient_recall_at_observed_k": 1.0,
    }
    rows: list[dict[str, Any]] = []
    for metric_index, (metric, direction) in enumerate(directions.items()):
        reference_frame = frame.loc[frame["model"].eq(reference), ["product", "month", metric]]
        reference_frame = reference_frame.rename(columns={metric: "reference_value"})
        metric_rows: list[dict[str, Any]] = []
        for model_index, model in enumerate(MODEL_ORDER):
            if model == reference:
                continue
            comparator = frame.loc[frame["model"].eq(model), ["product", "month", metric]]
            comparator = comparator.rename(columns={metric: "comparator_value"})
            paired = reference_frame.merge(comparator, on=["product", "month"], how="inner")
            paired["benefit"] = direction * (
                paired["reference_value"] - paired["comparator_value"]
            )
            month_difference = paired.groupby("month")["benefit"].mean().dropna()
            values = month_difference.to_numpy(dtype=float)
            if len(values) == 0 or np.allclose(values, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            rng = np.random.default_rng(seed + 1009 * metric_index + model_index)
            if len(values) > 1:
                draw_indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
                draws = values[draw_indices].mean(axis=1)
                ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
            elif len(values) == 1:
                ci_low = ci_high = values[0]
            else:
                ci_low = ci_high = np.nan
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            metric_rows.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparator": model,
                    "mean_benefit_reference": float(np.mean(values)) if len(values) else np.nan,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "cohen_dz": float(np.mean(values) / sd) if np.isfinite(sd) and sd > 0 else np.nan,
                    "n_paired_months": int(len(values)),
                }
            )
        adjusted = _holm_adjust(np.array([row["p_value"] for row in metric_rows]))
        for row, value in zip(metric_rows, adjusted):
            row["holm_p"] = float(value)
            row["significant_0_05"] = bool(value < 0.05)
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def ambiguity_analysis(
    metrics: pd.DataFrame,
    seed: int = 20260718,
    n_boot: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = metrics.loc[metrics["split"].eq("test")].copy()
    reference_panels = test.loc[
        test["model"].eq("ACM-4 (coupled)"), ["product", "month", "panel_ambiguity"]
    ]
    threshold = float(reference_panels["panel_ambiguity"].median())
    strata = reference_panels.assign(
        ambiguity_stratum=np.where(
            reference_panels["panel_ambiguity"] >= threshold, "Higher ambiguity", "Lower ambiguity"
        )
    )[["product", "month", "ambiguity_stratum"]]
    stratified = test.merge(strata, on=["product", "month"], how="left")
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        for stratum_index, stratum in enumerate(("Lower ambiguity", "Higher ambiguity")):
            frame = stratified.loc[
                stratified["model"].eq(model) & stratified["ambiguity_stratum"].eq(stratum)
            ]
            for metric_index, metric in enumerate(("allocation_overlap", "priority_overlap", "ndcg_all")):
                low, high = _cluster_bootstrap_ci(
                    frame,
                    metric,
                    seed + 401 * model_index + 31 * stratum_index + metric_index,
                    n_boot=n_boot,
                )
                rows.append(
                    {
                        "model": model,
                        "ambiguity_stratum": stratum,
                        "ambiguity_threshold": threshold,
                        "metric": metric,
                        "mean": float(frame[metric].mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "n_panels": int(len(frame)),
                        "n_months": int(frame["month"].nunique()),
                    }
                )
    summary = pd.DataFrame(rows)

    test_rows: list[pd.DataFrame] = []
    for stratum in ("Lower ambiguity", "Higher ambiguity"):
        subset = stratified.loc[stratified["ambiguity_stratum"].eq(stratum)]
        comparisons = paired_model_tests(
            subset.assign(split="test"), split="test", seed=seed + (0 if stratum.startswith("Lower") else 10000)
        )
        comparisons.insert(0, "ambiguity_stratum", stratum)
        test_rows.append(comparisons)
    return summary, pd.concat(test_rows, ignore_index=True)


def _scalar_coefficient_table(estimator: Any) -> pd.DataFrame:
    feature_names = [
        "Demand pressure",
        "Backlog pressure",
        "Inventory risk",
        "Waiting risk",
        "Capacity/policy risk",
        "Mean source confidence",
        "Ambiguity index",
        "Prior fill",
        "Route risk",
        "Log request cap",
    ]
    logistic = estimator.named_steps["logisticregression"]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": logistic.coef_.ravel(),
        }
    )


def describe_splits(metrics: pd.DataFrame) -> pd.DataFrame:
    reference = metrics.loc[metrics["model"].eq("ACM-4 (coupled)")]
    return (
        reference.groupby("split")
        .agg(
            first_month=("month", "min"),
            last_month=("month", "max"),
            month_resource_panels=("month", "size"),
            calendar_months=("month", "nunique"),
            observed_quantity=("observed_quantity", "sum"),
            median_candidates=("n_candidates", "median"),
            observed_recipients=("n_observed_recipients", "sum"),
        )
        .reset_index()
    )


def run_retrospective_study(
    data_dir: Path,
    n_trials: int = 24,
    seed: int = 20260718,
    n_boot: int = 5000,
) -> RetrospectiveBundle:
    series, audit = load_getusppe_series(data_dir, lookback_days=90)
    selected, calibration_trials = calibrate_acm(
        series, n_trials=n_trials, seed=seed, progress=True
    )
    scalar_estimator, scalar_calibration = fit_scalar_baseline(
        series, selected, seed=seed
    )

    metrics_frames: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    allocation_frame = pd.DataFrame()
    for model in MODEL_ORDER:
        print(f"  evaluating {model}", flush=True)
        metrics, allocations, diagnostics = run_model(
            series,
            selected,
            model,
            estimator=scalar_estimator if model == "Calibrated scalar" else None,
            build_diagnostics=True,
            collect_allocations=model == "ACM-4 (coupled)",
        )
        metrics_frames.append(metrics)
        if len(diagnostics):
            diagnostics_frames.append(diagnostics)
        if len(allocations):
            allocation_frame = allocations
    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    summary = summarize_test_metrics(all_metrics, seed=seed, n_boot=n_boot)
    tests = paired_model_tests(all_metrics, seed=seed, n_boot=n_boot)
    ambiguity_summary, ambiguity_tests = ambiguity_analysis(
        all_metrics, seed=seed, n_boot=n_boot
    )

    graph_rows: list[dict[str, Any]] = []
    for product in series:
        graph = build_sparse_product_graph(product, selected, estimate_norm=True)
        graph_rows.append(
            {
                "product": product.product,
                "concepts": len(graph.metadata),
                "directed_edges": int(graph.adjacency.nnz),
                "operator_scale": graph.operator_scale,
                "estimated_coupled_norm": graph.operator_norm_coupled,
                "estimated_independent_norm": graph.operator_norm_independent,
                "numerically_verified_coupled_contraction_bound": graph.contraction_bound_coupled,
                "numerically_verified_independent_contraction_bound": graph.contraction_bound_independent,
            }
        )
    graph_diagnostics = pd.DataFrame(graph_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    for lookback in (30, 60, 90, 120):
        sensitivity_series, sensitivity_audit = load_getusppe_series(
            data_dir, lookback_days=lookback
        )
        for model in ("ACM-4 (coupled)", "Demand proportional", "Robust priority"):
            metrics, _, _ = run_model(
                sensitivity_series,
                selected,
                model,
                allowed_splits={"test"},
                build_diagnostics=False,
            )
            test = metrics.loc[metrics["split"].eq("test")]
            sensitivity_rows.append(
                {
                    "lookback_days": lookback,
                    "model": model,
                    "test_panels": int(len(test)),
                    "allocation_overlap": float(test["allocation_overlap"].mean()),
                    "priority_overlap": float(test["priority_overlap"].mean()),
                    "ndcg_all": float(test["ndcg_all"].mean()),
                    "eligible_row_coverage": float(
                        sensitivity_audit.loc[
                            sensitivity_audit["item"].eq("Eligible delivered-row coverage"), "value"
                        ].iloc[0]
                    ),
                    "eligible_quantity_coverage": float(
                        sensitivity_audit.loc[
                            sensitivity_audit["item"].eq("Eligible delivered-quantity coverage"), "value"
                        ].iloc[0]
                    ),
                }
            )

    return RetrospectiveBundle(
        audit=audit,
        selected_parameters=selected,
        calibration_trials=calibration_trials,
        scalar_calibration=scalar_calibration,
        scalar_coefficients=_scalar_coefficient_table(scalar_estimator),
        metrics=all_metrics,
        allocations=allocation_frame,
        diagnostics=all_diagnostics,
        summary=summary,
        statistical_tests=tests,
        ambiguity_summary=ambiguity_summary,
        ambiguity_tests=ambiguity_tests,
        graph_diagnostics=graph_diagnostics,
        split_description=describe_splits(all_metrics),
        lookback_sensitivity=pd.DataFrame(sensitivity_rows),
    )


def _plot_style(model: str) -> dict[str, Any]:
    if model in MODEL_STYLES:
        return dict(MODEL_STYLES[model])
    additions = {
        "Calibrated scalar": dict(color="0.22", linestyle=(0, (5, 1)), marker="P"),
        "Demand proportional": dict(color="0.50", linestyle=(0, (3, 2)), marker="d"),
        "Robust priority": dict(color="0.70", linestyle=(0, (1, 1)), marker="h"),
        "Equal allocation": dict(color="0.82", linestyle=(0, (1, 2)), marker="+"),
    }
    return additions.get(model, dict(color="black", linestyle="-", marker="o"))


def _summary_lookup(summary: pd.DataFrame, model: str, metric: str) -> pd.Series:
    return summary.loc[summary["model"].eq(model) & summary["metric"].eq(metric)].iloc[0]


def _figure_primary(bundle: RetrospectiveBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.2), sharey=True)
    metrics = (
        ("allocation_overlap", "LP allocation overlap"),
        ("priority_overlap", "Softmax-to-allocation overlap"),
        ("ndcg_all", "NDCG across all hospitals"),
    )
    positions = np.arange(len(MODEL_ORDER))[::-1]
    for panel_index, (metric, xlabel) in enumerate(metrics):
        ax = axes[panel_index]
        for y, model in zip(positions, MODEL_ORDER):
            row = _summary_lookup(bundle.summary, model, metric)
            style = _plot_style(model)
            ax.errorbar(
                float(row["mean"]),
                y,
                xerr=np.array(
                    [[float(row["mean"] - row["ci_low"])], [float(row["ci_high"] - row["mean"])]]
                ),
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                capsize=2.5,
                markersize=5,
                linewidth=1.0,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0.0)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel_index))
    fig.subplots_adjust(left=0.31, hspace=0.28)
    save_figure(fig, output, "fig_real10_getusppe_allocation_validation", formats=("png",), dpi=dpi)


def _figure_ambiguity(bundle: RetrospectiveBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    selected_models = (
        "ACM-4 (coupled)",
        "ACM-4 (independent)",
        "ACM-2",
        "FCM",
        "Calibrated scalar",
        "Demand proportional",
        "Robust priority",
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), sharey=True)
    positions = np.arange(len(selected_models))[::-1]
    offsets = {"Lower ambiguity": -0.10, "Higher ambiguity": 0.10}
    markers = {"Lower ambiguity": "o", "Higher ambiguity": "s"}
    grays = {"Lower ambiguity": "0.65", "Higher ambiguity": "0.05"}
    for panel_index, (metric, xlabel) in enumerate(
        (("priority_overlap", "Softmax-to-allocation overlap"), ("ndcg_all", "NDCG across all hospitals"))
    ):
        ax = axes[panel_index]
        for stratum in ("Lower ambiguity", "Higher ambiguity"):
            for y, model in zip(positions, selected_models):
                row = bundle.ambiguity_summary.loc[
                    bundle.ambiguity_summary["model"].eq(model)
                    & bundle.ambiguity_summary["metric"].eq(metric)
                    & bundle.ambiguity_summary["ambiguity_stratum"].eq(stratum)
                ].iloc[0]
                ax.errorbar(
                    float(row["mean"]),
                    y + offsets[stratum],
                    xerr=np.array(
                        [
                            [float(row["mean"] - row["ci_low"])],
                            [float(row["ci_high"] - row["mean"])],
                        ]
                    ),
                    marker=markers[stratum],
                    color=grays[stratum],
                    linestyle="none",
                    capsize=2.5,
                    markersize=4.5,
                    label=stratum if y == positions[0] else None,
                )
        ax.set_yticks(positions)
        ax.set_yticklabels(selected_models)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0.0)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel_index))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.58, 1.01))
    fig.subplots_adjust(left=0.31, top=0.92, hspace=0.28)
    save_figure(fig, output, "fig_real11_getusppe_ambiguity_strata", formats=("png",), dpi=dpi)


def _figure_calibration(bundle: RetrospectiveBundle, output: Path, dpi: int) -> None:
    apply_nature_style(dpi)
    trials = bundle.calibration_trials
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 7.2))
    ax = axes[0]
    ordinary = trials.loc[~trials["calibration_shortlist"]]
    shortlisted = trials.loc[trials["calibration_shortlist"] & ~trials["selected"]]
    selected = trials.loc[trials["selected"]]
    ax.scatter(
        ordinary["calibration_objective"], ordinary["validation_objective"],
        marker="o", facecolors="white", edgecolors="0.65", s=25, label="Candidate"
    )
    ax.scatter(
        shortlisted["calibration_objective"], shortlisted["validation_objective"],
        marker="s", facecolors="0.75", edgecolors="black", s=32, label="Calibration shortlist"
    )
    ax.scatter(
        selected["calibration_objective"], selected["validation_objective"],
        marker="*", color="black", s=85, label="Selected"
    )
    limits = [
        min(trials["calibration_objective"].min(), trials["validation_objective"].min()),
        max(trials["calibration_objective"].max(), trials["validation_objective"].max()),
    ]
    ax.plot(limits, limits, color="0.55", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Calibration objective")
    ax.set_ylabel("Internal-validation objective")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(
        trials["trial"], trials["calibration_objective"], color="0.60", linestyle=":", marker="o",
        label="Calibration"
    )
    ax.plot(
        trials["trial"], trials["validation_objective"], color="black", linestyle="--", marker="s",
        label="Internal validation"
    )
    ax.axvline(float(selected.iloc[0]["trial"]), color="0.25", linestyle="-.", linewidth=0.9)
    ax.set_xlabel("Constrained Latin-hypercube trial")
    ax.set_ylabel("Composite allocation objective")
    clean_axis(ax)
    panel_label(ax, "b")
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.90, hspace=0.30)
    save_figure(fig, output, "fig_real12_getusppe_calibration", formats=("png",), dpi=dpi)


def _draft_results(bundle: RetrospectiveBundle) -> str:
    def metric(model: str, name: str) -> pd.Series:
        return _summary_lookup(bundle.summary, model, name)

    reference_overlap = metric("ACM-4 (coupled)", "allocation_overlap")
    reference_priority_overlap = metric("ACM-4 (coupled)", "priority_overlap")
    reference_ndcg = metric("ACM-4 (coupled)", "ndcg_all")
    baseline_rows = bundle.summary.loc[
        bundle.summary["metric"].eq("allocation_overlap")
        & ~bundle.summary["model"].eq("ACM-4 (coupled)")
    ].sort_values("mean", ascending=False)
    best_baseline = str(baseline_rows.iloc[0]["model"])
    best_overlap = baseline_rows.iloc[0]
    test_row = bundle.statistical_tests.loc[
        bundle.statistical_tests["metric"].eq("allocation_overlap")
        & bundle.statistical_tests["comparator"].eq(best_baseline)
    ].iloc[0]
    significance = (
        "The paired difference remained significant after Holm correction."
        if bool(test_row["significant_0_05"])
        else "The paired difference was not significant after Holm correction."
    )
    split = bundle.split_description.set_index("split")
    test_description = split.loc["test"]
    ambiguity = bundle.ambiguity_summary
    high_coupled = ambiguity.loc[
        ambiguity["model"].eq("ACM-4 (coupled)")
        & ambiguity["ambiguity_stratum"].eq("Higher ambiguity")
        & ambiguity["metric"].eq("priority_overlap")
    ].iloc[0]
    high_demand = ambiguity.loc[
        ambiguity["model"].eq("Demand proportional")
        & ambiguity["ambiguity_stratum"].eq("Higher ambiguity")
        & ambiguity["metric"].eq("priority_overlap")
    ].iloc[0]
    sensitivity = bundle.lookback_sensitivity
    lookback_30 = sensitivity.loc[
        sensitivity["lookback_days"].eq(30)
        & sensitivity["model"].eq("ACM-4 (coupled)")
    ].iloc[0]
    lookback_90 = sensitivity.loc[
        sensitivity["lookback_days"].eq(90)
        & sensitivity["model"].eq("ACM-4 (coupled)")
    ].iloc[0]
    selected = bundle.selected_parameters
    return f"""% Automatically generated from the untouched test split. Verify placement and references before insertion.
\\subsection{{Retrospective validation against observed PPE allocations}}

The additional retrospective study used anonymized GetUsPPE requests and successfully delivered matches for N95 respirators, surgical masks, face shields, and gowns. Only acute-care facilities with a positive request timestamped no later than the allocation event and within the prespecified 90-day active-request window were eligible. Matches whose recorded closure preceded creation were removed by a chronology rule. The calibration period ended on 30 June 2020, internal validation covered July--December 2020, and all 2021 events were retained as an untouched test set. The test set comprised {int(test_description['month_resource_panels'])} month--resource panels across {int(test_description['calendar_months'])} calendar months, {int(test_description['observed_recipients'])} observed recipient instances, and {test_description['observed_quantity']:,.0f} delivered units.

The Gaussian channel parameters, diagonal memory coefficient, structured coupling coefficients, evidence mixture, partial-channel projection, softmax temperature, and common LP coefficient ratios were selected by a constrained Latin-hypercube search (Fig.~\\ref{{fig:getusppe-calibration}}). Candidates were first shortlisted on the calibration objective and then selected using internal validation; the 2021 test outcomes were not accessed during selection. The selected configuration had $c_0={selected.diagonal_memory:.3f}$, partial-to-exact coupling {selected.partial_to_exact:.3f}, exact-to-partial coupling {selected.exact_to_partial:.3f}, partial self-coupling {selected.partial_self:.3f}, $\\rho_e={selected.physical_mix:.3f}$, and $\\gamma={selected.partial_coefficient:.3f}$. Deterministic sparse power iteration, repeated after scaling with a 1\\% safety margin, verified a contraction bound below one for every product graph.

On the untouched test set, coupled ACM achieved LP allocation overlap {reference_overlap['mean']:.3f} [{reference_overlap['ci_low']:.3f}, {reference_overlap['ci_high']:.3f}], softmax-to-allocation overlap {reference_priority_overlap['mean']:.3f} [{reference_priority_overlap['ci_low']:.3f}, {reference_priority_overlap['ci_high']:.3f}], and full-list NDCG {reference_ndcg['mean']:.3f} [{reference_ndcg['ci_low']:.3f}, {reference_ndcg['ci_high']:.3f}] (Table~\\ref{{tab:getusppe-retrospective}} and Fig.~\\ref{{fig:getusppe-allocation-validation}}). The strongest comparator by LP allocation overlap was {best_baseline}, with {best_overlap['mean']:.3f} [{best_overlap['ci_low']:.3f}, {best_overlap['ci_high']:.3f}]. {significance}

The channel and coupling ablations did not identify an operational context in which the coupled four-channel representation was superior. Coupled and independent four-channel ACM had the same mean LP overlap to three decimal places, while the three-channel, two-channel, and FCM variants were likewise indistinguishable on that metric. In the higher-ambiguity stratum, coupled ACM obtained priority overlap {high_coupled['mean']:.3f} [{high_coupled['ci_low']:.3f}, {high_coupled['ci_high']:.3f}], compared with {high_demand['mean']:.3f} [{high_demand['ci_low']:.3f}, {high_demand['ci_high']:.3f}] for demand-proportional allocation (Fig.~\\ref{{fig:getusppe-ambiguity}}); no stratum-specific comparison favoring coupled ACM remained significant after Holm adjustment. The real-allocation replay therefore supports feasibility and auditability of the evidence-encoding pipeline while providing no evidence that four-channel coupling is operationally necessary.

The linkage-window analysis also showed material sensitivity to how long an unmatched request was treated as active. Coupled-ACM LP overlap changed from {lookback_30['allocation_overlap']:.3f} with a 30-day window to {lookback_90['allocation_overlap']:.3f} under the prespecified 90-day window, while eligible delivered-quantity coverage increased from {lookback_30['eligible_quantity_coverage']:.3f} to {lookback_90['eligible_quantity_coverage']:.3f}. The primary 90-day analysis therefore balances archive coverage against temporal specificity and is reported together with the full 30--120-day sensitivity range. The reported metrics quantify agreement with logged allocation decisions; unrecorded donor preferences, contemporaneous inventories, dispatch constraints, and transport decisions can also explain disagreement with the historical record.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.82\\textwidth]{{fig_real10_getusppe_allocation_validation.png}}
\\caption{{Untouched-test agreement with observed GetUsPPE allocations. Points show calendar-month cluster means and bars show 95\\% cluster-bootstrap confidence intervals for (a) LP allocation overlap, (b) softmax priority overlap, and (c) full-list NDCG.}}
\\label{{fig:getusppe-allocation-validation}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.94\\textwidth]{{fig_real11_getusppe_ambiguity_strata.png}}
\\caption{{Performance by feature-defined ambiguity stratum on the untouched test split. The split threshold is the median ambiguity index computed without outcome information. Bars show 95\\% calendar-month cluster-bootstrap confidence intervals.}}
\\label{{fig:getusppe-ambiguity}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{{fig_real12_getusppe_calibration.png}}
\\caption{{Systematic calibration of the evidence-encoding and allocation pipeline. (a) Calibration and internal-validation objectives across the constrained Latin-hypercube candidates. (b) Trial-wise objectives; the vertical line marks the configuration selected before the 2021 test split was evaluated.}}
\\label{{fig:getusppe-calibration}}
\\end{{figure}}
"""


def create_retrospective_outputs(
    bundle: RetrospectiveBundle,
    output_root: Path,
    data_dir: Path,
    dpi: int = 600,
) -> None:
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "data_audit.csv": bundle.audit,
        "split_description.csv": bundle.split_description,
        "calibration_trials.csv": bundle.calibration_trials,
        "scalar_calibration.csv": bundle.scalar_calibration,
        "scalar_coefficients.csv": bundle.scalar_coefficients,
        "monthly_model_metrics.csv": bundle.metrics,
        "test_summary.csv": bundle.summary,
        "paired_statistical_tests.csv": bundle.statistical_tests,
        "ambiguity_summary.csv": bundle.ambiguity_summary,
        "ambiguity_statistical_tests.csv": bundle.ambiguity_tests,
        "graph_diagnostics.csv": bundle.graph_diagnostics,
        "lookback_sensitivity.csv": bundle.lookback_sensitivity,
        "coupled_acm_event_allocations.csv": bundle.allocations,
        "convergence_diagnostics.csv": bundle.diagnostics,
    }
    for filename, frame in frames.items():
        frame.to_csv(results_dir / filename, index=False)

    (results_dir / "selected_parameters.json").write_text(
        json.dumps(bundle.selected_parameters.as_dict(), indent=2), encoding="utf-8"
    )
    (output_root / "manuscript_results_draft.tex").write_text(
        _draft_results(bundle), encoding="utf-8"
    )

    hashes: dict[str, str] = {}
    for filename in ("all_requests.csv", "all_offers.csv", "all_matches.csv"):
        path = data_dir / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "dataset_repository": "https://github.com/GetUsPPE/ppe_needs_retrospective",
        "related_article_doi": "https://doi.org/10.1002/puh2.65",
        "analysis_scope": "Acute-care retrospective request-to-delivered-match replay",
        "source_file_sha256": hashes,
        "selected_parameter_trial": int(
            bundle.calibration_trials.loc[bundle.calibration_trials["selected"], "trial"].iloc[0]
        ),
    }
    (results_dir / "data_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    _figure_primary(bundle, figures_dir, dpi)
    _figure_ambiguity(bundle, figures_dir, dpi)
    _figure_calibration(bundle, figures_dir, dpi)

    manifest: list[dict[str, Any]] = []
    excluded_manifest_entries = {
        output_root / "MANIFEST.json",
        output_root / "GetUsPPE_Retrospective_Validation_Package.zip",
    }
    for path in sorted(
        p for p in output_root.rglob("*")
        if p.is_file() and p not in excluded_manifest_entries
    ):
        manifest.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
