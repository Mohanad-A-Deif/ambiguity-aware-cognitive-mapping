from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


NHS_WORKBOOK_URL = (
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2023/07/"
    "Weekly-covid-admissions-and-beds-publication-200429-up-to-210406v2.xlsx"
)
HHS_FACILITY_API = "https://api.delphi.cmu.edu/epidata/covid_hosp_facility/"
HHS_LOOKUP_API = "https://api.delphi.cmu.edu/epidata/covid_hosp_facility_lookup/"


@dataclass
class RealPanel:
    source: str
    dates: pd.DatetimeIndex
    node_ids: list[str]
    node_names: list[str]
    region_names: list[str]
    regions: np.ndarray
    resource_names: tuple[str, ...]
    demand: np.ndarray
    capacity: np.ndarray
    admissions_confirmed: np.ndarray
    admissions_suspected: np.ndarray
    reporting_confidence: np.ndarray
    long_data: pd.DataFrame
    selection_table: pd.DataFrame
    source_url: str
    notes: str

    @property
    def n_days(self) -> int:
        return int(self.demand.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.demand.shape[1])

    @property
    def n_resources(self) -> int:
        return int(self.demand.shape[2])


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        return destination
    request = Request(url, headers={"User-Agent": "ACM-real-world-validation/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 1024:
        raise RuntimeError(f"Downloaded payload from {url} is unexpectedly small ({len(payload)} bytes).")
    destination.write_bytes(payload)
    return destination


def _find_excel_header(path: Path, sheet_name: str) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=30)
    for row in range(len(preview)):
        values = {str(v).strip() for v in preview.iloc[row].tolist() if pd.notna(v)}
        if {"Code", "Name"}.issubset(values):
            return row
    raise ValueError(f"Could not locate Code/Name header in NHS sheet {sheet_name!r}.")


def _nhs_sheet_long(path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    header = _find_excel_header(path, sheet_name)
    frame = pd.read_excel(path, sheet_name=sheet_name, header=header)
    frame = frame[frame["Type 1 Acute?"].astype(str).str.strip().eq("Yes")].copy()
    date_columns = [
        col
        for col in frame.columns
        if isinstance(col, (pd.Timestamp, datetime, np.datetime64))
        or (isinstance(col, str) and str(col)[:4].isdigit() and "-" in str(col))
    ]
    if not date_columns:
        raise ValueError(f"No date columns found in NHS sheet {sheet_name!r}.")
    out = frame.melt(
        id_vars=["NHS England Region", "Code", "Name"],
        value_vars=date_columns,
        var_name="date",
        value_name=value_name,
    )
    out = out.rename(columns={"NHS England Region": "region", "Code": "node_id", "Name": "node_name"})
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


def _merge_on_panel(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    merged = frames[0]
    keys = ["date", "region", "node_id", "node_name"]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="outer", validate="one_to_one")
    return merged.sort_values(keys).reset_index(drop=True)


def _complete_node_panel(panel: pd.DataFrame, dates: pd.DatetimeIndex, node_ids: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([dates, node_ids], names=["date", "node_id"])
    base = panel.set_index(["date", "node_id"]).reindex(index).reset_index()
    lookup = panel.drop_duplicates("node_id").set_index("node_id")[["node_name", "region"]]
    base["node_name"] = base["node_id"].map(lookup["node_name"])
    base["region"] = base["node_id"].map(lookup["region"])
    return base


def _fill_numeric_by_node(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.sort_values(["node_id", "date"]).reset_index(drop=True).copy()
    for col in columns:
        out[col] = out.groupby("node_id", sort=False)[col].transform(
            lambda series: series.interpolate(limit_direction="both").ffill().bfill()
        )
        out[col] = out[col].fillna(0.0).clip(lower=0.0)
    return out


def load_nhs_panel(raw_dir: Path, regions: tuple[str, str] = ("London", "Midlands"), nodes_per_region: int = 6) -> RealPanel:
    workbook = _download(NHS_WORKBOOK_URL, raw_dir / "nhs_weekly_covid_admissions_beds_2020_2021.xlsx")
    sheet_map = {
        "Hosp ads & diag": "admissions",
        "All beds COVID": "all_beds_covid",
        "MV beds COVID": "mv_covid",
        "Adult G&A Beds Occupied COVID": "ga_covid",
        "Adult G&A Bed Occupied NonCOVID": "ga_noncovid",
        "Adult G&A Beds Unoccupied": "ga_unoccupied",
        "Adult CC Beds Occupied COVID": "cc_covid",
        "Adult CC Bed Occupied NonCOVID": "cc_noncovid",
        "Adult CC Beds Unoccupied": "cc_unoccupied",
    }
    frames = [_nhs_sheet_long(workbook, sheet, name) for sheet, name in sheet_map.items()]
    panel = _merge_on_panel(frames)
    start = pd.Timestamp("2020-11-17")
    end = pd.Timestamp("2021-04-06")
    panel = panel[panel["date"].between(start, end) & panel["region"].isin(regions)].copy()

    raw_measure_columns = list(sheet_map.values())
    panel["ga_capacity"] = panel[["ga_covid", "ga_noncovid", "ga_unoccupied"]].sum(axis=1, min_count=3)
    panel["cc_capacity"] = panel[["cc_covid", "cc_noncovid", "cc_unoccupied"]].sum(axis=1, min_count=3)
    selection = (
        panel.groupby(["region", "node_id", "node_name"], as_index=False)
        .agg(
            median_ga_capacity=("ga_capacity", "median"),
            median_cc_capacity=("cc_capacity", "median"),
            mean_mv_covid=("mv_covid", "mean"),
            completeness=("cc_capacity", lambda x: float(x.notna().mean())),
        )
    )
    selection["capacity_rank_value"] = selection["median_ga_capacity"].fillna(0) + 4.0 * selection["median_cc_capacity"].fillna(0)
    selected_parts: list[pd.DataFrame] = []
    for region in regions:
        eligible = selection[(selection["region"] == region) & (selection["completeness"] >= 0.95)].copy()
        eligible = eligible.sort_values(["capacity_rank_value", "node_id"], ascending=[False, True]).head(nodes_per_region)
        if len(eligible) != nodes_per_region:
            raise RuntimeError(f"NHS region {region!r} has only {len(eligible)} eligible trusts; expected {nodes_per_region}.")
        selected_parts.append(eligible)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selected_order"] = np.arange(1, len(selected) + 1)
    node_ids = selected["node_id"].tolist()
    dates = pd.date_range(start, end, freq="D")
    panel = _complete_node_panel(panel[panel["node_id"].isin(node_ids)], dates, node_ids)

    for col in raw_measure_columns:
        panel[f"_observed_{col}"] = panel[col].notna().astype(float)
    panel = _fill_numeric_by_node(panel, raw_measure_columns)
    panel["ga_capacity"] = panel["ga_covid"] + panel["ga_noncovid"] + panel["ga_unoccupied"]
    panel["cc_capacity"] = panel["cc_covid"] + panel["cc_noncovid"] + panel["cc_unoccupied"]
    # Mechanical-ventilation occupancy is observed directly; total adult
    # critical-care capacity is used as a conservative capacity envelope.
    panel["mv_capacity_proxy"] = panel["cc_capacity"]

    node_lookup = selected.set_index("node_id")
    node_names = [str(node_lookup.loc[node, "node_name"]) for node in node_ids]
    region_names = list(regions)
    region_map = {name: idx for idx, name in enumerate(region_names)}
    node_regions = np.array([region_map[str(node_lookup.loc[node, "region"])] for node in node_ids], dtype=int)
    resource_names = ("MV support", "Critical care", "G&A beds")
    t, h, r = len(dates), len(node_ids), len(resource_names)
    demand = np.zeros((t, h, r), dtype=float)
    capacity = np.zeros_like(demand)
    confidence = np.zeros_like(demand)
    admissions = np.zeros((t, h), dtype=float)
    node_pos = {node: idx for idx, node in enumerate(node_ids)}
    date_pos = {date: idx for idx, date in enumerate(dates)}
    for _, row in panel.iterrows():
        di = date_pos[pd.Timestamp(row["date"])]
        hi = node_pos[str(row["node_id"])]
        demand[di, hi] = [row["mv_covid"], row["cc_covid"], row["ga_covid"]]
        capacity[di, hi] = [row["mv_capacity_proxy"], row["cc_capacity"], row["ga_capacity"]]
        admissions[di, hi] = row["admissions"]
        confidence[di, hi, 0] = min(row["_observed_mv_covid"], row["_observed_cc_covid"])
        confidence[di, hi, 1] = min(
            row["_observed_cc_covid"],
            row["_observed_cc_noncovid"],
            row["_observed_cc_unoccupied"],
        )
        confidence[di, hi, 2] = min(
            row["_observed_ga_covid"],
            row["_observed_ga_noncovid"],
            row["_observed_ga_unoccupied"],
        )
    capacity = np.maximum(capacity, demand)
    confidence = np.where(confidence > 0, 1.0, 0.35)
    suspected = np.zeros_like(admissions)
    selected = selected[
        [
            "selected_order",
            "region",
            "node_id",
            "node_name",
            "median_ga_capacity",
            "median_cc_capacity",
            "mean_mv_covid",
            "completeness",
        ]
    ]
    return RealPanel(
        source="NHS England COVID-19 Hospital Activity",
        dates=dates,
        node_ids=node_ids,
        node_names=node_names,
        region_names=region_names,
        regions=node_regions,
        resource_names=resource_names,
        demand=demand,
        capacity=capacity,
        admissions_confirmed=admissions,
        admissions_suspected=suspected,
        reporting_confidence=confidence,
        long_data=panel,
        selection_table=selected,
        source_url="https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/",
        notes=(
            "Daily provider-level historical NHS SitRep data. The observed resource series are MV-bed occupancy, "
            "adult critical-care occupancy, and adult general-and-acute bed occupancy. Total critical-care and "
            "G&A capacity are reconstructed from occupied-COVID, occupied-non-COVID, and unoccupied beds."
        ),
    )


def _delphi_api(endpoint: str, params: dict[str, str]) -> list[dict]:
    request = Request(f"{endpoint}?{urlencode(params)}", headers={"User-Agent": "ACM-real-world-validation/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    if int(payload.get("result", -1)) != 1:
        raise RuntimeError(f"Delphi API request failed: {payload.get('message', 'unknown error')}")
    return list(payload["epidata"])


def _download_hhs_subset(raw_dir: Path, states: tuple[str, str], nodes_per_state: int) -> Path:
    """Cache a reproducible HHS subset obtained through CMU's HHS mirror.

    The official healthdata.gov Socrata endpoint is intermittently unavailable.
    CMU Delphi documents this endpoint as a public-domain mirror of the same HHS
    facility dataset.  Hospitals are screened by type, then selected using only
    staffed adult ICU capacity on four prespecified snapshot weeks; COVID outcome
    values never enter selection.
    """
    destination = raw_dir / f"hhs_facility_icu_{'_'.join(states).lower()}_2020_2021_v3.csv"
    if destination.exists() and destination.stat().st_size > 1024:
        return destination
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict] = []
    for state in states:
        metadata_rows.extend(_delphi_api(HHS_LOOKUP_API, {"state": state}))
    metadata = pd.DataFrame(metadata_rows)
    metadata = metadata[
        metadata["state"].isin(states)
        & metadata["hospital_subtype"].astype(str).str.contains("Short Term", case=False, na=False)
    ].copy()
    # The mirror retains a few historical aliases that differ only by a
    # leading zero (for example 050262 and 50262). Prefer the six-character
    # CMS identifier so that each physical facility can be selected once.
    metadata["hospital_pk"] = metadata["hospital_pk"].astype(str)
    metadata["_canonical_pk"] = metadata["hospital_pk"].str.lstrip("0")
    metadata["_pk_length"] = metadata["hospital_pk"].str.len()
    metadata = (
        metadata.sort_values(["state", "_canonical_pk", "_pk_length"], ascending=[True, True, False])
        .drop_duplicates(["state", "_canonical_pk"], keep="first")
        .drop(columns=["_canonical_pk", "_pk_length"])
    )
    snapshot_weeks = "20201204,20210205,20210402,20210604"
    snapshots: list[dict] = []
    hospital_pks = metadata["hospital_pk"].astype(str).drop_duplicates().tolist()
    for offset in range(0, len(hospital_pks), 80):
        snapshots.extend(
            _delphi_api(
                HHS_FACILITY_API,
                {
                    "hospital_pks": ",".join(hospital_pks[offset : offset + 80]),
                    "collection_weeks": snapshot_weeks,
                },
            )
        )
    snapshots_frame = pd.DataFrame(snapshots)
    capacity_col = "total_staffed_adult_icu_beds_7_day_avg"
    snapshots_frame[capacity_col] = pd.to_numeric(snapshots_frame[capacity_col], errors="coerce")
    snapshots_frame.loc[snapshots_frame[capacity_col] <= -999000, capacity_col] = np.nan
    screening = (
        snapshots_frame.groupby(["state", "hospital_pk"], as_index=False)
        .agg(
            hospital_name=("hospital_name", "last"),
            screening_median_icu_capacity=(capacity_col, "median"),
            screening_completeness=(capacity_col, lambda x: float(x.notna().mean())),
        )
    )
    selected_parts: list[pd.DataFrame] = []
    for state in states:
        eligible = screening[
            (screening["state"] == state) & (screening["screening_completeness"] >= 0.75)
        ].copy()
        eligible = eligible.sort_values(
            ["screening_median_icu_capacity", "hospital_pk"], ascending=[False, True]
        ).head(nodes_per_state)
        if len(eligible) != nodes_per_state:
            raise RuntimeError(
                f"HHS state {state!r} has only {len(eligible)} eligible hospitals; expected {nodes_per_state}."
            )
        selected_parts.append(eligible)
    selected_ids = pd.concat(selected_parts, ignore_index=True)["hospital_pk"].astype(str).tolist()
    rows = _delphi_api(
        HHS_FACILITY_API,
        {
            "hospital_pks": ",".join(selected_ids),
            "collection_weeks": "20201106-20210730",
        },
    )
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def load_hhs_panel(raw_dir: Path, states: tuple[str, str] = ("CA", "NY"), nodes_per_state: int = 6) -> RealPanel:
    csv_path = _download_hhs_subset(raw_dir, states, nodes_per_state)
    data = pd.read_csv(csv_path, dtype={"hospital_pk": str, "ccn": str, "collection_week": str})
    data["collection_week"] = pd.to_datetime(data["collection_week"], format="%Y%m%d")
    numeric = [col for col in data.columns if col not in {"hospital_pk", "collection_week", "state", "hospital_name"}]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data.loc[data[col] <= -999000, col] = np.nan
    data = data[data["state"].isin(states)].copy()
    capacity_col = "total_staffed_adult_icu_beds_7_day_avg"
    occ_col = "staffed_adult_icu_bed_occupancy_7_day_avg"
    covid_col = "staffed_icu_adult_patients_confirmed_covid_7_day_avg"
    confirmed_col = "previous_day_admission_adult_covid_confirmed_7_day_sum"
    suspected_col = "previous_day_admission_adult_covid_suspected_7_day_sum"
    selection = (
        data.groupby(["state", "hospital_pk"], as_index=False)
        .agg(
            hospital_name=("hospital_name", "last"),
            median_icu_capacity=(capacity_col, "median"),
            mean_covid_icu=(covid_col, "mean"),
            completeness=(capacity_col, lambda x: float(x.notna().mean())),
        )
    )
    selected_parts: list[pd.DataFrame] = []
    for state in states:
        eligible = selection[(selection["state"] == state) & (selection["completeness"] >= 0.80)].copy()
        eligible = eligible.sort_values(["median_icu_capacity", "hospital_pk"], ascending=[False, True]).head(nodes_per_state)
        if len(eligible) != nodes_per_state:
            raise RuntimeError(f"HHS state {state!r} has only {len(eligible)} eligible hospitals; expected {nodes_per_state}.")
        selected_parts.append(eligible)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selected_order"] = np.arange(1, len(selected) + 1)
    node_ids = selected["hospital_pk"].astype(str).tolist()
    start = pd.Timestamp("2020-11-06")
    end = pd.Timestamp("2021-07-30")
    dates = pd.date_range(start, end, freq="7D")
    panel = data[data["hospital_pk"].astype(str).isin(node_ids)].rename(
        columns={"collection_week": "date", "hospital_pk": "node_id", "hospital_name": "node_name", "state": "region"}
    )
    panel["node_id"] = panel["node_id"].astype(str)
    panel = _complete_node_panel(panel, dates, node_ids)
    observed_source_columns = [capacity_col, occ_col, covid_col, confirmed_col, suspected_col]
    for col in observed_source_columns:
        panel[f"_observed_{col}"] = panel[col].notna().astype(float)
    panel = _fill_numeric_by_node(panel, [capacity_col, occ_col, covid_col, confirmed_col, suspected_col])
    coverage_cols = [col for col in panel.columns if col.endswith("_coverage")]
    panel = _fill_numeric_by_node(panel, coverage_cols)
    node_lookup = selected.set_index(selected["hospital_pk"].astype(str))
    node_names = [str(node_lookup.loc[node, "hospital_name"]) for node in node_ids]
    region_names = list(states)
    region_map = {state: idx for idx, state in enumerate(region_names)}
    regions_array = np.array([region_map[str(node_lookup.loc[node, "state"])] for node in node_ids], dtype=int)
    t, h = len(dates), len(node_ids)
    demand = np.zeros((t, h, 1), dtype=float)
    capacity = np.zeros_like(demand)
    confirmed = np.zeros((t, h), dtype=float)
    suspected = np.zeros_like(confirmed)
    confidence = np.zeros_like(demand)
    node_pos = {node: idx for idx, node in enumerate(node_ids)}
    date_pos = {date: idx for idx, date in enumerate(dates)}
    relevant_coverage = [
        "total_staffed_adult_icu_beds_7_day_coverage",
        "staffed_adult_icu_bed_occupancy_7_day_coverage",
        "staffed_icu_adult_patients_confirmed_covid_7_day_coverage",
        "previous_day_admission_adult_covid_confirmed_7_day_coverage",
    ]
    for _, row in panel.iterrows():
        di = date_pos[pd.Timestamp(row["date"])]
        hi = node_pos[str(row["node_id"])]
        demand[di, hi, 0] = row[covid_col]
        capacity[di, hi, 0] = row[capacity_col]
        confirmed[di, hi] = row[confirmed_col]
        suspected[di, hi] = row[suspected_col]
        cover = np.mean([np.clip(float(row.get(col, 0.0)) / 7.0, 0.0, 1.0) for col in relevant_coverage])
        completeness = float(np.mean([row[f"_observed_{col}"] for col in observed_source_columns]))
        confidence[di, hi, 0] = np.clip(min(cover, 0.35 + 0.65 * completeness), 0.0, 1.0)
    capacity = np.maximum(capacity, demand)
    selected = selected[
        ["selected_order", "state", "hospital_pk", "hospital_name", "median_icu_capacity", "mean_covid_icu", "completeness"]
    ]
    return RealPanel(
        source="HHS COVID-19 Reported Patient Impact and Hospital Capacity by Facility",
        dates=dates,
        node_ids=node_ids,
        node_names=node_names,
        region_names=region_names,
        regions=regions_array,
        resource_names=("Adult ICU",),
        demand=demand,
        capacity=capacity,
        admissions_confirmed=confirmed,
        admissions_suspected=suspected,
        reporting_confidence=confidence,
        long_data=panel,
        selection_table=selected,
        source_url=(
            "https://catalog.data.gov/dataset/"
            "covid-19-reported-patient-impact-and-hospital-capacity-by-facility-raw"
        ),
        notes=(
            "Weekly facility-level HHS data retrieved through the documented CMU Delphi mirror of the HHS dataset. "
            "Counts suppressed with the HHS sentinel value were treated as missing, not as zero. Reporting coverage "
            "fields were retained as empirical confidence inputs."
        ),
    )
