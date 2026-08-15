#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config" / "real_stream_validation.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_fixed_lp(parameters: dict[str, float]) -> None:
    configured = CONFIG["fixed_lp_coefficients"]
    expected = {
        "lp_priority_weight": configured["priority"],
        "lp_demand_weight": configured["demand"],
        "lp_fairness_weight": configured["fairness"],
        "lp_lead_weight": configured["lead_time"],
    }
    for key, value in expected.items():
        assert abs(float(parameters[key]) - value) < 1e-12, (key, parameters[key])


def _assert_interval(value: float, domain: list[float], name: str) -> None:
    assert float(domain[0]) <= float(value) <= float(domain[1]), (name, value, domain)


def _assert_selected_domains(parameters: dict[str, float], study: str) -> None:
    common_mapping = {
        "mu_true": "mu_true",
        "mu_partial": "mu_partial_true",
        "sigma_exact": "sigma_exact",
        "sigma_partial": "sigma_partial",
        "beta_partial": "beta_partial",
        "diagonal_memory": "diagonal_memory",
        "partial_to_exact": "partial_to_exact",
        "exact_to_partial": "exact_to_partial",
        "partial_self": "partial_self",
        "physical_mix": "physical_mix",
    }
    for config_name, parameter_name in common_mapping.items():
        _assert_interval(
            parameters[parameter_name],
            CONFIG["shared_search_space"][config_name],
            parameter_name,
        )
    assert parameters["partial_coefficient"] in CONFIG["shared_search_space"][
        "partial_coefficient"
    ]
    assert abs(parameters["operator_target_norm"] - CONFIG["operator_target_norm"]) < 1e-12

    search = CONFIG[study]["search_space"]
    if study == "getusppe":
        _assert_interval(parameters["temperature"], search["temperature"], "temperature")
        source_domain = search["evidence_source_weights_log_uniform"]
        for name in ("weight_demand", "weight_backlog", "weight_inventory", "weight_route", "weight_capacity"):
            _assert_interval(parameters[name], source_domain, name)
    else:
        domains = {
            "temperature": "temperature_log_uniform",
            "weight_demand": "demand_weight_log_uniform",
            "weight_backlog": "backlog_weight_log_uniform",
            "weight_inventory": "inventory_weight_log_uniform",
            "weight_route": "route_weight_log_uniform",
            "weight_capacity": "capacity_weight_log_uniform",
        }
        for parameter_name, config_name in domains.items():
            _assert_interval(parameters[parameter_name], search[config_name], parameter_name)


def _assert_trial_protocol(trials: pd.DataFrame, study: str) -> None:
    protocol = CONFIG[study]["optimizer"]
    assert len(trials) == int(protocol["trials"])
    assert int(trials["calibration_shortlist"].sum()) == int(
        protocol["validation_shortlist"]
    )
    assert int(trials["selected"].sum()) == 1
    shortlisted = trials.loc[trials["calibration_shortlist"]]
    selected_value = float(trials.loc[trials["selected"], "validation_objective"].iloc[0])
    assert abs(selected_value - float(shortlisted["validation_objective"].max())) < 1e-12


def _assert_optimizer_comparison(path: Path) -> None:
    comparison = pd.read_csv(path)
    assert comparison["metric"].is_unique
    assert comparison[["preset_mean", "optimized_mean", "holm_p"]].notna().all().all()
    assert ((comparison["holm_p"] >= 0.0) & (comparison["holm_p"] <= 1.0)).all()


def _assert_manifest(root: Path) -> None:
    entries = json.loads((root / "MANIFEST.json").read_text())
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)), "Manifest contains duplicate paths"
    for entry in entries:
        path = root / entry["path"]
        assert path.is_file(), path
        assert path.stat().st_size == int(entry["bytes"]), path
        assert _sha256(path) == entry["sha256"], path


def validate_actual(root: Path, data_dir: Path) -> None:
    results = root / "results"
    trials = pd.read_csv(results / "optuna_trials.csv")
    _assert_trial_protocol(trials, "getusppe")
    selected = trials.loc[trials["selected"]].iloc[0]
    assert bool(selected["calibration_shortlist"])
    assert pd.notna(selected["validation_objective"])
    parameters = json.loads((results / "selected_parameters.json").read_text())
    _assert_fixed_lp(parameters)
    _assert_selected_domains(parameters, "getusppe")
    summary = pd.read_csv(results / "test_summary.csv")
    assert not summary.duplicated(["model", "metric"]).any()
    assert summary[["mean", "ci_low", "ci_high"]].notna().all().all()
    graph = pd.read_csv(results / "graph_diagnostics.csv")
    assert float(graph["max_contraction_bound"].max()) < 1.0
    coupled = graph.loc[graph["model"].eq("ACM-4 (coupled)")].set_index("product")
    independent = graph.loc[graph["model"].eq("ACM-4 (independent)")].set_index("product")
    assert (independent["max_operator_norm"] < coupled["max_operator_norm"]).all()
    _assert_optimizer_comparison(results / "optimizer_preset_comparison.csv")
    provenance = json.loads((results / "data_provenance.json").read_text())
    for name, digest in provenance["source_file_sha256"].items():
        assert _sha256(data_dir / name) == digest


def validate_ppe(root: Path, data_dir: Path) -> None:
    results = root / "results"
    trials = pd.read_csv(results / "optuna_trials.csv")
    _assert_trial_protocol(trials, "ppe_match")
    selected = trials.loc[trials["selected"]].iloc[0]
    assert bool(selected["calibration_shortlist"])
    assert pd.notna(selected["validation_objective"])
    parameters = json.loads((results / "selected_parameters.json").read_text())
    _assert_fixed_lp(parameters)
    _assert_selected_domains(parameters, "ppe_match")
    metrics = pd.read_csv(results / "product_week_metrics.csv")
    acm = metrics.loc[metrics["model"].str.startswith("ACM")]
    assert float(acm["contraction_bound"].max()) < 1.0
    summary = pd.read_csv(results / "test_summary.csv")
    assert not summary.duplicated(["model", "metric"]).any()
    assert summary[["mean", "ci_low", "ci_high"]].notna().all().all()
    context = pd.read_csv(results / "context_strata_summary.csv")
    assert set(context["context"]) == {"Feature ambiguity", "Supply scarcity"}
    _assert_optimizer_comparison(results / "optimizer_preset_comparison.csv")
    provenance = json.loads((results / "data_provenance.json").read_text())
    for name, digest in provenance["source_file_sha256"].items():
        assert _sha256(data_dir / name) == digest


def validate_figures(*roots: Path) -> None:
    for root in roots:
        for path in (root / "figures").glob("*.png"):
            with Image.open(path) as image:
                assert image.width >= 3500 and image.height >= 3500, path
                assert image.mode in {"L", "LA", "RGB", "RGBA"}, (path, image.mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-output", type=Path, required=True)
    parser.add_argument("--actual-data", type=Path, required=True)
    parser.add_argument("--ppe-output", type=Path, required=True)
    parser.add_argument("--ppe-data", type=Path, required=True)
    args = parser.parse_args()
    validate_actual(args.actual_output, args.actual_data)
    validate_ppe(args.ppe_output, args.ppe_data)
    validate_figures(args.actual_output, args.ppe_output)
    _assert_manifest(args.actual_output)
    _assert_manifest(args.ppe_output)
    print("V2 output validation passed.")


if __name__ == "__main__":
    main()
