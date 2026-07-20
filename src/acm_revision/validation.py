from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def validate_outputs(output_root: Path, expected_runs: int, expected_dpi: int) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    metric_path = output_root / "raw" / "per_run_metrics.csv"
    if not metric_path.exists():
        errors.append("Missing per_run_metrics.csv")
        metrics = pd.DataFrame()
    else:
        metrics = pd.read_csv(metric_path)
    if not metrics.empty:
        if metrics["seed"].nunique() != expected_runs:
            errors.append(f"Expected {expected_runs} paired seeds; found {metrics['seed'].nunique()}.")
        if metrics["model"].nunique() != 9:
            errors.append(f"Expected 9 evaluated models; found {metrics['model'].nunique()}.")
        if metrics.isna().any().any():
            errors.append("Per-run metrics contain missing values.")
        complement_error = float(np.max(np.abs(metrics["service_level"] + metrics["unmet_demand_rate"] - 1.0)))
        if complement_error > 1e-10:
            errors.append(f"Service/unmet complement error is {complement_error:.3g}.")
        for column in ["service_level", "unmet_demand_rate", "gini_fill", "jain_fairness", "max_min_fairness", "geographic_equity", "priority_weighted_equity"]:
            if not metrics[column].between(0.0, 1.0).all():
                errors.append(f"Metric {column} lies outside [0,1].")
    else:
        complement_error = np.nan

    stability_path = output_root / "tables" / "table_stability_theoretical.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        if not (stability["contraction_bound"] < 1.0).all():
            errors.append("At least one reported theoretical contraction bound is >= 1.")
    else:
        errors.append("Missing stability table.")

    figures = sorted((output_root / "figures").glob("fig*.png"))
    if len(figures) < 14:
        errors.append(f"Expected at least 14 PNG figures; found {len(figures)}.")
    figure_checks = []
    for path in figures:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0.0, 0.0))
                width, height = image.size
            if min(dpi) < expected_dpi * 0.97:
                errors.append(f"{path.name} reports {dpi} DPI; expected approximately {expected_dpi}.")
            if min(width, height) < 1000:
                warnings.append(f"{path.name} has a dimension below 1000 pixels.")
            figure_checks.append({"file": path.name, "width": width, "height": height, "dpi_x": dpi[0], "dpi_y": dpi[1], "valid_png": True})
        except Exception as exc:
            errors.append(f"Invalid image {path.name}: {exc}")
            figure_checks.append({"file": path.name, "valid_png": False, "error": str(exc)})

    pdf_files = [str(p.relative_to(output_root)) for p in output_root.rglob("*.pdf")]
    if pdf_files:
        errors.append(f"PDF output is forbidden but found: {pdf_files}")
    required_tables = [
        "table_main_performance_mean_sd.csv",
        "table_ablation.csv",
        "table_paired_statistical_tests.csv",
        "table_fairness_metrics.csv",
        "table_sensitivity_summary.csv",
        "table_stability_theoretical.csv",
        "table_signal_to_channel_mapping.csv",
        "table_baseline_settings.csv",
        "all_manuscript_tables.xlsx",
    ]
    for name in required_tables:
        if not (output_root / "tables" / name).exists():
            errors.append(f"Missing required table: {name}")

    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "paired_runs": int(metrics["seed"].nunique()) if not metrics.empty else 0,
        "models": int(metrics["model"].nunique()) if not metrics.empty else 0,
        "service_unmet_max_error": complement_error,
        "pdf_files_found": pdf_files,
        "figure_checks": figure_checks,
    }
    (output_root / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Output validation failed: " + " | ".join(errors))
    return report
