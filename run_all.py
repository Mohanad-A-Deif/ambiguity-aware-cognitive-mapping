#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from acm_revision.core import build_concept_graph
from acm_revision.experiments import build_experiment_bundle
from acm_revision.learning import train_learning_baselines
from acm_revision.real_world import run_real_world_validation
from acm_revision.reporting import create_all_outputs
from acm_revision.validation import validate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the ACM experiments, tables, and raster figures.")
    parser.add_argument("--profile", choices=["quick", "paper"], default="paper")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--real-data", type=Path, default=None, help="Optional real-world CSV using data/real_world_template.csv schema.")
    return parser.parse_args()


def file_manifest(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = config[args.profile]
    model_cfg = config["model"]
    output_root = args.output or (ROOT / "outputs" / args.profile)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    regions = np.repeat(np.arange(2), int(profile["n_hospitals"]) // 2)
    graph = build_concept_graph(
        n_hospitals=int(profile["n_hospitals"]),
        n_dcs=int(profile["n_dcs"]),
        n_suppliers=int(profile["n_suppliers"]),
        regions=regions,
        alpha=float(model_cfg["transfer_alpha"]),
        physical_evidence_mix=float(model_cfg["physical_evidence_mix"]),
        operator_target_norm=float(model_cfg["operator_target_norm"]),
        gaussian_mu=model_cfg["gaussian_mu"],
        gaussian_sigma=model_cfg["gaussian_sigma"],
        gaussian_beta=model_cfg["gaussian_beta"],
    )
    print(f"[1/4] Graph built: {len(graph.metadata)} concepts; contraction bound={graph.contraction_bound_coupled:.4f}", flush=True)

    learning = train_learning_baselines(
        graph=graph,
        n_scenarios=int(profile["training_scenarios"]),
        days=int(profile["days"]),
        interactions=int(profile["ppo_interactions"]),
        master_seed=int(profile["master_seed"]),
        reporting_noise=float(model_cfg["reporting_noise"]),
        missing_probability=float(model_cfg["reporting_missing_probability"]),
        temperature=float(model_cfg["priority_temperature"]),
    )
    print("[2/4] Learning baselines trained on disjoint synthetic scenarios.", flush=True)

    bundle = build_experiment_bundle(profile, model_cfg, graph, learning)
    print(f"[3/4] Main, ablation, statistical and sensitivity experiments complete; selected gamma={bundle.selected_partial_coefficient:.2f}", flush=True)

    create_all_outputs(bundle, learning, profile, model_cfg, output_root)
    if args.real_data is not None:
        run_real_world_validation(
            args.real_data,
            graph,
            model_cfg,
            bundle.selected_partial_coefficient,
            output_root,
            list(profile["output_formats"]),
            int(profile["dpi"]),
            int(profile["master_seed"]),
        )
        real_status = "completed"
    else:
        real_status = "not run - no real-world CSV supplied"
        note = output_root / "REAL_WORLD_VALIDATION_REQUIRED.txt"
        note.write_text(
            "External validation was not fabricated. Provide a populated CSV matching data/real_world_template.csv and rerun with --real-data PATH.\n",
            encoding="utf-8",
        )

    metadata = {
        "profile": args.profile,
        "elapsed_seconds": time.time() - started,
        "real_world_validation": real_status,
        "selected_partial_coefficient": bundle.selected_partial_coefficient,
        "n_evaluation_runs": int(profile["n_runs"]),
        "figure_formats": profile["output_formats"],
        "pdf_generated": False,
    }
    validation = validate_outputs(output_root, int(profile["n_runs"]), int(profile["dpi"]))
    metadata["validation_passed"] = bool(validation["passed"])
    (output_root / "RUN_SUMMARY.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = file_manifest(output_root)
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[4/4] Tables and raster figures written to {output_root}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
