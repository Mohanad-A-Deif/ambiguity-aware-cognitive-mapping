#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from acm_revision.real_datasets import load_hhs_panel, load_nhs_panel
from acm_revision.real_experiments import run_direct_validation, run_semi_empirical_experiment
from acm_revision.real_reporting import create_real_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-data ACM validation and semi-empirical allocation study.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "real_world")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    master_seed = int(config["paper"]["master_seed"])
    raw_dir = ROOT / "data" / "raw"

    print("[1/5] Loading and harmonizing the NHS England panel.", flush=True)
    nhs = load_nhs_panel(raw_dir)
    print("[2/5] Loading the HHS external panel through the documented CMU Delphi mirror.", flush=True)
    hhs = load_hhs_panel(raw_dir)
    print("[3/5] Calibrating on the NHS first half and evaluating the untouched NHS holdout and HHS panel.", flush=True)
    nhs_direct = run_direct_validation(nhs, model_cfg, master_seed)
    hhs_direct = run_direct_validation(hhs, model_cfg, master_seed, nhs_direct.selected_partial_coefficient)
    print(f"[4/5] Running {args.runs} paired semi-empirical allocation replications.", flush=True)
    semi = run_semi_empirical_experiment(
        nhs,
        model_cfg,
        nhs_direct.selected_partial_coefficient,
        master_seed,
        n_runs=args.runs,
    )
    create_real_outputs(nhs_direct, hhs_direct, semi, args.output, dpi=args.dpi)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": time.time() - started,
                "selected_partial_coefficient": nhs_direct.selected_partial_coefficient,
                "nhs_shape": list(nhs.demand.shape),
                "hhs_shape": list(hhs.demand.shape),
                "paired_runs": args.runs,
                "figures": 10,
                "formats": ["png"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
