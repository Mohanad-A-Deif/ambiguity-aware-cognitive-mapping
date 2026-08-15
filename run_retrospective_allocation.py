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
sys.path.insert(0, str(ROOT / "src"))

from acm_revision.retrospective_allocation import (  # noqa: E402
    create_retrospective_outputs,
    run_retrospective_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrospective validation against real GetUsPPE allocation and delivery records."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "retrospective_allocation"
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    required = [args.data_dir / name for name in ("all_requests.csv", "all_offers.csv", "all_matches.csv")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required GetUsPPE files are missing: {missing}")

    print("[1/4] Building timestamp-controlled request-to-delivery event panels.", flush=True)
    bundle = run_retrospective_study(
        args.data_dir,
        n_trials=args.trials,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    print("[2/4] Writing machine-readable results and manuscript-ready figures.", flush=True)
    create_retrospective_outputs(bundle, args.output, args.data_dir, dpi=args.dpi)
    print("[3/4] Checking selected configuration and test-panel counts.", flush=True)
    selected_trial = int(
        bundle.calibration_trials.loc[bundle.calibration_trials["selected"], "trial"].iloc[0]
    )
    test_panels = int(
        bundle.metrics.loc[
            bundle.metrics["model"].eq("ACM-4 (coupled)")
            & bundle.metrics["split"].eq("test")
        ].shape[0]
    )
    if bundle.graph_diagnostics["numerically_verified_coupled_contraction_bound"].max() >= 1.0:
        raise RuntimeError("The selected configuration violates the numerically verified contraction condition.")
    print("[4/4] Retrospective validation complete.", flush=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": time.time() - started,
                "selected_trial": selected_trial,
                "test_month_resource_panels": test_panels,
                "figure_dpi": args.dpi,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
