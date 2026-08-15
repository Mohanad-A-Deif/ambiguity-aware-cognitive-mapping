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

from acm_revision.retrospective_allocation_v2 import (  # noqa: E402
    create_v2_outputs,
    run_v2_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the leakage-controlled, event-local GetUsPPE V2 allocation replay."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "retrospective_allocation_v2",
    )
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    bundle = run_v2_study(
        args.data_dir,
        n_trials=args.trials,
        seed=args.seed,
        n_boot=args.bootstrap,
        progress=True,
    )
    create_v2_outputs(bundle, args.output, args.data_dir, dpi=args.dpi)
    selected = int(
        bundle.calibration_trials.loc[bundle.calibration_trials["selected"], "trial"].iloc[0]
    )
    max_bound = float(bundle.graph_diagnostics["max_contraction_bound"].max())
    if max_bound >= 1.0:
        raise RuntimeError(f"V2 contraction check failed: maximum bound={max_bound:.6f}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_trial": selected,
                "maximum_contraction_bound": max_bound,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
