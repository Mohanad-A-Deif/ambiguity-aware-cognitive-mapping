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

from acm_revision.ppe_match_operational_v2 import (  # noqa: E402
    create_ppe_match_outputs,
    run_ppe_match_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimized ACM policy on official PPE-Match real request streams."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ppe_match_operational_v2",
    )
    parser.add_argument("--trials", type=int, default=28)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    bundle = run_ppe_match_study(
        args.data_dir,
        n_trials=args.trials,
        seed=args.seed,
        n_boot=args.bootstrap,
        progress=True,
    )
    create_ppe_match_outputs(bundle, args.output, dpi=args.dpi)
    max_bound = float(
        bundle.metrics.loc[
            bundle.metrics["model"].str.startswith("ACM"), "contraction_bound"
        ].max()
    )
    if max_bound >= 1.0:
        raise RuntimeError(f"PPE-Match contraction check failed: {max_bound:.6f}")
    selected = int(bundle.trials.loc[bundle.trials["selected"], "trial"].iloc[0])
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
