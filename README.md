# Ambiguity-Aware Cognitive Mapping for Epidemic Medical-Resource Allocation

Reproducibility package for the manuscript:

> **Ambiguity-Aware Cognitive Mapping for Epidemic Medical-Resource Allocation: A Coupled Four-Channel Model with Observational and Real-Stream Evaluation**

The package contains the coupled four-channel Ambiguous Cognitive Map (ACM), evidence encoding, temporal calibration, allocation models, statistical analyses, frozen result artifacts, and manuscript figures.

## Evaluation scope

| Evaluation | Empirical inputs | Decision layer | Interpretation |
|---|---|---|---|
| Synthetic experiments | Simulated epidemic and supply-chain state | Simulated LP allocation | Controlled method comparison and stress testing |
| NHS ranking | Observed NHS hospital activity | Ranking only | Held-out observational ranking evaluation |
| HHS ranking | Observed HHS facility reports | Ranking only | External evaluation without recalibration |
| NHS allocation | Observed demand with simulated inventory, dispatch, supplier, and transport state | Simulated LP allocation | Semi-empirical allocation experiment |
| GetUsPPE replay | Timestamped requests, offers, and successful deliveries | Retrospective recipient-assignment replay conditional on the logged batch budget | Agreement with historical assignments |
| PPE-Match replay | Anonymized requests, offers, and donor-recipient distances | Counterfactual weekly matching policies | Operational trade-off evaluation on real streams |

The held-out results are endpoint-dependent. In the GetUsPPE replay, the calibrated scalar and demand baselines agreed more closely with historical recipient assignments than coupled ACM. In PPE-Match, ACM produced service/holding-time and coverage/distance trade-offs, while equal allocation attained the highest prespecified composite score. Coupled and independent four-channel variants were operationally indistinguishable in both replays. The four-channel state therefore serves primarily as an auditable ambiguity representation in these data rather than as evidence of universal operational superiority.

## Repository layout

```text
config/default.json                         Synthetic and NHS/HHS settings
config/real_stream_validation.json          Temporal splits, Optuna spaces, and fixed LP coefficients
data/raw/                                   Cached NHS/HHS public-data snapshots
data/README.md                              Data provenance, hashes, and acquisition instructions
src/acm_revision/                            Models, experiments, statistics, and reporting
run_all.py                                  Synthetic study pipeline
run_real_world_study.py                     NHS/HHS and semi-empirical pipeline
run_retrospective_allocation_v2.py          GetUsPPE retrospective replay
run_ppe_match_operational_v2.py             PPE-Match operational replay
validate_v2_outputs.py                      Frozen-output and provenance checks
scripts/                                    Manuscript-figure generators
reproducibility/figures/                     Final manuscript figures
reproducibility/results/real_streams/        Frozen V2 outputs and SHA-256 manifests
docs/REAL_STREAM_VALIDATION.md               Design and numerical results for the two replays
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the exact environment used to generate the frozen V2 artifacts:

```bash
python -m pip install -r requirements-lock.txt
```

## Reproduce the original pipelines

Fast synthetic installation check:

```bash
python run_all.py --profile quick
```

Synthetic paper experiments:

```bash
python run_all.py --profile paper --output outputs/paper
```

NHS/HHS observational evaluation and 30 paired semi-empirical runs:

```bash
python run_real_world_study.py --runs 30 --dpi 600 --output outputs/real_world
```

The NHS/HHS scripts use the cached source snapshots in `data/raw/`.

## Reproduce the real-stream evaluations

Acquire the external source files as described in `data/README.md`, then run:

```bash
python run_retrospective_allocation_v2.py \
  --data-dir external/ppe_needs_retrospective/data \
  --trials 36 --bootstrap 5000 --seed 20260718 --dpi 600 \
  --output outputs/retrospective_allocation_v2

python run_ppe_match_operational_v2.py \
  --data-dir external/MatchingPPE/data \
  --trials 28 --bootstrap 5000 --seed 20260718 --dpi 600 \
  --output outputs/ppe_match_operational_v2
```

The optimizer uses only calibration data; a shortlist is chosen on internal validation, and the temporal test interval remains untouched until final evaluation. Evidence/Gaussian, memory, coupling, mixing, partial-channel, temperature, and evidence-source parameters are calibrated. The priority, demand, fairness, and lead-time LP coefficients remain fixed at `1.80`, `0.20`, `0.16`, and `0.08`, respectively. Exact spaces and splits are recorded in `config/real_stream_validation.json`; selected configurations and every trial are stored with the frozen outputs.

## Validate frozen artifacts

```bash
python validate_v2_outputs.py \
  --actual-output reproducibility/results/real_streams/getusppe \
  --actual-data external/ppe_needs_retrospective/data \
  --ppe-output reproducibility/results/real_streams/ppe_match \
  --ppe-data external/MatchingPPE/data
```

The validator checks the unique selected trial, calibration-to-validation selection, fixed LP coefficients, contraction bounds, result-table integrity, source-file hashes, and figure dimensions. Each real-stream directory also contains a SHA-256 `MANIFEST.json` for its frozen artifacts.

## Regenerate the integrated manuscript figures

```bash
python scripts/make_workflow_figure.py
python scripts/make_real_stream_figure.py
```

Both scripts read repository-local frozen artifacts and write 600-dpi PNG files to `reproducibility/figures/`.

## Reproducibility settings

- Master seed: `20260718`
- Synthetic evaluation runs: `30`
- Semi-empirical paired runs: `30`
- Simulation horizon: `40` days
- Bootstrap resamples: `5000`
- Figure resolution: `600` dpi
- GetUsPPE TPE trials / validation shortlist: `36 / 8`
- PPE-Match TPE trials / validation shortlist: `28 / 7`

## Citation and license

Machine-readable citation metadata are provided in `CITATION.cff`. The software is released under the MIT License. Public source data remain subject to their upstream terms and attribution requirements.
