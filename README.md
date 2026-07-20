# Ambiguity-Aware Cognitive Mapping for Epidemic Medical-Resource Allocation

This repository is the reproducibility package for the manuscript:

> **Ambiguity-Aware Cognitive Mapping for Epidemic Medical-Resource Allocation: A Coupled Four-Channel Model with Real-Data Validation**

It contains the exact Python pipeline, fixed configuration, cached public-data snapshots, processed panels, machine-readable results, graph matrices, and final manuscript figures used in the revised study.

## Scope and interpretation

The code implements a four-channel Ambiguous Cognitive Map (ACM) with true, false, partially true, and partially false evidence channels. The coupled cognitive scores feed a common linear-programming allocation layer. The repository reproduces:

- 30 paired synthetic epidemic-supply-chain runs;
- channel-count and coupling ablations;
- convergence, stability, robustness, fairness, and sensitivity analyses;
- held-out NHS England ranking validation;
- external HHS ranking validation without recalibration; and
- a 30-run **semi-empirical** allocation experiment anchored to observed NHS demand.

The public hospital panels do not include complete inventory, dispatch, supplier, and realized transport records. The allocation study must therefore be described as semi-empirical, not as observed real-world allocation outcomes.

## Repository layout

```text
config/default.json            Fixed paper and quick-test configurations
data/raw/                      Cached public-data snapshots used in the study
data/real_world_template.csv   Schema for an optional external panel
src/acm_revision/              Model, experiments, statistics, and reporting code
run_all.py                     Synthetic study and manuscript-output pipeline
run_real_world_study.py        NHS/HHS validation and semi-empirical study
reproducibility/figures/       Ten final grayscale manuscript figures
reproducibility/results/       Processed panels, matrices, and result workbooks
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the results

Run a fast installation and pipeline check:

```bash
python run_all.py --profile quick
```

Reproduce the synthetic paper results:

```bash
python run_all.py --profile paper --output outputs/paper
```

Reproduce the NHS/HHS validation and 30 paired semi-empirical runs:

```bash
python run_real_world_study.py --runs 30 --dpi 600 --output outputs/real_world
```

The scripts use the cached source snapshots in `data/raw/`. If those files are absent, the real-data loader retrieves the official NHS workbook and the HHS facility series through the documented CMU Delphi mirror.

Generated outputs are written beneath `outputs/` and include figures, tables, raw per-run metrics, matrices, validation reports, run summaries, and SHA-256 manifests. Figures are grayscale 600-dpi PNG files; the pipeline does not generate PDF figures.

## Fixed reproducibility settings

- Master seed: `20260718`
- Synthetic evaluation runs: `30`
- Semi-empirical paired runs: `30`
- Simulation horizon: `40` days
- Paper figure resolution: `600` dpi
- Synthetic partial-evidence coefficient: selected by the coded calibration routine
- NHS partial-evidence coefficient: calibrated on the first half of the NHS panel; the second half is retained for evaluation
- HHS validation: performed without recalibration

All remaining settings are recorded in `config/default.json` and in `reproducibility/results/reproducibility_metadata.json`.

## Public data provenance

- NHS England, COVID-19 Hospital Activity: <https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/>
- U.S. HHS, COVID-19 Reported Patient Impact and Hospital Capacity by Facility: <https://catalog.data.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-by-facility>
- CMU Delphi EpiData API, used as the documented HHS retrieval mirror: <https://cmu-delphi.github.io/delphi-epidata/api/covid_hosp_facility.html>

The cached files are included to preserve the exact inputs used for the reported results. Source attribution and any upstream data-use terms remain applicable.

## Citation

Please cite the associated manuscript when using this code or its outputs. Machine-readable citation metadata are provided in `CITATION.cff`.

## License

The software is released under the MIT License. Public source data remain subject to their respective upstream terms and attribution requirements.

