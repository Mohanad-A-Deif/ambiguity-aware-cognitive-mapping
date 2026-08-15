# Real-stream allocation evaluation and temporal calibration

## Evaluation design

### GetUsPPE historical-assignment replay

Successfully delivered batches are replayed against acute-care requests timestamped before each decision. Donor supply is reconstructed from offers available by the decision time minus earlier commitments. The logged batch quantity defines the conditional dispatch budget and is excluded from evidence encoding. Route risk replaces request age in the lead-time channel. The cognitive operator is restricted and norm-scaled on the event-active graph, and the previous converged concept state is carried forward.

### PPE-Match operational replay

The `ppe-match==0.1.4` donor offers, recipient requests, and anonymized distance matrix are processed at seven-day decision intervals for respirators, surgical masks, face shields, and gowns. The replay compares ACM with proximity, demand-proportional, and equal-allocation policies on service, recipient coverage, fill inequality, unit-miles, holding time, and a prespecified composite score.

### Temporal optimization

Seeded Optuna TPE searches calibrate evidence/Gaussian, memory, coupling, mixing, partial-channel, temperature, and evidence-source parameters. Calibration determines the trial ordering, internal validation selects among a fixed shortlist, and test intervals are evaluated only after parameter selection. LP coefficient ratios remain fixed at the manuscript values. Softmax priorities are max-normalized within each active candidate set before entering the LP; this monotone normalization preserves priority order while maintaining a comparable objective scale across candidate-set sizes.

The complete search spaces, split definitions, objective weights, and fixed coefficients are recorded in `config/real_stream_validation.json`.

## Calibration effect on temporal test panels

| Dataset / metric | Frozen preset | Optimized | Relative change | Holm-adjusted p |
|---|---:|---:|---:|---:|
| GetUsPPE priority overlap | 0.0287 | 0.0319 | +11.1% | 0.0142 |
| GetUsPPE recipient AP | 0.0733 | 0.0812 | +10.8% | 0.0042 |
| GetUsPPE full-list NDCG | 0.2215 | 0.2328 | +5.1% | 0.0042 |
| GetUsPPE allocation overlap | 0.0081 | 0.0081 | 0.0% | 1.0000 |
| PPE-Match composite score | 0.1322 | 0.2003 | +51.6% | 0.0059 |
| PPE-Match recipient coverage | 0.00645 | 0.01434 | +122.5% | 0.0059 |
| PPE-Match fill Gini | 0.99648 | 0.99451 | -0.20% | 0.0059 |
| PPE-Match unit-miles | 1999.6 | 24.8 | -98.8% | 0.0059 |

These paired comparisons quantify the effect of data-adaptive calibration relative to the frozen preset. Comparisons with external allocation baselines are reported separately.

## GetUsPPE historical-assignment results

| Model | Allocation overlap | Recipient AP | Full-list NDCG |
|---|---:|---:|---:|
| ACM-4 coupled | 0.0081 | 0.0812 | 0.2328 |
| Calibrated scalar | 0.1104 | 0.1768 | 0.3733 |
| Demand proportional | 0.0509 | 0.1319 | 0.3616 |
| Equal allocation | 0.0509 | 0.0393 | 0.2717 |

No Holm-adjusted comparison favored coupled ACM. Coupled and independent four-channel allocation overlap was identical. The observed gap is compatible with the importance of proximity, logistics, donor preferences, and eligibility constraints that are only partly represented in the public archive.

## PPE-Match operational results

| Model | Composite score | Service | Coverage | Fill Gini | Unit-miles | Holding days |
|---|---:|---:|---:|---:|---:|---:|
| ACM-4 coupled | 0.2003 | 0.00986 | 0.01434 | 0.99451 | 24.82 | 4.18 |
| Proximity | 0.1842 | 0.00937 | 0.07803 | 0.94055 | 12.54 | 30.01 |
| Demand proportional | 0.1611 | 0.00986 | 0.00041 | 0.99965 | 1067.94 | 4.18 |
| Equal allocation | 0.2044 | 0.00986 | 0.02837 | 0.99105 | 28.06 | 4.18 |

After Holm correction, coupled ACM had higher service and shorter holding time than proximity matching, and higher coverage with lower unit-miles than demand-proportional allocation. Proximity retained higher coverage and lower fill inequality, while equal allocation attained the highest composite score. Coupled and independent ACM were operationally indistinguishable. The statistically detectable Gini differences among ACM-2, ACM-3, and ACM-4 were extremely small in magnitude.

## Interpretation

The two replays validate the end-to-end data-to-allocation implementation and characterize where the policy yields service/holding-time or coverage/distance compromises. The held-out evidence supports systematic dataset-specific calibration and an auditable ambiguity representation. It does not establish general dominance of ACM or empirical operational necessity for cross-channel coupling.

## Frozen evidence

- GetUsPPE: `reproducibility/results/real_streams/getusppe/`
- PPE-Match: `reproducibility/results/real_streams/ppe_match/`

Each directory contains selected parameters, all optimizer trials, split-level metrics, paired statistical tests, provenance hashes, diagnostic figures, and a SHA-256 manifest.
