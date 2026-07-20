# Data snapshots

The `raw/` directory contains the exact public-data snapshots used by the real-data pipeline.

| File | SHA-256 |
|---|---|
| `hhs_facility_icu_ca_ny_2020_2021_v3.csv` | `31351bca19d63451b65f40aae950df2367e5924b4741e3c98770e31c1b704cb1` |
| `nhs_weekly_covid_admissions_beds_2020_2021.xlsx` | `49c5c45759e042739243e21506420d35de5332d351734e8932015ccdaf8c6713` |

The NHS workbook originates from NHS England COVID-19 Hospital Activity. The HHS subset was retrieved from the CMU Delphi EpiData public mirror of the HHS facility-level dataset. Source URLs, retrieval logic, selection rules, and missing-value handling are defined in `src/acm_revision/real_datasets.py`.

These snapshots are included for exact reproducibility. Upstream attribution and data-use terms remain applicable. The files contain public facility-level aggregate data and no patient-level or personally identifiable records.

`real_world_template.csv` documents the accepted schema for an optional external panel.

