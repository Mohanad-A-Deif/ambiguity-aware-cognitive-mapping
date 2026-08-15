# Data provenance

## Cached observational snapshots

The repository includes the exact aggregate NHS/HHS snapshots used by the observational ranking and semi-empirical analyses.

| File | SHA-256 |
|---|---|
| `raw/hhs_facility_icu_ca_ny_2020_2021_v3.csv` | `31351bca19d63451b65f40aae950df2367e5924b4741e3c98770e31c1b704cb1` |
| `raw/nhs_weekly_covid_admissions_beds_2020_2021.xlsx` | `49c5c45759e042739243e21506420d35de5332d351734e8932015ccdaf8c6713` |

Sources:

- NHS England COVID-19 Hospital Activity: <https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-hospital-activity/>
- U.S. HHS COVID-19 Reported Patient Impact and Hospital Capacity by Facility: <https://catalog.data.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-by-facility>
- CMU Delphi EpiData facility API used for the HHS snapshot: <https://cmu-delphi.github.io/delphi-epidata/api/covid_hosp_facility.html>

Retrieval, selection, and missing-value rules are implemented in `src/acm_revision/real_datasets.py`. The snapshots contain aggregate facility reports and no patient-level records.

## GetUsPPE historical allocation stream

The retrospective replay requires three files from the public GetUsPPE archive:

| File | SHA-256 used in the study |
|---|---|
| `all_requests.csv` | `bc270e7f561e0da36aa47cc6a29adfd7f8766b16b6bc8066c97f184328bad64c` |
| `all_offers.csv` | `b1a576d46e44465554dff422f94eb9159cff6ffbdd5469a018a8a37f72160898` |
| `all_matches.csv` | `eaacac794180177c6e1209b9be5164b88546bbd3c1783ff8ba6c7d76052c9643` |

Acquisition:

```bash
mkdir -p external
git clone https://github.com/GetUsPPE/ppe_needs_retrospective.git \
  external/ppe_needs_retrospective
```

The archive is described in the associated public-health data article: <https://doi.org/10.1002/puh2.65>.

## PPE-Match operational stream

The counterfactual operational replay uses the anonymized data distributed with `ppe-match==0.1.4`:

| File | SHA-256 used in the study |
|---|---|
| `anon_donors.csv` | `6b277d4df5f04db4689a4f7bc0c386446d949ffdab2d592a0cfee55e8d88751a` |
| `anon_recipients.csv` | `18e34c7f6c4b2b6f273d9d1d8b8376ff3228869ed4a6f61420701c5032b8c1f2` |
| `anon_distance_matrix.csv` | `d7dab7aab20f54575bcb87f8ebb04ed04acd4b003e046a773fc8c61aefcb1d3d` |

Acquisition:

```bash
mkdir -p external
git clone https://github.com/samorani/MatchingPPE.git external/MatchingPPE
```

Use `external/MatchingPPE/data` as the `--data-dir`. The same files are distributed with `ppe-match==0.1.4`.

Upstream references:

- Package: <https://pypi.org/project/ppe-match/>
- Matching framework: <https://github.com/samorani/MatchingPPE>
- Project Stanley: <https://github.com/GetUsPPE/project_stanley>

## Storage boundary

The large GetUsPPE and PPE-Match raw files are not duplicated in this repository. Their exact hashes, processed outputs, selected parameters, optimizer trials, and run manifests are retained under `reproducibility/results/real_streams/`. Upstream attribution and data-use terms apply.

`real_world_template.csv` documents the schema accepted for an optional external facility panel.
