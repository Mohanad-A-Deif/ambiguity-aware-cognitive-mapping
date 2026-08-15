# Reproducibility readiness check

Verification date: 2026-08-15

## Package checks

- [x] Python sources compile successfully under the locked environment.
- [x] Both V2 command-line entry points load and expose their documented options.
- [x] `config/default.json` and `config/real_stream_validation.json` parse as valid JSON.
- [x] GetUsPPE source hashes match the recorded provenance.
- [x] PPE-Match source hashes match the official `MatchingPPE` repository files.
- [x] Each optimizer output contains exactly one selected validation candidate.
- [x] Trial counts and validation-shortlist sizes match the configured 36/8 and 28/7 protocols.
- [x] Selected parameters lie inside every declared search domain.
- [x] Priority, demand, fairness, and lead-time LP coefficients remain fixed.
- [x] Coupled-operator contraction bounds remain below one on all evaluated graphs.
- [x] Result tables contain unique model-metric rows and finite confidence intervals.
- [x] Both frozen-output SHA-256 manifests validate without missing or altered files.
- [x] The workflow and integrated real-stream figures regenerate byte-for-byte from repository-local inputs.
- [x] `git diff --check` reports no whitespace errors.

## Frozen-output validation command

```bash
python validate_v2_outputs.py \
  --actual-output reproducibility/results/real_streams/getusppe \
  --actual-data external/ppe_needs_retrospective/data \
  --ppe-output reproducibility/results/real_streams/ppe_match \
  --ppe-data external/MatchingPPE/data
```

Expected terminal message:

```text
V2 output validation passed.
```

The packaging check validates the frozen numerical results and their source provenance. It does not repeat the complete 36- and 28-trial optimizer runs; the commands for a full independent rerun are provided in `README.md`.
