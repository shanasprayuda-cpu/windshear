# Operational LLWAS runway-alert forecasting — Juanda

**Companion implementation and audit documentation** for the manuscript:
*Chronologically Validated Deep Learning for Multi-Horizon Prediction of Operational Windshear Alerts* (IJIES, Paper ID 20265914; revised manuscript).

> **Scope:** The target is the **recorded binary LLWAS runway-alert state** (WSA/MBA) at H+1 to H+10 minutes. This is not a claim to detect every windshear encounter experienced by aircraft. **Research software, not an operational warning system.**

## What is tested?

- **Observation network:** 10 LLWAS wind sensors (A2–A6 and B2–B6), converted to 20 horizontal `u/v` wind features (F1).
- **History / labels:** 60 one-minute historical observations per prediction; 10 exact future alert labels at H+1 … H+10.
- **Development:** January 2023–December 2024; 847,513 canonical valid windows.
- **Chronological validation:** 5 expanding-window folds with **69 excluded anchor minutes** at each boundary to prevent reuse of input/target raw timestamps.
- **Temporal external evaluation:** January 1–March 27, 2025; 70,810 canonical valid windows.
- **Model:** selected Conv1D–BiLSTM (historically named TCN–BiLSTM in experimental artifacts). **The convolutional implementation is not a dilated residual TCN.**
- **Selection:** model and 10 CSI-maximizing decision thresholds selected on development data, using pooled OOF predictions.
- **Reviewer experiments:** 4 capacity-matched model variants × 5 random seeds × 5 development folds (100 fits); nested retention restriction of original 5-minute Stage-A imputations to 0, 2, or 5 minutes.

**Important evaluation limitation:** January–March 2025 was separated chronologically from model fitting and final OOF threshold estimation but had been inspected in earlier exploratory work. It should **not** be described as a wholly untouched confirmatory test.

## Public release status

This repository contains code and documentation; the presence of source files **does not yet establish full, independently runnable reproduction**. Original LLWAS observations, some exact fold manifests, imputation masks, full checkpoints/predictions, and the original software environment may be unavailable in a public clone. See [reproducibility status](docs/REPRODUCIBILITY.md) and [artifact inventory](docs/ARTIFACT_INVENTORY.md).

The `requirements.txt` and `environment.example.yml` are **minimal templates, not a frozen environment**. Please record installed package versions from the original `windshear_gpu` environment before claiming exact reproducibility.

## Quick start

```bash
python -m pip install -r requirements.txt
python examples/generate_synthetic_llwas.py --out examples/synthetic_llwas_minute.csv
python scripts/check_release_package.py --root .
python scripts/reviewer_imputation_policies_v3.py --stage smoke
```

The synthetic dataset is a **demonstration of a file schema only**. It is not sufficient for training, for the canonical Stage-C manifests, or for reconstructing paper metrics. `--stage smoke` uses the experiment's internal synthetic test and performs **no scientific training**.

Before publishing, read the [upload and merge guide](docs/UPLOAD_GUIDE.md) and [owner-approval note](docs/RELEASE_CHECKLIST.md).

For the actual development → OOF → full fit → external test workflow, see [Windows runbook](docs/EXPERIMENT_RUNBOOK_WINDOWS.md). It requires separately authorized original data and the StageC/StageD1 directory layout.

## Reproduced results vs. inputs still needed

Published aggregate table values can be found under [`results_summary/`](results_summary/) and are copied from the revised manuscript / supplied experiment result files, **not recomputed by installing this package**. Do not present them as independently reproduced metrics. In particular, do not compare the 0.5332 architecture-selection AUPRC with the 0.3298 repeated-seed ablation AUPRC as if they were the same experimental protocol.

## Repository structure

| Path | Purpose |
|---|---|
| `config/stage_d1_final_focal.json` | Audited final focal-training settings and OOF thresholds |
| `config/experiment_registry.json` | Separate protocols and evaluation periods |
| `docs/EXPERIMENT_RUNBOOK_WINDOWS.md` | Exact observed Stage D1/V3 commands and required inputs |
| `docs/REVIEWER_TRACEABILITY.md` | Reviewer issues and evidence/limitations |
| `docs/UPLOAD_GUIDE.md` | Safe upload to existing GitHub repo |
| `docs/ARTIFACT_INVENTORY.md` | What is provided versus still missing |
| `docs/DATA_ACCESS.md` | Operational data-sharing safeguards |
| `docs/RESULTS_GUIDE.md` | Tables 2–9; non-interchangeability caveats |
| `results_summary/` | Aggregate paper tables and observed 2025 V3 evaluation CSV |
| `scripts/reviewer_imputation_policies_v3.py` | Original-flag-anchored 0/2/5 nested restriction experiment |
| `scripts/check_release_package.py` | Checks presence of public documentation and high-risk files |
| `scripts/check_aggregate_metrics.py` | Validates 60-row V3 aggregate confusion matrices and Table 8 mean |
| `examples/generate_synthetic_llwas.py` | Clearly fictional minute data demonstration |
| `tests/test_v3_nested.py` | Original synthetic V3 tests |

Other existing project source folders (`data`, `evaluation`, `models`, `runner`, `training`, and other scripts) must be checked against the local canonical study engine before promising an end-to-end automated rerun.

## Citation and code availability

If citing the implementation, link to this repository and the matching **commit SHA**. A permanent archive tag/release should be created only after the full contents and permissions have been reviewed. Use the manuscript's formal bibliographic details once published; do not invent a DOI or license.

## Contact, data access and license

This project uses observations gathered for an airport operational monitoring system. **Do not upload raw LLWAS records or derived per-minute operational time series without written permission from the data holder.** No software license is asserted by this documentation package; choose and add a `LICENSE` only with the code owners' approval.
