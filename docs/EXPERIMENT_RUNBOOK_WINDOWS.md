# Windows reproduction runbook (original local Stage D1 layout)

This is an **audit-first** runbook for the StageD1_v6.4.4 complete engine, not an assertion that a fresh public GitHub clone already contains all necessary datasets or code.

## 1. Environment capture (before any reproducibility claim)

```bat
conda activate windshear_gpu
python --version
python -m pip freeze > environment_original_freeze.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Review `environment_original_freeze.txt` for private URLs, local paths, credentials and internal infrastructure **before publishing**. `requirements.txt` in this kit is a minimal input list, not the frozen original environment.

## 2. Canonical data structure

The following **local** arrangement was observed in the research logs (drive prefix is user-specific):

```text
STUDY_ROOT/
  stageA_gap5/llwas_minute_preprocessed.csv           [private]
  stageB_uv/stageB_uv_minute_features.csv              [private]
  stageB2_physics/stageB2_physics_minute_features.csv  [private]
  stageC/manifest/development_windows.csv              [private]
  stageC/manifest/external_test_windows.csv            [private]
  stageC/manifest/fold_membership.csv                  [private]
  stageD1_data/F1/{X.npy,y.npy}                        [private]
  external_test_2025/stageB_uv/stageB_uv_minute_features.csv [private]
  StageD1_v6.4.4_Complete_Real_Training_Engine/
    models/tcn_bilstm.py
    scripts/run_final_focal_pipeline.py
    scripts/reviewer_imputation_policies_v3.py
    stageD1_data/F1_external/{X.npy,y.npy}             [private]
```

The final focal-loss script observed in the user's local environment uses StageD1_DATA under `STUDY_ROOT/stageD1_data/F1` and the Stage C manifest. It must be tested against actual provided code and filesystem layout; do not guess new paths.

## 3. Run core final focal experiment (requires private canonical inputs)

From the *StageD1_v6.4.4* engine directory:

```bat
python scripts\run_final_focal_pipeline.py --stage oof
python scripts\run_final_focal_pipeline.py --stage full
python scripts\run_final_focal_pipeline.py --stage external
```

The script's documented locked final settings are in `config/stage_d1_final_focal.json`. The 2025 period was inspected during broader exploratory work; **do not claim a wholly unseen test**.

## 4. Reviewer 1: nested imputation restriction (0/2/5 minutes)

Use the V3 script shipped in this package; **do not use the earlier failed strict replay script**. From the same engine directory, after copying the V3 script into its `scripts/` folder:

```bat
python scripts\reviewer_imputation_policies_v3.py --stage smoke
python scripts\reviewer_imputation_policies_v3.py --stage prepare --dev-minute "..\stageB_uv\stageB_uv_minute_features.csv"
python scripts\reviewer_imputation_policies_v3.py --stage oof --workers 4
python scripts\reviewer_imputation_policies_v3.py --stage full --workers 4
python scripts\reviewer_imputation_policies_v3.py --stage external --ext-minute "..\external_test_2025\stageB_uv\stageB_uv_minute_features.csv" --stage-d1-data ".\stageD1_data" --workers 4
```

Audited checkpoints before OOF training: development Stage A minute rows 1,052,640, 5-minute baseline 847,513 valid canonical windows; 644 development X windows checked against original X.npy. At external test: 123,432 minute rows, 70,810 baseline windows; 640 external X windows checked. A verified `PREPARE PASS` is mandatory; never bypass baseline-label assertions to obtain performance metrics.

Expected eligibility counts (data-dependent and not synthetic targets):

| Retained original imputation run max | Development eligible | External eligible |
|---|---:|---:|
| 0 minutes | 841,919 | 68,488 |
| 2 minutes | 846,118 | 70,369 |
| 5 minutes | 847,513 | 70,810 |

The 0/2/5 experiment **restricts original imputed values**, not newly replays three raw-data preprocessing algorithms. Final comparison on **68,488 matched common anchors** uses the same target timestamps, but different training samples and OOF-derived thresholds.

## 5. Reviewer 2: component-matched ablation

Documented 100 fits: 4 variants × 5 seeds `[11,22,33,44,55]` × 5 purged folds. The component training code and actual fold manifest must be checked on the canonical research engine before publishing exact run commands: these were not verifiably reconstructed from the public repository in this package. See `config/experiment_registry.json` and `results_summary/table9_component_ablation.csv`.

## 6. Release verification

Before public upload, check `docs/RELEASE_CHECKLIST.md` and run:

```bat
python scripts\check_release_package.py --root .
```

A PASS means the **documentation package checks** passed. It is **not** an independent rerun of reported research experiments.
