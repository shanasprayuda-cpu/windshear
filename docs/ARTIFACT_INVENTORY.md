# Required and available artifact inventory

## Included in this public documentation package

- `models/tcn_bilstm.py`, `scripts/run_final_focal_pipeline.py`: locally provided model/final-focal scripts; **compare to public repo before overwriting**.
- `config/stage_d1_final_focal.json`: frozen known hyperparameters and development-derived thresholds.
- `config/experiment_registry.json`: protocol separation, seeds, folds, and policy counts.
- `scripts/reviewer_imputation_policies_v3.py`, `tests/test_v3_nested.py`: original-flag-anchored nested-restriction experiment and synthetic tests.
- `results_summary/table2_...` through `table9_...`: aggregate tables transcribed from the latest Word manuscript.
- `results_summary/external_all_metrics.csv`: 60 aggregated V3 evaluation rows supplied by researcher, not raw operational data.
- `results_summary/oof_summary.csv`: 3 aggregated policy OOF rows supplied by researcher.
- `supplementary_for_owner_review/S1_45_predictor_definitions.csv`: 45-feature dictionary; review sharing permissions before publishing.

## Important artifacts NOT included in this kit

Original 10-second LLWAS observations; original minute wind u/v and imputation flags; exact timestamp-level Stage C fold manifests; full `.npy` history and target windows; final checkpoint `.pt` or `.pth`; per-anchor predicted probabilities and target labels; all 30-trial Optuna histories per architecture; 100 component-ablation checkpoints and seed-level predictions; bootstrap implementation and sampling replicates; original exact dependency lock.

These are not silently replaced by synthetic files, reconstructed numbers or guessed URLs. For restricted operational data, consider data-access procedures and privacy-preserving synthetic examples.
