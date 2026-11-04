# Aggregate table index and interpretation

The source-of-truth is the **latest user-provided revised IJIES manuscript** and its accompanying completed reviewer experiment result files. This folder duplicates aggregate evidence for machine reading; it **does not independently regenerate research output**.

| Paper table | CSV in `results_summary/` | Interpretation |
|---|---|---|
| Table 1 | table1_chronological_folds.csv | Validation period and fold membership totals |
| Table 2 | table2_architecture_selection.csv | Optuna/weighted-BCE architecture ranking, max AUPRC 0.5332 |
| Table 3 | table3_feature_representations.csv | F1/F2/F3, fixed larger backbone |
| Table 4 | table4_imbalance_losses.csv | Development pooled-OOF loss comparison |
| Table 5 | table5_monthly_2025.csv | Jan–Mar 2025 temporal stratification |
| Table 6 | table6_sensor_imputation.csv | Sensor-specific external imputation flags |
| Table 7 | table7_exposure_exclusion_deltas.csv | Exclusion minus full-sample metrics, each horizon |
| Table 8 | table8_nested_imputation_common_anchors.csv | 68,488 common anchors, 0/2/5 min restrictions |
| Table 9 | table9_component_ablation.csv | 4 capacity-matched variants × 5 seeds × 5 folds |

`external_all_metrics.csv` records 60 rows (three policies × ten horizons × two comparison populations), including confusion matrices and thresholds; `oof_summary.csv` contains three pooled development summaries. Do not infer significance from mean metrics; no paired block-bootstrap across 0/2/5 variants was reported.

Important: The architecture selection (0.5332), focal 5-seed ablation (0.3298), and imputation experiment (0.4397 mean matched external AUPRC for policy 5) used different model/protocol/aggregation contexts. Comparing these numbers as direct estimates of a common benchmark would be misleading.
