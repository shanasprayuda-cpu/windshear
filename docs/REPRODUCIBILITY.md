# Reproducibility status and explicit limitations

**Status: partial public implementation, not yet a turnkey end-to-end independent replication.**

| Item reviewer requested | Documented in paper / kit | Machine-readable public archive verified? |
|---|---|---|
| Feature input order, dimensions, derived geometry | Sections 2.2–2.3; Supplement S1 review copy | Supplement provided in this kit for owner approval |
| Fold periods and time-purge logic | Section 2.4 and Table 1 | Detailed fold membership *files*: **not included** |
| Development-locked final focal settings and thresholds | `config/stage_d1_final_focal.json` | Config JSON **included** |
| Full architecture-search trials and checkpoints | Table 2, methods | **Not included** |
| Exact original Stage A preprocess / masks | Methods; sensor-level Table 6 | Raw minute masks / private Stage A source **not included** |
| Focal loss OOF/full training code | Existing engine script seen locally | Existing repo version **not independently validated** |
| Original 2025 per-anchor model predictions | Aggregate metrics | **Not included** |
| New nested 0/2/5 sensitivity experiment source | V3 script and smoke tests | **Included**; original data absent |
| 0/2/5 per-horizon aggregate metrics | `results_summary/external_all_metrics.csv` | **Included**, not rerun here |
| 5-seed ablation predictions/checkpoints | Table 9 aggregate | **Not included** |
| Bootstrap 24/48/72h code and replicate outputs | Sections 2.8/3.7 | **Not included or audited for public reproduction** |
| Environment with precise versions, CUDA details | Minimal package declarations | **Exact original version lock not included** |

Never describe these available files as proof that anyone can reproduce the tables solely by cloning the public GitHub repository. Create a tagged release and supply authorized artifacts only after permission, provenance and reproducibility audits.

**Evaluation caveat:** The development period (2023–2024) did not use external labels during final training/OOF calibration, but the 2025 time period was examined during broader exploratory work. It is best characterized as chronologically separated *retrospective* testing.
