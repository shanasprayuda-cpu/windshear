# Reviewer traceability and verification status

Companion to IJIES Paper ID 20265914; read alongside the current revised manuscript (20-page 2 Nov draft). This file is a **status register**, not proof all comments are fully satisfied.

| Reviewer | Comment | Main response location | Evidence / limitation |
|---|---|---|---|
| R1 | 1. Purge overlapping time windows | §2.4, Table 1 | 69-anchor-minute purge; raw timestamps |
| R1 | 2. Development-only selection / one-time test | §§2.4–2.7, §3.10 | Final fit/threshold dev only; prior 2025 exploratory inspection admitted |
| R1 | 3. All 5 architectures | §3.1, Table 2 | Five rows and model-ranking results |
| R1 | 4. Exact predictor tensors | §2.3, Supplement S1 | 45 features; supplement must be authorized and attached |
| R1 | 5. Missingness and sensitivity | §§2.2, 2.8, 3.8; Tables 6–8 | Original 5-minute flags, exclusion, V3 nested 0/2/5; not independent raw preprocessing |
| R1 | 6. Dependence-aware paired uncertainty | §§2.8, 3.7 | 500 paired 24/48/72h blocks; uncertainty intervals exploratory |
| R1 | 7. LLWAS–PIREP contingency | §3.6, §3.10 | Invalid PPV claim withdrawn; full contingency unavailable |
| R1 | 8. Machine-readable full pipeline | §2.9, this kit | **Partial**; private manifests, masks, models, bootstrap code absent |
| R2 | 1. Novelty / recent baselines | Introduction, §3.10 | Novelty reframed as protocol; no same-data recent baseline reimplementation |
| R2 | 2. Strict one-time external | §2.4, §3.10 | **Partial**; 2025 explored in prior work |
| R2 | 3. Seasonal generalization | §3.5, Table 5 | Jan–Mar month stratification; not independent seasons/sites |
| R2 | 4. Ablation and multi-seed | §2.5, §3.9, Table 9 | 4×5×5=100 fits; augmentation/threshold policy not isolated |
| R2 | 5. Event-level verification | §§2.8, 3.7 | 65 reference LLWAS alert episodes; lead and false alerts |
| R2 | 6. Scope of windshear claim | Abstract, Introduction, §4 | Operational LLWAS alert states only |

**Before resubmission:** verify the point-by-point response letter matches the revised manuscript, fix formatting/editor remarks, attach authorized Supplement S1, and satisfy the editor's resubmission timing and package instructions.
