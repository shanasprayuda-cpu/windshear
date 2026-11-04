# Before committing/publishing this GitHub release

- [ ] Review **all** source files for API keys, usernames, absolute paths, operational identifiers, sensitive data and hidden credentials.
- [ ] Obtain permission to publish any LLWAS-derived source records, sensor geometry, location metadata, checkpoint weights, or per-minute predictions; omit these files if not authorized.
- [ ] Confirm current public repository code actually corresponds to audited StageA/B/C/D1 study scripts. The kit does **not** assert that independently.
- [ ] Replace example dependency versions with a safely reviewed freeze from the original `windshear_gpu` environment; record Python, PyTorch, CUDA/driver and OS versions.
- [ ] Verify main training and all model imports against a fresh clone plus separately authorized input data.
- [ ] Verify final 10 OOF thresholds and 69-anchor-minute fold purges against original run logs.
- [ ] Confirm one-time 2025 external-test claims are phrased as *temporally separated retrospective* assessment.
- [ ] Check and attach authorized Supplement S1 to the journal submission; do not rely only on an unverified GitHub copy.
- [ ] Verify all table CSV transcriptions against final (not superseded) manuscript.
- [ ] Add real, owner-approved `LICENSE` if code distribution terms are known; do not add an assumed MIT license.
- [ ] Create a release tag/commit SHA only after verifying final code/manifest contents and update journal + reviewer-response references.
- [ ] Keep `docs/REPRODUCIBILITY.md` explicit about unavailable masks, exact folds, checkpoints, full predictions and bootstrap script.
