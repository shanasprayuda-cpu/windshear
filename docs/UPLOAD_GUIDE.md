# Upload these materials to the existing GitHub repository

**Repository:** https://github.com/shanasprayuda-cpu/windshear

This package is a **proposed documentation/source overlay** and has not been pushed to GitHub. The public GitHub folder listing was visible; individual remote code files could not all be compared.

1. Make a local clone or backup of your existing repo, and inspect the original `models/` and `scripts/` sources. Compare with `docs/EXISTING_CODE_MERGE_NOTE.md`; avoid overwriting canonical model/training code blindly.
2. Add `README.md`, `.gitignore`, `requirements.txt`, `environment.example.yml`, `config/*.json`, `docs/*`, `examples/*`, `scripts/check_release_package.py`, `scripts/check_aggregate_metrics.py` and `results_summary/*`.
3. If missing from the actual repository, review and then add `scripts/reviewer_imputation_policies_v3.py`, `models/tcn_bilstm.py`, `scripts/run_final_focal_pipeline.py`, and `tests/test_v3_nested.py`.
4. **Do not publish `supplementary_for_owner_review/` until permission to disclose network geometry is confirmed.** It is ignored by the supplied `.gitignore`; share privately with the journal if appropriate. Never publish raw or per-minute LLWAS data without documented data-owner permission.
5. Run `python scripts/check_release_package.py --root .`, `python scripts/check_aggregate_metrics.py --results results_summary`, `python tests/test_v3_nested.py`, and `python scripts/reviewer_imputation_policies_v3.py --stage smoke`. These are local code/aggregate checks only, not an independent training reproduction.
6. Check `git status` and `git diff --stat` carefully; `.gitignore` does **not** retroactively remove tracked data. Use a new branch/review request if you are unsure.
7. Commit the reviewed changes. Record the actual commit SHA in the manuscript and response letter; do not claim full end-to-end reproducibility while private inputs/artifacts remain unavailable.

Example Windows commands (after inspecting files):

```bat
git switch -c ijies-reproducibility-docs
python scripts\check_release_package.py --root .
python scripts\check_aggregate_metrics.py --results results_summary
git status --short
git diff --stat
```

**Publishing via GitHub web UI:** Open your repository, choose Add file → Upload files or create new file, use the proposed relative paths and verify the content. Upload **only** reviewed public-appropriate files.
