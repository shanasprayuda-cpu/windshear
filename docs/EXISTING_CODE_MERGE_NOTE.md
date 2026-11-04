# IMPORTANT — compare existing GitHub code before overwriting

The package includes `models/tcn_bilstm.py` and `scripts/run_final_focal_pipeline.py` copied from local source artifacts supplied during manuscript development. The live public GitHub site was not fully available for source-to-source comparison. **Do not blindly overwrite** identically named files already present in `models/` or `scripts/`. Compare checksums, imports, and behavior against the canonical StageD1_v6.4.4 engine first.

If the repository already has an equivalent model and final pipeline, retain the correctly audited original and upload only missing documentation and reviewer V3 code. The bundled files are not evidence of a full public reproducibility run.
