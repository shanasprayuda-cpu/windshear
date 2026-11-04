#!/usr/bin/env python3
"""Conservative pre-commit check; passing this is NOT research reproduction."""
import argparse
import json
from pathlib import Path

REQUIRED = [
    'README.md', 'requirements.txt', '.gitignore',
    'config/stage_d1_final_focal.json', 'config/experiment_registry.json',
    'docs/REPRODUCIBILITY.md', 'docs/ARTIFACT_INVENTORY.md',
    'docs/EXPERIMENT_RUNBOOK_WINDOWS.md', 'docs/REVIEWER_TRACEABILITY.md',
    'docs/DATA_ACCESS.md', 'docs/RESULTS_GUIDE.md',
    'models/tcn_bilstm.py', 'scripts/run_final_focal_pipeline.py',
    'scripts/check_aggregate_metrics.py',
    'scripts/reviewer_imputation_policies_v3.py',
    'examples/generate_synthetic_llwas.py',
]
RISKY_EXTENSIONS = {'.npy', '.npz', '.pt', '.pth', '.ckpt', '.parquet', '.pem', '.key', '.p12'}
RISKY_FOLDERS = {'stageA_gap5', 'stageB_uv', 'stageB2_physics', 'stageC',
                 'stageD1_data', 'external_test_2025'}

def audit(root: Path):
    missing = [f for f in REQUIRED if not (root/f).exists()]
    suspicious = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if path.suffix.lower() in RISKY_EXTENSIONS or any(p in RISKY_FOLDERS for p in rel.parts):
            suspicious.append(str(rel))
        if '.env' == path.name or path.name.startswith('.env.'):
            suspicious.append(str(rel))
    cfg = json.loads((root/'config/stage_d1_final_focal.json').read_text(encoding='utf-8')) if not missing else {}
    thresholds = cfg.get('threshold_selection', {}).get('thresholds_h1_to_h10', [])
    errors = []
    if len(thresholds) != 10: errors.append('Expected 10 OOF thresholds')
    if missing: errors.append(f'Missing required files: {missing}')
    if suspicious: errors.append(f'Potential restricted/private file paths: {suspicious[:20]}')
    return errors

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    args=parser.parse_args()
    errors=audit(args.root.resolve())
    if errors:
        for item in errors: print('FAIL:', item)
        raise SystemExit(1)
    print('PASS: Documentation package structure and simple private-file guardrails.')
    print('IMPORTANT: This is NOT a full secret scan, scientific audit or reproduction.')
