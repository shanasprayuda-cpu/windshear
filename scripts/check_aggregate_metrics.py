#!/usr/bin/env python3
"""Audit published V3 aggregate CSV arithmetic, no private input data required."""
import argparse
import csv
from pathlib import Path


def audit(folder):
    with (folder/'external_all_metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(rows) != 60:
        raise ValueError(f'Expected 60 rows, got {len(rows)}')
    seen=set()
    for r in rows:
        policy=int(r['policy']);h=int(r['horizon']);comparison=r['comparison']
        key=(policy,h,comparison)
        if key in seen:raise ValueError(f'Duplicate: {key}')
        seen.add(key)
        if policy not in (0,2,5) or h not in range(1,11) or comparison not in ('policy_eligible','matched_common_anchors'):
            raise ValueError(f'Unexpected identity {key}')
        tp,fp,fn,tn=(int(r[k]) for k in ('TP','FP','FN','TN'))
        n=int(r['N']);positive=int(r['Positives'])
        if tp+fp+fn+tn!=n or tp+fn!=positive:raise ValueError(f'Confusion matrix mismatch {key}')
        expected={'CSI':tp/(tp+fp+fn) if (tp+fp+fn) else 0,
                  'POD':tp/(tp+fn) if (tp+fn) else 0,
                  'FAR':fp/(tp+fp) if (tp+fp) else 0}
        for metric,value in expected.items():
            if abs(float(r[metric])-value)>1e-9:
                raise ValueError(f'{metric} mismatch {key}')
    for policy in (0,2,5):
        subset=[r for r in rows if int(r['policy'])==policy and r['comparison']=='matched_common_anchors']
        if len(subset)!=10 or any(int(r['N'])!=68488 for r in subset):
            raise ValueError(f'Invalid matched common anchors for {policy}')
    means={}
    for policy in (0,2,5):
        subset=[r for r in rows if int(r['policy'])==policy and r['comparison']=='matched_common_anchors']
        means[policy]={metric:sum(float(r[metric]) for r in subset)/10 for metric in ('AUPRC','CSI','POD','FAR')}
    with (folder/'table8_nested_imputation_common_anchors.csv').open(newline='',encoding='utf-8') as f:
        tab=list(csv.DictReader(f))
    for row in tab:
        policy=int(row['Restriction'].split()[0])
        for metric in ('AUPRC','CSI','POD','FAR'):
            if abs(float(row['Mean '+metric])-means[policy][metric])>0.000055:
                raise ValueError(f'Table 8 {metric} does not match 60-row CSV policy {policy}')
    return means


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results_summary'))
    means=audit(p.parse_args().results)
    print('PASS: 60 external metric rows, 3 policies, 2 populations, 10 horizons; confusion matrices and Table 8 means consistent.')
    for policy,metrics in means.items():print('policy',policy,{k:round(v,4) for k,v in metrics.items()})
    print('Aggregate-data audit only — does NOT reproduce model fitting or independent inference.')
