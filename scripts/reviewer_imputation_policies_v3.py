#!/usr/bin/env python3
"""Reviewer 1: original-flag-anchored NESTED imputation restriction (0/2/5 minutes).

This is a NEW experiment, separate from the existing 2025 imputation-exposure
exclusion analysis. Never substitute the old exclusion results for this test.

Source: original Stage-A *one-minute*, post-5-min-fill CSV with ten original
per-sensor *_imputed flags. Undo ONLY flagged fills, then retain eligible original filled runs.
The five-minute baseline is EXACTLY the original Stage-A data (not replay-inferred).
The nested policy MUST pass run-level structural audits and match the baseline and
canonical Stage-C/Stage-D1 labels; otherwise execution stops.

Data: F1 = 10 sensors x (u,v), lookback 60 min, H+1..H+10.
Train: original Conv1D-BiLSTM, focal alpha=0.6 gamma=2, AdamW fixed LR,
       five original purged folds, 50 epochs, development-only OOF thresholds.
Test: after OOF and full training, Jan-Mar 2025 once per scenario; also report
      identical-timestamp (common-anchor) comparisons.

Windows use minute-level NumPy memory maps to avoid storing three enormous
(847513,60,20) tensors. All train/val/test anchors are subsets of the original
Stage-C eligibility manifests; NO folds or target definitions are invented.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

SENSORS = tuple([f'A{i}' for i in range(2,7)]+[f'B{i}' for i in range(2,7)])
FEATURES = tuple(f'{s}_{axis}_ms' for s in SENSORS for axis in ('u','v'))
FLAGS = tuple(f'{s}_imputed' for s in SENSORS)
POLICIES = (0, 2, 5)
LOOKBACK = 60
HORIZONS = 10
SEED = 2026
EXPECTED_DEV = 847_513
EXPECTED_EXT = 70_810
LR = 0.00036954127204552557
WEIGHT_DECAY = 4.98413619067967e-05
ALPHA = 0.6
GAMMA = 2.0
BATCH = 128
EPOCHS = 50
FOLD_COUNTS = {1:(331300,125212),2:(456512,124394),3:(580975,81241),4:(662216,106429),5:(768645,78868)}


def write_json(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    tmp.replace(p)


def file_id(p):
    p=Path(p)
    stat=p.stat()
    return {'path':str(p.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def require_file(path):
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing required file: {p}')
    return p


def as_binary(series, name):
    v=pd.to_numeric(series,errors='coerce')
    if v.isna().any() or not v.isin((0,1)).all():
        raise ValueError(f'{name}: all rows must contain explicit 0/1 values')
    return v.to_numpy(dtype=np.uint8)


def utc_minutes(series, name):
    ts=pd.to_datetime(series,utc=True,errors='coerce')
    if ts.isna().any():
        raise ValueError(f'{name}: invalid UTC timestamps')
    ts=ts.dt.tz_convert(None)
    if not (ts.dt.second.eq(0)&ts.dt.microsecond.eq(0)).all():
        raise ValueError(f'{name}: timestamps must be exact whole UTC minutes')
    return ts.to_numpy(dtype='datetime64[m]')


def read_minute(path):
    path=require_file(path)
    cols=list(pd.read_csv(path,nrows=0).columns)
    required=['timestamp','timestamp_present','label','label_valid',*FEATURES,*FLAGS]
    missing=[c for c in required if c not in cols]
    if missing:
        raise ValueError(f'{path.name} lacks original Stage-A columns: {missing}. '
                         'Need 5-minute processed Stage-A minute output WITH original per-sensor imputation flags; '
                         'X.npy or predictions alone are insufficient.')
    use=required+([c for c in ('any_imputed','all_sensors_available') if c in cols])
    df=pd.read_csv(path,usecols=use,low_memory=False)
    timestamps=utc_minutes(df['timestamp'],path.name)
    mins=timestamps.astype(np.int64)
    if len(mins)>1 and not np.all(np.diff(mins)==1):
        ix=int(np.where(np.diff(mins)!=1)[0][0])
        raise ValueError(f'{path.name}: minute timeline not continuous at row {ix}. '
                         'Add globally absent minutes as explicit rows; never bridge gaps by forward-fill.')
    present=as_binary(df['timestamp_present'],'timestamp_present')
    lv=as_binary(df['label_valid'],'label_valid')
    y=pd.to_numeric(df['label'],errors='coerce').to_numpy(dtype=np.float32)
    if np.any((lv==1)&(~np.isin(y,[0,1]))):
        raise ValueError('Invalid positive/negative operational target where label_valid=1')
    if np.any((present==0)&(lv!=0)):
        raise ValueError('Globally absent timestamps cannot carry valid target labels')
    wind=df[list(FEATURES)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
    flags=np.stack([as_binary(df[c],c) for c in FLAGS],axis=1).astype(bool)
    if np.any(flags & (present[:,None]==0)):
        raise ValueError('Imputation flag on a globally absent timestamp')
    if 'any_imputed' in df:
        any_flag=as_binary(df['any_imputed'],'any_imputed')
        if not np.array_equal(any_flag,flags.any(axis=1).astype(np.uint8)):
            raise ValueError('any_imputed differs from OR of sensor-specific imputation flags')
    if 'all_sensors_available' in df:
        available=as_binary(df['all_sensors_available'],'all_sensors_available')
        if not np.array_equal(available, np.isfinite(wind).all(axis=1).astype(np.uint8)):
            raise ValueError('Stage-A all_sensors_available inconsistent with UV fields')
    flagged_features=np.repeat(flags,2,axis=1)
    if not np.isfinite(wind[flagged_features]).all():
        raise ValueError('A flagged filled sensor has non-finite u/v (unexpected Stage-A data)')
    # Undo forward fill: a sensor flag means BOTH wind components were filled.
    restored=wind.copy()
    restored[flagged_features]=np.nan
    return {'times':timestamps, 'wind_observed':restored,'wind_original':wind,
            'flags':flags,'present':present,'label_valid':lv,'label':y,
            'path':str(path.resolve()),'rows':len(df)}


def bounded_fill(raw, present, cap):
    """Generic synthetic-control fill implementation; NOT used for real data in V3.

    Fill complete *runs* of <= cap; do NOT partially fill longer runs.

    A run cannot cross a globally absent minute. A missing (u,v) pair is the
    minimal eligible unit. Missing only one component is never silently filled.
    """
    if cap not in POLICIES:
        raise ValueError('Only audited policies 0,2,5 are permitted')
    x=np.array(raw,copy=True,dtype=np.float32)
    n,dim=x.shape
    if dim!=20:
        raise ValueError('Wind array must have 20 columns')
    imputed=np.zeros((n,10),dtype=bool)
    # One pass per sensor, vectorized over individual short runs.
    for j in range(10):
        i=2*j
        paired_missing=~np.isfinite(x[:,i]) & ~np.isfinite(x[:,i+1])
        partial=(np.isfinite(x[:,i]) ^ np.isfinite(x[:,i+1])) & (present==1)
        if partial.any():
            raise ValueError(f'{SENSORS[j]}: partial u/v missing pairs need original Stage-A normalization')
        missing=paired_missing & (present==1)
        if cap==0 or not missing.any():
            continue
        starts=np.flatnonzero(missing & ~np.r_[False,missing[:-1]])
        ends=np.flatnonzero(missing & ~np.r_[missing[1:],False])+1
        for start,end in zip(starts,ends):
            if end-start>cap or start==0 or present[start-1]!=1:
                continue
            seed=x[start-1,i:i+2]
            if np.isfinite(seed).all():
                x[start:end,i:i+2]=seed
                imputed[start:end,j]=True
    return x,imputed



def original_flag_nested_policy(data, cap, audit=True):
    """STRICT nested restriction of the *observed original* Stage-A five-minute fills.

    Five-minute variant is unchanged original Stage A outputs, copied verbatim.
    Zero-minute variant removes all observed originally imputed sensor pairs.
    Two-minute variant retains complete ORIGINAL flagged runs of length <=2
    and removes originally imputed runs of length 3..5. It NEVER attempts to
    fill originally missing unflagged minutes (the source of 28 discrepancies).

    This is a controlled 'restriction of original fills' sensitivity experiment,
    not proof of re-running the unavailable Stage-A source code with new caps.
    Refuse any ambiguous partial original filled gap or >5-minute filled run.
    """
    if cap not in POLICIES: raise ValueError('Invalid nested policy')
    original=data['wind_original']
    raw=data['wind_observed']
    flags=data['flags']
    present=data['present']
    n=len(original)
    if original.shape!=(n,20) or flags.shape!=(n,10):
        raise ValueError('Unexpected 20 components / 10 masks')
    if np.any(flags & (present[:,None]!=1)):
        raise ValueError('Stage-A source flags on globally absent minute')
    assert np.array_equal(np.isfinite(original), np.isfinite(raw) | np.repeat(flags,2,axis=1))
    result=raw.copy()
    retained=np.zeros_like(flags)
    original_run_counts={str(i):0 for i in range(1,6)}
    sensors_report=[]
    for sensor in range(10):
        pair=slice(2*sensor, 2*sensor+2)
        fl=flags[:,sensor]
        # Check original observed raw data is either a complete pair or missing pair.
        p=np.isfinite(raw[:,pair])
        partial=(p[:,0]!=p[:,1]) & (present==1)
        if np.any(partial):
            ix=int(np.flatnonzero(partial)[0])
            raise ValueError(f'{SENSORS[sensor]}: partial original raw u/v at row {ix}; STOP')
        missing=(~p[:,0]) & (present==1)
        # Every flag run must be exactly an entire raw-missing run,
        # never a partial run (which would obscure original policy semantics).
        begins=np.flatnonzero(fl & ~np.r_[False,fl[:-1]])
        ends=np.flatnonzero(fl & ~np.r_[fl[1:],False])+1
        for b,e in zip(begins,ends):
            ln=int(e-b)
            if ln>5:
                raise ValueError(f'{SENSORS[sensor]}: original Stage-A flagged run {b}:{e} length {ln}>5; STOP')
            if b==0 or present[b-1]!=1 or not np.isfinite(raw[b-1,pair]).all():
                raise ValueError(f'{SENSORS[sensor]}: original fill at {b}:{e} missing a valid source seed; STOP')
            if not missing[b:e].all():
                raise ValueError(f'{SENSORS[sensor]}: source flagged nonmissing raw wind {b}:{e}; STOP')
            if b>0 and missing[b-1]:
                raise ValueError(f'{SENSORS[sensor]}: ORIGINAL imputed run begins inside longer missing gap, first={b}; STOP')
            if e<n and missing[e]:
                raise ValueError(f'{SENSORS[sensor]}: ORIGINAL imputed run ends before raw missing gap, last={e-1}; STOP')
            if not np.allclose(original[b:e,pair],raw[b-1,pair][None,:],rtol=1e-6,atol=1e-6):
                raise ValueError(f'{SENSORS[sensor]}: original filled values not forward copies of prior raw pair at {b}; STOP')
            original_run_counts[str(ln)]+=1
            if ln<=cap:
                retained[b:e,sensor]=True
                result[b:e,pair]=original[b:e,pair]
        sensors_report.append({'sensor':SENSORS[sensor], 'original_filled_minutes':int(fl.sum()),
                               'retained_filled_minutes':int(retained[:,sensor].sum())})
    if cap==5:
        if not np.array_equal(retained,flags):
            raise ValueError('Nested 5-minute retained mask differs from source original flags; STOP')
        if not np.allclose(result,original,atol=1e-6,rtol=1e-6,equal_nan=True):
            raise ValueError('Nested 5-minute result differs from source original Stage A values; STOP')
        # Preserve exact original bit patterns/float32 as exposed to Stage B.
        result=original.copy()
    if cap==0 and (retained.any() or not np.allclose(result,raw,equal_nan=True)):
        raise ValueError('Nested 0-minute MUST equal original unfilled data')
    if np.any(retained & ~flags):
        raise ValueError('Counterfactual introduced fills absent in original Stage A; STOP')
    if not np.array_equal(np.isfinite(result), np.isfinite(raw) | np.repeat(retained,2,axis=1)):
        raise ValueError('Counterfactual has unexplained nonfinite output; STOP')
    info={'policy':cap,'method':'original_stageA_flag_anchored_complete_run_restriction',
          'not_an_independent_stageA_10second_reprocessing':True,
          'original_filled_run_counts_by_length':original_run_counts,
          'original_imputed_sensor_minutes':int(flags.sum()),
          'retained_imputed_sensor_minutes':int(retained.sum()),
          'per_sensor':sensors_report}
    return result,retained,info

def verify_replayed_5(data, replay, masks):
    original=data['wind_original']
    if not np.array_equal(np.isfinite(original), np.isfinite(replay)):
        bad=int(np.flatnonzero((np.isfinite(original)!=np.isfinite(replay)).any(axis=1))[0])
        raise ValueError(f'5-min baseline reconstruction mismatch in availability at raw-minute row {bad}')
    same=np.isclose(original,replay,atol=1e-6,rtol=1e-6,equal_nan=True)
    if not np.all(same):
        bad=int(np.flatnonzero(~np.all(same,axis=1))[0])
        raise ValueError(f'5-min baseline reconstruction mismatched existing Stage A at row {bad}. '
                         'Different gap-filling semantics detected: do NOT train until reconciled.')
    if not np.array_equal(masks,data['flags']):
        first=int(np.flatnonzero(np.any(masks!=data['flags'],axis=1))[0])
        raise ValueError(f'5-min baseline imputation flags differ at row {first}; '
                         'cannot claim a controlled alternative policy with these inputs')


def raw_indices_from_manifest(path,minutes,expected,kind):
    path=require_file(path)
    cols=pd.read_csv(path,nrows=0).columns.tolist()
    needed=['anchor_idx','anchor_time']+[f'y_h{i}' for i in range(1,11)]
    missing=[c for c in needed if c not in cols]
    if missing:raise ValueError(f'{path}: missing {missing}')
    m=pd.read_csv(path,usecols=needed,low_memory=False)
    if len(m)!=expected:raise ValueError(f'{kind}: canonical manifest has {len(m)} windows, expected {expected}')
    anchors=pd.to_numeric(m['anchor_idx'],errors='raise').to_numpy(dtype=np.int64)
    if len(np.unique(anchors))!=len(anchors) or (anchors<59).any() or (anchors+10>=len(minutes)).any():
        raise ValueError(f'{kind}: duplicate or out-of-bounds Stage-C raw anchor indices')
    if not np.array_equal(utc_minutes(m['anchor_time'],'anchor_time'),minutes[anchors]):
        raise ValueError(f'{kind}: anchor_idx/anchor_time do not match supplied Stage-A minute timeline. '
                         'Likely WRONG SOURCE CSV or timeline origin; do NOT proceed.')
    y=m[[f'y_h{i}' for i in range(1,11)]].to_numpy(dtype=np.float32)
    if not np.isin(y,[0,1]).all():raise ValueError(f'{kind}: manifest targets are not binary')
    return anchors,y


def validate_all_target_labels(data, anchors, manifest_y):
    # Only original Stage-C candidates are checked, preserving the exact target.
    for h in range(1,11):
        ix=anchors+h
        good=(data['present'][ix]==1)&(data['label_valid'][ix]==1)
        if not np.all(good):
            raise ValueError(f'Original Stage-C includes {int((~good).sum())} invalid H+{h} labels')
        if not np.array_equal(data['label'][ix], manifest_y[:,h-1]):
            raise ValueError(f'Stage-A operational labels disagree with original Stage-C H+{h} targets')


def baseline_window_check(data,anchors,kind,stagec,data_root):
    """Confirm exact temporal sample/label alignment with locked Stage-D1 data."""
    source=Path(data_root)/'F1' if kind=='dev' else Path(data_root)/'F1_external'
    xp=source/'X.npy'
    if not xp.is_file():
        raise FileNotFoundError(f'{kind}: canonical Stage-D1 5-min baseline {xp} is required for verification')
    X=np.load(xp,mmap_mode='r')
    if X.shape!=(len(anchors),60,20):
        raise ValueError(f'{kind}: canonical Stage-D1 X shape {X.shape} differs from manifest')
    rng=np.random.default_rng(2026)
    sample=np.unique(np.r_[np.arange(min(5,len(X))), np.linspace(0,len(X)-1,128,dtype=int),
                           rng.integers(0,len(X),size=min(512,len(X)))])
    replay=data['wind_original']
    for k in sample:
        a=int(anchors[k]); actual=np.asarray(X[k]); expected=replay[a-59:a+1]
        if not np.allclose(actual,expected,rtol=1e-6,atol=1e-6,equal_nan=True):
            raise ValueError(f'{kind}: 5-min replay vs canonical X.npy mismatch at manifest row {k}, anchor {a}')
    print(f'   {kind}: audited {len(sample)} baseline X windows against original Stage-D1 X.npy',flush=True)
    yp=source/'y.npy'
    if not yp.is_file():raise FileNotFoundError(f'{kind}: canonical Stage-D1 {yp} missing')
    Y=np.load(yp,mmap_mode='r')
    if Y.shape!=(len(X),10):raise ValueError(f'{kind}: original y shape mismatch')
    return Y


def valid_anchors(x, present, anchors):
    good=(present==1)&np.isfinite(x).all(axis=1)
    bad_count=(~good).astype(np.int32)
    cumulative=np.r_[0,np.cumsum(bad_count,dtype=np.int64)]
    return (cumulative[anchors+1]-cumulative[anchors-59])==0


def audit_stagec_folds(stagec,anchors):
    order=np.argsort(anchors)
    sorted_a=anchors[order]
    fold_rows={}
    for f in range(1,6):
        pairs={}
        for role in ('train','val'):
            file=stagec/'folds'/f'fold_{f}_{role}.csv'
            df=pd.read_csv(require_file(file),usecols=['anchor_idx'])
            vals=df['anchor_idx'].to_numpy(dtype=np.int64)
            pos=np.searchsorted(sorted_a,vals)
            if np.any(pos>=len(sorted_a)) or not np.array_equal(sorted_a[pos],vals):
                raise ValueError(f'Fold {f} {role}: anchors absent from canonical Stage-C development manifest')
            pairs[role]=order[pos]
            pairs[role+'_anchor']=vals
        if (len(pairs['train']),len(pairs['val']))!=FOLD_COUNTS[f]:
            raise ValueError(f'Fold {f} counts differ from audited Stage C: {len(pairs["train"])} / {len(pairs["val"])}')
        if len(np.intersect1d(pairs['train'],pairs['val'])):
            raise ValueError(f'Fold {f}: identical training/validation anchors')
        if max(pairs['train_anchor'])+10 >= min(pairs['val_anchor'])-59:
            raise ValueError(f'Fold {f}: 60-min history/10-min target raw-timestamp overlap')
        fold_rows[f]=pairs
    return fold_rows


def save_array(path,x):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,np.asarray(x))


def prep_one(kind, src, manifest, root,stagec,data_root):
    data=read_minute(src)
    print(f'[{kind}] Stage-A minute rows: {data["rows"]:,}',flush=True)
    anchors,y=raw_indices_from_manifest(manifest,data['times'],EXPECTED_DEV if kind=='dev' else EXPECTED_EXT,kind)
    validate_all_target_labels(data,anchors,y)
    Y=baseline_window_check(data,anchors,kind,stagec,data_root)
    if not np.array_equal(np.asarray(Y,dtype=np.float32),y):
        raise ValueError(f'{kind}: canonical Stage-D1 y.npy is not the exact Stage-C target')
    baseline_orig_good=valid_anchors(data['wind_original'],data['present'],anchors)
    if not baseline_orig_good.all():
        raise ValueError(f'{kind}: Stage-C canonical 5-min baseline contains incomplete historical input window')
    # V3 strict alternative: preserve original Stage-A five-minute output exactly.
    # Independently re-filling missing rows would introduce 28 non-original fills.
    # First AUDIT EVERY source filled run, then write anything.
    policies={}
    for cap in POLICIES:
        policies[cap]=original_flag_nested_policy(data,cap)
    x0,f0,info0=policies[0]
    x2,f2,info2=policies[2]
    x5,f5,info5=policies[5]
    if not np.array_equal(f5,data['flags']):raise ValueError('Original five-minute flags changed; STOP')
    if not np.allclose(x5,data['wind_original'],rtol=0,atol=0,equal_nan=True):
        raise ValueError('Original five-minute values changed; STOP')
    if (f2&~f5).any() or (f0&~f2).any():
        raise ValueError('Nested imputation masks not monotonic')
    valid_by_cap={cap:valid_anchors(policies[cap][0],data['present'],anchors) for cap in POLICIES}
    if not valid_by_cap[5].all():raise ValueError('V3 five-minute policy lost canonical anchors')
    if np.any(valid_by_cap[0]&~valid_by_cap[2]) or np.any(valid_by_cap[2]&~valid_by_cap[5]):
        raise ValueError('0/2/5 eligible windows not nested; STOP')
    print(f'[{kind}] audited ORIGINAL Stage-A 5-minute output, all imputed runs, locked X/y, and nested restrictions PASS',flush=True)
    out=root/'prepared'/kind
    out.mkdir(parents=True,exist_ok=True)
    save_array(out/'canonical_anchors.npy',anchors)
    for cap in POLICIES:
        xx,flags,info=policies[cap]
        valid=valid_by_cap[cap]
        save_array(out/f'uv_policy_{cap}.npy',xx)
        save_array(out/f'valid_rows_policy_{cap}.npy',np.flatnonzero(valid).astype(np.int64))
        print(f'  cap {cap}: {int(valid.sum()):,}/{len(valid):,} canonical eligible anchors; '
              f'{int(flags.any(axis=1).sum()):,} original-flag-filled source minutes retained',flush=True)
    meta={'kind':kind,'source':file_id(src),'manifest':file_id(manifest),
          'canonical_X_and_y_verified':True,'five_min_original_baseline_unchanged':True,
          'original_windows':len(anchors),'original_minute_rows':data['rows'],
          'policy_valid_counts':{str(cap):int(valid_by_cap[cap].sum()) for cap in POLICIES},
          'original_stageA_filled_run_counts':info5['original_filled_run_counts_by_length'],
          'original_stageA_imputed_sensor_minutes':info5['original_imputed_sensor_minutes'],
          'raw_10second_available':False,
          'reconstruction_source':'Original Stage-A one-minute UV + exact original per-sensor imputation flags',
          'experiment_type':'Nested restriction of ORIGINAL Stage-A 5-minute fills (0, <=2, <=5); NOT independent Stage-A reprocessing',
          'reviewer_caveat':'Sensitivity to retaining original Stage-A filled runs of length <=0/2/5. The original unfilled Stage-A rejected gaps are NEVER newly filled.'}
    write_json(out/'audit.json',meta)
    if kind=='dev':
        folds=audit_stagec_folds(stagec, anchors)
        supports=[]
        for cap in POLICIES:
            allow=np.zeros(len(anchors),dtype=bool)
            allow[np.load(out/f'valid_rows_policy_{cap}.npy')]=True
            for fold,pair in folds.items():
                for role in ('train','val'):
                    rows=pair[role][allow[pair[role]]]
                    support=y[rows]
                    rec={'policy':cap,'fold':fold,'role':role,'N':len(rows)}
                    for h in range(10):
                        positives=int(support[:,h].sum())
                        rec[f'H{h+1}_positive']=positives
                        rec[f'H{h+1}_negative']=len(rows)-positives
                        if role=='val' and (positives==0 or positives==len(rows)):
                            raise ValueError(f'policy={cap}, fold={fold}, H+{h+1}: '
                                             'validation contains only one class; AP undefined')
                    supports.append(rec)
        pd.DataFrame(supports).to_csv(out/'development_support_audit.csv',index=False)
        print('  Fold-by-horizon class-support audit PASS; output: development_support_audit.csv',flush=True)
    return data, anchors


class MinuteDataset(Dataset):
    """Separate memmaps per worker: safe with Windows spawn multiprocessing."""
    def __init__(self,minute_path,y_path,anchor_path,manifest_rows):
        self.minute_path=str(minute_path)
        self.y_path=str(y_path)
        self.anchor_path=str(anchor_path)
        self.rows=np.asarray(manifest_rows,dtype=np.int64)
        self._minute=None;self._labels=None;self._anchors=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if self._minute is None:
            self._minute=np.load(self.minute_path,mmap_mode='r')
            self._labels=np.load(self.y_path,mmap_mode='r')
            self._anchors=np.load(self.anchor_path,mmap_mode='r')
        k=int(self.rows[index]); a=int(self._anchors[k]);x=self._minute[a-59:a+1]
        if x.shape!=(60,20) or not np.isfinite(x).all():
            raise RuntimeError(f'Invalid window in data loader at row {k}')
        return torch.from_numpy(np.array(x,dtype=np.float32,copy=True)),torch.from_numpy(
            np.array(self._labels[k],dtype=np.float32,copy=True))


def make_loader(path,y_path,anchor_path,rows,args,train=False):
    gen=torch.Generator().manual_seed(SEED)
    opts=dict(dataset=MinuteDataset(path,y_path,anchor_path,rows), batch_size=args.batch,
              shuffle=train, num_workers=args.workers,pin_memory=torch.cuda.is_available(),
              drop_last=False,generator=gen)
    if args.workers>0:opts.update(persistent_workers=True,prefetch_factor=4)
    return DataLoader(**opts)


def set_seed(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def make_model(engine_root):
    if str(engine_root) not in sys.path:sys.path.insert(0,str(engine_root))
    from models.tcn_bilstm import TCNBiLSTM
    model=TCNBiLSTM(hidden_size=32,num_layers=1,dropout=0.25)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count!=41994:raise ValueError(f'Unexpected final model parameters {count}; expected 41,994')
    return model


def focal_loss(logits,y):
    bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    p=torch.sigmoid(logits)
    pt=p*y+(1-p)*(1-y)
    at=ALPHA*y+(1-ALPHA)*(1-y)
    return (at*(1-pt).clamp(min=1e-8).pow(GAMMA)*bce).mean()


def predict(model,loader,device):
    model.eval(); yy=[];pp=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=device.type=='cuda')
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(xb)
            pp.append(torch.sigmoid(logits).float().cpu().numpy())
            yy.append(yb.numpy().astype(np.uint8))
    if not pp:raise ValueError('Prediction loader is empty')
    return np.concatenate(yy),np.concatenate(pp)


def train(engine_root,out,prefix,minute_path,y_path,anchor_path,rows,args):
    done=out/'COMPLETE.json'
    ck=out/'last_epoch.pt'
    if done.is_file() and ck.is_file() and not args.overwrite:
        meta=json.loads(done.read_text(encoding='utf8'))
        if (meta.get('data_policy')==prefix and meta.get('epochs')==args.epochs
                and meta.get('n_train')==len(rows) and meta.get('batch_size')==args.batch
                and meta.get('minute_array_identity')==file_id(minute_path)
                and meta.get('original_y_identity')==file_id(y_path)):
            print(f'  SKIP trained: {out}',flush=True)
            return ck
        raise RuntimeError(f'Existing checkpoint {ck} does not match requested configuration; use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    set_seed()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(engine_root).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    loader=make_loader(minute_path,y_path,anchor_path,rows,args,True)
    history=[];start=time.monotonic()
    for ep in range(1,args.epochs+1):
        total=0.;count=0;model.train()
        for xb,yb in loader:
            xb=xb.to(dev,non_blocking=dev.type=='cuda')
            yb=yb.to(dev,non_blocking=dev.type=='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
                logits=model(xb);loss=focal_loss(logits,yb)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss for policy {prefix}, epoch {ep}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.item())*len(xb);count+=len(xb)
        history.append({'epoch':ep,'loss':total/count})
        if ep==1 or ep%10==0 or ep==args.epochs:
            print(f'    {prefix}: epoch {ep}/{args.epochs}, loss={total/count:.6f}',flush=True)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    torch.save({'model_state_dict':model.cpu().state_dict(),'policy':prefix,'epochs':args.epochs,'seed':SEED,
                'lr':LR,'weight_decay':WEIGHT_DECAY,'loss':'Focal','alpha':ALPHA,'gamma':GAMMA},ck)
    write_json(done,{'data_policy':prefix,'epochs':args.epochs,'n_train':len(rows),
                     'seed':SEED,'batch_size':args.batch,'model_parameters':41994,
                     'minute_array_identity':file_id(minute_path),'original_y_identity':file_id(y_path),
                     'duration_seconds':round(time.monotonic()-start,3),
                     'development_only':True,'external_accessed':False})
    del model,loader
    if dev.type=='cuda':torch.cuda.empty_cache()
    return ck


def get_predictions(engine_root,ck,minute_path,y_path,anchor_path,rows,args):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(engine_root).to(dev)
    state=torch.load(ck,map_location=dev,weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    ds=make_loader(minute_path,y_path,anchor_path,rows,args,False)
    y,p=predict(model,ds,dev)
    del ds,model
    if dev.type=='cuda':torch.cuda.empty_cache()
    return y,p


def score(y,p,t):
    binary=p>=t
    tp=int(np.sum(binary & (y==1)));fp=int(np.sum(binary & (y==0)))
    fn=int(np.sum((~binary)&(y==1)));tn=int(np.sum((~binary)&(y==0)))
    total=tp+fp+fn
    return {'CSI':tp/total if total else 0.,'POD':tp/(tp+fn) if tp+fn else 0.,
            'FAR':fp/(tp+fp) if tp+fp else 0.,'TP':tp,'FP':fp,'FN':fn,'TN':tn}


def find_threshold(y,p):
    """Development-only exact 0.001..0.999 CSI grid search without O(N*999)."""
    order=np.argsort(p,kind='stable')
    ys=y[order].astype(np.uint8);ps=p[order]
    positives=int(ys.sum());n=len(ys)
    cs=np.cumsum(ys,dtype=np.int64)
    cut=np.searchsorted(ps,np.arange(1,1000,dtype=np.float64)/1000.,side='left')
    below=np.where(cut>0,cs[np.maximum(cut-1,0)],0)
    tp=positives-below
    pred=n-cut
    fp=pred-tp
    fn=positives-tp
    denom=tp+fp+fn
    csi=np.divide(tp,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0)
    pod=np.divide(tp,positives,out=np.zeros_like(tp,dtype=float),where=positives!=0)
    far=np.divide(fp,pred,out=np.zeros_like(tp,dtype=float),where=pred!=0)
    best=np.lexsort((np.arange(len(csi)),far,-pod,-csi))[0]
    return float((best+1)/1000.)


def metrics(y,p,threshold):
    if y.sum()<=0 or y.sum()>=len(y):
        raise ValueError('AUPRC undefined: validation/test set contains only one class; do not report NaN')
    r=score(y,p,threshold)
    r.update(AUPRC=float(average_precision_score(y,p)),ROC_AUC=float(roc_auc_score(y,p)),
             N=len(y),Positives=int(y.sum()),threshold=float(threshold))
    return r


def ensure_prepared(root,kind):
    p=root/'prepared'/kind/'audit.json'
    if not p.is_file():raise FileNotFoundError(f'{p} missing; run --stage prepare first')
    meta=json.loads(p.read_text(encoding='utf8'))
    if meta.get('experiment_type')!='Nested restriction of ORIGINAL Stage-A 5-minute fills (0, <=2, <=5); NOT independent Stage-A reprocessing':
        raise ValueError(f'{kind} audit is from a different policy experiment; STOP')
    if not meta.get('five_min_original_baseline_unchanged') or not meta.get('canonical_X_and_y_verified'):
        raise ValueError(f'{kind} prepare audit incomplete')
    return meta


def read_prepared(root,kind,cap):
    p=root/'prepared'/kind
    return p/f'uv_policy_{cap}.npy',p/'canonical_anchors.npy',np.load(p/f'valid_rows_policy_{cap}.npy')


def make_dev_folds(stagec,anchors,policy_valid_rows):
    f=audit_stagec_folds(stagec,anchors)
    eligible=np.zeros(len(anchors),dtype=bool)
    eligible[policy_valid_rows]=True
    folds={}
    for fold,pair in f.items():
        tr=pair['train'][eligible[pair['train']]]
        va=pair['val'][eligible[pair['val']]]
        if not len(tr) or not len(va):
            raise ValueError(f'Policy removed all train/val windows from fold {fold}')
        if anchors[tr].max()+10 >= anchors[va].min()-59:
            raise ValueError(f'Policy has timestamp leakage in fold {fold}')
        folds[fold]=(tr,va)
    return folds


def oof(args,engine,stagec,stage_d1,root):
    ensure_prepared(root,'dev')
    orig_anchors=np.load(root/'prepared'/'dev'/'canonical_anchors.npy',mmap_mode='r')
    y_path=stage_d1/'F1'/'y.npy'
    summary=[]
    for cap in POLICIES:
        x_path,a_path,rows=read_prepared(root,'dev',cap)
        folds=make_dev_folds(stagec,orig_anchors,rows)
        all_ys=[];all_ps=[];all_rows=[]
        for fold,(tr,va) in folds.items():
            run=root/f'policy_{cap}'/'oof'/f'fold_{fold}'
            print(f'policy={cap} fold={fold} train={len(tr):,} val={len(va):,}',flush=True)
            predfile=run/'validation_preds.npz'
            ck=train(engine,run,str(cap),x_path,y_path,a_path,tr,args)
            if predfile.is_file() and not args.overwrite:
                with np.load(predfile) as saved:
                    if not np.array_equal(saved['rows'],va):
                        raise ValueError(f'Existing predictions are for different fold rows: {predfile}')
                    yt=saved['y'];prob=saved['p']
            else:
                yt,prob=get_predictions(engine,ck,x_path,y_path,a_path,va,args)
                np.savez_compressed(predfile,rows=va,y=yt,p=prob)
            if len(yt)!=len(va) or prob.shape!=(len(va),10):
                raise ValueError('Validation predictions shape mismatch')
            all_ys.append(yt);all_ps.append(prob);all_rows.append(va)
        yy=np.concatenate(all_ys);pp=np.concatenate(all_ps);which=np.concatenate(all_rows)
        if len(np.unique(which))!=len(which):raise ValueError('OOF anchor appears in two validation folds')
        threshold=[]; rows_metrics=[]
        for h in range(10):
            t=find_threshold(yy[:,h],pp[:,h]);threshold.append(t)
            r=metrics(yy[:,h],pp[:,h],t)
            rows_metrics.append({'policy':cap,'horizon':h+1,**r})
        out=root/f'policy_{cap}'/'oof'
        pd.DataFrame(rows_metrics).to_csv(out/'oof_metrics.csv',index=False)
        write_json(out/'thresholds.json',{'policy':cap,'thresholds':threshold,
                   'selected_from':'development-only pooled OOF','method':'max CSI; ties max POD then min FAR',
                   'grid':'0.001..0.999','N_OOF':len(yy),'no_external_data_accessed':True,'epochs':args.epochs,
                   'seed':SEED})
        summary.append({'policy':cap,'total_OOF':len(yy),'mean_OOF_AUPRC':float(np.mean([r['AUPRC'] for r in rows_metrics]))})
    pd.DataFrame(summary).to_csv(root/'oof_summary.csv',index=False)
    print('Development-only OOF completed; NO external 2025 data opened in this stage.',flush=True)


def full(args,engine,stage_d1,root):
    ensure_prepared(root,'dev')
    y_path=stage_d1/'F1'/'y.npy'
    for cap in POLICIES:
        threshold=root/f'policy_{cap}'/'oof'/'thresholds.json'
        if not threshold.is_file():raise FileNotFoundError(f'{threshold} missing; run --stage oof first')
        x_path,a_path,rows=read_prepared(root,'dev',cap)
        train(engine,root/f'policy_{cap}'/'full',str(cap),x_path,y_path,a_path,rows,args)
    print('Full development fits complete. External period has NOT been opened.',flush=True)


def external(args,engine,stagec,stage_d1,root):
    for cap in POLICIES:
        if not (root/f'policy_{cap}'/'full'/'COMPLETE.json').is_file():
            raise FileNotFoundError(f'Policy {cap} full fit missing. Run --stage full first.')
        if not (root/f'policy_{cap}'/'oof'/'thresholds.json').is_file():
            raise FileNotFoundError(f'Policy {cap} OOF thresholds missing')
    if not args.ext_minute:
        raise ValueError('--ext-minute required ONLY for --stage external / all')
    prep_one('external',args.ext_minute,stagec/'manifest'/'external_test_windows.csv',
             root,stagec,stage_d1)
    ypath=stage_d1/'F1_external'/'y.npy'
    cap_records=[]; cached={}; predictions={}
    anchors=np.load(root/'prepared'/'external'/'canonical_anchors.npy')
    common_rows=np.load(root/'prepared'/'external'/'valid_rows_policy_0.npy')
    for cap in POLICIES:
        x_path,a_path,rows=read_prepared(root,'external',cap)
        ck=root/f'policy_{cap}'/'full'/'last_epoch.pt'
        tcfg=json.loads((root/f'policy_{cap}'/'oof'/'thresholds.json').read_text(encoding='utf8'))
        if not tcfg['no_external_data_accessed']:
            raise ValueError('Thresholds must be development-only')
        th=tcfg['thresholds']
        yt,prob=get_predictions(engine,ck,x_path,ypath,a_path,rows,args)
        cap_records.append((cap,rows,yt,prob,th))
        out=root/f'policy_{cap}'/'external'
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'external_predictions.npz',anchor_idx=anchors[rows],rows=rows,y=yt,p=prob)
        rowscore=[]
        for h in range(10):
            rowscore.append({'policy':cap,'comparison':'policy_eligible', 'horizon':h+1,
                            **metrics(yt[:,h],prob[:,h],th[h])})
        pd.DataFrame(rowscore).to_csv(out/'external_metrics_policy_eligible.csv',index=False)
        cached[cap]=dict(rows=rows,y=yt,p=prob,th=th)
    common_scores=[];detail=[]
    for cap in POLICIES:
        z=cached[cap];rows=z['rows'];pos=np.searchsorted(rows,common_rows)
        if np.any(pos>=len(rows)) or not np.array_equal(rows[pos],common_rows):
            raise ValueError('Common-anchor expected subset of every policy; ordering mismatch')
        y=z['y'][pos];p=z['p'][pos];th=z['th']
        for h in range(10):
            common_scores.append({'policy':cap,'comparison':'matched_common_anchors',
                                  'horizon':h+1,**metrics(y[:,h],p[:,h],th[h])})
        detail.append({'policy':cap,'N_policy_eligible':len(rows),'N_common':len(common_rows)})
    pd.DataFrame(common_scores).to_csv(root/'external_metrics_common_anchors.csv',index=False)
    pd.DataFrame(detail).to_csv(root/'external_policy_support.csv',index=False)
    pd.concat([pd.read_csv(root/f'policy_{cap}'/'external'/'external_metrics_policy_eligible.csv')
               for cap in POLICIES]+[pd.DataFrame(common_scores)]).to_csv(root/'external_all_metrics.csv',index=False)
    write_json(root/'external_audit.json',{'policies':list(POLICIES),'shared_anchor_count':len(common_rows),
        'note':'External test inspected retrospectively during broader research; not fully blind; compare STRICT NESTED restriction of original imputation flags',
        'comparison_types':['all_eligible_per_policy','matched_common_anchors'],
        'tuning':'OOF thresholds from development only, each cap independently; no external threshold tuning'})
    print('EXTERNAL test finished. Results saved to external_all_metrics.csv',flush=True)


def locate(stagec, engine):
    base=engine.parent
    roots=[base/x for x in ('stageA','stageA_v1_2','StageA','stageB','stageB_uv','stageB_data','external_2025','stageA_external')]
    found=[]
    for root in roots:
        if root.exists():
            for path in root.rglob('*.csv'):
                try:
                    cols=set(pd.read_csv(path,nrows=0).columns)
                    if {'timestamp','A2_u_ms','A2_imputed','B6_v_ms','B6_imputed'}.issubset(cols):
                        found.append(str(path))
                except (OSError,UnicodeError,ValueError):pass
    print('Stage-A minute tables with original per-sensor imputation flags:')
    for p in found:print(' ',p)
    if not found:
        print('  NONE automatically located. Search your Stage-A output folder for a minute CSV '
              'containing timestamp, 20 *_u_ms/*_v_ms, ten *_imputed flags, label, label_valid.')
    print('Stage C manifest:',stagec/'manifest'/'development_windows.csv')


def smoke(engine):
    present=np.ones(18,dtype=np.uint8)
    present[8]=0
    x=np.full((18,20),np.nan,dtype=np.float32)
    x[0:3]=1
    x[3:5,2:4]=np.nan
    x[5:8]=1
    x[9:12]=2
    x[12:15]=2
    x[15:18]=2
    # Run-length = two, no fill at cap0, fill cap2.
    a=np.ones((12,20),dtype=np.float32);a[3:5,0:2]=np.nan;a[8:11,2:4]=np.nan
    b0,_=bounded_fill(a,np.ones(12,dtype=np.uint8),0)
    b2,_=bounded_fill(a,np.ones(12,dtype=np.uint8),2)
    b5,mask=bounded_fill(a,np.ones(12,dtype=np.uint8),5)
    assert np.isnan(b0[3:5,0:2]).all() and np.isfinite(b2[3:5,0:2]).all()
    assert np.isnan(b2[8:11,2:4]).all() and np.isfinite(b5[8:11,2:4]).all()
    assert mask[8:11,1].all()
    gap=np.ones((8,20),dtype=np.float32);gap[2:4,0:2]=np.nan
    pres=np.ones(8,dtype=np.uint8);pres[1]=0;gap[1,:]=np.nan
    r,_=bounded_fill(gap,pres,5)
    assert np.isnan(r[2:4,0:2]).all(),'Fill bridged globally absent minute!'
    # synthetic nested policy: one 2-min flagged run + one 4-min run.
    sd=np.ones((16,20),dtype=np.float32)
    sf=np.zeros((16,10),dtype=bool)
    sf[3:5,0]=True;sf[9:13,1]=True
    orig=sd.copy()
    raw=orig.copy();raw[3:5,0:2]=np.nan;raw[9:13,2:4]=np.nan
    nested={'wind_original':orig,'wind_observed':raw,'flags':sf,'present':np.ones(16,dtype=np.uint8)}
    a0,f0,_=original_flag_nested_policy(nested,0)
    a2,f2,_=original_flag_nested_policy(nested,2)
    a5,f5,_=original_flag_nested_policy(nested,5)
    assert np.isnan(a0[3:5,0:2]).all() and np.isnan(a0[9:13,2:4]).all()
    assert np.isfinite(a2[3:5,0:2]).all() and np.isnan(a2[9:13,2:4]).all()
    assert np.isfinite(a5[3:5,0:2]).all() and np.isfinite(a5[9:13,2:4]).all()
    assert not f0.any() and f2[3:5,0].all() and not f2[9:13,1].any() and np.array_equal(f5,sf)
    # synthetic UNFILLED gap: bounded fill would incorrectly invent an original Stage-A fill.
    raw[6:8,4:6]=np.nan;orig[6:8,4:6]=np.nan
    nested={'wind_original':orig,'wind_observed':raw,'flags':sf,'present':np.ones(16,dtype=np.uint8)}
    for cap in POLICIES:
        nx,nf,_=original_flag_nested_policy(nested,cap)
        assert np.isnan(nx[6:8,4:6]).all() and not nf[6:8,2].any()
    model=make_model(engine)
    inp=torch.ones(2,60,20)
    out=model(inp)
    assert out.shape==(2,10) and torch.isfinite(out).all()
    loss=focal_loss(out,torch.zeros(2,10));loss.backward()
    print('PASS synthetic gap-policy smoke, missing-timestamp safety, model forward/backward; NO research metrics.')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage',choices=['locate','smoke','prepare','oof','full','external','all'],default='locate')
    ap.add_argument('--engine-root',default='.',help='Stage D1 engine root containing models/tcn_bilstm.py')
    ap.add_argument('--stagec',default=None,help='Canonical stageC root (default engine-root parent/stageC)')
    ap.add_argument('--stage-d1-data',default=None,help='Original stageD1_data root (default engine-root parent/stageD1_data)')
    ap.add_argument('--dev-minute',default=None,help='Original Stage A dev minute CSV WITH ten sensor imputation flags')
    ap.add_argument('--ext-minute',default=None,help='Original Stage A 2025 minute CSV WITH ten sensor imputation flags')
    ap.add_argument('--results-dir',default=None)
    ap.add_argument('--batch',type=int,default=BATCH)
    ap.add_argument('--epochs',type=int,default=EPOCHS)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    engine=Path(args.engine_root).resolve()
    stagec=Path(args.stagec).resolve() if args.stagec else engine.parent/'stageC'
    stage_d1=Path(args.stage_d1_data).resolve() if args.stage_d1_data else engine.parent/'stageD1_data'
    root=Path(args.results_dir).resolve() if args.results_dir else engine/'results'/'reviewer_imputation_nested_0_2_5'
    if args.stage=='locate':return locate(stagec,engine)
    if not (engine/'models'/'tcn_bilstm.py').exists():
        ap.error(f'Original final TCNBiLSTM not found in {engine}/models/tcn_bilstm.py')
    if args.stage=='smoke':return smoke(engine)
    if args.workers<0 or args.batch<=0 or args.epochs<=0:ap.error('Invalid workers, batch, or epochs')
    if args.epochs!=50:
        print('WARNING: nonstandard epochs; resulting metrics are NOT comparable to final 50-epoch study',flush=True)
    root.mkdir(parents=True,exist_ok=True)
    protocol={'source':'Original Stage A one-minute UV and imputation flags; Stage C original manifest',
       'scope':'nested original-flag restriction sensitivity; not raw Stage-A gap refilling; not prior exposed/unexposed test',
       'policies_max_fill_minutes':[0,2,5], 'fill_semantics':'strict nested restriction of ORIGINAL Stage-A imputed runs; never synthesize new imputations or bridge original disallowed gaps',
       'lookback':60,'horizons':10,'features':list(FEATURES),'model':'original TCNBiLSTM Conv1D-BiLSTM 41994 params',
       'seed':SEED,'epochs':args.epochs,'batch':args.batch,'workers':args.workers,
       'optimizer':'AdamW','learning_rate':LR,'weight_decay':WEIGHT_DECAY,'scheduler':'none',
       'focal_alpha':ALPHA,'focal_gamma':GAMMA,'threshold_source':'per-policy development pooled OOF',
       'evaluation':['policy eligible','common anchors'],
       'engine':str(engine),'stagec':str(stagec),'stage_d1_data':str(stage_d1)}
    protocol['experiment_type']='NESTED originally-imputed run restriction, not source Stage-A reprocessing'
    existing=root/'protocol.json'
    if existing.exists():
        prev=json.loads(existing.read_text(encoding='utf8'))
        if prev.get('experiment_type')!=protocol['experiment_type']:
            raise ValueError('Previous experimental protocol at '+str(root)+' is different! Provide a NEW --results-dir for V3 (never mix incompatible outputs).')
    write_json(root/'protocol.json',protocol)
    if args.stage in ('prepare','all'):
        if not args.dev_minute:ap.error('--dev-minute is required for prepare/all')
        prep_one('dev',args.dev_minute,stagec/'manifest'/'development_windows.csv',root,stagec,stage_d1)
        a=np.load(root/'prepared'/'dev'/'canonical_anchors.npy')
        audit_stagec_folds(stagec,a)
        print('Development PREPARE PASS. All canonical folds remain unchanged.',flush=True)
    if args.stage in ('oof','all'):oof(args,engine,stagec,stage_d1,root)
    if args.stage in ('full','all'):full(args,engine,stage_d1,root)
    if args.stage in ('external','all'):external(args,engine,stagec,stage_d1,root)
    print('FINISHED stage:',args.stage,'RESULTS:',root,flush=True)

if __name__=='__main__':main()
