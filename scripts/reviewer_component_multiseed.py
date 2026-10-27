#!/usr/bin/env python
"""Reviewer-2 controlled component/capacity ablation + five-seed chronological CV.

Uses ORIGINAL physical-unit F1 windows and the ORIGINAL 5 purged Stage-C folds.
No external 2025 files are opened. NO model results are built into this script.

Python >=3.9, PyTorch >=2.0, numpy, pandas, scikit-learn.
Place this script in the engine's ``scripts`` folder, or pass --engine-root.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

VARIANTS = ("full", "bilstm_only", "conv_only", "conv_transformer")
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
REQUIRED_SHAPE = (847513, 60, 20)
EXPECTED_COUNTS = {1:(331300,125212),2:(456512,124394),3:(580975,81241),4:(662216,106429),5:(768645,78868)}
LR = 0.00036954127204552557
WEIGHT_DECAY = 4.98413619067967e-05
ALPHA = 0.6
GAMMA = 2.0
BATCH = 128
NUM_EPOCHS = 50


def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMOnly(nn.Module):
    """No Conv1D; one bidirectional recurrent layer, matched capacity."""
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(20,61,num_layers=1,bidirectional=True,batch_first=True)
        self.head = nn.Linear(122,10)
    def forward(self,x):
        z,_=self.rnn(x)
        return self.head(z[:,-1,:])


class ConvOnly(nn.Module):
    """No recurrence; use global temporal mean to exploit entire 60-min history."""
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(20,107,3,padding=1),nn.ReLU(),
                                nn.Conv1d(107,107,3,padding=1),nn.ReLU())
        self.head=nn.Linear(107,10)
    def forward(self,x):
        h=self.conv(x.transpose(1,2))
        return self.head(h.mean(dim=-1))


class ConvTransformer(nn.Module):
    """Same 64-channel conv front-end; one 4-head encoder; no future leakage."""
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(20,64,3,padding=1),nn.ReLU(),
                                nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        self.transformer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=64,
                                                     dropout=0.0,batch_first=True,activation='relu')
        self.head=nn.Linear(64,10)
        self.register_buffer('pos_encoding',self._positional_encoding(60,64),persistent=False)
    @staticmethod
    def _positional_encoding(length:int,d:int):
        pos=torch.arange(length,dtype=torch.float32).unsqueeze(1)
        terms=torch.exp(torch.arange(0,d,2,dtype=torch.float32)*(-np.log(10000.0)/d))
        enc=torch.zeros(length,d,dtype=torch.float32)
        enc[:,0::2]=torch.sin(pos*terms)
        enc[:,1::2]=torch.cos(pos*terms)
        return enc
    def forward(self,x):
        h=self.conv(x.transpose(1,2)).transpose(1,2)
        h=self.transformer(h+self.pos_encoding[:h.size(1)].to(h.dtype))
        return self.head(h[:,-1,:])


def build_model(kind:str,engine_root:Path):
    if kind=='full':
        sys.path.insert(0,str(engine_root))
        try:
            from models.tcn_bilstm import TCNBiLSTM
        except Exception as e:
            raise RuntimeError(f"Cannot import original models/tcn_bilstm.py from {engine_root}: {e}") from e
        return TCNBiLSTM(input_size=20,tcn_channels=64,hidden_size=32,
                          num_layers=1,dropout=0.25,output_size=10)
    if kind=='bilstm_only': return LSTMOnly()
    if kind=='conv_only': return ConvOnly()
    if kind=='conv_transformer': return ConvTransformer()
    raise ValueError(kind)


def focal_loss(logits,y):
    bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    p=torch.sigmoid(logits)
    pt=p*y+(1-p)*(1-y)
    alpha=ALPHA*y+(1-ALPHA)*(1-y)
    return (alpha*((1-pt).clamp(min=1e-8)**GAMMA)*bce).mean()


class LazyMemmapData(Dataset):
    """Lazy-open numpy arrays PER WORKER: avoids duplicating huge X.npy on Windows."""
    def __init__(self,x_path:Path,y_path:Path,rows:np.ndarray):
        self.x_path=str(x_path)
        self.y_path=str(y_path)
        self.rows=np.asarray(rows,dtype=np.int64)
        self._X=None
        self._y=None
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        if self._X is None:
            self._X=np.load(self.x_path,mmap_mode='r')
            self._y=np.load(self.y_path,mmap_mode='r')
        row=int(self.rows[index])
        return (torch.from_numpy(np.array(self._X[row],dtype=np.float32,copy=True)),
                torch.from_numpy(np.array(self._y[row],dtype=np.float32,copy=True)))


def dataloader(x_path,y_path,rows,batch,workers,shuffle,seed):
    g=torch.Generator()
    g.manual_seed(seed)
    kwargs=dict(dataset=LazyMemmapData(x_path,y_path,rows),batch_size=batch,shuffle=shuffle,
                drop_last=False,num_workers=workers,pin_memory=torch.cuda.is_available(),generator=g)
    if workers>0:
        kwargs.update(persistent_workers=True,prefetch_factor=4)
    return DataLoader(**kwargs)


def load_indices(root:Path,strict:bool=True):
    """Load the canonical per-fold Stage-C CSVs, as used by stagec_loader.py.

    IMPORTANT: ``stageC/manifest/fold_membership.csv`` may use different
    split-role spellings; it is NOT the authoritative source for this loader.
    No folds are regenerated or inferred here.
    """
    data=root.parent/'stageD1_data'/'F1'
    stagec=root.parent/'stageC'
    x_path=data/'X.npy'; y_path=data/'y.npy'
    dev_path=stagec/'manifest'/'development_windows.csv'
    for p in (x_path,y_path,dev_path):
        if not p.is_file(): raise FileNotFoundError(f'Missing required file: {p}')
    X=np.load(x_path,mmap_mode='r'); y=np.load(y_path,mmap_mode='r')
    if X.shape!=REQUIRED_SHAPE or y.shape!=(REQUIRED_SHAPE[0],10):
        raise ValueError(f'Wrong shapes: X={X.shape}; y={y.shape}; expected {REQUIRED_SHAPE} and ({REQUIRED_SHAPE[0]},10)')
    d=pd.read_csv(dev_path,usecols=['anchor_idx'])
    if d['anchor_idx'].isna().any():
        raise ValueError(f'Null anchor_idx in {dev_path}')
    anchors=d['anchor_idx'].to_numpy(dtype=np.int64)
    if len(anchors)!=len(X) or len(np.unique(anchors))!=len(anchors):
        raise ValueError('Development anchors are not unique or are not aligned in count with X.npy')
    order=np.argsort(anchors)
    sorted_a=anchors[order]

    def to_rows(vals,source):
        vals=np.asarray(vals,dtype=np.int64)
        if not len(vals): raise ValueError(f'Empty list of anchors in {source}')
        pos=np.searchsorted(sorted_a,vals)
        safe=np.minimum(pos,len(sorted_a)-1)
        found=(pos<len(sorted_a)) & (sorted_a[safe]==vals)
        if not np.all(found):
            examples=vals[~found][:10].tolist()
            raise ValueError(f'{len(vals)-int(found.sum())} anchors from {source} missing in development_windows.csv. Examples: {examples}')
        return order[pos].astype(np.int64)

    folds={}
    for fold in range(1,6):
        # Match the verified production stagec_loader.py:
        #   stageC/folds/fold_N_train.csv and fold_N_val.csv.
        train_file=stagec/'folds'/f'fold_{fold}_train.csv'
        val_file=stagec/'folds'/f'fold_{fold}_val.csv'
        for csv_file in (train_file,val_file):
            if not csv_file.is_file():
                raise FileNotFoundError(f'Canonical Stage-C fold file missing: {csv_file} (do NOT regenerate folds)')
        t=pd.read_csv(train_file,usecols=['anchor_idx'])['anchor_idx'].to_numpy(dtype=np.int64)
        v=pd.read_csv(val_file,usecols=['anchor_idx'])['anchor_idx'].to_numpy(dtype=np.int64)
        if len(t)==0 or len(v)==0:
            raise ValueError(f'Empty train/val membership for fold {fold}: train={len(t)}, val={len(v)}')
        if len(np.unique(t))!=len(t) or len(np.unique(v))!=len(v):
            raise ValueError(f'Duplicate anchor_idx inside train or validation for fold {fold}')
        tr=to_rows(t,train_file)
        va=to_rows(v,val_file)
        if len(np.intersect1d(tr,va)):
            raise ValueError(f'Overlapping train and validation rows in fold {fold}')
        # Complete raw support of one anchor a: historical [a-59,a], targets [a+1,a+10].
        # Non-overlap of those supports requires max(train_anchor)+10 < min(val_anchor)-59.
        if int(t.max())+10 >= int(v.min())-59:
            raise ValueError(f'Raw-timestamp supports overlap in fold {fold}: train max {t.max()}, val min {v.min()}')
        if strict and (len(t),len(v))!=EXPECTED_COUNTS[fold]:
            raise ValueError(f'Fold {fold} count {len(t)}/{len(v)} differs from audited expected {EXPECTED_COUNTS[fold]}')
        folds[fold]=dict(train=tr,val=va,train_anchor=t,val_anchor=v)
    check_rows=np.unique(np.linspace(0,len(X)-1,17,dtype=np.int64))
    if not np.isfinite(np.asarray(X[check_rows])).all():
        raise ValueError('Non-finite X values in check sample')
    vals=np.unique(np.asarray(y[check_rows]))
    if not np.isin(vals,[0,1]).all():
        raise ValueError(f'Non-binary target values in check sample: {vals}')
    print('Fold source: canonical stageC/folds/fold_N_train.csv and fold_N_val.csv',flush=True)
    return x_path,y_path,folds


def predict(model,loader,device):
    model.eval(); ybuf=[]; pbuf=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=device.type=='cuda')
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(xb)
            pbuf.append(torch.sigmoid(logits).float().cpu().numpy())
            ybuf.append(yb.numpy().astype(np.uint8))
    return np.concatenate(ybuf,axis=0),np.concatenate(pbuf,axis=0)


def safe_ap(y,p):
    y=np.asarray(y,dtype=np.uint8)
    return float(average_precision_score(y,p)) if (y.sum()>0 and y.sum()<len(y)) else float('nan')


def finite_mean(values):
    """Mean of defined metrics, or NaN without misleading NumPy warnings."""
    arr=np.asarray(values,dtype=float)
    good=arr[np.isfinite(arr)]
    return float(good.mean()) if good.size else float('nan')


def finite_std(values,ddof=0):
    """SD of defined metrics, or NaN when there are too few observations."""
    arr=np.asarray(values,dtype=float)
    good=arr[np.isfinite(arr)]
    return float(good.std(ddof=ddof)) if good.size>ddof else float('nan')


def audit_validation_targets(y_path,fold_data,mode):
    """Check event support on the FULL canonical validation folds, not pilot slices."""
    labels=np.load(y_path,mmap_mode='r')
    print('Full-fold validation target audit (positive labels H+1..H+10):',flush=True)
    missing=[]
    for fold,indices in fold_data.items():
        val=np.asarray(labels[indices['val']],dtype=np.uint8)
        n=val.shape[0]
        counts=val.sum(axis=0,dtype=np.int64)
        print(f' fold {fold}: positives={counts.tolist()}, N={n:,}',flush=True)
        for h,num in enumerate(counts,start=1):
            if num==0 or num==n: missing.append((fold,h,int(num)))
    if missing:
        msg=(f'UNDEFINED full-fold AUPRC: no class variation for {missing}. '
             'Do not report mean AUPRC without explicitly handling these horizons.')
        if mode=='run':raise ValueError(msg)
        print(' WARNING: '+msg,flush=True)
    else:
        print(' Full-fold class support PASS: all ten AUPRC values defined in all five folds.',flush=True)


def compute_scores(y,p,thr=0.5):
    q=(p>=thr)
    tp=int(np.sum(q&(y==1))); fp=int(np.sum(q&(y==0)))
    fn=int(np.sum((~q)&(y==1))); tn=int(np.sum((~q)&(y==0)))
    d=tp+fp+fn
    return dict(AUPRC=safe_ap(y,p),CSI=tp/d if d else 0.0,
                POD=tp/(tp+fn) if tp+fn else 0.0,FAR=fp/(tp+fp) if tp+fp else 0.0,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                TP=tp,FP=fp,FN=fn,TN=tn)


def optimal_threshold(y,p):
    """Efficient exact search on locked 0.001 ... 0.999 grid (ties POD up, FAR down)."""
    order=np.argsort(p,kind='stable')
    ps=np.asarray(p)[order]; ys=np.asarray(y,dtype=np.uint8)[order]
    pos=np.cumsum(ys,dtype=np.int64)
    positives=int(pos[-1]); n=len(ys)
    thresholds=np.arange(1,1000,dtype=np.float64)/1000.0
    cut=np.searchsorted(ps,thresholds,side='left')
    tp=positives-np.where(cut>0,pos[np.maximum(cut-1,0)],0)
    predicted=n-cut
    fp=predicted-tp; fn=positives-tp
    denom=tp+fp+fn
    csi=np.divide(tp,denom,out=np.zeros_like(tp,dtype=np.float64),where=denom!=0)
    pod=tp/positives if positives else np.zeros_like(csi)
    far=np.divide(fp,predicted,out=np.zeros_like(csi),where=predicted!=0)
    best=sorted(range(len(thresholds)),key=lambda i:(-csi[i],-pod[i],far[i],i))[0]
    return float(thresholds[best])


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)


def train_fold(root,x_path,y_path,fold_id,indices,variant,seed,args,run_dir):
    fold_dir=run_dir/f'fold_{fold_id}'
    done=fold_dir/'COMPLETE.json'
    pred_path=fold_dir/'validation_predictions.npz'
    if done.is_file() and pred_path.is_file() and not args.overwrite:
        print(f'  SKIP complete: {fold_dir}',flush=True)
        return
    fold_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=='pilot':
        # Pilot subset only; development manifest remains unchanged.
        tr=indices['train'][-min(1024,len(indices['train'])):]
        va=indices['val'][:min(512,len(indices['val']))]
        epochs=2
    else:
        tr=indices['train'];va=indices['val'];epochs=args.epochs
    seed_everything(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(variant,root).to(device)
    params=sum(v.numel() for v in model.parameters() if v.requires_grad)
    base_params=41994
    if abs(params-base_params)/base_params>0.03:
        raise RuntimeError(f'Unmatched capacity: {variant}: {params} parameters; baseline {base_params}')
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    tr_loader=dataloader(x_path,y_path,tr,args.batch,args.workers,True,seed)
    start=time.monotonic()
    hist=[]
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for xb,yb in tr_loader:
            xb=xb.to(device,non_blocking=device.type=='cuda')
            yb=yb.to(device,non_blocking=device.type=='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(xb); loss=focal_loss(logits,yb)
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss in {variant}, seed{seed}, fold{fold_id}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach().item())*len(xb);count+=len(xb)
        epoch_loss=total/count
        hist.append({'epoch':epoch,'train_loss':epoch_loss})
        if epoch==1 or epoch%10==0 or epoch==epochs:
            print(f'  {variant} seed={seed} fold={fold_id} epoch={epoch}/{epochs} loss={epoch_loss:.6f}',flush=True)
    pd.DataFrame(hist).to_csv(fold_dir/'training_history.csv',index=False)
    va_loader=dataloader(x_path,y_path,va,args.batch,args.workers,False,seed)
    yt,yp=predict(model,va_loader,device)
    if yt.shape!=(len(va),10) or yp.shape!=(len(va),10):
        raise RuntimeError('Unexpected prediction shape')
    horizon_metrics=[]
    for h in range(10):
        scores=compute_scores(yt[:,h],yp[:,h],0.5)
        horizon_metrics.append({'horizon':h+1,'N':len(va),'positives':int(yt[:,h].sum()),'threshold':0.5,**scores})
    class_counts=[m['positives'] for m in horizon_metrics]
    undefined=[h+1 for h,n_pos in enumerate(class_counts) if n_pos==0 or n_pos==len(va)]
    print(f'  {variant} seed={seed} fold={fold_id} validation positives H+1..H+10: {class_counts}',flush=True)
    if undefined:
        msg=(f'Undefined AUPRC for horizons {undefined}: validation subset has only one class. '
             'This is expected in a short pilot if positives are rare; pilot scores are NOT journal results.')
        if args.mode=='run':raise ValueError(msg)
        print('  PILOT NOTE: '+msg,flush=True)
    pd.DataFrame(horizon_metrics).to_csv(fold_dir/'validation_metrics_fixed05.csv',index=False)
    np.savez_compressed(pred_path,y_true=yt,p_prob=yp.astype(np.float32),
                        anchor_idx=indices['val_anchor'][:len(va)] if args.mode!='pilot' else indices['val_anchor'][:len(va)])
    meta={'variant':variant,'seed':seed,'fold':fold_id,'epochs':epochs,'batch_size':args.batch,
          'training_samples':len(tr),'validation_samples':len(va),'trainable_parameters':params,
          'optimizer':'AdamW','learning_rate':LR,'weight_decay':WEIGHT_DECAY,
          'loss':'Focal','alpha':ALPHA,'gamma':GAMMA,'checkpoint':'last_epoch',
          'mean_fold_AUPRC':finite_mean([m['AUPRC'] for m in horizon_metrics]),
          'runtime_seconds':round(time.monotonic()-start,2),'pilot_only':args.mode=='pilot',
          'external_test_accessed':False,'parameter_count_relative_difference':round((params-base_params)/base_params,6)}
    # A complete run can be safely skipped in future invocations.
    torch.save({'variant':variant,'seed':seed,'fold':fold_id,'epoch':epochs,
                'model_state_dict':model.cpu().state_dict(),'configuration':meta},fold_dir/'last_epoch_model.pt')
    save_json(done,meta)
    del tr_loader,va_loader,model
    if torch.cuda.is_available():torch.cuda.empty_cache()


def report_one_run(root_dir:Path,variant:str,seed:int,folds:list[int]):
    run_dir=root_dir/variant/f'seed_{seed}'
    ylist=[];plist=[];foldscores=[]
    for fold in folds:
        fp=run_dir/f'fold_{fold}'/'validation_predictions.npz'
        if not fp.exists():
            print(f'Missing fold {fold}: {fp}');return None
        with np.load(fp) as v:
            y=v['y_true'];p=v['p_prob']
        ylist.append(y);plist.append(p)
        foldscores.append(finite_mean([safe_ap(y[:,h],p[:,h]) for h in range(10)]))
    y=np.concatenate(ylist);p=np.concatenate(plist)
    records=[]
    for h in range(10):
        yi=y[:,h];pi=p[:,h]
        thr=optimal_threshold(yi,pi)
        d={'variant':variant,'seed':seed,'horizon':h+1,'N':len(yi),'n_positive':int(yi.sum()),
           'threshold_OOF_CSI':thr,'AUPRC':safe_ap(yi,pi)}
        d.update({'fixed05_'+k:v for k,v in compute_scores(yi,pi,0.5).items() if k!='AUPRC'})
        d.update({'OOF_CSI_optimized_'+k:v for k,v in compute_scores(yi,pi,thr).items() if k!='AUPRC'})
        records.append(d)
    pd.DataFrame(records).to_csv(run_dir/'pooled_OOF_horizon_metrics.csv',index=False)
    info={'variant':variant,'seed':seed,'folds':folds,'N_OOF':int(len(y)),
          'mean_fold_horizon_AUPRC':finite_mean(foldscores),
          'sd_fold_mean_AUPRC_ddof0':finite_std(foldscores,ddof=0),
          'mean_pooled_horizon_AUPRC':finite_mean([q['AUPRC'] for q in records]),
          'OOF_thresholds':[q['threshold_OOF_CSI'] for q in records],
          'external_test_accessed':False,'complete_fivefold':folds==[1,2,3,4,5],
          'caution':'OOF CSI-optimized categorical scores are selected on these same OOF predictions; descriptive, not independent testing.'}
    save_json(run_dir/'run_summary.json',info)
    return info


def summary_all(root_dir,variants,seeds,folds):
    summaries=[]
    for v in variants:
        for seed in seeds:
            p=root_dir/v/f'seed_{seed}'/'run_summary.json'
            if p.is_file():
                with open(p,encoding='utf-8') as f:summaries.append(json.load(f))
    if not summaries:return
    df=pd.DataFrame(summaries)
    df[['variant','seed','mean_fold_horizon_AUPRC','sd_fold_mean_AUPRC_ddof0','N_OOF']].to_csv(root_dir/'summary_by_seed.csv',index=False)
    grouped=[]
    for v,g in df.groupby('variant',sort=False):
        arr=g['mean_fold_horizon_AUPRC'].to_numpy(dtype=float)
        grouped.append({'variant':v,'seeds_completed':len(arr),'mean_across_seeds':float(np.mean(arr)),
                        'sd_across_seeds_ddof1':float(np.std(arr,ddof=1)) if len(arr)>1 else np.nan,
                        'complete_5seeds_5folds':bool(set(g['seed'])==set(DEFAULT_SEEDS) and all(g['complete_fivefold'])),
                        'min_seed_score':float(np.min(arr)),'max_seed_score':float(np.max(arr))})
    pd.DataFrame(grouped).to_csv(root_dir/'summary_across_seeds.csv',index=False)
    if 'full' in set(df['variant']):
        pivot=df.pivot(index='seed',columns='variant',values='mean_fold_horizon_AUPRC')
        diff_rows=[]
        for v in [x for x in variants if x!='full' and x in pivot.columns]:
            for seed,record in pivot[['full',v]].dropna().iterrows():
                diff_rows.append({'seed':int(seed),'reference':'full','alternative':v,
                                  'full_AUPRC':float(record['full']),
                                  'alternative_AUPRC':float(record[v]),
                                  'delta_full_minus_alternative':float(record['full']-record[v])})
        if diff_rows:
            pd.DataFrame(diff_rows).to_csv(root_dir/'paired_seed_differences.csv',index=False)
    print('\nSummary:',root_dir/'summary_across_seeds.csv')


def parse_ints(value:str):
    return [int(p.strip()) for p in value.split(',') if p.strip()]


def smoke(engine_root:Path):
    print('Model smoke test on tiny synthetic inputs; NOT research results.')
    device=torch.device('cpu')
    x=torch.randn(3,60,20,device=device)
    y=torch.randint(0,2,(3,10),device=device).float()
    results=[]
    for variant in VARIANTS:
        m=build_model(variant,engine_root).to(device)
        params=sum(p.numel() for p in m.parameters() if p.requires_grad)
        o=m(x)
        assert o.shape==(3,10), (variant,o.shape)
        loss=focal_loss(o,y);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
        results.append({'variant':variant,'parameters':params,'relative_difference_pct':round(100*(params-41994)/41994,3)})
        print(f" {variant:18s}  params={params:6d}  delta={results[-1]['relative_difference_pct']:+.3f}%  ✓")
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',choices=['smoke','check','pilot','run','summarize'],default='smoke')
    ap.add_argument('--engine-root',default='.',help='StageD1_v6..._Engine root directory')
    ap.add_argument('--variants',default=','.join(VARIANTS))
    ap.add_argument('--seeds',default=','.join(map(str,DEFAULT_SEEDS)))
    ap.add_argument('--folds',default='1,2,3,4,5')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args()
    engine_root=Path(args.engine_root).resolve()
    if not (engine_root/'models'/'tcn_bilstm.py').is_file():
        ap.error(f'--engine-root must contain models/tcn_bilstm.py, got {engine_root}')
    variants=[v.strip() for v in args.variants.split(',') if v.strip()]
    if not variants or any(v not in VARIANTS for v in variants):ap.error(f'--variants must be subset of {VARIANTS}')
    seeds=parse_ints(args.seeds);folds=parse_ints(args.folds)
    if not seeds or len(set(seeds))!=len(seeds) or not folds or any(f not in range(1,6) for f in folds):ap.error('Invalid --seeds or --folds')
    if args.workers<0 or args.batch<1 or args.epochs<1:ap.error('Invalid worker/batch/epoch count')
    print(f'PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}; mode={args.mode}',flush=True)
    if args.mode=='smoke':
        smoke(engine_root);return
    if args.mode=='run' and (args.epochs!=50 or args.batch!=128):
        ap.error('Publication run uses --epochs 50 and --batch 128; use --mode pilot for quick test')
    if args.mode=='run' and not torch.cuda.is_available():
        ap.error('CUDA GPU is required for full publication runs; CPU only supports smoke/check/pilot.')
    if args.mode=='run' and len(folds)!=5:
        print('CAUTION: Partial fold execution; DO NOT report as a completed 5-fold CV.',flush=True)
    folder=Path(args.output_dir).resolve() if args.output_dir else engine_root/'results'/('reviewer_component_PILOT' if args.mode=='pilot' else 'reviewer_component_5seed')
    if args.mode=='summarize':
        for variant in variants:
            for seed in seeds:
                report_one_run(folder,variant,seed,[1,2,3,4,5])
        summary_all(folder,variants,seeds,[1,2,3,4,5]);return
    x_path,y_path,fold_data=load_indices(engine_root,strict=True)
    print('Data and Stage-C boundary audit PASS; external 2025 NEVER opened.',flush=True)
    for k,val in fold_data.items():
        print(f" fold {k}: train={len(val['train']):,}; val={len(val['val']):,}")
    audit_validation_targets(y_path,fold_data,args.mode)
    smoke(engine_root)
    if args.mode=='check':return
    folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/('protocol_request_'+('_'.join(map(str,seeds)))+'.json'),{'mode':args.mode,'data':'F1 physical-unit windows','folds':folds,
              'seeds':seeds,'variants':variants,'epoch_count':2 if args.mode=='pilot' else args.epochs,
              'optimizer':'AdamW','loss':'focal','alpha':ALPHA,'gamma':GAMMA,'lr':LR,'weight_decay':WEIGHT_DECAY,
              'selection':'mean fold mean horizon AUPRC','external_2025_used':False,
              'component_capacity_matching':'within 3% trainable parameters',
              'caution':'Do not present pilot results as complete reviewer experiment; AUPRC may be undefined on a 512-window pilot without positive targets.'})
    for variant in variants:
        for seed in seeds:
            rdir=folder/variant/f'seed_{seed}'
            for f in folds:
                print(f'\n=== {variant}, seed {seed}, fold {f} ===',flush=True)
                train_fold(engine_root,x_path,y_path,f,fold_data[f],variant,seed,args,rdir)
            report_one_run(folder,variant,seed,[1,2,3,4,5] if args.mode=='run' else folds)
            summary_all(folder,variants,seeds,folds)
    print('\nFINISHED. Upload summary_by_seed.csv, summary_across_seeds.csv, and protocol_request_*.json (do not submit pilot metrics).',flush=True)


if __name__=='__main__':
    main()
