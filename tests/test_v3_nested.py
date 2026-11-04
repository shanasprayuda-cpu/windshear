import importlib.util
from pathlib import Path
import numpy as np
import sys
script=Path(__file__).resolve().parents[1]/'scripts'/'reviewer_imputation_policies_v3.py'
spec=importlib.util.spec_from_file_location('impv3',script)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def base(n=32):
    raw=np.full((n,20),1.7,dtype=np.float32)
    orig=raw.copy()
    flags=np.zeros((n,10),bool)
    # 1-minute gap in A2, 2-minute gap in A3, 3-minute in A4, 5-minute in B6
    for st,end,s in [(2,3,0),(6,8,1),(11,14,2),(21,26,9)]:
        raw[st:end,2*s:2*s+2]=np.nan
        orig[st:end,2*s:2*s+2]=orig[st-1,2*s:2*s+2]
        flags[st:end,s]=1
    # 2 minute source missing but not filled because of a source QA barrier.
    raw[17:19,8:10]=np.nan
    orig[17:19,8:10]=np.nan
    return {'wind_observed':raw,'wind_original':orig,'flags':flags,'present':np.ones(n,np.uint8)}

def test():
    d=base()
    results={cap:m.original_flag_nested_policy(d,cap) for cap in (0,2,5)}
    a0, f0,_=results[0]
    a2, f2,_=results[2]
    a5, f5,_=results[5]
    assert np.allclose(a0,d['wind_observed'],equal_nan=True)
    assert np.allclose(a5,d['wind_original'],equal_nan=True)
    assert not f0.any() and np.array_equal(f5,d['flags'])
    assert f2[2,0] and f2[6:8,1].all() and not f2[11:14,2].any() and not f2[21:26,9].any()
    assert all(np.isnan(a[:,8:10]).all(axis=1)[17:19].all() for a in (a0,a2,a5))
    # Partial original Stage A fill within a single raw gap must stop.
    d=base();d['flags'][6,1]=False;d['wind_original'][6,2:4]=np.nan
    try:m.original_flag_nested_policy(d,2)
    except ValueError as e:assert ('inside longer missing gap' in str(e) or 'missing a valid source seed' in str(e))
    else:raise AssertionError('Partial original fill was not rejected')
    # Original fill longer than five minutes must stop.
    d=base();d['wind_observed'][21:27,18:20]=np.nan;d['wind_original'][21:27,18:20]=1.7;d['flags'][26,9]=True
    try:m.original_flag_nested_policy(d,5)
    except ValueError as e:assert 'length' in str(e)
    else:raise AssertionError('Original fill > 5 was not rejected')
    # Global entirely absent minutes protected when original has no fill.
    d=base();d['present'][17:19]=0
    try:m.original_flag_nested_policy(d,5)
    except ValueError:raise
    # Numeric 0 is valid wind component (never missing).
    d=base();d['wind_observed'][0,:]=0;d['wind_original'][0,:]=0
    m.original_flag_nested_policy(d,0)
    print('PASS: V3 nested 0/2/5; source rejected gaps preserved; partial fills rejected; >5 rejected; global gaps; zeros valid')

if __name__=='__main__':test()
