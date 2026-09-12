"""Repeat timing without treating repeats as independent cases."""
from pathlib import Path
import json,sys,platform,random,os
import numpy as np,scipy
from scipy.spatial.transform import Rotation
from probe import CASES,solve,LOW,HIGH,STEP,OFF,ANG,EPS

def independent_pose(q):
    T=np.eye(4)
    for i in range(7):
        a=ANG[i];c,s=np.cos(a),np.sin(a)
        origin=np.eye(4);origin[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]];origin[:3,3]=OFF[i]
        z=np.eye(4);c,s=np.cos(q[i]),np.sin(q[i]);z[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        T=T@origin@z
    tip=np.eye(4);tip[2,3]=.107
    return T@tip

def audit(row,case):
    q=np.array(row['q']);prev=np.array(case['previous']);T=independent_pose(q)
    ep=float(np.linalg.norm(T[:3,3]-case['p']))
    er=float(Rotation.from_matrix(np.array(case['R'])@T[:3,:3].T).magnitude())
    ok=bool(np.isfinite(q).all() and ep<=EPS[0] and er<=EPS[3] and np.all(q>=LOW) and np.all(q<=HIGH) and np.all(np.abs(q-prev)<=STEP))
    return dict(accepted=ok,position_error_m=ep,orientation_error_rad=er,step_max=float(np.max(np.abs(q-prev)/STEP)))

# warm up each method; warm-up data are not included in summaries.
for mode in ('least_squares','minimax'):solve(CASES[1],mode)
jobs=[(i,m,k) for i in range(len(CASES)) for m in ('least_squares','minimax') for k in range(11)]
random.Random(160912).shuffle(jobs)
rows=[]
for i,m,k in jobs:
    r=solve(CASES[i],m);r['repeat']=k;r['independent_check']=audit(r,CASES[i]);rows.append(r)
    assert r['accepted']==r['independent_check']['accepted']
summary=[]
for c in CASES:
    for m in ('least_squares','minimax'):
        rr=[r for r in rows if r['case']==c['id'] and r['mode']==m]
        summary.append(dict(case=c['id'],method=m,accepted_repeats=sum(r['accepted'] for r in rr),repeats=len(rr),time_median_ms=float(np.median([r['elapsed_ms'] for r in rr])),time_p95_ms=float(np.quantile([r['elapsed_ms'] for r in rr],.95)),position_mm=rr[0]['position_mm'],orientation_deg=rr[0]['orientation_deg'],q=rr[0]['q'],max_step_fraction=rr[0]['max_step_fraction']))
metadata=dict(python=sys.version,scipy=scipy.__version__,numpy=np.__version__,platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},scope='Five archived Panda input rows, three trajectory UIDs. Manually reconstructed public Panda FK; recorded residuals independently matched. Not original verifier runtime, not an online whole-trajectory comparison, not a TRAC/Pink rerun. 11 repeats are timing repeats only.',network='Container internet unavailable. Source content retrieved through GitHub tools and transcribed; no repository binaries were executed.')
Path(__file__).with_name('repeated_results.json').write_text(json.dumps(dict(metadata=metadata,summary=summary,raw=rows),indent=2))
Path(__file__).with_name('recorded_inputs.json').write_text(json.dumps(CASES,indent=2))
print(json.dumps(summary,indent=2))
