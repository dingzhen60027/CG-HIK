"""Geometry-defined task-set coverage probe; local development, not a fresh paper test.

No new algorithm or outcome-based input filtering. All inputs saved before any
comparison, including FK-centred controls (alpha=0) and tolerance offsets.
"""
from pathlib import Path
import os, sys, json, time, hashlib, csv, platform
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent
REF=BASE/'base/task_balance_completion'
sys.path[:0]=[str(REF),str(REF/'reference')]
# Match provided local package's compiled geometry, explicit recorded fallback.
try:
 import numba
 accelerator='numba'
except (ImportError,AttributeError) as err:
 import types
 stub=types.ModuleType('numba');stub.njit=lambda *a,**k:lambda f:f
 sys.modules['numba']=stub;accelerator='numpy_optional_fallback:'+repr(err)
from panda_model_local import LOW,HIGH,VEL,STEP,EPS,fk_jac,residual_jac,OFF,ANG
from bounded_gn_reference import BoundedGN,Settings as GNS,box_qp
from inexact_balance import solve,Settings as BS
OUT=BASE/'boundary_measured';OUT.mkdir(exist_ok=False)

def independent_fk(q):
 T=np.eye(4)
 for j in range(7):
  a=ANG[j];ca,sa=np.cos(a),np.sin(a)
  O=np.eye(4);O[:3,:3]=[[1,0,0],[0,ca,-sa],[0,sa,ca]];O[:3,3]=OFF[j]
  ca,sa=np.cos(q[j]),np.sin(q[j]);Q=np.eye(4);Q[:3,:3]=[[ca,-sa,0],[sa,ca,0],[0,0,1]]
  T=T@O@Q
 O=np.eye(4);O[2,3]=.107
 return T@O

def check(q,prev,p,R):
 T=independent_fk(q);ep=float(np.linalg.norm(T[:3,3]-p));er=float(Rotation.from_matrix(R@T[:3,:3].T).magnitude())
 accepted=bool(np.isfinite(q).all() and ep<=EPS[0] and er<=EPS[3] and np.all(q>=LOW-1e-9) and np.all(q<=HIGH+1e-9) and np.all(np.abs(q-prev)<=STEP))
 return accepted,ep,er

config=dict(seed=202609140714,anchors_per_family=30,families=['local','near_singular','near_limit','high_utilization'],alpha=[0.,.5,.95],dt=.02,position_tolerance=float(EPS[0]),orientation_tolerance=float(EPS[3]),kappa=1.,methods=['gn','tight','relative'],repeats=1,source='user supplied task_balance_completion_package.zip, numerical files unmodified', witness_step='All joints at 99.9% of velocity*dt before physical-limit clipping; dedicated near-boundary stress, not representative frequency',scope='one local Panda development coverage probe; not original-server verification, no input screening by solver results',offset='fixed independent unit translation and rotation directions per anchor; same direction for all alpha levels',accelerator=accelerator)
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
rng=np.random.default_rng(config['seed']);span=HIGH-LOW
inputs=[]
for family in config['families']:
 for j in range(config['anchors_per_family']):
  if family=='near_singular':
   pool=rng.uniform(LOW+.05*span,HIGH-.05*span,size=(64,7))
   prev=pool[np.argmin([np.linalg.svd(fk_jac(q)[2],compute_uv=False)[-1] for q in pool])]
  else:prev=rng.uniform(LOW+.07*span,HIGH-.07*span)
  if family=='near_limit':
   axis=int(rng.integers(7));prev[axis]=HIGH[axis]-rng.uniform(.001,.008)*span[axis]
  scale=.999
  direction=rng.choice([-1.,1.],7)
  witness=np.clip(prev+scale*VEL*.02*direction,LOW,HIGH)
  p,R,J=fk_jac(witness)
  dp=rng.normal(size=3);dp/=np.linalg.norm(dp)
  dr=rng.normal(size=3);dr/=np.linalg.norm(dr)
  for alpha in config['alpha']:
   pd=p+alpha*EPS[0]*dp;Rd=Rotation.from_rotvec(alpha*EPS[3]*dr).as_matrix()@R
   ok,ep,er=check(witness,prev,pd,Rd)
   assert ok,(family,j,alpha,ep,er)
   inputs.append(dict(anchor=f'{family}_{j:03d}',family=family,alpha=alpha,previous=prev.tolist(),q_witness=witness.tolist(),target_position=pd.tolist(),target_rotation=Rd.tolist(),direction_p=dp.tolist(),direction_R=dr.tolist(),witness_position_error=ep,witness_orientation_error=er))
raw=json.dumps(inputs,separators=(',',':')).encode();(OUT/'inputs.json').write_bytes(raw)
(OUT/'input_seal.json').write_text(json.dumps({'inputs_sha256':hashlib.sha256(raw).hexdigest(),'count':len(inputs),'solver_calls':0,'created_before_comparison':True},indent=2))

def invoke(method,prev,p,R):
 ev=lambda q:residual_jac(q,p,R);verify=lambda q:check(q,prev,p,R)[0]
 if method=='gn':
  return BoundedGN(LOW,HIGH,VEL,settings=GNS(posture_weight=1.)).solve(prev,.02,ev,lambda q:SimpleNamespace(accepted=verify(q)))
 return solve(prev,.02,LOW,HIGH,VEL,ev,verify,box_qp,settings=BS(posture_weight=1.,forcing=None if method=='tight' else .25))
# Synthetic midpoint warmup; does not use saved inputs or witnesses.
q=(LOW+HIGH)/2;p,R,J=fk_jac(q+.2*STEP)
for method in config['methods']:
 for _ in range(4):invoke(method,q,p,R)
rows=[];order_rng=np.random.default_rng(config['seed']+1)
for index,i in enumerate(inputs):
 prev=np.array(i['previous']);p=np.array(i['target_position']);R=np.array(i['target_rotation'])
 for method in order_rng.permutation(config['methods']):
  at=time.perf_counter_ns();out=invoke(method,prev,p,R);elapsed=time.perf_counter_ns()-at
  q=np.array(out['q']);ok,ep,er=check(q,prev,p,R)
  assert ok==out['accepted']
  rows.append(dict(anchor=i['anchor'],family=i['family'],alpha=i['alpha'],method=str(method),q=q.tolist(),accepted=ok,position_error=ep,orientation_error=er,latency_ns=elapsed,within_20ms=ok and elapsed<=20_000_000,status=out.get('status',out.get('internal_status')),evaluations=out['evaluations'],iterations=out['iterations'],dual_updates=out.get('dual_updates')))
 if (index+1)%60==0:print(index+1,'/',len(inputs),flush=True)
(OUT/'records.json').write_text(json.dumps(rows,separators=(',',':')))
summary=[]
for alpha in config['alpha']:
 for family in ['all']+config['families']:
  for method in config['methods']:
   rs=[r for r in rows if r['alpha']==alpha and r['method']==method and (family=='all' or r['family']==family)]
   summary.append(dict(alpha=alpha,family=family,method=method,n=len(rs),success=sum(r['accepted'] for r in rs),on_time=sum(r['within_20ms'] for r in rs),p50_ms=float(np.median([r['latency_ns']/1e6 for r in rs])),mean_ms=float(np.mean([r['latency_ns']/1e6 for r in rs]))))
with (OUT/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
print(json.dumps([r for r in summary if r['family']=='all'],indent=2))
