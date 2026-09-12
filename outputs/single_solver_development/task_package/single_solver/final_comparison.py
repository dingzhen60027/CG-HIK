"""Same-machine Panda development comparison. No TRAC/Pink rerun is claimed."""
import json,time,platform,hashlib,os
from pathlib import Path
import numpy as np,scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from single_core import solve_frame
from panda_model import *

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'data/panda_reconstructed.json').read_text())

class EarlyStop(Exception):
 def __init__(self,x,why):self.x=x.copy();self.why=why

def trf_frame(prev,p,R,deadline):
 lo=np.maximum(LOW+1e-12,prev-STEP*(1-1e-12));hi=np.minimum(HIGH-1e-12,prev+STEP*(1-1e-12))
 low=(lo-prev)/STEP;high=(hi-prev)/STEP
 x0=np.minimum(np.maximum(np.zeros(7),low),high)
 initial=admissible(prev,prev,p,R)
 if initial:return prev.copy(),0,0,'initial_admissible'
 lastx=None;e=A=None;calls=0
 def calc(x):
  nonlocal lastx,e,A,calls
  if lastx is None or not np.array_equal(x,lastx):
   q=prev+STEP*x;e,J=residual_jac(q,p,R);A=J*STEP;lastx=x.copy();calls+=1
  return e
 def jac(x):calc(x);return A
 def callback(intermediate_result):
  x=intermediate_result.x;q=prev+STEP*x
  if admissible(q,prev,p,R):raise EarlyStop(x,'accepted')
  if time.perf_counter_ns()>=deadline:raise EarlyStop(x,'deadline')
 try:
  res=least_squares(calc,x0,jac=jac,bounds=(low,high),method='trf',tr_solver='exact',x_scale=1.,max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10,callback=callback)
  x=res.x;status=str(res.status)
 except EarlyStop as ex:x=ex.x;status=ex.why
 return prev+STEP*x,calls,0,status


def fk_independent(q):
 T=np.eye(4)
 for i in range(7):
  a=ANG[i];c,s=np.cos(a),np.sin(a)
  O=np.eye(4);O[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]];O[:3,3]=OFF[i]
  c,s=np.cos(q[i]),np.sin(q[i]);Z=np.eye(4);Z[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
  T=T@O@Z
 end=np.eye(4);end[2,3]=.107;T=T@end
 return T[:3,3],T[:3,:3]

def independent_check(q,prev,p,R):
 a,B=fk_independent(q)
 ep=float(np.linalg.norm(a-p));er=float(np.linalg.norm(Rotation.from_matrix(R@B.T).as_rotvec()))
 vel=float(np.max(np.abs(q-prev)/STEP))
 ok=bool(np.isfinite(q).all() and ep<=EPS[0] and er<=EPS[3] and np.all(q>=LOW-1e-9) and np.all(q<=HIGH+1e-9) and vel<=1.)
 return ok,ep,er,vel


def run():
 out=ROOT/'measured';out.mkdir(exist_ok=True)
 # Every signature and the independent checking path are warmed before timing.
 for row in DATA[:2]:
  prev=np.array(row['initial_q']);p=np.array(row['target_position'][0]);R=np.array(row['target_rotation'][0])
  for w in [0.,1.]:
   solve_frame(prev,np.zeros(7),p,R,0.,1.,.01,w)
  trf_frame(prev,p,R,time.perf_counter_ns()+20_000_000)
  independent_check(prev,prev,p,R)
 records=[];meta=[]
 jobs=[(mi,rep,k) for mi in ['trf','single_box','single_center'] for rep in range(3) for k in range(40)]
 rng=np.random.default_rng(2026091207)
 for jj in rng.permutation(len(jobs)):
  method,rep,k=jobs[jj];row=DATA[k];prev=np.array(row['initial_q']);frames=[]
  for t,(p,R) in enumerate(zip(row['target_position'],row['target_rotation'])):
   p=np.array(p);R=np.array(R);old=prev.copy();start=time.perf_counter_ns()
   if method=='trf':q,ev,qi,status=trf_frame(old,p,R,start+20_000_000);it=ev
   else:
    q,_,it,ev,qi=solve_frame(old,np.zeros(7),p,R,0.,1.,.01,1. if method=='single_center' else 0.);status='bounded_gn'
   ok=bool(admissible(q,old,p,R));e,J=residual_jac(q,p,R)
   dt=time.perf_counter_ns()-start
   if ok:prev=q.copy()
   frames.append(dict(frame=t,previous_q=old.tolist(),returned_q=q.tolist(),state_q=prev.tolist(),accepted=ok,latency_ns=int(dt),position_error=float(np.linalg.norm(e[:3])*EPS[0]),orientation_error=float(np.linalg.norm(e[3:])*EPS[3]),iterations=int(it),evaluations=int(ev),qp_iterations=int(qi),status=status))
  path=out/f'{method}_r{rep}_{k:02d}.json';path.write_text(json.dumps(dict(method=method,repeat=rep,site=row['site_id'],uid=row['uid'],family=row['family'],frames=frames)))
  meta.append(dict(method=method,repeat=rep,site=row['site_id'],uid=row['uid'],family=row['family'],complete=all(f['accepted'] for f in frames),deadline_complete=all(f['accepted'] and f['latency_ns']<=20_000_000 for f in frames),first_failure=next((f['frame'] for f in frames if not f['accepted']),None),total_ms=sum(f['latency_ns'] for f in frames)/1e6,filename=path.name))
  if len(meta)%40==0:print(len(meta),'/',len(jobs),flush=True)
 results={}
 for method in ['trf','single_box','single_center']:
  rr=[m for m in meta if m['method']==method];lat=[];err=[];acc=[]
  for row in rr:
   f=json.loads((out/row['filename']).read_text())['frames'];lat += [x['latency_ns']/1e6 for x in f]
   err += [[x['position_error'],x['orientation_error']] for x in f if x['accepted']]
   q=np.array([x['state_q'] for x in f]);acc.append(float(np.sqrt(np.mean((np.diff(q,n=2,axis=0)/.02**2)**2))))
  results[method]=dict(completions=[sum(m['complete'] for m in rr if m['repeat']==r) for r in range(3)],deadline_completions=[sum(m['deadline_complete'] for m in rr if m['repeat']==r) for r in range(3)],mean_total_ms=sum(m['total_ms'] for m in rr)/3,latency_quantiles_ms=np.percentile(lat,[50,95,99]).tolist(),max_latency_ms=max(lat),late_calls=sum(t>20 for t in lat),accepted_errors_p95=np.percentile(err,95,axis=0).tolist(),command_acceleration_rms_mean=float(np.mean(acc)),failed_uids=sorted(set(m['site'] for m in rr if not m['complete'])))
 print(json.dumps(results,indent=2),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(results=results,meta=meta,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),processor=platform.processor(),threads=os.environ.get('OPENBLAS_NUM_THREADS')),scope='Locally reconstructed Panda development trajectories; parameters chosen after local exploration. Not an independent test. No local TRAC/Pink.',jit_excluded=True),indent=2))
 # Read-only independent FK validation. Does not call a solver.
 violations=[];max_de=max_dr=0.;count=0;acceptedcount=0
 for row in meta:
  origin=DATA[int(row['site'][-2:])]
  for f,p,R in zip(json.loads((out/row['filename']).read_text())['frames'],origin['target_position'],origin['target_rotation']):
   verdict,ep,er,vel=independent_check(np.array(f['returned_q']),np.array(f['previous_q']),np.array(p),np.array(R));count+=1
   max_de=max(max_de,abs(ep-f['position_error']));max_dr=max(max_dr,abs(er-f['orientation_error']))
   if f['accepted']:acceptedcount+=1
   if verdict!=f['accepted']:violations.append([row['method'],row['site'],row['repeat'],f['frame'],ep,er,vel])
 check=dict(frames=count,accepted_frames=acceptedcount,acceptance_mismatches=violations,max_position_discrepancy=max_de,max_rotation_discrepancy=max_dr)
 (out/'independent_check.json').write_text(json.dumps(check,indent=2));print(check,flush=True)

if __name__=='__main__':run()
