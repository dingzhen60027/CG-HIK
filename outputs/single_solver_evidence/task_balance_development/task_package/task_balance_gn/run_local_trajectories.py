"""Local same-input full-sequence DEVELOPEMENT check. Not server/fresh evidence."""
import sys,json,types,time,argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
try:import numba
except (ImportError,AttributeError):
 m=types.ModuleType('numba');m.njit=lambda *a,**k:lambda f:f;sys.modules['numba']=m
from panda_model_local import LOW,HIGH,VEL,STEP,residual_jac,admissible
from check_selected_input import independent_fk
from bounded_gn_reference import BoundedGN,Settings as OldSettings
from task_excess_gn import solve_task_excess
from balanced_gn import solve_balanced,Settings

def inspect(q,p,R,previous):
 T=independent_fk(q)
 ep=float(np.linalg.norm(T[:3,3]-p));er=float(Rotation.from_matrix(R@T[:3,:3].T).magnitude())
 legal=bool(ep<=.001 and er<=np.deg2rad(.5) and np.all(q>=LOW-1e-9) and np.all(q<=HIGH+1e-9) and np.all(np.abs(q-previous)<=STEP))
 return legal,ep,er

def run(method):
 data=json.loads((ROOT/'data/panda_reconstructed.json').read_text())
 first=data[0];q=np.array(first['initial_q']);p=np.array(first['target_position'][0]);R=np.array(first['target_rotation'][0])
 old=BoundedGN(LOW,HIGH,VEL,settings=OldSettings(posture_weight=0 if method=='point_k0' else 1))
 def solve(q,p,R):
  ev=lambda x:residual_jac(x,p,R)
  ve=lambda x:admissible(x,q,p,R)
  if method.startswith('point'):return old.solve(q,.02,ev,lambda x:SimpleNamespace(accepted=ve(x)))
  if method=='excess':return solve_task_excess(q,.02,LOW,HIGH,VEL,ev,ve)
  return solve_balanced(q,.02,LOW,HIGH,VEL,ev,ve,settings=Settings(posture_weight=0. if method=='balanced_k0' else 1.))
 for _ in range(3):solve(q,p,R)
 allrows=[];trajs=[]
 for d in data:
  q=np.array(d['initial_q']);rr=[]
  for i,(pp,r) in enumerate(zip(d['target_position'],d['target_rotation'])):
   previous=q.copy();pp=np.array(pp);r=np.array(r);t=time.perf_counter_ns();out=solve(previous,pp,r)
   qq=np.array(out['q']);ok,ep,er=inspect(qq,pp,r,previous);elapsed=time.perf_counter_ns()-t
   if ok:q=qq.copy()
   row=dict(uid=d['uid'],site_id=d['site_id'],frame=i,previous=previous.tolist(),q=qq.tolist(),accepted=ok,
      position_error_m=ep,orientation_error_rad=er,elapsed_ns=elapsed,status=out.get('status',out.get('internal_status')),
      iterations=out['iterations'],dual_updates=out.get('dual_updates'),velocity_utilization=float(np.max(np.abs(qq-previous)/STEP)))
   rr.append(row)
  acc=np.array([d['initial_q']]+[x['q'] if x['accepted'] else x['previous'] for x in rr]);ar=np.diff(acc,n=2,axis=0)/.02**2
  tr=dict(uid=d['uid'],site_id=d['site_id'],family=d['family'],complete=all(x['accepted'] for x in rr),
     deadline_complete=all(x['accepted'] and x['elapsed_ns']<=20_000_000 for x in rr),cumulative_ns=sum(x['elapsed_ns'] for x in rr),
     acceleration_rms=float(np.sqrt(np.mean(ar**2))),first_failure=next((x['frame'] for x in rr if not x['accepted']),None))
  print(method,d['site_id'],tr['complete'],tr['cumulative_ns']/1e6,flush=True)
  trajs.append(tr);allrows+=rr
 errors=np.array([[x['position_error_m']/.001,x['orientation_error_rad']/np.deg2rad(.5)] for x in allrows if x['accepted']])
 times=np.array([x['elapsed_ns'] for x in allrows])/1e6
 summary=dict(method=method,trajectories=len(trajs),completed=sum(x['complete'] for x in trajs),deadline_completed=sum(x['deadline_complete'] for x in trajs),
    time_ms_quantiles=np.percentile(times,[50,95,99]).tolist(),cumulative_ms=float(sum(times)),
    edge_fraction=float(np.mean(errors.max(axis=1)>.9)),normalized_error_p95=np.percentile(errors,95,axis=0).tolist(),
    acceleration_rms=float(np.mean([x['acceleration_rms'] for x in trajs])),
    total_dual_updates=sum(x['dual_updates'] or 0 for x in allrows),scope='One local sweep, old reconstructed Panda development trajectories; Numba FK/box QP plus independent Python FK final check; not original server, not fresh test')
 output=ROOT/f'new_local_{method}.json'
 if output.exists():raise FileExistsError('Refusing to overwrite an earlier local run')
 output.write_text(json.dumps(dict(summary=summary,trajectories=trajs,rows=allrows),indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method');run(p.parse_args().method)
