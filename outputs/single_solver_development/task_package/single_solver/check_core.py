import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from bounded_gn import box_qp,BoundedGN
from panda_model import *
from final_comparison import DATA,independent_check
rng=np.random.default_rng(2026091211);err=0.;kkt=0.
for _ in range(100):
 n=7;A=rng.normal(size=(10,n));H=A.T@A+.1*np.eye(n);g=rng.normal(size=n)*5
 lo=-rng.uniform(.01,1,n);hi=rng.uniform(.01,1,n)
 x,_=box_qp(H,g,lo,hi)
 ref=minimize(lambda y:.5*y@H@y+g@y,np.zeros(n),jac=lambda y:H@y+g,method='SLSQP',bounds=list(zip(lo,hi)),options=dict(ftol=1e-12,maxiter=200))
 err=max(err,abs((.5*x@H@x+g@x)-ref.fun))
 grad=H@x+g;projected=x-np.clip(x-grad,lo,hi);kkt=max(kkt,np.max(np.abs(projected)))
assert kkt<1e-7,(kkt,err)
engine=BoundedGN(LOW,HIGH,VEL)
counts=0;complete=0;errors=[];saved=[]
for row in DATA:
 prev=np.array(row['initial_q']);success=True;fs=[]
 for t,(p,R) in enumerate(zip(row['target_position'],row['target_rotation'])):
  p,R=np.array(p),np.array(R);before=prev.copy()
  def evaluate(q):return residual_jac(q,p,R)
  def verify(q):
   ok,ep,er,vel=independent_check(q,before,p,R)
   return SimpleNamespace(accepted=ok,ep=ep,er=er,vel=vel)
  result=engine.solve(before,.02,evaluate,verify)
  q=result['q'];counts+=1;success=success and result['accepted']
  if result['accepted']:prev=q.copy()
  fs.append(dict(frame=t,previous_q=before.tolist(),q=q.tolist(),accepted=result['accepted'],latency_ns=result['total_latency_ns']))
 complete+=int(success);saved.append(dict(site_id=row['site_id'],uid=row['uid'],complete=bool(success),frames=fs))
print(dict(qp_cases=100,max_objective_difference=err,max_projected_KKT=kkt,portable_complete=complete,portable_frames=counts))
Path('measured/portable_check.json').write_text(json.dumps(dict(qp_cases=100,max_objective_difference=err,max_projected_KKT=kkt,portable_complete=complete,portable_frames=counts,trajectories=saved)))
