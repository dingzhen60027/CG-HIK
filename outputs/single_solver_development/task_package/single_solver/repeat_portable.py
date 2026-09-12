"""Explicitly warmed timing of the portable engine + independent Python FK verifier."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter_ns
import numpy as np,json
from bounded_gn import BoundedGN
from panda_model import *
from final_comparison import DATA,independent_check
eng=BoundedGN(LOW,HIGH,VEL)
row=DATA[0];prev=np.array(row['initial_q']);p=np.array(row['target_position'][0]);R=np.array(row['target_rotation'][0])
def warmcheck(q):
 ok,ep,er,v=independent_check(q,prev,p,R);return SimpleNamespace(accepted=ok)
for _ in range(5):eng.solve(prev,.02,lambda q:residual_jac(q,p,R),warmcheck)
res=[]
for rep in range(3):
 for row in DATA:
  prev=np.array(row['initial_q']);frames=[]
  for frame,(p,R) in enumerate(zip(row['target_position'],row['target_rotation'])):
   p=np.array(p);R=np.array(R);old=prev.copy()
   def check(q):
    ok,ep,er,v=independent_check(q,old,p,R)
    return SimpleNamespace(accepted=ok,position_error=ep,orientation_error=er,velocity_utilization=v)
   start=perf_counter_ns();out=eng.solve(old,.02,lambda q:residual_jac(q,p,R),check);q=out['q'];v=out['verification']
   # Include API result extraction in the timing, not JSON disk writing.
   ff=dict(frame=frame,q=q.tolist(),previous_q=old.tolist(),accepted=out['accepted'],position_error=v.position_error,orientation_error=v.orientation_error,velocity_utilization=v.velocity_utilization)
   ff['latency_ns']=perf_counter_ns()-start;frames.append(ff)
   if out['accepted']:prev=q.copy()
  res.append(dict(repeat=rep,site_id=row['site_id'],uid=row['uid'],complete=all(f['accepted'] for f in frames),deadline_complete=all(f['accepted'] and f['latency_ns']<=20_000_000 for f in frames),frames=frames))
lat=np.array([f['latency_ns']/1e6 for r in res for f in r['frames']]);summary=dict(completions=[sum(x['complete'] for x in res if x['repeat']==i) for i in range(3)],deadline_completions=[sum(x['deadline_complete'] for x in res if x['repeat']==i) for i in range(3)],latency_quantiles_ms=np.percentile(lat,[50,95,99]).tolist(),max_latency_ms=float(lat.max()),late_calls=int(sum(lat>20)),mean_total_ms=float(lat.sum()/3),includes_independent_python_verification=True,jit_warmed=True)
Path('measured/portable_repeated.json').write_text(json.dumps(dict(summary=summary,runs=res)))
print(summary)
