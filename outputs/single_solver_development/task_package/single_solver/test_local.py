import json,time,sys
import numpy as np
from pathlib import Path
from single_core import solve_frame
from panda_model import *
data=json.loads(Path('data/panda_reconstructed.json').read_text())

def run(alpha,tol,damp,weight=0.,tag='',ids=None):
 rows=[]
 for row in data:
  if ids is not None and int(row['site_id'][-2:]) not in ids:continue
  prev=np.array(row['initial_q']);delta=np.zeros(7);oklist=[];times=[];qs=[];errors=[];first=None;work=[]
  for i,(p,R) in enumerate(zip(row['target_position'],row['target_rotation'])):
   p=np.array(p);R=np.array(R);start=time.perf_counter_ns()
   q,ok,it,ev,qi=solve_frame(prev,delta,p,R,alpha,tol,damp,weight)
   final=admissible(q,prev,p,R);times.append((time.perf_counter_ns()-start)/1e6)
   if not final and first is None:first=i
   oklist.append(bool(final))
   newprev=q.copy() if final else prev.copy();delta=newprev-prev;prev=newprev
   qs.append(prev.copy());e,_=residual_jac(q,p,R);errors.append([float(np.linalg.norm(e[:3])),float(np.linalg.norm(e[3:]))]);work.append([int(it),int(ev),int(qi)])
  rows.append(dict(uid=row['uid'],site=row['site_id'],family=row['family'],complete=all(oklist),first_failure=first,accepted=oklist,times_ms=times,commands=np.array(qs).tolist(),errors=errors,work=work))
 summary=dict(alpha=alpha,tol=tol,damp=damp,weight=weight,complete=sum(r['complete'] for r in rows),failed=[(r['site'],r['first_failure']) for r in rows if not r['complete']],total_ms=sum(sum(r['times_ms']) for r in rows),quantiles=np.percentile([t for r in rows for t in r['times_ms']],[50,95,99]).tolist())
 print(tag,summary,flush=True)
 if tag:
  Path('results').mkdir(exist_ok=True);Path(f'results/{tag}.json').write_text(json.dumps(dict(summary=summary,rows=rows)))
 return summary
if __name__=='__main__':
 q=np.array(data[0]['initial_q']);solve_frame(q,np.zeros(7),np.array(data[0]['target_position'][0]),np.array(data[0]['target_rotation'][0]));admissible(q,q,np.zeros(3),np.eye(3))
 for a,t,d in [(0.,1.,.01),(0.,.1,.01),(1.,1.,.01),(1.,.5,.01),(1.,.1,.01),(1.,.1,.1),(0.,.01,.001)]:run(a,t,d,tag=f'a{a}_t{t}_d{d}')
