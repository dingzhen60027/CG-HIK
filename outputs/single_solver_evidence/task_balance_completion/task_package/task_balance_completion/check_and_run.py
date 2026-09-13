from pathlib import Path
import sys,json,time,types,argparse,os
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from bounded_gn_reference import box_qp,BoundedGN,Settings as GNSettings
from panda_model_local import LOW,HIGH,VEL,STEP,residual_jac,admissible
from check_selected_input import independent_fk
from balanced_old_local import solve_balanced,Settings as OldSettings
from inexact_balance import solve,Settings,dual_direction,task_value


def checks():
 rng=np.random.default_rng(739102);rows=[]
 for i in range(48):
  n=6+i%2;e=rng.normal(size=6)*3.;G=rng.normal(size=(6,n));lo=-rng.uniform(.1,1.,n);hi=rng.uniform(.1,1.,n)
  lam=.01;posture=np.zeros(n);w=np.zeros(n)
  H=G.T@G+lam*np.eye(n);g=G.T@e;d,nit=box_qp(H,g,lo,hi)
  d1,info=dual_direction(e,G,lo,hi,lam,0.,posture,w,box_qp,(H,g,d,nit),lambda d:False,Settings(forcing=.25),time.perf_counter_ns()+int(1e12),True)
  def obj(x):return x[-1]+.5*lam*(x[:-1]@x[:-1])
  def jac(x):return np.r_[lam*x[:-1],1.]
  def constraint(x):
   r=e+G@x[:-1];return np.array([x[-1]-r[:3]@r[:3],x[-1]-r[3:]@r[3:]])
  def cj(x):
   r=e+G@x[:-1];return np.array([np.r_[-2*r[:3]@G[:3],1.],np.r_[-2*r[3:]@G[3:],1.]])
  r=minimize(obj,np.r_[np.zeros(n),task_value(e)],jac=jac,method='SLSQP',bounds=list(zip(lo,hi))+[(None,None)],constraints={'type':'ineq','fun':constraint,'jac':cj},options={'ftol':1e-11,'maxiter':1000})
  opt=obj(r.x);U=info['upper'];L=info['lower'];P0=info['pzero']
  good=bool(np.min(constraint(r.x))>=-1e-7 and np.isfinite(opt))
  assert good,(i,r.message,constraint(r.x))
  assert L<=opt+1e-6*max(1,abs(opt)),(i,L,opt)
  assert U>=opt-1e-6*max(1,abs(opt)),(i,U,opt)
  frac=(P0-U)/(P0-opt) if P0-opt>1e-10 else None
  if info['reason']=='relative_progress': assert frac>=.8-1e-6,(i,frac)
  rows.append(dict(id=i,n=n,reference_success=bool(r.success),reference_objective=opt,upper=U,lower=L,reason=info['reason'],dual_updates=info['dual_updates'],achieved_fraction=frac))
 # The analytical convex failure example, not a robot outcome.
 e=np.array([0.,0,0,1.4,0,0]);G=np.array([[1.],[0],[0],[-.5],[0],[0]])
 lo=np.array([0.]);hi=np.array([1.]);H=G.T@G+.01*np.eye(1);g=G.T@e;d,nit=box_qp(H,g,lo,hi)
 candidate,info=dual_direction(e,G,lo,hi,.01,0.,np.zeros(1),np.zeros(1),box_qp,(H,g,d,nit),lambda d:False,Settings(forcing=.25),time.perf_counter_ns()+int(1e12),True)
 (ROOT/'results').mkdir(exist_ok=True)
 (ROOT/'results/math_checks.json').write_text(json.dumps(dict(random_tests=rows,example=dict(point_direction=d.tolist(),relative_direction=candidate.tolist(),trace=info)),indent=2))
 print('48 references checked; relative stops',sum(r['reason']=='relative_progress' for r in rows),'mean QPs',np.mean([r['dual_updates'] for r in rows]),flush=True)
 # Verify that only the external checker can grant acceptance.
 out=solve(np.zeros(1),.02,np.array([-2.]),np.array([2.]),np.array([1.]),lambda q:(np.zeros(6),np.zeros((6,1))),lambda q:False,box_qp)
 assert not out['accepted'] and out['verification_calls']==1


def check_selected():
 src=json.loads((ROOT/'reference/selected_input_source.json').read_text());print(src.keys())

def verify_values(q,previous,p,R):
 T=independent_fk(q)
 ep=float(np.linalg.norm(T[:3,3]-p));er=float(Rotation.from_matrix(R@T[:3,:3].T).magnitude())
 ok=bool(np.isfinite(q).all() and ep<=.001 and er<=.00872664626 and np.all(q>=LOW-1e-9) and np.all(q<=HIGH+1e-9) and np.all(np.abs(q-previous)<=STEP))
 return ok,ep,er


def invoke(method,kappa,q,p,R):
 ev=lambda x:residual_jac(x,p,R)
 vi=lambda x:verify_values(x,q,p,R)[0]
 if method=='gn':
  out=BoundedGN(LOW,HIGH,VEL,settings=GNSettings(posture_weight=kappa)).solve(q,.02,ev,lambda x:SimpleNamespace(accepted=vi(x)))
 elif method=='old_local_balance':
  out=solve_balanced(q,.02,LOW,HIGH,VEL,ev,vi,settings=OldSettings(posture_weight=kappa))
 else:
  out=solve(q,.02,LOW,HIGH,VEL,ev,vi,box_qp,settings=Settings(posture_weight=kappa,forcing=None if method=='cached_tight' else .25))
 return out


def run(method,kappa,repeat):
 data=json.loads((ROOT/os.environ.get('DATA_FILE','panda_development.json')).read_text())
 q=np.array(data[0]['initial_q']);p=np.array(data[0]['target_position'][0]);R=np.array(data[0]['target_rotation'][0])
 for _ in range(4):invoke(method,kappa,q,p,R)
 allrows=[];trajs=[]
 for item in data:
  q=np.array(item['initial_q']);rows=[]
  for i,(pp,rr) in enumerate(zip(item['target_position'],item['target_rotation'])):
   previous=q.copy();p=np.array(pp);R=np.array(rr)
   at=time.perf_counter_ns();out=invoke(method,kappa,previous,p,R);elapsed=time.perf_counter_ns()-at
   qq=np.asarray(out['q']);ok,ep,er=verify_values(qq,previous,p,R)
   assert ok==out['accepted']
   if ok:q=qq.copy()
   row=dict(uid=item['uid'],site_id=item['site_id'],frame=i,previous_q=previous.tolist(),q=qq.tolist(),accepted=ok,position_error=ep,orientation_error=er,
     total_latency_ns=elapsed,evaluations=out['evaluations'],iterations=out['iterations'],dual_updates=out.get('dual_updates',0),status=out.get('status',out.get('internal_status')),
     reason_counts=out.get('reason_counts',{}),verification_calls=out.get('verification_calls'))
   rows.append(row)
  states=np.array([item['initial_q']]+[r['q'] if r['accepted'] else r['previous_q'] for r in rows])
  aa=np.diff(states,n=2,axis=0)/.02**2
  trajs.append(dict(uid=item['uid'],site_id=item['site_id'],family=item['family'],complete=all(r['accepted'] for r in rows),deadline_complete=all(r['accepted'] and r['total_latency_ns']<=20_000_000 for r in rows),time_ns=sum(r['total_latency_ns'] for r in rows),acceleration_rms=float(np.sqrt(np.mean(aa**2))),first_failure=next((r['frame'] for r in rows if not r['accepted']),None)))
  allrows+=rows
 times=np.array([r['total_latency_ns'] for r in allrows])/1e6
 errors=np.array([[r['position_error']/.001,r['orientation_error']/.00872664626] for r in allrows if r['accepted']])
 reasons={}
 for row in allrows:
  for key,val in row['reason_counts'].items():reasons[key]=reasons.get(key,0)+val
 summary=dict(method=method,kappa=kappa,repeat=repeat,complete=sum(t['complete'] for t in trajs),deadline_complete=sum(t['deadline_complete'] for t in trajs),
   latency_ms=np.percentile(times,[50,95,99]).tolist(),cumulative_ms=float(times.sum()),edge_pct=float(np.mean(errors.max(axis=1)>.9)*100),
   acceleration_rms=float(np.mean([t['acceleration_rms'] for t in trajs])),evaluations=sum(r['evaluations'] for r in allrows),dual_updates=sum(r['dual_updates'] for r in allrows),reason_counts=reasons,
   measurement='local; same warm Numba geometry and box QP for all variants; independent Python FK in timed final check; one source-reconstructed Panda development set; no server TRAC/Pink benchmark')
 out=ROOT/'results'/f"{os.environ.get('RUN_TAG','')}{method}_k{kappa}_r{repeat}.json"
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(dict(summary=summary,trajectories=trajs,rows=allrows),separators=(',',':')))
 print(json.dumps(summary),flush=True)


def selected():
 inp=json.loads((ROOT/'reference/selected_input_source.json').read_text())['input']
 q=np.array(inp['previous_q']);p=np.array(inp['target_position']);R=np.array(inp['target_rotation']);rows=[]
 for method in ['gn','old_local_balance','cached_tight','relative']:
  invoke(method,0,q,p,R)
  for repeat in range(3):
   at=time.perf_counter_ns();out=invoke(method,0,q,p,R);ns=time.perf_counter_ns()-at
   ok,ep,er=verify_values(out['q'],q,p,R)
   rows.append(dict(method=method,repeat=repeat,accepted=ok,q=out['q'].tolist(),position_error=ep,orientation_error=er,elapsed_ns=ns,evaluations=out['evaluations'],iterations=out['iterations'],dual_updates=out.get('dual_updates',0),status=out.get('status',out.get('internal_status'))))
 (ROOT/'results/selected.json').write_text(json.dumps(dict(input=inp,results=rows),indent=2))
 print(json.dumps(rows,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action');a.add_argument('--kappa',type=int,default=1);a.add_argument('--repeat',type=int,default=0);args=a.parse_args()
 if args.action=='math':checks()
 elif args.action=='selected':selected()
 else:run(args.action,args.kappa,args.repeat)
