import sys,json,types
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
try:import numba
except (ImportError,AttributeError):
    stub=types.ModuleType('numba');stub.njit=lambda *a,**k:lambda f:f;sys.modules['numba']=stub
from balanced_gn import minimax_step,solve_balanced,Settings
from panda_model_local import LOW,HIGH,VEL,STEP,residual_jac
from check_selected_input import source,prev,pd,Rd,inspect

# Scalar model: damping made tiny for algebra verification, not robot setting.
e=np.array([0.,0,0,1.4,0,0]);G=np.array([[1.],[0],[0],[-.5],[0],[0]])
d,info=minimax_step(e,G,np.array([0.]),np.array([2.]),damping=1e-9)
print('scalar',d,info['gap'],info['dual_updates'])
# Deterministic convex micro-checks with independent epigraph optimization.
rng=np.random.default_rng(2026091307);checks=[]
for n in (1,6,7):
 for k in range(12):
    ee=rng.normal(size=6);GG=rng.normal(size=(6,n));ll=-rng.uniform(.1,1,n);hh=rng.uniform(.1,1,n);lam=.01
    dd,ii=minimax_step(ee,GG,ll,hh,damping=lam,max_dual_updates=32)
    def fun(y):return y[-1]+.5*lam*(y[:-1]@y[:-1])
    def jac(y):return np.r_[lam*y[:-1],1.]
    def con(y):
     r=ee+GG@y[:-1];return np.array([y[-1]-(r[:3]@r[:3]),y[-1]-(r[3:]@r[3:])])
    def cjac(y):
     r=ee+GG@y[:-1];return np.c_[-2*np.array([GG[:3].T@r[:3],GG[3:].T@r[3:]]),np.ones(2)]
    oo=minimize(fun,np.r_[np.zeros(n),max(np.sum(ee[:3]**2),np.sum(ee[3:]**2))],jac=jac,
        constraints=[dict(type='ineq',fun=con,jac=cjac)],bounds=list(zip(ll,hh))+[(None,None)],method='SLSQP',options=dict(ftol=1e-12,maxiter=500))
    checks.append(dict(n=n,reference_success=bool(oo.success),relative_difference=float((ii['primal_upper']-oo.fun)/max(1,abs(oo.fun))),
        dual_gap=ii['gap'],converged=ii['converged']))
# model warm-up, then exact old selected source.
residual_jac(prev,pd,Rd)
r=solve_balanced(prev,.02,LOW,HIGH,VEL,lambda q:residual_jac(q,pd,Rd),lambda q:inspect(q)['accepted'],
    settings=Settings(deadline_ms=10000.),keep_trace=True)
res=dict(scope='Local development prototype; 36 convex problems and one previously observed Panda input, not robot benchmark or original-server timing',
         scalar=dict(step=d.tolist(),expected_minimax=14/15,info=info),qp_checks=checks,
         selected_input=source,robot={**r,'q':r['q'].tolist(),'independent_verification':inspect(r['q'])})
output=ROOT/'new_prototype_checks.json'
if output.exists():raise FileExistsError('Move or rename your earlier check output before rerunning')
output.write_text(json.dumps(res,indent=2))
print('qp differences',max(abs(x['relative_difference']) for x in checks),'gap',max(x['dual_gap'] for x in checks),
     'unconverged',sum(not x['converged'] for x in checks))
print('robot',r['accepted'],inspect(r['q']),r['iterations'],r['dual_updates'],r['elapsed_ns']/1e6)
