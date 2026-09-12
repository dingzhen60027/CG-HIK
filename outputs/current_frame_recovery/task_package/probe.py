"""Local numerical probe. Public Panda chain; not the user's frozen runtime.
Source of kinematic constants: justagist/franka_panda_description/robots/panda_arm.urdf,
blob 407642f8156a754b7e9c74bc4bab4209fd330d06, retrieved through GitHub connector.
Recorded inputs: CG-HIK commit 8e754127bb56ec6054ef197f7f1b4e2324611b91,
outputs/task_recourse/numerical_completion_development/reports/first_failure_inputs.csv.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, least_squares

LOW=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
HIGH=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
STEP=np.array([2.175]*4+[2.61]*3)*.02+1e-4
EPS=np.array([.001]*3+[.00872664626]*3)
OFF=np.array([[0,0,.333],[0,0,0],[0,-.316,0],[.0825,0,0],[-.0825,.384,0],[0,0,0],[.088,0,0]])
ANG=np.array([0,-1.57079632679,1.57079632679,1.57079632679,-1.57079632679,1.57079632679,1.57079632679])
ORIG=Rotation.from_rotvec(np.column_stack([ANG,np.zeros(7),np.zeros(7)])).as_matrix()

def skew(x):
    a,b,c=x
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0.]])

def fk_jac(q):
    p=np.zeros(3); R=np.eye(3); axes=[]; centers=[]
    for i in range(7):
        p=p+R@OFF[i];R=R@ORIG[i]
        axes.append(R[:,2].copy());centers.append(p.copy())
        c,s=np.cos(q[i]),np.sin(q[i])
        R=R@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    p=p+R@np.array([0,0,.107])
    axes=np.array(axes);centers=np.array(centers)
    J=np.vstack([np.cross(axes,p-centers).T,axes.T])
    return p,R,J

def residual_jac(q,pd,Rd):
    p,R,J=fk_jac(q)
    er=Rotation.from_matrix(Rd@R.T).as_rotvec();theta=np.linalg.norm(er)
    K=skew(er)
    a=(1./12+theta**2/720) if theta<1e-5 else (1/theta**2-(1+np.cos(theta))/(2*theta*np.sin(theta)))
    Jr_inv=np.eye(3)+.5*K+a*K@K
    e=np.r_[pd-p,er]/EPS
    A=-np.vstack([J[:3],Jr_inv@J[3:]])/EPS[:,None]
    return e,A

CASES=[
 dict(id='panda_trajectory14_pink_r1_f122',source='pink_qp',previous=[-2.580960908165891,.045225535505242616,2.8972999999999995,-.5738462902928353,-2.8972999999999995,2.0299278701804253,-1.7646994928786397],p=[.20345807338386102,.045433628449392,1.1610273074352238],R=[[.34441197335250046,.8332610360490273,-.43250021781952736],[-.9093383559548955,.18154820955192075,-.3743581199837129],[-.2334183947350436,.5222224558662657,.8202435977124855]],returned=[-2.578042024040462,.036008619969428904,2.8972999999999995,-.5589157909530367,-2.8972999999999995,2.0324139277175353,-1.7685271154541125],expected=[.0012992839328331773,.004631253155571994]),
 dict(id='panda_trajectory14_tar_r0_f115',source='tar_corrected',previous=[-2.1438080350350557,.05329899431860638,2.45049230073697,-.5585506551483025,-2.897299999758884,2.0671755660802966,-1.752459466423062],p=[.20861375575358249,.026903607247617728,1.1614330491751486],R=[[.34229102437705455,.8425791812818971,-.41580906423650105],[-.9074290555980031,.18164137962248073,-.3789180891239665],[-.24374036121952192,.507017487350456,.8267550446374108]],expected=[.0034535995934168867,.007814299369663286]),
 dict(id='panda_trajectory14_trac_r1_f115',source='trac_task_5ms',previous=[-2.271496215287015,.05635762476489381,2.5662068857508156,-.5700076469642632,-2.896034749633398,2.0642965613790434,-1.7430298825193924],p=[.20861375575358249,.026903607247617728,1.1614330491751486],R=[[.34229102437705455,.8425791812818971,-.41580906423650105],[-.9074290555980031,.18164137962248073,-.3789180891239665],[-.24374036121952192,.507017487350456,.8267550446374108]],expected=[.0032967721333230274,.006124485041746941]),
 dict(id='panda_trajectory26_tar_r1_f72',source='tar_corrected',previous=[-1.1639954548486624,.5353856535970403,.24494794116457477,-.9575487106914551,2.8909076134262737,2.7952076472041076,.5137465510857151],p=[.37719423030748733,-.6378831642217682,.6422111927532251],R=[[.5275670504201451,.6604035474662989,.534359580998522],[-.10416908689423132,.6745619459455053,-.7308317059472961],[-.8431024900014291,.3298989778343376,.424670290669419]],expected=[.012542362744525656,.024533186881132853]),
 dict(id='panda_trajectory12_tar_r0_f88',source='tar_corrected',previous=[1.0553777074897555,-.06954642181667957,1.250238389752356,-.27990444430191647,1.2944146799097758,2.887950844810127,1.9241793077097602],p=[-.07726759269901129,.07668869761084363,1.1698773132063292],R=[[-.6556085626488005,-.6131555357539996,-.44070137457897657],[.675307884141976,-.7372338854612545,.021105917260096793],[-.3378411967141688,-.28377189272711234,.8974056154832164]],expected=[.0071159456362062746,.015926583070896755])
]

class Found(Exception):
    def __init__(self,x):self.x=x.copy()

def solve(case,mode):
    prev=np.array(case['previous']);pd=np.array(case['p']);Rd=np.array(case['R'])
    lo=np.nextafter(np.maximum(LOW,prev-STEP),np.inf);hi=np.nextafter(np.minimum(HIGH,prev+STEP),-np.inf)
    bounds=np.column_stack([(lo-prev)/STEP,(hi-prev)/STEP]);x0=np.clip(np.zeros(7),bounds[:,0],bounds[:,1])
    calls=0;start=time.perf_counter()
    cache_x=None;cache_r=None;cache_J=None
    def evaluate(x):
        nonlocal calls,cache_x,cache_r,cache_J
        if cache_x is not None and np.array_equal(cache_x,x):return cache_r,cache_J
        r,J=residual_jac(prev+x*STEP,pd,Rd);calls+=1
        cache_x=x.copy();cache_r=r;cache_J=J*STEP
        return cache_r,cache_J
    def accepted(x):
        r,_=evaluate(x)
        return np.linalg.norm(r[:3])<=.99999 and np.linalg.norm(r[3:])<=.99999 and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1])
    iters=0
    def callback(x):
        nonlocal iters
        iters+=1
        u=x[:7]
        if accepted(u):raise Found(u)
    status=''
    try:
        if mode=='minimax':
            r,_=evaluate(x0);t0=max(np.linalg.norm(r[:3]),np.linalg.norm(r[3:]))
            def fun(v):return v[-1]
            def jac(v):return np.r_[np.zeros(7),1.]
            def con(v):
                r,_=evaluate(v[:7]);return v[-1]-np.array([np.linalg.norm(r[:3]),np.linalg.norm(r[3:])])
            def cjac(v):
                r,J=evaluate(v[:7]);a=np.empty((2,8));a[:,-1]=1
                for k in range(2):
                    sl=slice(k*3,k*3+3);norm=max(np.linalg.norm(r[sl]),1e-16);a[k,:7]=-r[sl]@J[sl]/norm
                return a
            result=minimize(fun,np.r_[x0,t0],jac=jac,method='SLSQP',bounds=list(bounds)+[(0,None)],constraints=[dict(type='ineq',fun=con,jac=cjac)],callback=callback,options=dict(maxiter=50,ftol=1e-10))
            x=result.x[:7];status=str(result.message)
        elif mode=='least_squares':
            result=least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],bounds=(bounds[:,0],bounds[:,1]),max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10,callback=callback)
            x=result.x;status=str(result.message)
        else:raise ValueError(mode)
    except Found as ex:x=ex.x;status='early_verified'
    elapsed=time.perf_counter()-start;q=prev+x*STEP;r,_=evaluate(x)
    verdict=bool(np.linalg.norm(r[:3])<=1 and np.linalg.norm(r[3:])<=1 and np.all(q>=LOW) and np.all(q<=HIGH) and np.all(np.abs(q-prev)<=STEP))
    return dict(case=case['id'],mode=mode,accepted=verdict,position_mm=float(np.linalg.norm(r[:3])),orientation_deg=float(np.linalg.norm(r[3:])*EPS[3]*180/np.pi),max_step_fraction=float(np.max(np.abs(q-prev)/STEP)),calls=calls,iterations=iters,elapsed_ms=elapsed*1000,status=status,q=q.tolist())

if __name__=='__main__':
    agreement=[]
    for c in CASES:
        q=np.array(c.get('returned',c['previous']));e,_=residual_jac(q,np.array(c['p']),np.array(c['R']))
        actual=[np.linalg.norm(e[:3])*EPS[0],np.linalg.norm(e[3:])*EPS[3]]
        agreement.append(dict(case=c['id'],recorded=c['expected'],recomputed=list(map(float,actual)),absolute_difference=list(map(float,np.abs(np.array(c['expected'])-actual)))))
    print('model residual agreement',json.dumps(agreement,indent=2))
    # Central differences check analytic residual derivatives.
    c=CASES[0];q=np.array(c['previous']);pd=np.array(c['p']);rd=np.array(c['R']);e,J=residual_jac(q,pd,rd)
    fd=np.column_stack([(residual_jac(q+1e-6*np.eye(7)[i],pd,rd)[0]-residual_jac(q-1e-6*np.eye(7)[i],pd,rd)[0])/2e-6 for i in range(7)])
    print('jac max abs',np.max(np.abs(J-fd)))
    rows=[solve(c,m) for c in CASES for m in ['least_squares','minimax']]
    print(json.dumps(rows,indent=2))
    Path(__file__).with_name('probe_results.json').write_text(json.dumps(dict(model_agreement=agreement,jacobian_error=float(np.max(np.abs(J-fd))),results=rows),indent=2))
