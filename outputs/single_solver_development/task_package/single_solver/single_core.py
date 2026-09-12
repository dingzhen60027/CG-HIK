"""Single bound-constrained predictor-corrector kernel; no secondary solver."""
import numpy as np
from numba import njit
from panda_model import residual_jac,LOW,HIGH,STEP,EPS

@njit(cache=True)
def box_qp(H,g,lo,hi):
 n=len(g);x=np.minimum(np.maximum(np.zeros(n),lo),hi);state=np.zeros(n,np.int64)
 for k in range(n):
  if abs(x[k]-lo[k])<1e-14:state[k]=-1
  if abs(x[k]-hi[k])<1e-14:state[k]=1
 for it in range(50):
  ids=np.where(state==0)[0]
  if len(ids)>0:
   grad=H@x+g
   Hf=np.empty((len(ids),len(ids))); gf=np.empty(len(ids))
   for i in range(len(ids)):
    gf[i]=grad[ids[i]]
    for j in range(len(ids)):Hf[i,j]=H[ids[i],ids[j]]
   d=np.linalg.solve(Hf,-gf);alpha=1.;hit=-1;side=0
   for j in range(len(ids)):
    i=ids[j]
    if d[j]>1e-14 and (hi[i]-x[i])/d[j]<alpha:alpha=(hi[i]-x[i])/d[j];hit=i;side=1
    if d[j]<-1e-14 and (lo[i]-x[i])/d[j]<alpha:alpha=(lo[i]-x[i])/d[j];hit=i;side=-1
   x[ids]+=max(0.,alpha)*d
   x=np.minimum(np.maximum(x,lo),hi)
   if hit>=0:state[hit]=side;continue
  grad=H@x+g;worst=1e-9;release=-1
  for i in range(n):
   violation=-state[i]*grad[i]
   # lower requires g>=0, upper g<=0 => state*g<=0
   violation=state[i]*grad[i]
   if state[i]!=0 and violation>worst:worst=violation;release=i
  if release<0:return x,it+1
  state[release]=0
 return x,50

@njit(cache=True)
def solve_frame(prev,prev_delta,pd,Rd,alpha=0.,tol=.5,damp=.01,center_weight=0.,maxiter=30):
 lo=np.maximum(LOW+1e-12,prev-STEP*(1-1e-12));hi=np.minimum(HIGH-1e-12,prev+STEP*(1-1e-12))
 q=np.minimum(np.maximum(prev+alpha*prev_delta,lo),hi);lam=damp
 qp_iters=0;evals=0
 bestq=q.copy();bestcost=1e100
 for it in range(maxiter):
  e,A=residual_jac(q,pd,Rd);evals+=1;cost=.5*(e@e)
  ep=np.linalg.norm(e[:3]);er=np.linalg.norm(e[3:])
  if cost<bestcost:bestcost=cost;bestq=q.copy()
  if ep<=tol and er<=tol:return q,True,it,evals,qp_iters
  G=A*STEP;H=G.T@G+lam*np.eye(7);g=G.T@e
  if center_weight>0:
   # globally continuous quadratic posture cost, normalized to joint range
   span=HIGH-LOW;mid=.5*(HIGH+LOW);w=STEP/span
   H+=center_weight*np.diag(w*w);g+=center_weight*w*((q-mid)/span)
  d,qi=box_qp(H,g,np.maximum((lo-q)/STEP,-1.),np.minimum((hi-q)/STEP,1.));qp_iters+=qi
  if np.linalg.norm(d)<1e-12:break
  accepted=False
  for kk in range(8):
   a=2.**(-kk);qt=q+a*STEP*d;et,_=residual_jac(qt,pd,Rd);evals+=1
   ct=.5*(et@et)
   if ct<cost-1e-12:
    q=qt;lam=max(damp/100,lam*.5);accepted=True;break
  if not accepted:
   lam*=10
   if lam>1e8:break
 e,_=residual_jac(bestq,pd,Rd)
 ok=np.linalg.norm(e[:3])<=1 and np.linalg.norm(e[3:])<=1
 return bestq,ok,it+1,evals,qp_iters
