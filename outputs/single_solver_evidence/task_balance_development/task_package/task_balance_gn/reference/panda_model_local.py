"""Panda chain reconstructed from the already checked public URDF constants.
This is a LOCAL model, not the user's server native library.
"""
import numpy as np
from numba import njit
LOW=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
HIGH=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
VEL=np.array([2.175]*4+[2.61]*3)
STEP=VEL*.02+1e-4
EPS=np.array([.001]*3+[.00872664626]*3)
OFF=np.array([[0.,0,.333],[0,0,0],[0,-.316,0],[.0825,0,0],[-.0825,.384,0],[0,0,0],[.088,0,0]])
ANG=np.array([0.,-1.57079632679,1.57079632679,1.57079632679,-1.57079632679,1.57079632679,1.57079632679])
ORIG=np.array([[[1.,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]] for a in ANG])

@njit(cache=True)
def skew(x):
 a,b,c=x;return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])

@njit(cache=True)
def fk_jac(q):
 p=np.zeros(3);R=np.eye(3);axes=np.empty((7,3));centers=np.empty((7,3))
 for i in range(7):
  p=p+R@OFF[i];R=R@ORIG[i];axes[i]=R[:,2];centers[i]=p
  c,s=np.cos(q[i]),np.sin(q[i]);R=R@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
 p=p+R[:,2]*.107
 J=np.empty((6,7))
 for i in range(7):
  J[:3,i]=np.cross(axes[i],p-centers[i]);J[3:,i]=axes[i]
 return p,R,J

@njit(cache=True)
def rot_log(M):
 c=min(1.,max(-1.,.5*(np.trace(M)-1.)));v=np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])
 t=np.arccos(c)
 if t<1e-6:return (.5+t*t/12)*v
 if np.pi-t<1e-5:
  # robust eigen-axis branch, only for uncommon large angular residuals.
  vals,V=np.linalg.eigh(.5*(M+M.T));axis=V[:,-1]
  if axis@v<0:axis=-axis
  return t*axis
 return t/(2*np.sin(t))*v

@njit(cache=True)
def residual_jac(q,pd,Rd):
 p,R,J=fk_jac(q);er=rot_log(Rd@R.T);t=np.linalg.norm(er);K=skew(er)
 a=1/12+t*t/720 if t<1e-5 else 1/(t*t)-(1+np.cos(t))/(2*t*np.sin(t))
 Jr=np.eye(3)+.5*K+a*K@K
 e=np.empty(6);e[:3]=pd-p;e[3:]=er
 A=np.empty((6,7));A[:3]=-J[:3];A[3:]=-Jr@J[3:]
 return e/EPS,A/EPS.reshape(6,1)

@njit(cache=True)
def admissible(q,prev,p,R):
 e,J=residual_jac(q,p,R)
 return np.linalg.norm(e[:3])<=1 and np.linalg.norm(e[3:])<=1 and np.all(q>=LOW-1e-9) and np.all(q<=HIGH+1e-9) and np.all(np.abs(q-prev)<=STEP)
