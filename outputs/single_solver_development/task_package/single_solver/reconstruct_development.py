"""Reconstruct (not download) the original Panda generator from source.
Source: CG-HIK f72f89d..., revision_compute_allocation/data.py,
continuation_mechanism/tolerance_comparison.py and kinematics/base.py.
Reconstructed FK has tiny floating-point differences from the original backend.
"""
import numpy as np,json,hashlib
from pathlib import Path
from panda_model import *

def reconstruct():
 out=[]
 for i,family in enumerate([f for f in ['smooth','near_singular','joint_limit_return','high_curvature'] for _ in range(10)]):
  seed=970909000+i;rng=np.random.default_rng(seed)
  if family=='near_singular':
   pool=rng.uniform(LOW+.05*(HIGH-LOW),HIGH-.05*(HIGH-LOW),size=(128,7))
   sigma=np.array([np.linalg.svd(fk_jac(q)[2],compute_uv=False)[-1] for q in pool]);center=pool[np.argmin(sigma)]
  else:center=rng.uniform(LOW+.22*(HIGH-LOW),HIGH-.22*(HIGH-LOW))
  u=np.linspace(0,1,151);amp=rng.uniform(.1,.35,7)*rng.choice([-1.,1.],7);phase=rng.uniform(-np.pi,np.pi,7)
  if family=='near_singular':off=np.cos(np.pi*u[:,None])*amp
  elif family=='high_curvature':off=amp*(.55*np.sin(2*np.pi*u[:,None]+phase)+.35*np.sin(8*np.pi*u[:,None]+phase)+.1*np.sin(18*np.pi*u[:,None]))
  else:off=amp*np.sin(2*np.pi*u[:,None]+phase)
  if family=='joint_limit_return':
   k=int(rng.integers(7));center[k]=HIGH[k]-.002;off[:,k]=-.3*(1-np.sin(np.pi*u)**2)
  factor=1.
  for k in range(7):
   if off[:,k].max()>0:factor=min(factor,(HIGH[k]-center[k]-1e-6)/off[:,k].max())
   if off[:,k].min()<0:factor=min(factor,(center[k]-LOW[k]-1e-6)/(-off[:,k].min()))
  off*=min(1.,max(factor,1e-6));util=np.max(np.abs(np.diff(off,axis=0))/(VEL*.02));off*=min(1.,(.92 if family=='high_curvature' else .65)/max(util,1e-12))
  ref=center+off
  ps=[];Rs=[]
  for q in ref[1:]:
   p,R,J=fk_jac(q);ps.append(p.tolist());Rs.append(R.tolist())
  uid=hashlib.sha256(f'revision_trajectory:panda:{seed}:{family}:0'.encode()).hexdigest()
  out.append(dict(site_id=f'trajectory_{i:02d}',uid=uid,family=family,seed=seed,initial_q=ref[0].tolist(),target_position=ps,target_rotation=Rs,dt=.02,reference_q=ref.tolist()))
 return out

if __name__=='__main__':
 out=reconstruct();Path('data/panda_reconstructed.json').write_text(json.dumps(out))
 import sys;sys.path.insert(0,'/mnt/data/current_ik_workbench');from probe import CASES
 check=[]
 for case in CASES:
  i=int(case['id'].split('trajectory')[1].split('_')[0]);frame=int(case['id'].split('_f')[1]);row=out[i]
  check.append(dict(case=case['id'],position_difference=float(np.max(np.abs(np.array(row['target_position'][frame])-case['p']))),rotation_difference=float(np.max(np.abs(np.array(row['target_rotation'][frame])-case['R'])))))
 print(check)
 Path('data/reconstruction_checks.json').write_text(json.dumps(check,indent=2))
 print('uids 14/26/12',[out[i]['uid'] for i in [14,26,12]])
