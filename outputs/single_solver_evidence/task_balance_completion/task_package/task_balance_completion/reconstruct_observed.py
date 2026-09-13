"""Post-hoc local reconstruction of previously observed 160-Panda generator.
NOT a new independent test. Source recipe: revision_compute_allocation/data.py;
seeds and enumeration: single_solver_evidence.yaml and run_single_solver_evidence.py.
Must compare all inputs in original repository before accepting as identical.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'reference'))
from panda_model_local import LOW,HIGH,VEL,fk_jac,STEP
families=['smooth','near_singular','joint_limit_return','high_curvature'];rows=[]
for i,fam in enumerate(f for f in families for _ in range(40)):
 seed=202609130000+i;rng=np.random.default_rng(seed)
 if fam=='near_singular':
  pool=rng.uniform(LOW+.05*(HIGH-LOW),HIGH-.05*(HIGH-LOW),(128,7))
  center=pool[np.argmin([np.linalg.svd(fk_jac(q)[2],compute_uv=False)[-1] for q in pool])]
 else:center=rng.uniform(LOW+.22*(HIGH-LOW),HIGH-.22*(HIGH-LOW))
 u=np.linspace(0.,1.,151)
 amp=rng.uniform(.1,.35,7)*rng.choice([-1.,1.],7);phase=rng.uniform(-np.pi,np.pi,7)
 if fam=='near_singular':offset=np.cos(np.pi*u[:,None])*amp
 elif fam=='high_curvature':offset=amp*(.55*np.sin(2*np.pi*u[:,None]+phase)+.35*np.sin(8*np.pi*u[:,None]+phase)+.1*np.sin(18*np.pi*u[:,None]))
 else:offset=amp*np.sin(2*np.pi*u[:,None]+phase)
 if fam=='joint_limit_return':
  j=int(rng.integers(7));center[j]=HIGH[j]-.002;offset[:,j]=-.30*(1-np.sin(np.pi*u)**2)
 factor=1.
 for k in range(7):
  if offset[:,k].max()>0:factor=min(factor,(HIGH[k]-center[k]-1e-6)/offset[:,k].max())
  if offset[:,k].min()<0:factor=min(factor,(center[k]-LOW[k]-1e-6)/(-offset[:,k].min()))
 offset*=min(1.,max(factor,1e-6))
 util=np.max(abs(np.diff(offset,axis=0))/(VEL*.02));maxu=.92 if fam=='high_curvature' else .65
 offset*=min(1.,maxu/max(util,1e-12));ref=center+offset
 assert (ref>=LOW).all() and (ref<=HIGH).all() and (abs(np.diff(ref,axis=0))<=STEP).all()
 pose=[fk_jac(q) for q in ref[1:]]
 uid=hashlib.sha256(f'revision_trajectory:panda:{seed}:{fam}:0'.encode()).hexdigest()
 rows.append(dict(uid=uid,site_id=f'trajectory_{i:03d}',family=fam,seed=seed,initial_q=ref[0].tolist(),dt=.02,target_position=[v[0].tolist() for v in pose],target_rotation=[v[1].tolist() for v in pose]))
(ROOT/'panda_observed160.json').write_text(json.dumps(rows,separators=(',',':')))
src=json.loads((ROOT/'reference/selected_input_source.json').read_text())['input'];t=rows[94]
checks=dict(observed_source='existing 160-trajectory generator; not fresh',assumed_no_seed_collisions=True,uid094=t['uid'],matches_selected_uid=t['uid']==src['uid'],target094_frame53_max_position_difference=float(np.max(abs(np.array(t['target_position'][53])-src['target_position']))),target094_frame53_max_rotation_difference=float(np.max(abs(np.array(t['target_rotation'][53])-src['target_rotation']))),all_target_identity_status='Only one historical target crosschecked locally; full array comparison required in original repository.')
(ROOT/'results/reconstruction_check.json').write_text(json.dumps(checks,indent=2));print(checks)
