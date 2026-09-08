"""Geometry-only 300-frame reference paths; no algorithm outcome is inspected."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import yaml

from ..types import IKQuery
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.data import near_singular_center
from .study import ROOT,context,sha,write_json,utc,code_hashes


def reference_path(kin,family,seed,frames,dt,within_family):
    rng=np.random.default_rng(seed)
    center=near_singular_center(kin,rng,128) if family=='near_singular' else kin.random_configuration(rng,.22)
    t=np.linspace(0,1,frames)
    amplitude=rng.uniform(.2,.65,kin.nq)*rng.choice([-1.,1.],kin.nq)
    phase=rng.uniform(-np.pi,np.pi,kin.nq)
    level=None;limit_joint=None
    if family=='near_singular':
        offsets=amplitude*np.cos(2*np.pi*t[:,None])
    elif family=='high_curvature':
        # Four a priori levels, five paths each. The sharpest level has abrupt
        # velocity reversals; all positions remain continuous and rate bounded.
        level=within_family%4
        slope_levels=((1.,1.5,.75,1.25),(1.,2.,.4,1.6),
                      (1.,-1.,2.,.5),(1.,-2.,2.5,-1.))
        slopes=np.array(slope_levels[level])[np.minimum((4*t).astype(int),3)]
        progress=np.cumsum(slopes)/(frames-1);progress-=progress[0]
        offsets=amplitude*(.5*np.sin(2*np.pi*progress[:,None]+phase)+
            .35*np.sin(12*np.pi*progress[:,None]+phase)+.15*np.sin(26*np.pi*progress[:,None]))
    elif family in ('smooth','joint_limit_return'):
        offsets=amplitude*np.sin(4*np.pi*t[:,None]+phase)
        if family=='joint_limit_return':
            limit_joint=int(rng.integers(kin.nq))
            center[limit_joint]=kin.limits.upper[limit_joint]-.002
            offsets[:,limit_joint]=-.5*(1-np.sin(np.pi*t)**2)
    else:raise ValueError(family)
    bound_factor=1.
    for j in range(kin.nq):
        high=offsets[:,j].max();low=offsets[:,j].min()
        if high>0:bound_factor=min(bound_factor,(kin.limits.upper[j]-center[j]-1e-6)/high)
        if low<0:bound_factor=min(bound_factor,(center[j]-kin.limits.lower[j]-1e-6)/(-low))
    offsets*=max(1e-8,min(1.,bound_factor))
    peak=np.max(np.abs(np.diff(offsets,axis=0))/(kin.limits.velocity*dt))
    cap=.92 if family=='high_curvature' else .75
    rate_factor=min(1.,cap/max(peak,1e-12));offsets*=rate_factor
    return center+offsets,dict(abruptness_level=level,limit_joint=limit_joint,
        bound_amplitude_factor=bound_factor,rate_amplitude_factor=rate_factor,
        reference_rate_utilization=float(np.max(np.abs(np.diff(offsets,axis=0))/(kin.limits.velocity*dt))),
        generation='geometry only; fixed waveform, range/rate amplitude contraction; no solver calls or screening')


def generate(cfg):
    targets=[];witnesses=[];identities=[]
    for ri,robot in enumerate(cfg['robots']):
        _,kin,v,_=context(robot,cfg)
        for fi,family in enumerate(cfg['formal']['families']):
            for j in range(cfg['formal']['trajectories_per_family']):
                seed=cfg['formal']['seed_base']+ri*10000+fi*100+j
                q,info=reference_path(kin,family,seed,cfg['formal']['frames'],cfg['dt'],j)
                uid=hashlib.sha256(f'crik_fresh:{robot}:{family}:{seed}'.encode()).hexdigest()
                poses=[kin.forward(x) for x in q];hashes=[];checks=[]
                for t,pose in enumerate(poses):
                    query=IKQuery(pose,q[max(0,t-1)],cfg['dt'])
                    check=v.check(q[t],query)
                    if not check.accepted:raise AssertionError((robot,seed,t,check))
                    hashes.append(query_digest(query));checks.append(bool(check.accepted))
                if len(set(hashes))!=len(hashes):raise AssertionError('duplicate query within trajectory')
                common=dict(robot=robot,uid=uid,site_id=f'trajectory_{fi*20+j:03d}',
                            family=family,seed=seed,dt=cfg['dt'])
                targets.append(dict(**common,initial_q=q[0].tolist(),
                    target_position=[p.position.tolist() for p in poses],
                    target_rotation=[p.rotation.tolist() for p in poses]))
                witnesses.append(dict(**common,reference_q=q.tolist(),verified=checks,
                    meaning='one reference-state feasible path, not a promise for method-specific feedback states'))
                identities.append(dict(**common,frames=len(q),query_hashes=hashes,**info,
                    min_limit_margin=float(np.min(np.minimum(q-kin.limits.lower,kin.limits.upper-q))),
                    min_unscaled_sigma=float(min(kin.min_singular_value(x) for x in q))))
    return targets,witnesses,identities


def preserve_inventory():
    # Git object identities are independent of timestamps and cover every old
    # tracked manuscript, output, solver and configuration, not selected figures.
    output=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=ROOT,text=True)
    return output.splitlines()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/correction_reserve.yaml')
    p.add_argument('--folder',default='outputs/correction_reserve_ik/formal_protocol');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());folder=Path(args.folder);folder.mkdir(parents=True,exist_ok=False)
    targets,witnesses,identities=generate(cfg)
    write_json(folder/'online_targets.json',targets)
    write_json(folder/'reference_witnesses.json',witnesses)
    write_json(folder/'identities.json',identities)
    write_json(folder/'frozen_configuration.json',cfg)
    write_json(folder/'baseline_git_inventory.json',preserve_inventory())
    write_json(folder/'selection_seal.json',dict(utc=utc(),configuration=cfg,code_hashes=code_hashes(),
        generation_source_sha256=sha(__file__),
        files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
        trajectories=len(targets),frames=sum(len(x['target_position']) for x in targets),
        outcome_access_before_generation=False,parameters_may_change_after_outcomes=False))
    print(f'Fixed {len(targets)} paths, {len(targets)*cfg["formal"]["frames"]} verified reference frames',flush=True)


if __name__=='__main__':main()
