#!/usr/bin/env python3
"""One entry point for the authorized Phase-1 scan work (no formal runs)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/process-scan-mpl')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import h5py
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confik.process_scan.models import context,physical_model
from confik.process_scan.geometry_adapter import GeometrySettings
from confik.process_scan.task import ScanTask
from confik.process_scan.planning import initialize,spline_basis,optimize,dense_validate,retime
from confik.process_scan.collision import Clearance
from confik.process_scan.execution import execute
from confik.correction_reserve.geometry import residual_linearization

OUT=ROOT/'outputs/process_scan/phase1_baselines'


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False)+'\n')


def endpoints(task,adapter):
    limits=adapter.public.limits
    seed=np.array([0.,-.4,0.,-2.,0.,1.7,.8]) if task.robot=='panda' else np.array([0.,-1.4,1.8,-1.9,-1.57,0.])
    bounds=(limits.lower,limits.upper);ends=[];records=[]
    # Common pre-comparison reference construction, not a measured baseline.
    # Follow the whole fixed raster so the end configuration belongs to the
    # SAME continuous branch, not an independently chosen endpoint solution.
    old_settings=adapter.settings
    adapter.settings=GeometrySettings(max_iterations=100,budget_s=10.)
    try:
        for index,target in enumerate(task.poses(np.linspace(0,1,401))):
            box=bounds if index==0 else (seed-.35,seed+.35)
            if index==0:
                # Reference-data construction ONLY. A single conventional TRF
                # establishes the common start before comparing ANY baseline.
                # It is not in B0/B1/G and not a runtime fallback chain.
                scale=np.array([.00005]*3+[np.deg2rad(.05)]*3)
                def fun(x):return residual_linearization(adapter.native,target,x,scale)[0]
                def jac(x):return residual_linearization(adapter.native,target,x,scale)[1]
                ref=least_squares(fun,seed,jac=jac,bounds=bounds,max_nfev=1000,
                                  ftol=1e-10,xtol=1e-10,gtol=1e-10)
                seed=ref.x
            result=adapter.solve(target,seed,bounds,box,.2,(.00005,np.deg2rad(.05)),method='gn')
            records.append(result);seed=result['q']
            if index==0:ends.append(seed)
            if not result['accepted']:return [seed,seed],records
        ends.append(seed)
    finally:adapter.settings=old_settings
    return ends,records


def smoke(robot,family,direction,nctrl,attempt):
    task=ScanTask(family,0,direction,robot)
    root=OUT/'calibration'/f'{robot}_{family}_{direction}_{nctrl}_{attempt}'
    if root.exists():raise FileExistsError('Calibration record exists; use a distinct recorded invocation.')
    root.mkdir(parents=True)
    model,data,adapter,modelmeta=physical_model(robot,task)
    (root/'scene.xml').write_text(modelmeta.pop('xml'));write(root/'model.json',modelmeta)
    collision=Clearance(model);ends,records=endpoints(task,adapter)
    write(root/'endpoints.json',records)
    if not all(r['accepted'] for r in records):print('endpoint calibration failed',root,flush=True);return
    for method in ('B0','B1','G'):
        initial=initialize(task,adapter,*ends,method)
        write(root/f'{method}_initialization.json',initial)
        print(robot,method,'initial',initial['feasible'],initial['time'],flush=True)
        if not initial['feasible']:continue
        basis=spline_basis(task,nctrl);coeff=np.linalg.lstsq(basis(initial['s']),initial['q'],rcond=None)[0]
        coeff[0]=ends[0];coeff[-1]=ends[1]
        np.savez_compressed(root/f'{method}_initial.npz',s=initial['s'],q=initial['q'],coeff=coeff,knots=basis.t)
        if method!='B0':
            result=optimize(task,adapter,basis,coeff,collision=collision)
            write(root/f'{method}_smoothing.json',result);coeff=result['coeff']
            print(robot,method,'smoothing',result['status'],result['sample_constraint_violation'],result['elapsed'],flush=True)
        validation=dense_validate(task,adapter,basis,coeff,collision.numeric)
        write(root/f'{method}_validation.json',validation)
        print(robot,method,'dense',validation['feasible'],validation['min_clearance'],flush=True)
        if validation['feasible']:
            timing=retime(task,adapter,basis,coeff);write(root/f'{method}_timing.json',timing)
            np.savez_compressed(root/f'{method}_path.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')})
            print(robot,method,'time',timing['duration'],flush=True)


def physics(robot,family,direction,nctrl,attempt,calibration_id='initial'):
    task=ScanTask(family,0,direction,robot);root=OUT/'calibration'/f'{robot}_{family}_{direction}_{nctrl}_{attempt}'
    source=root/'B0_path.npz';raw=np.load(source);basis=spline_basis(task,nctrl)
    timing={k:raw[k] for k in ('s','x','u','t')};timing['duration']=float(timing['t'][-1])
    target=root/f'physical_calibration_{calibration_id}'
    if target.exists():raise FileExistsError(target)
    target.mkdir()
    model,data,adapter,meta=physical_model(robot,task)
    (target/'scene.xml').write_text(meta.pop('xml'));write(target/'model.json',meta)
    metric,history,profiles,mask=execute(model,data,task,adapter,basis,raw['coeff'],timing,timing['duration'])
    with h5py.File(target/'execution.h5','w') as f:
        for key in history[0]:f.create_dataset(key,data=np.array([row[key] for row in history]),compression='gzip')
    np.savez_compressed(target/'scan_samples.npz',t=[r['t'] for r in profiles],s=[r['s'] for r in profiles],
        points=[r['points'] for r in profiles],raw_valid=[r['raw_valid'] for r in profiles],
        quality_valid=[r['quality_valid'] for r in profiles],coverage_mask=mask)
    write(target/'metrics.json',metric);print(json.dumps(clean(metric),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','physics','prepare','run','report','render']);p.add_argument('--robot',choices=['panda','ur5e'],default='panda')
    p.add_argument('--family',choices=['plane','cylinder','saddle'],default='plane');p.add_argument('--direction',choices=['u','v'],default='u');p.add_argument('--controls',type=int,default=64)
    p.add_argument('--attempt',default='initial')
    p.add_argument('--calibration-id',default='initial')
    a=p.parse_args()
    if a.phase=='smoke':smoke(a.robot,a.family,a.direction,a.controls,a.attempt)
    elif a.phase=='physics':physics(a.robot,a.family,a.direction,a.controls,a.attempt,a.calibration_id)
    elif a.phase=='prepare':
        from confik.process_scan.study import prepare
        prepare(endpoints)
    elif a.phase=='run':
        from confik.process_scan.study import run
        run()
    elif a.phase=='report':
        from confik.process_scan.reporting import report
        report()
    else:
        from confik.process_scan.rendering import render
        render()


if __name__=='__main__':main()
