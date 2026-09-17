"""Phase 1.5 baseline closure. No proposed transition optimizer lives here."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import gzip
import json
import itertools
import subprocess
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import make_interp_spline, BSpline
from .models import ROOT, physical_model
from .task import ScanTask
from .collision import Clearance
from .planning import spline_basis, dense_validate, retime, numerical_grid, joint_time_calibrated
from .planning import initialize,optimize
from .execution import execute
from .study import write, clean, sha, save_execution, FROZEN
from ..correction_reserve.geometry import residual_linearization
from ..types import Pose

OUT=ROOT/'outputs/process_scan/phase15_baselines'
OLD=ROOT/'outputs/process_scan/phase1_baselines'


def reference(task, adapter, collision, root, controls=64,tilt_deg=0.,joint_interior=1e-8):
    """Independent multistart IK layers and continuity shortest path.

    Only the prescribed poses, URDF and deterministic seeds are inputs. No
    baseline path, method outcome, or optimized coefficient is ever loaded.
    """
    if (root/'summary.json').exists():
        return json.loads((root/'summary.json').read_text())
    root.mkdir(parents=True,exist_ok=True);start=perf_counter()
    rng=np.random.default_rng(2026091701)
    lim=adapter.public.limits;lo=lim.lower+joint_interior;hi=lim.upper-joint_interior
    scale=np.array([.00005]*3+[np.deg2rad(.05)]*3)
    grid=numerical_grid(np.linspace(0,1,401),task.knots)
    layers=[];costs=[];parents=[];log=[]
    seed0=np.array([0.,-.4,0.,-2.,0.,1.7,.8]) if task.robot=='panda' else np.array([0.,-1.4,1.8,-1.9,-1.57,0.])
    poses=task.poses(grid)
    if tilt_deg:
        centers=task.references(grid)[0]
        axis='x' if task.direction=='u' else 'y'
        poses=[Pose(c-.15*(p.rotation@Rotation.from_euler(axis,tilt_deg,degrees=True).as_matrix())[:,2],
                    p.rotation@Rotation.from_euler(axis,tilt_deg,degrees=True).as_matrix()) for c,p in zip(centers,poses)]
    for index,target in enumerate(poses):
        seeds=[seed0,*rng.uniform(lo,hi,size=(39,len(lo)))] if index==0 else list(layers[-1])
        candidates=[];records=[]
        for seed in seeds:
            answer=least_squares(lambda q:residual_linearization(adapter.native,target,q,scale)[0],
                np.clip(seed,lo,hi),jac=lambda q:residual_linearization(adapter.native,target,q,scale)[1],
                bounds=(lo,hi),method='dogbox',max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            q=answer.x;pose=adapter.native.forward(q)
            r=residual_linearization(adapter.native,target,q,scale)[0]
            good=bool(np.linalg.norm(r[:3])<=1 and np.linalg.norm(r[3:])<=1 and collision.numeric(q)>=.005)
            records.append(dict(seed=seed,q=q,status=int(answer.status),nfev=answer.nfev,legal=good,residual=r))
            if good and all(np.linalg.norm(q-old)>1e-3 for old in candidates):candidates.append(q)
        # Preserve distinct branches, with a fixed cap unrelated to baseline outcomes.
        candidates=np.asarray(candidates[:16]);log.append(dict(index=index,s=grid[index],attempts=records))
        if len(candidates)==0:break
        if index==0:
            cc=np.zeros(len(candidates));pp=np.full(len(candidates),-1)
        else:
            delta=candidates[:,None,:]-layers[-1][None,:,:]
            edge=np.sum((delta/.15)**2,axis=2)
            edge[np.max(abs(delta),axis=2)>.35]=np.inf
            total=edge+costs[-1][None,:];pp=np.argmin(total,axis=1);cc=total[np.arange(len(pp)),pp]
            keep=np.isfinite(cc);candidates=candidates[keep];cc=cc[keep];pp=pp[keep]
            if len(candidates)==0:break
        layers.append(candidates);costs.append(cc);parents.append(pp)
        if index%100==0:print('REFERENCE',root.name,index,len(candidates),flush=True)
    complete=len(layers)==len(grid);summary=dict(status='reference_unresolved',task=asdict(task),
        source='independent_multistart_bounded_dogbox_continuity_graph',baseline_outputs_used=False,
        tool_incidence_variant_deg=tilt_deg,nominal_surface_centers_unchanged=True,
        keypose_count=len(grid),completed_layers=len(layers),controls=controls,elapsed_s=perf_counter()-start)
    if complete:
        index=int(np.argmin(costs[-1]));sequence=[]
        for k in range(len(layers)-1,-1,-1):sequence.append(layers[k][index]);index=int(parents[k][index])
        q=np.array(sequence[::-1]);basis=spline_basis(task,controls)
        coeff=np.linalg.lstsq(basis(grid),q,rcond=None)[0];coeff[0]=q[0];coeff[-1]=q[-1]
        valid=dense_validate(task,adapter,basis,coeff,collision.numeric)
        fit='common_control_least_squares'
        if not valid['feasible']:
            interp=make_interp_spline(grid,q,k=3)
            basis=BSpline(interp.t,np.eye(len(interp.c)),3);coeff=interp.c
            valid=dense_validate(task,adapter,basis,coeff,collision.numeric)
            fit='independent_C2_interpolating_reference_not_baseline_coefficients'
        np.savez_compressed(root/'reference.npz',s=grid,q=q,coeff=coeff,knots=basis.t,
                            dense_s=valid['s'],dense_q=valid['q'])
        summary.update(status='reference_available' if valid['feasible'] else 'reference_unresolved',
                       dense_feasible=valid['feasible'],min_clearance=valid['min_clearance'],
                       common_start=q[0],common_end=q[-1],path_cost=float(min(costs[-1])))
        summary['fit']=fit
        write(root/'dense_check.json',valid)
    with gzip.open(root/'candidate_graph.json.gz','wt') as f:
        json.dump(clean(dict(log=log,layers=layers,costs=costs,parents=parents)),f)
    summary['elapsed_s']=perf_counter()-start;write(root/'summary.json',summary)
    print('REFERENCE DONE',root,summary['status'],summary['elapsed_s'],flush=True)
    return clean(summary)


def repair_references():
    for slot in ('panda_cylinder_p0_u','panda_cylinder_p1_u','panda_cylinder_p1_v'):
        d=json.loads((OLD/'inputs'/slot/'identity.json').read_text())
        task=ScanTask(d['family'],d['placement'],d['direction'],d['robot'])
        model,data,adapter,meta=physical_model(task.robot,task)
        collision=Clearance(model);attempts=[]
        for tilt in (0.,6.,-6.,9.,-9.):
            info=reference(task,adapter,collision,OUT/'reference_audit_final'/slot/f'tilt_{tilt:+g}',tilt_deg=tilt,joint_interior=.005)
            attempts.append(info)
            if info['status']=='reference_available':break
        write(OUT/'reference_audit_final'/slot/'selection.json',dict(attempts=attempts,
            selected_tilt_deg=tilt if info['status']=='reference_available' else None,
            status=info['status'],selection='first densely verified variant in fixed order; no method outcome'))


def calibrate_execution():
    # Fixed before the new physical outcomes: matched before/after margins on
    # six isolated paths. No comparison-scene outcome chooses these paths.
    for robot in ('panda','ur5e'):
        for family,direction in (('plane','u'),('cylinder','v'),('saddle','u')):
            for label,extension,overlap,reserve,controls in (
                ('before',0.,0.,.98,64),('after',.003,.001,.90,96)):
                root=OUT/'execution_calibration'/f'{robot}_{family}_{direction}'/label
                if (root/'execution_metrics.json').exists():continue
                task=ScanTask(family,0,direction,robot,extension,overlap,True)
                model,data,adapter,meta=physical_model(robot,task);collision=Clearance(model)
                info=reference(task,adapter,collision,root/'reference',controls)
                write(root/'settings.json',dict(task=asdict(task),reserve=reserve,controls=controls,
                    quality_thresholds_unchanged=True,comparison_scene=False))
                (root/'scene.xml').write_text(meta['xml'])
                if info['status']!='reference_available':continue
                path=np.load(root/'reference/reference.npz');basis=spline_basis(task,controls)
                timing=retime(task,adapter,basis,path['coeff'],speed_reserve=reserve)
                write(root/'timing.json',timing)
                if not timing['feasible']:continue
                np.savez_compressed(root/'planned_path.npz',coeff=path['coeff'],knots=basis.t,
                                    **{k:timing[k] for k in ('s','x','u','t')})
                metric,hist,profiles,mask=execute(model,data,task,adapter,basis,path['coeff'],timing,timing['duration'],state_stride=1)
                save_execution(root,metric,hist,profiles,mask)
                print('CALIBRATION',robot,family,label,metric['quality_completed'],metric['acceleration_utilization_max'],metric['max_hole_diameter_upper_bound_m'],flush=True)


def calibrate_b2():
    for family,direction in (('plane','u'),('cylinder','v'),('saddle','u')):
        root=OUT/'b2_calibration'/family;root.mkdir(parents=True,exist_ok=True)
        if (root/'summary.json').exists():continue
        task=ScanTask(family,0,direction,'panda',.003,.001,True)
        model,data,adapter,meta=physical_model('panda',task);collision=Clearance(model)
        source=OUT/'execution_calibration'/f'panda_{family}_{direction}'/'after/reference/reference.npz'
        fast=np.load(source);basis=spline_basis(task,96);grid=fast['s'];qfast=basis(grid)@fast['coeff']
        limits=adapter.public.limits;scale=np.array([.00005]*3+[np.deg2rad(.05)]*3)
        slow=[];last_null=None;log=[]
        # Fixed 0.22 rad, four-cycle redundant excursion, projected to exactly
        # the independently constructed fast path's poses. No baseline output.
        for s,q in zip(grid,qfast):
            J=adapter.native.jacobian(q);null=np.linalg.svd(J,full_matrices=True)[2][-1]
            if last_null is not None and null@last_null<0:null=-null
            last_null=null;seed=q+.22*np.sin(8*np.pi*s)*null;target=adapter.native.forward(q)
            answer=least_squares(lambda a:residual_linearization(adapter.native,target,a,scale)[0],
                np.clip(seed,limits.lower+1e-8,limits.upper-1e-8),
                jac=lambda a:residual_linearization(adapter.native,target,a,scale)[1],
                bounds=(limits.lower+1e-8,limits.upper-1e-8),method='dogbox',max_nfev=160,
                ftol=1e-10,xtol=1e-10,gtol=1e-10)
            slow.append(answer.x);log.append(dict(s=s,status=int(answer.status),nfev=answer.nfev))
        coeff=np.linalg.lstsq(basis(grid),slow,rcond=None)[0];coeff[0]=fast['coeff'][0];coeff[-1]=fast['coeff'][-1]
        valid=dense_validate(task,adapter,basis,coeff,collision.numeric)
        write(root/'construction.json',dict(projection=log,validation=valid,amplitude_rad=.22,cycles=4))
        if not valid['feasible']:
            write(root/'summary.json',dict(status='slow_path_construction_unresolved'));continue
        initial_time=retime(task,adapter,basis,coeff,speed_reserve=.90)
        fasttime=retime(task,adapter,basis,fast['coeff'],speed_reserve=.90)
        np.savez_compressed(root/'initial_slow.npz',coeff=coeff,knots=basis.t,**{k:initial_time[k] for k in ('s','x','u','t')})
        np.savez_compressed(root/'known_faster.npz',coeff=fast['coeff'],knots=basis.t,**{k:fasttime[k] for k in ('s','x','u','t')})
        result=joint_time_calibrated(task,adapter,basis,coeff,initial_time,collision,derivative_check=True)
        with gzip.open(root/'optimization.json.gz','wt') as f:json.dump(clean(result),f)
        best=min(result['incumbents'],key=lambda r:r['objective'])
        metrics={}
        for name,co,ti in (('initial',coeff,initial_time),('known_faster',fast['coeff'],fasttime),('adopted',best['coeff'],best['timing'])):
            target=root/name;target.mkdir(exist_ok=True)
            (target/'scene.xml').write_text(meta['xml'])
            metric,hist,profiles,mask=execute(model,data,task,adapter,basis,co,ti,initial_time['duration'],state_stride=1)
            save_execution(target,metric,hist,profiles,mask);metrics[name]=metric
            np.savez_compressed(target/'planned_path.npz',coeff=co,knots=basis.t,**{k:ti[k] for k in ('s','x','u','t')})
        passed=bool(best['source']!='verified_initial_path' and metrics['adopted']['quality_completed'] and
                    metrics['initial']['quality_completed'] and metrics['known_faster']['quality_completed'] and
                    metrics['adopted']['simulated_completion_time_s']<metrics['initial']['simulated_completion_time_s'])
        write(root/'summary.json',dict(status='validated' if passed else 'not_validated',
            known_faster_s=fasttime['duration'],initial_s=initial_time['duration'],adopted_s=best['objective'],
            derivative_check=result['derivative_check'],optimizer_status=result['status'],incumbents=len(result['incumbents']),
            metrics=metrics))
        print('B2 CALIBRATION',family,passed,initial_time['duration'],best['objective'],result['status'],flush=True)


def code_hashes():
    files=[ROOT/p for p in FROZEN]+list((ROOT/'src/confik/process_scan').glob('*.py'))
    files+=[ROOT/'scripts/run_process_scan_phase15.py',ROOT/'configs/process_scan_phase15.yaml']
    return {str(p.relative_to(ROOT)):sha(p) for p in files if p.name not in ('reporting.py','rendering.py','phase15_reporting.py')}


def prepare_core():
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/process_scan_phase15.yaml').read_text())
    if (OUT/'inputs/seal.json').exists():raise FileExistsError('Core inputs already frozen')
    # Do not enter the comparison until all three independent B2 checks pass.
    for family in ('plane','cylinder','saddle'):
        check=json.loads((OUT/'b2_calibration'/family/'summary.json').read_text())
        if check['status']!='validated':raise RuntimeError(f'B2 calibration not closed: {family}')
    records=[]
    for robot,family,placement,direction in itertools.product(('panda','ur5e'),('plane','cylinder','saddle'),(0,1),('u','v')):
        slot=f'{robot}_{family}_p{placement}_{direction}';root=OUT/'inputs'/slot
        task=ScanTask(family,placement,direction,robot,.003,.001)
        model,data,adapter,meta=physical_model(robot,task);collision=Clearance(model)
        attempts=[];selected=None
        for tilt in (0.,6.,-6.,9.,-9.):
            rr=root/'reference'/f'tilt_{tilt:+g}'
            info=reference(task,adapter,collision,rr,96,tilt_deg=tilt,joint_interior=.005)
            attempts.append(dict(tilt=tilt,root=str(rr.relative_to(OUT)),status=info['status']))
            if info['status']=='reference_available':selected=rr;break
        old=json.loads((OLD/'inputs'/slot/'identity.json').read_text())
        identity=dict(**{k:old[k] for k in ('slot','scene_uid','cluster','robot','family','placement','direction')},
            task=asdict(task),original_identity_sha256=sha(OLD/'inputs'/slot/'identity.json'),
            reference_geometry_feasible=selected is not None,attempts=attempts,
            reference_source=None if selected is None else str(selected.relative_to(OUT)),
            common_start=info.get('common_start') if selected else None,common_end=info.get('common_end') if selected else None,
            selected_tilt_deg=tilt if selected else None,split='same_24_development_scenes_calibrated_route',
            no_method_outcome_used=True,quality_thresholds_unchanged=True)
        root.mkdir(parents=True,exist_ok=True);(root/'scene.xml').write_text(meta['xml']);write(root/'identity.json',identity);records.append(identity)
    inputs=OUT/'inputs'
    write(inputs/'seal.json',dict(config=cfg,scenes=records,code_hashes=code_hashes(),
        files={str(p.relative_to(OUT)):sha(p) for p in inputs.rglob('*') if p.is_file()},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),formal_scenes_run=False))


def core_one(identity,method,repeat,cfg,model,data,adapter,collision):
    root=OUT/'runs'/identity['slot']/f'{method}_r{repeat}'
    if (root/'metrics.json').exists():return
    if root.exists():raise RuntimeError(f'Unfinished condition: {root}')
    root.mkdir(parents=True);start=perf_counter();task=ScanTask(**identity['task']);basis=spline_basis(task,96)
    metric=dict(**{k:identity[k] for k in ('slot','scene_uid','cluster','robot','family','placement','direction')},
        method=method,repeat=repeat,status='started',planner_feasible=False,physical_run=False,
        quality_completed=None,execution_completed=None,T_init=None,T_smooth=None,T_opt=None,T_retime=None,T_validate=None)
    write(root/'manifest.json',dict(identity=identity,method=method,repeat=repeat,code_hashes=code_hashes(),
          physical_qpos_runtime_overwrite=False,old_online_interface_called=False))
    if not identity['reference_geometry_feasible']:
        metric['status']='reference_unresolved';write(root/'metrics.json',metric);return
    raw=np.load(OUT/identity['reference_source']/'reference.npz')
    refbasis=BSpline(raw['knots'],np.eye(len(raw['coeff'])),3)
    ref_time=retime(task,adapter,refbasis,raw['coeff'],speed_reserve=.90)
    if not ref_time['feasible']:raise RuntimeError('Common reference failed timing')
    # Common reference timeout construction is a shared fixture, not a solver.
    start=perf_counter();snapshot=OUT/'shared_b1'/identity['slot']/'path.npz';optim=None;initial=None
    if method=='B2':
        if not snapshot.exists():
            metric['status']='common_b1_initialization_missing';write(root/'metrics.json',metric);return
        path=np.load(snapshot);coeff=path['coeff'];timing={k:path[k] for k in ('s','x','u','t')}
        timing.update(feasible=True,duration=float(timing['t'][-1]));prior=json.loads((snapshot.parent/'cost.json').read_text())
        metric['T_init']=prior['total_plan_wall_s'];metric['shared_initial_sha256']=sha(snapshot)
        np.savez_compressed(root/'path_initialization.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')})
        optim=joint_time_calibrated(task,adapter,basis,coeff,timing,collision)
        metric['T_opt']=optim['elapsed']
    else:
        reference_pose=None
        if method in ('B1','G') and identity['selected_tilt_deg']!=0:
            # A full-process-domain reference supplies admissible tool targets,
            # identically for TB and GN. It is not a baseline path/fallback.
            reference_pose=lambda ss:[adapter.native.forward(q) for q in refbasis(ss)@raw['coeff']]
        initial=initialize(task,adapter,np.array(identity['common_start']),np.array(identity['common_end']),method,reference_pose=reference_pose)
        metric.update(T_init=initial['time'],initialization_success=initial['feasible'],
            initialization_residual_evaluations=sum(r['residual_evaluations'] for r in initial['records']),
            initialization_nodes=len(initial['records']),
            tb_dual_continuation_calls=sum(len(r['trace']) for r in initial['records']) if method=='B1' else 0)
        np.savez_compressed(root/'path_initialization.npz',s=initial['s'],q=initial['q'])
        with gzip.open(root/'initialization_records.json.gz','wt') as f:json.dump(clean(initial['records']),f)
        if not initial['feasible']:
            metric.update(status='geometric_initialization_failed',total_plan_wall_s=perf_counter()-start);write(root/'metrics.json',metric);return
        coeff=np.linalg.lstsq(basis(initial['s']),initial['q'],rcond=None)[0]
        coeff[0]=identity['common_start'];coeff[-1]=identity['common_end']
        if method!='B0':
            optim=optimize(task,adapter,basis,coeff,collision=collision)
            metric['T_smooth']=optim['elapsed']
    if optim is not None:
        with gzip.open(root/'optimizer_trace.json.gz','wt') as f:json.dump(clean(optim),f)
        metric.update(solver_return_status=optim['status'],checkpoint_selected=10.,
            feasible_checkpoint_count=sum(v is not None for v in optim['checkpoints'].values()))
        selected=optim['checkpoints']['10.0']
        if selected is None:
            metric.update(status='no_verified_path_at_primary_checkpoint',total_plan_wall_s=perf_counter()-start+(metric['T_init'] if method=='B2' else 0.));write(root/'metrics.json',metric);return
        coeff=selected['coeff']
    tick=perf_counter();valid=dense_validate(task,adapter,basis,coeff,collision.numeric);metric['T_validate']=perf_counter()-tick
    metric.update(dense_geometry_feasible=valid['feasible'],dense_min_clearance_m=valid['min_clearance'])
    np.savez_compressed(root/'dense_validation.npz',s=valid['s'],q=valid['q'],process=np.array([[r[k] for k in ('standoff_m','center_error_m','incidence_rad','line_error_rad')] for r in valid['process']]))
    if valid['feasible']:
        tick=perf_counter();timing=retime(task,adapter,basis,coeff,speed_reserve=.90);metric['T_retime']=perf_counter()-tick
        metric['planner_feasible']=timing['feasible']
    metric['total_plan_wall_s']=perf_counter()-start+(metric['T_init'] if method=='B2' else 0.)
    if metric['planner_feasible']:
        metric.update(status='planner_feasible',predicted_execution_s=timing['duration'])
        np.savez_compressed(root/'planned_path.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')});write(root/'timing.json',timing)
        if method=='B1' and repeat==0:
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(snapshot,coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')});write(snapshot.parent/'cost.json',metric)
        if repeat==0:
            result,hist,profiles,mask=execute(model,data,task,adapter,basis,coeff,timing,ref_time['duration'],state_stride=1)
            save_execution(root,result,hist,profiles,mask);metric.update(result,physical_run=True)
    else:metric['status']='dense_geometry_or_timing_failed'
    write(root/'metrics.json',metric)
    print('CORE',identity['slot'],method,repeat,metric['status'],metric['quality_completed'],flush=True)


def run_core():
    seal=json.loads((OUT/'inputs/seal.json').read_text())
    if seal['code_hashes']!=code_hashes():raise RuntimeError('Core numerical source changed after seal')
    for p,digest in seal['files'].items():
        if sha(OUT/p)!=digest:raise RuntimeError(f'Changed input: {p}')
    rng=np.random.default_rng(2026091711)
    for identity in seal['scenes']:
        task=ScanTask(**identity['task']);model,data,adapter,meta=physical_model(task.robot,task);collision=Clearance(model)
        q=(adapter.public.limits.lower+adapter.public.limits.upper)/2;pose=adapter.public.forward(q)
        for _ in range(3):adapter.solve(pose,q,(adapter.public.limits.lower,adapter.public.limits.upper),(q-.35,q+.35),.15,(.001,np.deg2rad(.5)))
        for repeat in range(3):
            order=list(rng.permutation(['B0','B1','G','B2']))
            if repeat==0 and order.index('B2')<order.index('B1'):
                i,j=order.index('B1'),order.index('B2');order[i],order[j]=order[j],order[i]
            for method in order:core_one(identity,method,repeat,seal['config'],model,data,adapter,collision)
