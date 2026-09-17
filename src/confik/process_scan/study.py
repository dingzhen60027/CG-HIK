"""Phase-1 orchestration, immutable scene inputs, and per-condition records."""
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import platform
import os
import traceback
from pathlib import Path
import subprocess
from time import perf_counter
import h5py
import numpy as np
import yaml
from .models import ROOT,physical_model
from .task import ScanTask
from .planning import initialize,spline_basis,optimize,dense_validate,retime,timed_validate
from .collision import Clearance
from .execution import execute

OUT=ROOT/'outputs/process_scan/phase1_baselines'
FROZEN=['src/confik/task_balance_gn.py','src/confik/bounded_gn.py','src/confik/correction_reserve/geometry.py',
        'src/confik/correction_reserve/native_geometry.py','src/confik/solvers/verifier.py']


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False)+'\n')


def hashes():
    files=FROZEN+['configs/process_scan_phase1.yaml','scripts/run_process_scan.py']
    # Read-only reporting/rendering may be completed while frozen solvers run.
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'src/confik/process_scan').glob('*.py')
              if p.name not in ('reporting.py','rendering.py')]
    return {p:sha(ROOT/p) for p in sorted(files)}


def prepare(reference_constructor):
    target=OUT/'inputs'
    if (target/'seal.json').exists():raise FileExistsError('Inputs already sealed')
    cfg=yaml.safe_load((ROOT/'configs/process_scan_phase1.yaml').read_text());records=[]
    for robot,family,placement,direction in itertools.product(cfg['robots'],cfg['families'],cfg['placements'],cfg['directions']):
        slot=f'{robot}_{family}_p{placement}_{direction}';root=target/slot
        if (root/'identity.json').exists():records.append(json.loads((root/'identity.json').read_text()));continue
        task=ScanTask(family,placement,direction,robot);model,data,adapter,meta=physical_model(robot,task)
        collision=Clearance(model);ends,raw=reference_constructor(task,adapter)
        pose_ok=all(r['accepted'] for r in raw);q=np.array([r['q'] for r in raw])
        clearance=np.array([collision.numeric(row) for row in q]);ref_ok=bool(pose_ok and len(q)==401 and min(clearance)>=.005)
        identity=dict(slot=slot,**asdict(task),center=task.center,world_R=task.world_R,
            common_start=ends[0],common_end=ends[1],reference_geometry_feasible=ref_ok,
            reference_pose_ok=pose_ok,reference_min_clearance=float(min(clearance)),
            cluster=f'{robot}_{family}_p{placement}',config_sha256=sha(ROOT/'configs/process_scan_phase1.yaml'),
            source_model_sha256=hashlib.sha256(meta['xml'].encode()).hexdigest(),fk_max_error=meta['fk_max_error'],
            no_method_outcome_used=True,split='development')
        identity['scene_uid']=hashlib.sha256(json.dumps(clean(identity),sort_keys=True).encode()).hexdigest()
        root.mkdir(parents=True,exist_ok=True);(root/'scene.xml').write_text(meta['xml'])
        np.savez_compressed(root/'reference_geometry.npz',q=q,s=np.linspace(0,1,len(q)),clearance=clearance)
        write(root/'identity.json',identity);records.append(clean(identity))
        with gzip.open(root/'reference_construction.json.gz','wt') as f:json.dump(clean(raw),f)
        print('INPUT',slot,'reference',ref_ok,'clearance',min(clearance),flush=True)
    seal=dict(config=cfg,code_hashes=hashes(),scenes=records,
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()},
        git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        platform=platform.platform(),formal_test_outcomes_generated=False,
        cpu=Path('/proc/cpuinfo').read_text(),affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        calibration_status='1ms extrema and actual rays checked before comparisons',
        boundmpc_task_status='not_evaluable_semantic_adaptation_incomplete')
    write(target/'seal.json',seal)


def save_execution(root,metric,history,profiles,mask):
    with h5py.File(root/'execution.h5','w') as f:
        f.attrs['qpos_runtime_overwrite']=False;f.attrs['source']='torque_actuators_mj_step'
        for key in history[0]:f.create_dataset(key,data=np.array([r[key] for r in history]),compression='gzip')
    np.savez_compressed(root/'scan_samples.npz',t=[r['t'] for r in profiles],s=[r['s'] for r in profiles],
        points=[r['points'] for r in profiles],raw_valid=[r['raw_valid'] for r in profiles],
        quality_valid=[r['quality_valid'] for r in profiles],geomids=[r['geomids'] for r in profiles],
        ranges=[r['ranges'] for r in profiles],coverage_mask=mask)
    write(root/'execution_metrics.json',metric)


def run_one(identity,method,repeat,cfg,model,data,adapter,collision):
    slot=identity['slot'];root=OUT/'runs'/slot/f'{method}_r{repeat}'
    if (root/'metrics.json').exists():return
    if root.exists():raise RuntimeError(f'Unfinished run requires explicit audit before resuming: {root}')
    root.mkdir(parents=True);task=ScanTask(identity['family'],identity['placement'],identity['direction'],identity['robot'])
    basis=spline_basis(task,cfg['control_points']);start=perf_counter();snapshot=OUT/'shared_b1'/slot/'path.npz'
    metric=dict(scene_uid=identity['scene_uid'],slot=slot,cluster=identity['cluster'],robot=identity['robot'],
        family=identity['family'],placement=identity['placement'],direction=identity['direction'],method=method,repeat=repeat,
        implementation_ready=method!='B3',planner_feasible=False,execution_completed=None,quality_completed=None,
        T_init=None,T_smooth=None,T_opt=None,T_retime=None,T_validate=None,total_plan_wall_s=None,
        predicted_execution_s=None,physical_run=False,status='started')
    manifest=dict(identity=identity,method=method,repeat=repeat,code_hashes=hashes(),
        old_online_interface_called=False,physical_rate_in_geometry=False,nearest_projection_claim=False,
        controller='common_torque_limited_computed_torque_PD',timing_repeats_are_not_independent_scenes=True)
    write(root/'manifest.json',manifest)
    if method=='B3':
        metric.update(status='not_evaluable',reason='Official example ran; upstream seven-joint IIWA state packing and decoupled Cartesian error box are not yet verified against Panda/UR5e scanner-ray constraints. No surrogate MPC substituted.')
        write(root/'metrics.json',metric);return
    if not identity['reference_geometry_feasible']:
        metric.update(status='common_reference_unavailable');write(root/'metrics.json',metric);return
    initial=None;timing=None;coeff=None;optim=None
    if method=='B2':
        if not snapshot.exists():
            metric.update(status='common_b1_initialization_missing');write(root/'metrics.json',metric);return
        raw=np.load(snapshot);coeff=raw['coeff'];timing={k:raw[k] for k in ('s','x','u','t')};timing.update(feasible=True,duration=float(raw['t'][-1]))
        np.savez_compressed(root/'path_initialization.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')})
        prior=json.loads((snapshot.parent/'cost.json').read_text());metric['T_init']=prior['total_plan_wall_s']
        manifest.update(shared_initial_sha256=sha(snapshot),shared_initial_source='B1_repeat0_primary_checkpoint')
        optim=optimize(task,adapter,basis,coeff,mode='joint_time',timing=timing,budget=30.,collision=collision)
        metric['T_opt']=optim['elapsed'];selected=optim['checkpoints']['10.0']
        if selected is not None:coeff=selected['coeff'];timing=selected['timing']
    else:
        initial=initialize(task,adapter,np.array(identity['common_start']),np.array(identity['common_end']),method)
        metric.update(T_init=initial['time'],initialization_success=initial['feasible'],
            initialization_residual_evaluations=sum(r['residual_evaluations'] for r in initial['records']),
            initialization_nodes=len(initial['records']),
            tb_dual_continuation_calls=sum(len(r['trace']) for r in initial['records']) if method=='B1' else 0,
            initialization_node_process_legal=sum(task.process_legal(adapter.native.forward(q).position,adapter.native.forward(q).rotation,s)
                                                    for s,q in zip(initial['s'],initial['q'])))
        np.savez_compressed(root/'path_initialization.npz',s=initial['s'],q=initial['q'])
        with gzip.open(root/'initialization_records.json.gz','wt') as f:json.dump(clean(initial['records']),f)
        if not initial['feasible']:
            metric.update(status='geometric_initialization_failed',total_plan_wall_s=perf_counter()-start)
            write(root/'metrics.json',metric);return
        coeff=np.linalg.lstsq(basis(initial['s']),initial['q'],rcond=None)[0]
        coeff[0]=identity['common_start'];coeff[-1]=identity['common_end']
        if method!='B0':
            optim=optimize(task,adapter,basis,coeff,budget=30.,collision=collision)
            metric['T_smooth']=optim['elapsed'];selected=optim['checkpoints']['10.0']
            if selected is not None:coeff=selected['coeff'];timing=selected['timing']
    if optim is not None:
        with gzip.open(root/'optimizer_trace.json.gz','wt') as f:json.dump(clean(optim),f)
        metric['solver_return_status']=optim['status'];metric['checkpoint_selected']=10.
        metric['feasible_checkpoint_count']=sum(v is not None for v in optim['checkpoints'].values())
        if selected is None:
            metric.update(status='no_verified_path_at_primary_checkpoint',total_plan_wall_s=perf_counter()-start+(metric['T_init'] if method=='B2' else 0.))
            write(root/'metrics.json',metric);return
    tick=perf_counter();validation=dense_validate(task,adapter,basis,coeff,collision.numeric)
    metric['T_validate']=perf_counter()-tick
    metric['dense_geometry_feasible']=validation['feasible'];metric['dense_min_clearance_m']=validation['min_clearance']
    if initial is not None:
        metric['smoothing_displacement_rms_rad']=float(np.sqrt(np.mean((basis(initial['s'])@coeff-initial['q'])**2)))
    np.savez_compressed(root/'dense_validation.npz',s=validation['s'],q=validation['q'],
        process=np.array([[r[k] for k in ('standoff_m','center_error_m','incidence_rad','line_error_rad')] for r in validation['process']]))
    if validation['feasible']:
        tick=perf_counter();timing=retime(task,adapter,basis,coeff);metric['T_retime']=perf_counter()-tick
        metric['planner_feasible']=timing['feasible']
    metric['total_plan_wall_s']=perf_counter()-start+(metric['T_init'] if method=='B2' else 0.)
    metric['planning_cost_scope']='All measured work including 30 s checkpoint experiment and validation; selected path is the verified 10 s incumbent, not a 10 s return-time guarantee.'
    if metric['planner_feasible']:
        metric['predicted_execution_s']=timing['duration'];metric['status']='planner_feasible'
        np.savez_compressed(root/'planned_path.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')})
        write(root/'timing.json',timing)
        if method=='B1' and repeat==0:
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if snapshot.exists():raise FileExistsError(snapshot)
            np.savez_compressed(snapshot,coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')})
            write(snapshot.parent/'cost.json',metric)
        if repeat==0:
            # Common timeout is derived from the independent saved reference
            # geometry, not from this method's optimized predicted time.
            reference=np.load(OUT/'inputs'/slot/'reference_geometry.npz')
            refcoef=np.linalg.lstsq(basis(reference['s']),reference['q'],rcond=None)[0]
            ref_timing=retime(task,adapter,basis,refcoef)
            if not ref_timing['feasible']:raise RuntimeError('Common timeout reference retiming failed')
            result,history,profiles,mask=execute(model,data,task,adapter,basis,coeff,timing,ref_timing['duration'])
            save_execution(root,result,history,profiles,mask);metric.update(result,physical_run=True)
    else:metric['status']='dense_geometry_or_timing_failed'
    write(root/'manifest.json',manifest);write(root/'metrics.json',metric)
    print('RUN',slot,method,repeat,metric['status'],'quality',metric['quality_completed'],flush=True)


def run():
    sealpath=OUT/'inputs/numerical_seal_grid_repair.json'
    if not sealpath.exists():sealpath=OUT/'inputs/seal.json'
    seal=json.loads(sealpath.read_text());cfg=seal['config']
    if hashes()!=seal['code_hashes']:raise RuntimeError('Code/config differs from input seal; document and reseal only BEFORE comparison')
    for rel,digest in seal['files'].items():
        if sha(OUT/rel)!=digest:raise RuntimeError('Input identity changed')
    rng=np.random.default_rng(cfg['schedule_seed'])
    for identity in seal['scenes']:
        task=ScanTask(identity['family'],identity['placement'],identity['direction'],identity['robot'])
        model,data,adapter,meta=physical_model(identity['robot'],task);collision=Clearance(model)
        # One uniform non-study geometry/QP and FK warmup before measurement.
        midpoint=(adapter.public.limits.lower+adapter.public.limits.upper)/2
        pose=adapter.public.forward(midpoint)
        for _ in range(3):adapter.solve(pose,midpoint,(adapter.public.limits.lower,adapter.public.limits.upper),(midpoint-.35,midpoint+.35),.15,(.001,np.deg2rad(.5)))
        for repeat in range(3):
            order=list(rng.permutation(['B0','B1','B2','B3','G']))
            if repeat==0 and order.index('B2')<order.index('B1'):
                i,j=order.index('B1'),order.index('B2');order[i],order[j]=order[j],order[i]
            for method in order:
                try:run_one(identity,method,repeat,cfg,model,data,adapter,collision)
                except Exception as error:
                    root=OUT/'runs'/identity['slot']/f'{method}_r{repeat}'
                    if (root/'metrics.json').exists():raise
                    root.mkdir(parents=True,exist_ok=True)
                    write(root/'implementation_error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                    write(root/'metrics.json',dict(scene_uid=identity['scene_uid'],slot=identity['slot'],cluster=identity['cluster'],
                          robot=identity['robot'],family=identity['family'],placement=identity['placement'],direction=identity['direction'],
                          method=method,repeat=repeat,implementation_ready=False,status='implementation_error',
                          planner_feasible=False,quality_completed=None,execution_completed=None,physical_run=False))
                    print('IMPLEMENTATION ERROR',identity['slot'],method,repeat,repr(error),flush=True)
