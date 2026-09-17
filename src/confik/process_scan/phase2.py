"""Phase 2 fixed 24-scene development. No formal-scene entry point."""
from pathlib import Path
from time import perf_counter
import gzip
import json
import os
import platform
import shutil
import subprocess
import traceback
import numpy as np
from scipy.interpolate import BSpline
import yaml
from . import phase15
from .models import ROOT,physical_model
from .task import ScanTask
from .collision import Clearance
from .study import write,sha,clean,save_execution,FROZEN
from .planning import spline_basis,dense_validate,retime,optimize
from .execution import execute
from .phase2_graph import load_library,build_graph,shortest_path,transition_controls
from .phase2_optimize import optimize_transition_time

OUT=ROOT/'outputs/process_scan/phase2_development'
SOURCE=ROOT/'outputs/process_scan/phase15_baselines'
METHODS=('B0','B1','B2-common','A1','A2','Proposed')


def dump_gz(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(clean(value),f)


def sources():
    paths=[ROOT/p for p in FROZEN]+list((ROOT/'src/confik/process_scan').glob('*.py'))
    paths+=[ROOT/'scripts/run_process_scan_phase2.py',ROOT/'configs/process_scan_phase2.yaml']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths
            if 'reporting' not in p.name and 'render' not in p.name}


def prepare():
    if (OUT/'inputs/seal.json').exists():raise FileExistsError('Phase2 input seal already exists')
    cfg=yaml.safe_load((ROOT/'configs/process_scan_phase2.yaml').read_text())
    old=json.loads((SOURCE/'inputs/seal.json').read_text());records=[]
    for identity in old['scenes']:
        task=ScanTask(**identity['task']);model,data,adapter,meta=physical_model(task.robot,task);collision=Clearance(model)
        library=load_library(identity,SOURCE,task,adapter,collision,cfg)
        root=OUT/'inputs'/identity['slot'];root.mkdir(parents=True,exist_ok=True)
        write(root/'identity.json',identity)
        shutil.copyfile(SOURCE/'inputs'/identity['slot']/'scene.xml',root/'scene.xml')
        dest=OUT/identity['reference_source'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/identity['reference_source']/'reference.npz',dest/'reference.npz')
        dump_gz(root/'candidate_library.json.gz',library)
        refbasis=BSpline(library['reference_knots'],np.eye(len(library['reference_coeff'])),3)
        ref_time=retime(task,adapter,refbasis,library['reference_coeff'],speed_reserve=.90)
        assert ref_time['feasible'];write(root/'timeout_reference.json',ref_time)
        records.append(identity)
        print('PREPARE',identity['slot'],len(library['endpoints']),sum(map(len,library['endpoints'])),flush=True)
    write(OUT/'inputs/seal.json',dict(config=cfg,scenes=records,source_seal_sha256=sha(SOURCE/'inputs/seal.json'),
        code_hashes=sources(),files={str(p.relative_to(OUT)):sha(p) for p in (OUT/'inputs').rglob('*') if p.is_file()},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text(),affinity=sorted(os.sched_getaffinity(0)),
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        formal_test_run=False,old_methods_and_verifier_unchanged=True,
        candidate_library_source='Phase15 independent multistart reference only; no baseline planned_path reads'))


def library_read(identity):
    with gzip.open(OUT/'inputs'/identity['slot']/'candidate_library.json.gz','rt') as f:l=json.load(f)
    for key in ('grid','reference_q','reference_coeff','reference_knots'):l[key]=np.asarray(l[key])
    l['layers']=[np.asarray(x) for x in l['layers']]
    return l


def fit_graph(task,adapter,collision,library,graph,kind,cfg,root):
    start=perf_counter();selected=shortest_path(library,graph,kind);dump_gz(root/'selection.json.gz',selected)
    result=dict(feasible=False,reason=selected['status'],kind=kind,elapsed_s=0.)
    if not selected['feasible']:result['elapsed_s']=perf_counter()-start;write(root/'fit_summary.json',result);return result
    basis=spline_basis(task,cfg['control_points']);coeff=np.linalg.lstsq(basis(selected['s']),selected['q'],rcond=None)[0]
    coeff[0]=selected['q'][0];coeff[-1]=selected['q'][-1]
    np.savez_compressed(root/'raw_fit.npz',coeff=coeff,knots=basis.t)
    valid=dense_validate(task,adapter,basis,coeff,collision.numeric)
    # Common constrained C2 fitting, not an unconstrained interpolation accepted
    # as a path. It is charged to A1/A2/Proposed/B2-common alike when necessary.
    if not valid['feasible']:
        smoothing=optimize(task,adapter,basis,coeff,collision=collision,budget=cfg['geometry_fit_budget_s'])
        dump_gz(root/'constrained_fit.json.gz',smoothing)
        good=[x for x in smoothing['incumbents'] if x['available_s']<=cfg['geometry_fit_budget_s']]
        if not good:
            result.update(reason='graph_connected_but_no_verified_C2_fit',elapsed_s=perf_counter()-start)
            write(root/'fit_summary.json',result);return result
        coeff=min(good,key=lambda h:h['objective'])['coeff'];valid=dense_validate(task,adapter,basis,coeff,collision.numeric)
    timed=retime(task,adapter,basis,coeff,speed_reserve=cfg['common_speed_reserve']) if valid['feasible'] else None
    endpoint_error=float(max(np.max(abs(coeff[0]-selected['q'][0])),np.max(abs(coeff[-1]-selected['q'][-1]))))
    feasible=bool(valid['feasible'] and timed and timed['feasible'] and endpoint_error<1e-10)
    result.update(feasible=feasible,reason='verified_graph_initial_path' if feasible else 'graph_fit_or_timing_failed',
                  endpoint_error_rad=endpoint_error,elapsed_s=perf_counter()-start,
                  selected_node_fit_displacement_max_rad=float(np.max(abs(basis(task.knots)@coeff-selected['q'][library['endpoint_indices']]))))
    if feasible:
        np.savez_compressed(root/'path.npz',coeff=coeff,knots=basis.t,**{k:timed[k] for k in ('s','x','u','t')})
        write(root/'timing.json',timed)
        np.savez_compressed(root/'dense.npz',s=valid['s'],q=valid['q'])
    write(root/'fit_summary.json',result);return result


def graph_stage(identity,repeat,task,adapter,collision,cfg):
    root=OUT/'graphs'/identity['slot']/f'r{repeat}'
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    if root.exists():raise RuntimeError(f'Unfinished graph stage must be audited: {root}')
    root.mkdir(parents=True);start=perf_counter();library=library_read(identity)
    graph=build_graph(library,task,adapter,collision,cfg)
    dump_gz(root/'graph.json.gz',graph)
    graphwall=perf_counter()-start
    result=dict(shared_graph_s=graphwall,library_preparation_s=library['elapsed_s'],statistics=graph['statistics'])
    for kind in ('distance','time'):
        target=root/kind;target.mkdir()
        fit=fit_graph(task,adapter,collision,library,graph,kind,cfg,target)
        result[kind]=fit
        result[kind]['charged_initialization_s']=library['elapsed_s']+graphwall+fit['elapsed_s']
        write(target/'cost.json',result[kind])
    write(root/'summary.json',result)
    print('GRAPH',identity['slot'],repeat,result['statistics'],result['distance']['feasible'],result['time']['feasible'],flush=True)
    return result


def run_new(identity,method,repeat,cfg,task,model,data,adapter,collision,shared):
    root=OUT/'runs'/identity['slot']/f'{method}_r{repeat}'
    if (root/'metrics.json').exists():return
    if root.exists():raise RuntimeError(f'Unfinished condition must be audited: {root}')
    root.mkdir(parents=True);start=perf_counter();kind='distance' if method=='A1' else 'time'
    initialroot=OUT/'graphs'/identity['slot']/f'r{repeat}'/kind;fit=shared[kind]
    metric=dict(**{k:identity[k] for k in ('slot','scene_uid','cluster','robot','family','placement','direction')},
        method=method,repeat=repeat,planner_feasible=False,physical_run=False,quality_completed=None,execution_completed=None,
        T_init=fit['charged_initialization_s'],T_opt=None,status=fit['reason'],total_plan_wall_s=fit['charged_initialization_s'],
        shared_graph_statistics=shared['statistics'],endpoint_closure_error_rad=fit.get('endpoint_error_rad'),
        planning_cost_scope='Includes charged independent candidate validation, shared graph construction, own constrained fit, optimization, dense checks and TOPPRA; historical input-library acquisition excluded and disclosed.')
    write(root/'manifest.json',dict(identity=identity,method=method,repeat=repeat,code_hashes=sources(),
                                  shared_initial_source=str(initialroot.relative_to(OUT)),old_evidence_not_overwritten=True))
    if not fit['feasible']:write(root/'metrics.json',metric);return
    path=np.load(initialroot/'path.npz');coeff=path['coeff'];basis=BSpline(path['knots'],np.eye(len(coeff)),3)
    timing=json.loads((initialroot/'timing.json').read_text())
    for k in ('s','x','u','t'):timing[k]=np.asarray(timing[k])
    shutil.copyfile(initialroot/'path.npz',root/'path_initialization.npz')
    metric['shared_initial_sha256']=sha(initialroot/'path.npz')
    metric['first_legal_path_s']=metric['T_init']
    initial_coeff=coeff.copy();history=[dict(available_s=0.,objective=timing['duration'],coeff=coeff,timing=timing,source='graph_initial_path')]
    if method!='A2':
        controls,intervals=transition_controls(task,basis,cfg['boundary_layer_parameter_m'])
        local=method!='B2-common'
        result=optimize_transition_time(task,adapter,basis,coeff,timing,collision,budget=cfg['optimization_budget_s'],
            speed_reserve=cfg['common_speed_reserve'],grid_n=cfg['optimizer_grid_base_nodes'],
            active_controls=controls if local else None,active_intervals=intervals if local else None)
        dump_gz(root/'optimizer_trace.json.gz',result);metric['T_opt']=result['elapsed']
        metric['solver_return_status']=result['status'];metric['active_control_count']=len(controls) if local else len(coeff)-2
        history=[history[0]]+[h for h in result['incumbents'] if h['source']!='verified_initial_path']
        chosen=min(history,key=lambda h:h['objective'])
        coeff=chosen['coeff'];timing=chosen['timing']
        metric['selected_path_available_s']=metric['T_init']+chosen['available_s']
        metric['adopted_updates']=len(history)-1
        # The support mask freezes the scan interior exactly, not approximately.
        if local:
            fixed=np.setdiff1d(np.arange(len(coeff)),controls)
            np.testing.assert_allclose(coeff[fixed],initial_coeff[fixed],atol=1e-12,rtol=0)
    else:
        metric.update(selected_path_available_s=metric['T_init'],adopted_updates=0,active_control_count=0)
    write(root/'checkpoint_paths.json',{str(deadline):{
        'post_graph_budget_s':deadline,'best_plan_duration_s':min(h['objective'] for h in history if h['available_s']<=deadline),
        'physical_quality_not_yet_verified':True} for deadline in cfg['checkpoints_s']})
    dump_gz(root/'verified_history.json.gz',history)
    tick=perf_counter();valid=dense_validate(task,adapter,basis,coeff,collision.numeric);metric['T_validate']=perf_counter()-tick
    end_error=float(max(np.max(abs(coeff[0]-identity['common_start'])),np.max(abs(coeff[-1]-identity['common_end']))))
    metric['endpoint_closure_error_rad']=end_error
    np.savez_compressed(root/'dense_validation.npz',s=valid['s'],q=valid['q'],
        process=np.array([[r[k] for k in ('standoff_m','center_error_m','incidence_rad','line_error_rad')] for r in valid['process']]))
    tick=perf_counter();timing=retime(task,adapter,basis,coeff,speed_reserve=cfg['common_speed_reserve']) if valid['feasible'] else None
    metric['T_retime']=perf_counter()-tick
    metric['total_plan_wall_s']=metric['T_init']+perf_counter()-start
    metric['planner_feasible']=bool(valid['feasible'] and timing and timing['feasible'] and end_error<1e-10)
    metric['dense_min_clearance_m']=valid['min_clearance']
    altered=[]
    for i,segment in enumerate(task.segments):
        if not segment[5]:
            g=np.linspace(task.knots[i],task.knots[i+1],31)
            if np.max(abs(basis(g)@(coeff-initial_coeff)))>1e-8:altered.append(i)
    metric['transitions_actually_modified']=altered
    if metric['planner_feasible']:
        metric.update(status='planner_feasible',predicted_execution_s=timing['duration'])
        np.savez_compressed(root/'planned_path.npz',coeff=coeff,knots=basis.t,**{k:timing[k] for k in ('s','x','u','t')});write(root/'timing.json',timing)
        if repeat==0:
            common=json.loads((OUT/'inputs'/identity['slot']/'timeout_reference.json').read_text())
            result,hist,profiles,mask=execute(model,data,task,adapter,basis,coeff,timing,common['duration'],state_stride=1)
            save_execution(root,result,hist,profiles,mask);metric.update(result,physical_run=True)
    else:metric['status']='dense_geometry_timing_or_endpoint_failed'
    write(root/'metrics.json',metric)
    print('PHASE2',identity['slot'],method,repeat,metric['status'],metric['quality_completed'],metric['total_plan_wall_s'],flush=True)


def run(slots=None):
    seal=json.loads((OUT/'inputs/seal.json').read_text());cfg=seal['config']
    if seal['code_hashes']!=sources():raise RuntimeError('Numerical code changed after seal')
    for path,digest in seal['files'].items():
        if sha(OUT/path)!=digest:raise RuntimeError('Changed sealed input '+path)
    rng=np.random.default_rng(cfg['seed'])
    for identity in seal['scenes']:
        orders=[list(rng.permutation(METHODS)) for _ in range(cfg['repeats'])]
        if slots and identity['slot'] not in slots:continue
        task=ScanTask(**identity['task']);model,data,adapter,meta=physical_model(task.robot,task);collision=Clearance(model)
        q=np.array(identity['common_start']);pose=adapter.native.forward(q)
        for _ in range(3):adapter.solve(pose,q,(adapter.public.limits.lower,adapter.public.limits.upper),(q-.35,q+.35),.15,(.001,np.deg2rad(.5)))
        for repeat,order in enumerate(orders):
            shared=graph_stage(identity,repeat,task,adapter,collision,cfg)
            for method in order:
                if method in ('B0','B1'):
                    oldroot=phase15.OUT
                    try:
                        phase15.OUT=OUT
                        phase15.core_one(identity,method,repeat,cfg,model,data,adapter,collision)
                    finally:phase15.OUT=oldroot
                else:run_new(identity,method,repeat,cfg,task,model,data,adapter,collision,shared)


def integration(slot):
    """Interface-only smoke on a fixed development scene, separate from timed runs."""
    cfg=yaml.safe_load((ROOT/'configs/process_scan_phase2.yaml').read_text())
    identity=json.loads((SOURCE/'inputs'/slot/'identity.json').read_text());task=ScanTask(**identity['task'])
    model,data,adapter,meta=physical_model(task.robot,task);collision=Clearance(model)
    root=OUT/'integration'/slot
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    library=load_library(identity,SOURCE,task,adapter,collision,cfg)
    graph=build_graph(library,task,adapter,collision,cfg);dump_gz(root/'graph.json.gz',graph)
    write(root/'candidates.json',library['candidate_checks'])
    for kind in ('distance','time'):
        dest=root/kind;dest.mkdir();result=fit_graph(task,adapter,collision,library,graph,kind,cfg,dest)
        print('INTEGRATION',slot,kind,clean(result),flush=True)
        if result['feasible']:
            path=np.load(dest/'path.npz');basis=BSpline(path['knots'],np.eye(len(path['coeff'])),3)
            timing=json.loads((dest/'timing.json').read_text())
            for k in ('s','x','u','t'):timing[k]=np.asarray(timing[k])
            controls,intervals=transition_controls(task,basis,cfg['boundary_layer_parameter_m'])
            check=optimize_transition_time(task,adapter,basis,path['coeff'],timing,collision,budget=5.,
                derivative_check=True,active_controls=controls,active_intervals=intervals)
            dump_gz(dest/'kernel_check.json.gz',check)
            print('KERNEL',check['status'],check['derivative_check'],flush=True)
    print('GRAPH WORK',graph['elapsed_s'],graph['statistics'],flush=True)
