"""One fixed development comparison of existing solver tolerance settings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import csv_write, digest, json_write
from ..revision_compute_allocation.data import trajectories
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..types import IKQuery, Pose
from .independent_selection import metadata_keys
from .tolerance_solvers import METHODS, STRICT_EPS, ITERATIONS, ToleranceSolver, cartesian_box, feedback

BASELINE = 'b247641bb98807339c98723dcdefb4d29fc8f103'
PARENT = Path('outputs/continuation_mechanism_study')
OUT = PARENT/'tolerance_matched_solver_comparison'
PRIOR = PARENT/'nonlinear_continuation_reference'
FAMILIES = ('smooth','near_singular','joint_limit_return','high_curvature')
FRAMES, DT, SEED_BASE = 150, .02, 970909000
LIBRARY = 'tmp/tolerance_solver_build/libtolerance_trac.so'
UPSTREAM = 'tmp/revision_dependencies/trac_ik'


def now():
    return datetime.now(timezone.utc).isoformat()


def schedule():
    return [dict(site_id=f'trajectory_{i:02d}',family=f,seed=SEED_BASE+i,frames=FRAMES)
            for i,f in enumerate(f for f in FAMILIES for _ in range(10))]


def run_summary(rows):
    bad = next((r for r in rows if not r['accepted']),None)
    return dict(complete=bad is None, accepted_frames=sum(r['accepted'] for r in rows),
        all_frames_within_20ms=all(r['returned_within_20ms'] for r in rows),
        complete_within_20ms=all(r['accepted'] and r['returned_within_20ms'] for r in rows),
        first_failure_frame=bad['frame'] if bad else None,
        first_failure_kind=bad['failure_kind'] if bad else None,
        first_failure=bad, total_latency_ns=sum(r['total_latency_ns'] for r in rows))


class ToleranceComparison:
    def __init__(self, root='.'):
        self.root = Path(root).resolve(); self.out = self.root/OUT
        self.cfg = load_config(self.root/'configs/paper_v2.yaml')
        self.kin = load_robot(self.cfg,'panda')
        self.verifier = SolutionVerifier(self.kin,VerifierConfig(**self.cfg['verifier']))

    def solver(self, method):
        return ToleranceSolver(method,self.kin,self.verifier,self.cfg,self.root/LIBRARY,
            resolve_path(self.cfg,self.cfg['robots']['panda']['urdf']))

    def prepare(self):
        if self.out.exists(): raise FileExistsError(self.out)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()==BASELINE
        prior=json.loads((self.root/PRIOR/'protocol.json').read_text())
        protected=dict(prior['protected_files'])
        manifest=self.root/PRIOR/'delivery_manifest.json'
        protected.update(json.loads(manifest.read_text())['files'])
        protected[str(manifest.relative_to(self.root))]=digest(manifest)
        for p,h in protected.items(): assert digest(self.root/p)==h,p
        upstream=self.root/UPSTREAM
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
        assert sha=='90162ac2ecc6ea8f88c6e99df6ee01efd217a3fb'
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip()
        names=['src/confik/continuation_mechanism/tolerance_comparison.py',
            'src/confik/continuation_mechanism/tolerance_solvers.py',
            'src/confik/continuation_mechanism/tolerance_native/tolerance_adapter.cpp',
            'src/confik/continuation_mechanism/tolerance_native/CMakeLists.txt',
            'tests/test_tolerance_solver_comparison.py','src/confik/solvers/dls.py',
            'src/confik/solvers/verifier.py','src/confik/continuation_mechanism/observation.py',
            'src/confik/revision_compute_allocation/data.py','src/confik/revision_compute_allocation/native/trac_adapter.cpp',
            'configs/paper_v2.yaml',LIBRARY,'tmp/revision_dependencies/build/librevision_trac.so']
        names += [str(p.relative_to(self.root)) for p in (upstream/'trac_ik_lib').rglob('*') if p.is_file()]
        cases=json.loads((self.root/PRIOR/'inputs.json').read_text())
        fixed=[]
        for c in cases:
            # A fixed initial candidate is the ACTUAL previous state for the next target.
            # The old witness is a label only and is never supplied to solver().
            q=IKQuery(Pose(**c['targets'][0]),np.asarray(c['q0']),DT)
            witnesses=sorted((self.root/PRIOR/'successful_witnesses').glob(f'{c["case_id"]}_L1_*.json'))
            assert witnesses
            w=json.loads(witnesses[0].read_text())['validation']
            assert np.array_equal(w['q0'],q.previous_q)
            assert self.verifier.check(np.asarray(w['path'][0]),q).accepted
            fixed.append(dict(input_id=c['case_id'],site_id=c['site_id'],role=c['role'],uid=c['uid'],
                frame=c['frame']+1,previous_q=c['q0'],target_position=q.target.position.tolist(),
                target_rotation=q.target.rotation.tolist(),dt=DT,query_hash=query_digest(q),
                known_legal_command_witness=str(witnesses[0].relative_to(self.root))))
        protocol=dict(baseline=BASELINE,created_utc=now(),methods=METHODS,schedule=schedule(),
            scope='mechanism development plus new independent reference-feasible trajectories; no method/threshold selection, no total gate',
            robot='panda',dt=DT,frames=FRAMES,public_verifier=asdict(self.verifier.config),
            native=dict(version='2.2.0',upstream_sha=sha,mode='Speed',epsilon=STRICT_EPS,
                strict_bounds=[0.]*6,task_bounds=cartesian_box(self.verifier.config).tolist(),budgets_ms=[5,20],
                residual='target-frame translation and rotation vector from diffRelative, not RPY; same norms as public world-frame error',
                interpretation='inscribed component box, NOT full equality with position/orientation norm balls; per-axis max(bounds, epsilon), no additive tolerance',
                source='https://github.com/traclabs/trac_ik/tree/'+sha),
            dls=dict(max_iterations=ITERATIONS,strict_position=STRICT_EPS,strict_orientation=STRICT_EPS,
                original_parameters=self.cfg['solver'],task_position=self.verifier.config.position_tolerance,
                task_orientation=self.verifier.config.orientation_tolerance,
                domain='unchanged AdaptiveDLS clipping operation on query-specific representable URDF/velocity intersection through kinematics adapter; no direction/damping/line-search change',
                timing='25 iterations are not a time budget; report measured <=5/20ms proportions'),
            shared_acceptance='public verifier alone on returned q, irrespective of internal convergence; internal/native and public results separately retained; no post-solve clipping or extra search',
            feedback='only verified q updates actual previous; failed return never fed forward, target advances normally',
            inputs='current target, actual previous_q, dt only; reference initial state shared, no future target or q_ref sent to solver',
            fixed_input_count=len(fixed),fixed_trac_repeats=5,fixed_dls_repeats=1,
            full_trac_repeats=3,full_dls_repeats=1,
            order_seed=970909901,search_random_seed_control=False,
            statistics='40 complete trajectories are independent units, repeats nested; retain all 150-frame runs; describe repeat range and per-UID completion counts; no significance gate',
            timing='from raw pose/previous conversion through bounds, native setup/solve or DLS, final public verification; model load/URDF parsing and offline metrics/serialization excluded; no deadline feedback rejection, deadline success separately counted',
            protected_files=protected,measurement_sources={n:digest(self.root/n) for n in names},
            environment=dict(python=platform.python_version(),numpy=np.__version__,
                backend='URDFKinematics public FK/Jacobian; KDL inside native TRAC-IK',
                thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                load_average=list(os.getloadavg()),affinity=sorted(os.sched_getaffinity(0)),shared_workstation=True))
        json_write(self.out/'protocol.json',protocol)
        json_write(self.out/'fixed_inputs.json',fixed)
        print(f'Fixed {len(fixed)} known-feasible inputs, six settings, and 40 new seeds.',flush=True)

    def check(self):
        p=json.loads((self.out/'protocol.json').read_text())
        for group in ['protected_files','measurement_sources']:
            for path,h in p[group].items(): assert digest(self.root/path)==h,path
        return p

    def sample(self):
        self.check()
        json_write(self.out/'sampling_started.json',dict(utc=now()))
        prior=dict(uids=set(),seeds=set(),hashes=set());files={}
        for path in sorted((self.root/'outputs').rglob('*identit*.json')):
            if self.out in path.parents: continue
            metadata_keys(json.loads(path.read_text()),prior)
            files[str(path.relative_to(self.root))]=digest(path)
        arrays=[];identities=[];new_hashes=set()
        for i,site in enumerate(schedule()):
            cfg=dict(trajectory_seeds={'panda':site['seed']},_source=self.cfg,
                trajectories_per_family=1,trajectory_frames=FRAMES,dt=DT,trajectory_families=[site['family']])
            ds,ids=trajectories(self.kin,cfg,'panda');identity=dict(ids[0],site_id=site['site_id'])
            assert site['seed'] not in prior['seeds'] and identity['uid'] not in prior['uids']
            assert not set(identity['query_hashes']) & (prior['hashes']|new_hashes)
            new_hashes.update(identity['query_hashes']);ds.trajectory_id[:]=i
            arrays.append(ds);identities.append(identity)
        joined=QueryDataset(**{k:np.concatenate([getattr(d,k) for d in arrays]) for k in QueryDataset.__dataclass_fields__})
        with (self.out/'reference_trajectories.npz').open('xb') as f:
            np.savez_compressed(f,**{k:getattr(joined,k) for k in QueryDataset.__dataclass_fields__})
        # The online stage loads only this stripped target/initial-state record.
        online=[dict(site_id=s['site_id'],uid=s['uid'],family=s['family'],seed=s['seed'],
            initial_q=ds.previous_q[0].tolist(),target_position=ds.target_position.tolist(),
            target_rotation=ds.target_rotation.tolist(),dt=DT)
            for s,ds in zip(identities,arrays)]
        json_write(self.out/'trajectory_identities.json',identities)
        json_write(self.out/'online_targets.json',online)
        json_write(self.out/'execution_seal.json',dict(utc=now(),solver_calls=0,
            files={n:digest(self.out/n) for n in ['protocol.json','fixed_inputs.json','reference_trajectories.npz','trajectory_identities.json','online_targets.json']},
            old_identity_metadata=files,prior_counts={k:len(v) for k,v in prior.items()},
            distinct_new_queries=len(new_hashes),all_reference_frames_verified=40*FRAMES))
        print('All 40 trajectories / 6000 reference transitions fixed and verified before study solver calls.',flush=True)

    def sealed(self):
        self.check()
        for n,h in json.loads((self.out/'execution_seal.json').read_text())['files'].items():
            assert digest(self.out/n)==h,n

    def fixed(self):
        self.sealed();json_write(self.out/'fixed_started.json',dict(utc=now()))
        inputs=json.loads((self.out/'fixed_inputs.json').read_text())
        solvers={m:self.solver(m) for m in METHODS}
        rng=np.random.default_rng(970909901)
        jobs=[(i,m,r) for i in inputs for m in METHODS for r in range(5 if m.startswith('trac') else 1)]
        rows=[]
        try:
            for ix in rng.permutation(len(jobs)):
                i,m,r=jobs[ix]
                result=solvers[m].solve(i['target_position'],i['target_rotation'],i['previous_q'],i['dt'])
                rows.append(dict(input_id=i['input_id'],site_id=i['site_id'],uid=i['uid'],repeat=r,
                    frame=i['frame'],previous_q=i['previous_q'],target_position=i['target_position'],
                    target_rotation=i['target_rotation'],dt=i['dt'],query_hash=i['query_hash'],**result))
        finally:
            for s in solvers.values():s.close()
        json_write(self.out/'fixed_records.json',rows)
        csv_write(self.out/'fixed_records.csv',rows)
        json_write(self.out/'fixed_completed.json',dict(utc=now(),calls=len(rows)))
        print(f'{len(rows)} fixed-input solver calls completed.',flush=True)

    def online(self):
        self.sealed();assert (self.out/'fixed_completed.json').exists()
        json_write(self.out/'online_started.json',dict(utc=now()))
        inputs=json.loads((self.out/'online_targets.json').read_text())
        solvers={m:self.solver(m) for m in METHODS}
        jobs=[(i,m,r) for i in inputs for m in METHODS for r in range(3 if m.startswith('trac') else 1)]
        rng=np.random.default_rng(970909902);summaries=[]
        (self.out/'online_runs').mkdir()
        try:
            for ix in rng.permutation(len(jobs)):
                i,m,r=jobs[ix];rid=f'{i["site_id"]}_{m}_r{r}'
                previous=np.asarray(i['initial_q']);rows=[]
                for t,(p,rot) in enumerate(zip(i['target_position'],i['target_rotation'])):
                    result=solvers[m].solve(p,rot,previous,i['dt'])
                    row=dict(run_id=rid,site_id=i['site_id'],uid=i['uid'],family=i['family'],repeat=r,
                        frame=t,previous_q=previous.tolist(),target_position=p,target_rotation=rot,dt=i['dt'],**result)
                    previous=feedback(previous,result);row['accepted_state_q']=previous.tolist();rows.append(row)
                with gzip.open(self.out/'online_runs'/f'{rid}.jsonl.gz','xt',encoding='utf8') as f:
                    for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
                summary=dict(run_id=rid,site_id=i['site_id'],uid=i['uid'],family=i['family'],method=m,repeat=r,
                    frames=FRAMES,initial_q=i['initial_q'],**run_summary(rows))
                summaries.append(summary)
                if len(summaries)%20==0: print(f'{len(summaries)}/{len(jobs)} full 150-frame runs saved.',flush=True)
        finally:
            for s in solvers.values():s.close()
        json_write(self.out/'online_run_summaries.json',summaries)
        csv_write(self.out/'online_run_summaries.csv',[{k:v for k,v in r.items() if k!='first_failure'} for r in summaries])
        json_write(self.out/'online_completed.json',dict(utc=now(),runs=len(summaries),frames=len(summaries)*FRAMES))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sample','fixed','online']);p.add_argument('--root',default='.')
    a=p.parse_args();getattr(ToleranceComparison(a.root),a.stage)()


if __name__=='__main__': main()
