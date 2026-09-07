"""Reuse frozen mechanism samples for a nonlinear finite-horizon reference."""
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
import scipy

from ..config import load_config,load_robot,resolve_path
from ..revision_compute_allocation.common import digest,json_write,csv_write
from ..solvers.verifier import SolutionVerifier,VerifierConfig
from ..types import IKQuery,Pose
from .nonlinear_reference_math import HorizonProblem,SearchBudget,search_one,validate_path

BASELINE='9f00d6160b3e556a2201a401975e2e37447edbc0'
PARENT=Path('outputs/continuation_mechanism_study')
OUT=PARENT/'nonlinear_continuation_reference'
HORIZONS=(1,5,10,30)
INITIALIZATIONS=('hold','historical_prefix','dls_prediction')


def read_jsonl(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def target_list(rows):
    return [dict(position=r['target_position'],rotation=r['target_rotation']) for r in rows]


def successful_prefix(rows):
    prefix=[]
    for r in rows:
        if not r['accepted']:
            break
        prefix.append(r['q'])
    return prefix


def group_rows(path,site_ids):
    grouped={}
    for row in read_jsonl(path):
        if row['site_id'] in site_ids:
            grouped.setdefault(row['run_id'],[]).append(row)
    return {k:sorted(v,key=lambda r:r['frame']) for k,v in grouped.items()}


class NonlinearReference:
    def __init__(self,root='.'):
        self.root=Path(root).resolve();self.out=self.root/OUT
        self.cfg=load_config(self.root/'configs/paper_v2.yaml')
        self.kin=load_robot(self.cfg,'panda')
        self.verifier=SolutionVerifier(self.kin,VerifierConfig(**self.cfg['verifier']))

    def prepare(self):
        if self.out.exists():raise FileExistsError(self.out)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()==BASELINE
        protected={}
        for folder in [PARENT,PARENT/'residual_configuration_disentanglement',PARENT/'independent_configuration_selection']:
            manifest=self.root/folder/'delivery_manifest.json'
            for name,h in json.loads(manifest.read_text())['files'].items():
                assert digest(self.root/name)==h,name
                protected[name]=h
            protected[str(manifest.relative_to(self.root))]=digest(manifest)
        oldp=json.loads((self.root/PARENT/'protocol.json').read_text())
        for name,h in oldp['frozen_inputs'].items():
            assert digest(self.root/name)==h,name
            protected[name]=h
        independent=self.root/PARENT/'independent_configuration_selection'
        states=json.loads((independent/'state_results.json').read_text())
        controls=sorted([s for s in states if s.get('group')=='all_success'],key=lambda s:s['uid'])[:2]
        chosen=['state_14','state_18']+[s['site_id'] for s in controls]
        pools=json.loads((independent/'candidate_pools.json').read_text())
        runs=json.loads((independent/'continuation_runs.json').read_text())
        raw=group_rows(independent/'continuation_raw.jsonl.gz',chosen)
        cases=[]; selfchecks=[]
        for sid in chosen:
            block=next(b for b in pools if b['site']['site_id']==sid)
            for c in block['candidates']:
                rr=sorted([r for r in runs if r['site_id']==sid and r['candidate_id']==c['candidate_id']],key=lambda r:r['repeat'])
                rows=raw[rr[0]['run_id']]
                assert len(rows)==30 and np.array_equal(rows[0]['previous_q'],c['q'])
                case=dict(case_id=f'{sid}_{c["candidate_id"]}',site_id=sid,candidate_id=c['candidate_id'],
                    uid=block['site']['uid'],role='control' if sid not in ['state_14','state_18'] else 'focal',
                    frame=74,q0=c['q'],current_previous_q=block['current']['previous_q'],
                    current_target=dict(position=block['current']['target_position'],rotation=block['current']['target_rotation']),
                    targets=target_list(rows),historical_prefix=successful_prefix(rows),
                    ordinary_run_ids=[r['run_id'] for r in rr],ordinary_completed=sum(r['complete'] for r in rr),
                    ordinary_prefixes=[r['prefix_frames'] for r in rr],
                    historical_initializer_repeat=0,horizons=HORIZONS)
                cases.append(case)
                success=next((r for r in rr if r['complete']),None)
                if success:
                    selfchecks.append(dict(case_id=case['case_id'],source='independent_successful_witnesses',run_id=success['run_id']))
        residual=self.root/PARENT/'residual_configuration_disentanglement'
        original_pool=next(b for b in json.loads((self.root/PARENT/'candidate_pools.json').read_text()) if b['site']['site_id']=='case_01_pre_failure')
        matched_pool=next(b for b in json.loads((residual/'matched_candidates.json').read_text()) if b['site']['site_id']=='case_01_pre_failure')
        oldruns=json.loads((self.root/PARENT/'continuation_runs.json').read_text())
        matchedruns=json.loads((residual/'new_continuation_runs.json').read_text())
        oldraw=group_rows(self.root/PARENT/'continuation_raw.jsonl.gz',{'case_01_pre_failure'})
        matchedraw=group_rows(residual/'new_continuation_raw.jsonl.gz',{'case_01_pre_failure'})
        centers=[c for c in original_pool['candidates'] if c['candidate_id'] in ['candidate_00','candidate_05']]
        for c in centers+matched_pool['candidates']:
            is_center=c['candidate_id'].startswith('candidate')
            rr=sorted([r for r in (oldruns if is_center else matchedruns)
                if r['site_id']=='case_01_pre_failure' and r['candidate_id']==c['candidate_id']
                and r['variant']==('trac_5ms_interior' if is_center else 'ordinary_matched')],key=lambda r:r['repeat'])
            assert len(rr)==5
            rows=(oldraw if is_center else matchedraw)[rr[0]['run_id']]
            assert len(rows)==30 and np.array_equal(rows[0]['previous_q'],c['q'])
            site=original_pool['site']
            case=dict(case_id=f'case_01_{c["candidate_id"]}',site_id='case_01_pre_failure',candidate_id=c['candidate_id'],
                uid=site['uid'],role='prior_pose_matched_reversal',frame=33,q0=c['q'],
                current_previous_q=site['previous_q'],current_target=dict(position=site['target_position'],rotation=site['target_rotation']),
                targets=target_list(rows),historical_prefix=successful_prefix(rows),
                ordinary_run_ids=[r['run_id'] for r in rr],ordinary_completed=sum(r['complete'] for r in rr),
                ordinary_prefixes=[r['contiguous_success_frames'] for r in rr],historical_initializer_repeat=0,horizons=HORIZONS)
            cases.append(case)
            success=next((r for r in rr if r['complete']),None)
            if success:
                selfchecks.append(dict(case_id=case['case_id'],source=str((residual/success['witness']).relative_to(self.root)),run_id=success['run_id']))
        # The first-failure-input check uses each candidate's ACTUAL frame-101
        # state (repeat 0), never the other candidate's state or command.
        for cid in ['candidate_04','candidate_01']:
            rid=f'state_14_{cid}_r0';rows=raw[rid]
            before=next(r for r in rows if r['frame']==101)
            at=next(r for r in rows if r['frame']==102)
            assert before['accepted'] and np.array_equal(before['q'],at['previous_q'])
            cases.append(dict(case_id=f'failure_input_{cid}',site_id='state_14_frame_102',candidate_id=cid,
                uid=before['uid'],role='first_failure_input',frame=101,q0=before['q'],
                current_previous_q=before['previous_q'],current_target=dict(position=before['target_position'],rotation=before['target_rotation']),
                targets=target_list([at]),historical_prefix=successful_prefix([at]),
                ordinary_run_ids=[rid],ordinary_completed=int(at['accepted']),ordinary_prefixes=[int(at['accepted'])],
                historical_initializer_repeat=0,old_next_frame=at,horizons=[1]))
        assert len(cases)==34
        for case in cases:
            query=IKQuery(Pose(**case['current_target']),np.asarray(case['current_previous_q']),.02)
            assert self.verifier.check(np.asarray(case['q0']),query).accepted
        names=['src/confik/continuation_mechanism/nonlinear_reference.py',
               'src/confik/continuation_mechanism/nonlinear_reference_math.py',
               'tests/test_nonlinear_continuation_reference.py',
               'src/confik/solvers/dls.py','src/confik/solvers/verifier.py',
               'src/confik/geometry.py','src/confik/kinematics/urdf.py','configs/paper_v2.yaml',
               str(resolve_path(self.cfg,self.cfg['robots']['panda']['urdf']))]
        protocol=dict(baseline=BASELINE,created_utc=datetime.now(timezone.utc).isoformat(),
            scope='observed mechanism development, not fresh/formal test; no new candidates, no TRAC-IK reruns',
            controls=controls,case_count=len(cases),candidate_horizon_problems=sum(len(c['horizons']) for c in cases),
            horizons=HORIZONS,initializations=INITIALIZATIONS,budget=asdict(SearchBudget()),
            mathematical_problem='min rho>=0, fixed q_t; true nonlinear FK position/orientation balls at each next target, exact URDF ranges, abs joint difference <= rho*(v_i*dt+velocity_tolerance)',
            parameterization='z_k=(q_(t+k)-fixed_q_t)/(v_i*dt+velocity_tolerance); all 7*L joints optimized jointly plus rho',
            coordinates='world translation and Log(R_target R(q)^T); derivative -J_position and -Jr_inverse(error)*J_world_rotation',
            numerical_interior='fixed 1e-8 reserve in squared normalized pose-ball constraints, only to construct interior solutions; public epsilon/verifier unchanged; found path is feasible upper bound, not global optimum',
            optimizer='SciPy SLSQP analytic objective/constraint Jacobians; no optimizer tuning after observed results',
            initializers='hold; this candidate repeat-0 accepted prefix then final-state hold; sequential existing DLS with 25 iterations/target (initial guess only, may violate speed before optimization)',
            fairness='same three starts, 100 SLSQP iterations and 15s major-iteration callback limit per start at every horizon/candidate; no cross-horizon warm starts; DLS generation and verification times separately included in total',
            best='lowest actual max normalized joint step of an evaluated pose/range-feasible path; original optimizer rho and constraint defects also retained',
            fixed_current_candidate='immutable and excluded from decision vector; no future q_ref used',
            selfcheck='known witness constraint evaluation only, not new discovery or fair speed baseline',
            first_failure_check='fixed repeat 0, candidate04 and candidate01 each own frame101 q, common target102, L=1 and same three initializers',
            interpretation='rho>1 or local failure means not found in this search, never mathematical infeasibility; no total gate',
            environment=dict(python=platform.python_version(),scipy=scipy.__version__,numpy=np.__version__,
                backend='URDFKinematics',threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                cpu_count=os.cpu_count(),load_average=list(os.getloadavg()),shared_workstation=True),
            protected_files=protected,measurement_sources={n:digest(self.root/n) for n in names})
        json_write(self.out/'protocol.json',protocol)
        json_write(self.out/'inputs.json',cases)
        json_write(self.out/'known_witness_selection.json',selfchecks)
        json_write(self.out/'input_seal.json',dict(files={n:digest(self.out/n) for n in ['protocol.json','inputs.json','known_witness_selection.json']},
                    optimizer_calls=0,created_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Fixed {len(cases)} inputs / {protocol["candidate_horizon_problems"]} horizons / 3 starts.',flush=True)

    def check(self):
        p=json.loads((self.out/'protocol.json').read_text())
        for group in ['protected_files','measurement_sources']:
            for n,h in p[group].items():
                assert digest(self.root/n)==h,n
        for n,h in json.loads((self.out/'input_seal.json').read_text())['files'].items():
            assert digest(self.out/n)==h,n
        return p,json.loads((self.out/'inputs.json').read_text())

    def selfcheck(self):
        p,cases=self.check();lookup={c['case_id']:c for c in cases}
        chosen=json.loads((self.out/'known_witness_selection.json').read_text())
        independent_witnesses={w['run']['run_id']:w for w in read_jsonl(self.root/PARENT/'independent_configuration_selection/successful_witnesses.jsonl.gz')}
        checks=[]
        for record in chosen:
            case=lookup[record['case_id']]
            if record['source']=='independent_successful_witnesses':
                w=independent_witnesses[record['run_id']]
            else:
                w=json.loads((self.root/record['source']).read_text())
            path=np.asarray([r['q'] for r in w['frames']])
            assert np.array_equal(w['frames'][0]['previous_q'],case['q0'])
            for L in HORIZONS:
                targets=[Pose(**t) for t in case['targets'][:L]]
                result=validate_path(self.kin,self.verifier,case['q0'],targets,path[:L])
                assert result['found_legal_continuation'],(record,L)
                problem=HorizonProblem(self.kin,self.verifier,case['q0'],targets,interior=0.)
                x=problem.pack(path[:L])
                assert min(problem.pose_constraints(x))>=0 and min(problem.step_constraints(x))>=-1e-12
                checks.append(dict(case_id=case['case_id'],run_id=record['run_id'],horizon=L,actual_rho=result['actual_rho'],verified=True))
        json_write(self.out/'known_witness_selfcheck.json',dict(checks=checks,optimizer_calls=0,
            interpretation='known legal path recognition only; not evidence of a newly recovered failed candidate'))
        print(f'{len(checks)} known-witness horizon checks passed without optimization.',flush=True)

    def run(self):
        p,cases=self.check()
        assert (self.out/'known_witness_selfcheck.json').exists()
        json_write(self.out/'run_started.json',dict(utc=datetime.now(timezone.utc).isoformat(),
            input_seal_sha256=digest(self.out/'input_seal.json')))
        rng=np.random.default_rng(970908001)
        jobs=[(c,L) for c in cases for L in c['horizons']]
        summaries=[]
        for index in rng.permutation(len(jobs)):
            case,L=jobs[index]
            for kind in INITIALIZATIONS:
                path=self.out/'runs'/f'{case["case_id"]}_L{L}_{kind}.json'
                if path.exists():raise FileExistsError(path)
                result=search_one(self.kin,self.verifier.config,np.asarray(case['q0']),
                    [Pose(**t) for t in case['targets'][:L]],case['historical_prefix'][:L],kind,self.cfg,
                    SearchBudget(**p['budget']))
                result.update(case_id=case['case_id'],site_id=case['site_id'],candidate_id=case['candidate_id'],
                              uid=case['uid'],frame=case['frame'],role=case['role'])
                json_write(path,result)
                if result['found_legal_continuation']:
                    json_write(self.out/'successful_witnesses'/path.name,dict(case_id=case['case_id'],horizon=L,
                        initialization=kind,validation=result['validation'],source_run=str(path.relative_to(self.out)),
                        scope='fixed-current-candidate nonlinear reference; full original-verifier checks, no speed scaling of executed target sequence'))
                summaries.append(dict(case_id=case['case_id'],horizon=L,initialization=kind,
                    found=result['found_legal_continuation'],best_found_rho=result['best_found_rho'],
                    optimizer_status=result['optimizer']['code'],total_ms=result['timing_ns']['total']/1e6,
                    fk_calls=result['counts']['total']['fk_calls'],jacobian_calls=result['counts']['total']['geometric_jacobian_calls'],
                    run_file=str(path.relative_to(self.out))))
            local=summaries[-3:]
            print(f'{case["case_id"]} L={L}: {sum(r["found"] for r in local)}/3 starts verified; rho={[r["best_found_rho"] for r in local]}',flush=True)
        json_write(self.out/'run_summary.json',summaries)
        csv_write(self.out/'run_summary.csv',summaries)
        json_write(self.out/'run_completed.json',dict(runs=len(summaries),utc=datetime.now(timezone.utc).isoformat(),
            input_seal_sha256=digest(self.out/'input_seal.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','selfcheck','run']);parser.add_argument('--root',default='.')
    args=parser.parse_args();getattr(NonlinearReference(args.root),args.stage)()


if __name__=='__main__':main()
