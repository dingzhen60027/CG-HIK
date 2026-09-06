"""Bounded Panda continuation study; old outputs/runtime are read-only inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import csv_write, digest, json_write
from ..revision_compute_allocation.benchmark import read_records
from ..revision_compute_allocation.policies import build_internal
from ..solvers.dls import AdaptiveDLS
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..types import IKQuery, Pose
from .observation import DiagnosticTracIK, metrics, observe_internal, refine_candidate

BASELINE = 'bea96172a955f5140d796d3615349c77f9d28e51'
OLD = Path('outputs/revision_compute_allocation/03_feasible_trajectory_benchmark')
OUT = Path('outputs/continuation_mechanism_study')
METHODS = ('full_cghik', 'always_hard', 'trac_ik_5ms')
REPEATS = 5
HORIZON = 30


class Study:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.out = self.root/OUT
        self.cfg = load_config(self.root/'configs/paper_v2.yaml')
        self.kin = load_robot(self.cfg, 'panda')
        self.verifier = SolutionVerifier(self.kin, VerifierConfig(**self.cfg['verifier']))
        self.dataset = QueryDataset.load(self.root/OLD/'panda_trajectories.npz')
        self.identities = json.loads((self.root/OLD/'panda_identities.json').read_text())
        self.records = read_records(self.root/OLD/'panda_raw_records.jsonl.gz')
        self.old = defaultdict(dict)
        for row in self.records:
            self.old[(row['uid'], row['method'])][row['frame']] = row
        self.ix = {}
        self.family = {}
        for ti, identity in enumerate(self.identities):
            indices = np.flatnonzero(self.dataset.trajectory_id == ti)
            self.ix[identity['uid']] = indices[np.argsort(self.dataset.time_index[indices])]
            self.family[identity['uid']] = identity['family']

    def query(self, uid, frame, previous):
        i = self.ix[uid][frame]
        return IKQuery(Pose(self.dataset.target_position[i], self.dataset.target_rotation[i]),
                       np.asarray(previous, dtype=float), .02)

    def reference(self, uid, frame):
        return self.dataset.reference_q[self.ix[uid][frame]]

    def trac(self, budget):
        return DiagnosticTracIK(self.root/'tmp/revision_dependencies/build/librevision_trac.so',
                  resolve_path(self.cfg, self.cfg['robots']['panda']['urdf']), self.kin, self.verifier, budget)

    def internal(self):
        return build_internal(self.root, self.cfg, 'panda', self.kin, None,
                              ['always_hard', 'full_cghik'])

    def add_state_context(self, row, uid, frame, query):
        row.update(target_position=query.target.position.tolist(), target_rotation=query.target.rotation.tolist(),
                   previous_q=query.previous_q.tolist(), query_hash=query_digest(query), dt=query.dt,
                   previous_to_reference_max_rad=float(np.max(np.abs(query.previous_q-self.reference(uid,max(0,frame-1))))))
        if row.get('q') is not None:
            q = np.asarray(row['q'])
            row['to_reference_max_rad'] = float(np.max(np.abs(q-self.reference(uid,frame))))
            for method in METHODS:
                other = self.old[(uid,method)][frame]['command_q']
                row[f'to_{method}_command_max_rad'] = None if other is None else float(np.max(np.abs(q-other)))
        return row

    def prepare(self):
        if self.out.exists():
            raise FileExistsError('diagnostic output directory already exists')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()!=BASELINE:
            raise RuntimeError('unexpected evidence baseline')
        cases = []
        for uid in sorted(self.ix):
            full = self.old[(uid,'full_cghik')]
            hard = self.old[(uid,'always_hard')]
            trac = self.old[(uid,'trac_ik_5ms')]
            if all(r['accepted'] for r in hard.values()) and not all(r['accepted'] for r in full.values()):
                cases.append(dict(uid=uid, role='cghik_lost', focal='full_cghik',
                                  first_failure=min(t for t,r in full.items() if not r['accepted'])))
            if not all(r['accepted'] for r in trac.values()):
                cases.append(dict(uid=uid, role='trac_failed', focal='trac_ik_5ms',
                                  first_failure=min(t for t,r in trac.items() if not r['accepted'])))
        assert len(cases)==7
        for family in ['near_singular','smooth']:
            good = [uid for uid in sorted(self.ix) if self.family[uid]==family and
                    all(all(r['accepted'] for r in self.old[(uid,m)].values()) for m in METHODS)]
            failed_frames=[c['first_failure'] for c in cases if self.family[c['uid']]==family]
            cases.append(dict(uid=good[0], role='successful_control', focal='always_hard', first_failure=None,
                              matched_frame=int(np.median(failed_frames))-1))
        sites=[]
        for ci,case in enumerate(cases):
            uid=case['uid'];case.update(case_id=f'case_{ci:02d}',family=self.family[uid])
            preframe=case.get('matched_frame', (case['first_failure'] or 1)-1)
            for label,frame in [('first_divergence',0),('pre_failure',preframe)]:
                row=self.old[(uid,case['focal'])][frame]
                query=self.query(uid,frame,row['previous_q'])
                site=dict(case,site_id=f'{case["case_id"]}_{label}',site_kind=label,frame=frame,
                          previous_q=query.previous_q.tolist(),target_position=query.target.position.tolist(),
                          target_rotation=query.target.rotation.tolist(),query_hash=query_digest(query),
                          continuation_frames=min(HORIZON,149-frame))
                if case['role']=='successful_control':site['site_kind']='initial_control' if frame==0 else 'matched_control'
                sites.append(site)
        protected=[self.root/OLD/x for x in ['panda_trajectories.npz','panda_identities.json','panda_raw_records.jsonl.gz']]
        protected += [self.root/'paper'/x for x in ['main.tex','main.pdf','references.bib']]
        seal=json.loads((self.root/'outputs/revision_compute_allocation/selection_and_identity_seal.json').read_text())
        frozen=dict(seal['frozen_release'])
        frozen.update({str(p.relative_to(self.root)):digest(p) for p in protected})
        sources={str(p.relative_to(self.root)):digest(p) for p in (self.root/'src/confik/continuation_mechanism').glob('*.py')}
        protocol=dict(role='small post-hoc mechanism study, not a formal test or new method',baseline=BASELINE,
          robot='panda',cases=cases,sites=sites,repeats=REPEATS,horizon=HORIZON,dt=.02,
          selection='all three CG-HIK lost and all four TRAC-IK failed paths; smallest-UID jointly successful path per represented family',
          first_divergence='frame 0; current input identical across original methods; short-window scope only',
          failure_window='focal first failure +/- 5 frames retained from old logs; fresh solver replay at +/- 2 frames',
          candidate_sources=['existing focal/other-method command','offline q_ref if currently reachable',
                             'frozen hard and CG-HIK on common input','five TRAC-IK calls on common input',
                             'previous-state DLS with 25 iterations','tight bounded residual refinement of focal candidate'],
          candidate_filter='unchanged public verifier; deduplicate max joint difference <= 1e-10, retain aliases and every attempt',
          closest_rule='minimum Euclidean distance to common previous_q among current legal pool, no future outcomes',
          primary_continuation='5 ms official TRAC-IK Speed, previous_q seed, all unique legal candidates, five repeats',
          numerical_controls='pre-failure/control sites: original-focal, nearest, refined-focal candidates; 100 ms and one-ULP interior bounds each five repeats',
          strict_refinement='scipy least_squares trf; common one-step bounds; max_nfev=200; ftol=xtol=gtol=1e-12; report residual <=1e-7 m/rad',
          boundary_control='implementation diagnostic only; nextafter bounds inward, never relax verifier or repair old records',
          feedback='accept updates previous_q; fail holds it; targets advance through every fixed window frame',
          randomness='no TRAC-IK random seed control asserted; official concurrent workers, repeated interleaved trials; local schedule seed 960701 controls order only',
          statistical_unit='selected site nested within nine trajectories; five numerical repetitions are not five independent trajectories; descriptive counts, no population p-values',
          figures='two quantitative trajectory grids; raw state/residual/velocity-margin traces, all five suffix repeats; first-failure markers; SVG/PDF and PNG previews',
          frozen_inputs=frozen,measurement_sources=sources,
          backend='repository URDFKinematics FK/geometric Jacobian; independent KDL FK in official TRAC-IK')
        json_write(self.out/'protocol.json',protocol)
        self.original_windows(cases)
        print(f'Prepared {len(cases)} cases / {len(sites)} same-input sites',flush=True)

    def original_windows(self,cases):
        rows=[];failures=[]
        for case in cases:
            uid=case['uid'];center=case['first_failure'] if case['first_failure'] is not None else case['matched_frame']+1
            for method in METHODS:
                series=self.old[(uid,method)]
                bad=[t for t,r in series.items() if not r['accepted']]
                f=min(bad) if bad else None
                if f is not None:
                    source=series[f]
                    failures.append(dict(case_id=case['case_id'],uid=uid,role=case['role'],family=case['family'],method=method,
                      first_failure=f,solver_return_code=source.get('solver_return_code'),reject_reason=source['reject_reason'],
                      verification_reasons=source['verification_reasons'],raw_candidate_retained=source['command_q'] is not None))
                for t in range(max(0,center-5),min(150,center+6)):
                    source=series[t];query=self.query(uid,t,source['previous_q'])
                    row=metrics(self.kin,self.verifier,query,source['command_q'])
                    row.update(case_id=case['case_id'],uid=uid,family=case['family'],method=method,frame=t,
                               old_accepted=source['accepted'],old_return_code=source.get('solver_return_code'),
                               old_reject_reason=source['reject_reason'],old_verifier_reasons=source['verification_reasons'])
                    self.add_state_context(row,uid,t,query)
                    held=metrics(self.kin,self.verifier,query,query.previous_q)
                    row.update(held_state_position_error=held['position_error'],held_state_orientation_error=held['orientation_error'])
                    rows.append(row)
        csv_write(self.out/'old_first_failures.csv',failures)
        json_write(self.out/'old_failure_windows.json',rows)

    def check_seal(self):
        p=json.loads((self.out/'protocol.json').read_text())
        for path,h in p['frozen_inputs'].items():
            if digest(self.root/path)!=h:raise AssertionError(f'changed frozen input: {path}')
        return p

    def diagnostics(self):
        p=self.check_seal();methods=self.internal();tracs={b:self.trac(b) for b in [5,100]}
        rows=[];internal=[]
        for case in p['cases']:
            uid=case['uid'];center=case['first_failure'] if case['first_failure'] is not None else case['matched_frame']+1
            for t in range(max(0,center-2),min(150,center+3)):
                old=self.old[(uid,case['focal'])][t];query=self.query(uid,t,old['previous_q'])
                if case['focal']!='trac_ik_5ms':
                    result,traces=observe_internal(methods[case['focal']],query)
                    internal.append(dict(case_id=case['case_id'],uid=uid,frame=t,query_hash=query_digest(query),
                       accepted=result.accepted,old_accepted=old['accepted'],reject_reason=result.reject_reason,traces=traces,
                       accepted_command_matches_old=bool(result.accepted and np.array_equal(result.q,old['command_q']))))
                for repeat in range(REPEATS):
                    for budget,interior in [(5,False),(100,False),(5,True)]:
                        row=tracs[budget].observe(query,interior=interior)
                        row.update(case_id=case['case_id'],uid=uid,family=case['family'],focal=case['focal'],
                                   frame=t,repeat=repeat,original_first_failure=case['first_failure'])
                        self.add_state_context(row,uid,t,query);rows.append(row)
            print(f'diagnostic replay {case["case_id"]} / {case["role"]}',flush=True)
        json_write(self.out/'fresh_failure_replays.json',rows)
        json_write(self.out/'internal_stage_traces.json',internal)
        for m in tracs.values():m.close()

    def candidates(self):
        p=self.check_seal();methods=self.internal();trac=self.trac(5);dls=AdaptiveDLS(self.kin)
        attempts=[];all_sites=[]
        for site in p['sites']:
            uid=site['uid'];t=site['frame'];query=self.query(uid,t,site['previous_q']);local=[]
            def add(name,q,detail=None):
                row=metrics(self.kin,self.verifier,query,q)
                row.update(site_id=site['site_id'],source=name,detail=detail)
                self.add_state_context(row,uid,t,query);local.append(row)
            focal=site['focal']
            add('original_focal',self.old[(uid,focal)][t]['command_q'],focal)
            for method in METHODS:
                if method!=focal:add('original_'+method,self.old[(uid,method)][t]['command_q'])
            add('offline_reference',self.reference(uid,t),'diagnostic only; not a future seed')
            for method,m in methods.items():
                result,traces=observe_internal(m,query)
                add('common_input_'+method,result.q,dict(accepted=result.accepted,route=result.entry_action,traces=traces))
            for repeat in range(REPEATS):
                obs=trac.observe(query)
                add(f'common_input_trac_{repeat}',obs['q'] if obs['solver_ok'] else None,obs)
            prev=dls.solve(query.target,query.previous_q,25,seed_source='diagnostic_previous_first')
            add('previous_first_dls',prev.q if prev.converged else None,dict(status=prev.reason,iterations=prev.iterations))
            original=local[0]
            if original['verifier_accepted']:
                refined=refine_candidate(self.kin,self.verifier,query,original['q'])
                add('refined_focal',refined['q'],refined)
            pool=[]
            for row in local:
                if not row['verifier_accepted']:continue
                q=np.asarray(row['q'])
                same=next((c for c in pool if np.max(np.abs(np.asarray(c['q'])-q))<=1e-10),None)
                if same is not None:same['aliases'].append(row['source'])
                else:pool.append(dict(candidate_id=f'candidate_{len(pool):02d}',q=q.tolist(),aliases=[row['source']],
                         initial_metrics={k:v for k,v in row.items() if k not in ['detail']},
                         distance_to_previous=float(np.linalg.norm(q-query.previous_q))))
            if not pool:raise AssertionError('no current-frame legal candidate at selected site')
            nearest=min(pool,key=lambda c:(c['distance_to_previous'],c['candidate_id']))['candidate_id']
            all_sites.append(dict(site=site,candidates=pool,nearest_candidate_id=nearest))
            attempts.extend(local)
            print(f'candidate pool {site["site_id"]}: {len(pool)} legal distinct / {len(local)} attempts',flush=True)
        json_write(self.out/'candidate_attempts.json',attempts)
        json_write(self.out/'candidate_pools.json',all_sites)
        trac.close()

    def continuations(self):
        self.check_seal()
        pools=json.loads((self.out/'candidate_pools.json').read_text())
        tracs={b:self.trac(b) for b in [5,100]}
        rng=np.random.default_rng(960701);summaries=[]
        path=self.out/'continuation_raw.jsonl.gz'
        if path.exists():raise FileExistsError(path)
        with gzip.open(path,'xt',encoding='utf8') as stream:
            for block in pools:
                site=block['site'];uid=site['uid'];t0=site['frame'];jobs=[]
                for c in block['candidates']:
                    for repeat in range(REPEATS):jobs.append((c,repeat,5,False))
                    special=(c['candidate_id']==block['nearest_candidate_id'] or
                             any(a in c['aliases'] for a in ['original_focal','refined_focal']))
                    if t0>0 and special:
                        for repeat in range(REPEATS):
                            jobs.extend([(c,repeat,100,False),(c,repeat,5,True)])
                for ji in rng.permutation(len(jobs)):
                    c,repeat,budget,interior=jobs[ji]
                    variant=f'trac_{budget}ms'+('_interior' if interior else '')
                    run_id=f'{site["site_id"]}_{c["candidate_id"]}_{variant}_r{repeat}'
                    q=np.asarray(c['q']).copy();first_failure=None;accepted_count=0;rows=[]
                    query0=self.query(uid,t0,site['previous_q'])
                    assert self.verifier.check(q,query0).accepted
                    initial=dict(frame=t0,query_hash=query_digest(query0),previous_q=query0.previous_q.tolist(),
                         target_position=query0.target.position.tolist(),target_rotation=query0.target.rotation.tolist(),dt=.02,
                         accepted=True,q=q.tolist(),source_aliases=c['aliases'],**{k:v for k,v in metrics(self.kin,self.verifier,query0,q).items() if k not in ['q']})
                    for t in range(t0+1,t0+1+site['continuation_frames']):
                        query=self.query(uid,t,q)
                        row=tracs[budget].observe(query,interior=interior)
                        row.update(run_id=run_id,site_id=site['site_id'],uid=uid,frame=t,repeat=repeat,
                                   candidate_id=c['candidate_id'],variant=variant)
                        self.add_state_context(row,uid,t,query)
                        if row['accepted']:
                            q=np.asarray(row['q']).copy();accepted_count+=1
                            assert self.verifier.check(q,query).accepted
                        elif first_failure is None:first_failure=t
                        row['accepted_state_q']=q.tolist()
                        held=metrics(self.kin,self.verifier,query,q)
                        row.update(state_position_error=held['position_error'],state_orientation_error=held['orientation_error'])
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    summary=dict(run_id=run_id,site_id=site['site_id'],case_id=site['case_id'],uid=uid,role=site['role'],
                       family=site['family'],site_kind=site['site_kind'],frame=t0,candidate_id=c['candidate_id'],
                       aliases=c['aliases'],nearest=c['candidate_id']==block['nearest_candidate_id'],repeat=repeat,variant=variant,
                       frames=len(rows),complete=first_failure is None,accepted_frames=accepted_count,first_failure=first_failure,
                       first_failure_kind=next((r['failure_kind'] for r in rows if not r['accepted']),None),
                       contiguous_success_frames=len(rows) if first_failure is None else first_failure-t0-1,
                       minimum_accepted_velocity_margin_rad=min((r['velocity_margin_rad'] for r in rows if r['accepted']),default=None),
                       total_latency_ns=sum(r['latency_ns'] for r in rows))
                    if first_failure is None:
                        wp=self.out/'successful_witnesses'/f'{run_id}.json'
                        json_write(wp,dict(run=summary,initial=initial,frames=rows,
                            scope='verified suffix witness including reachable current candidate, not a whole-original-trajectory completion claim'))
                        summary['witness']=str(wp.relative_to(self.out))
                    summaries.append(summary)
                stream.flush()
                print(f'continuations {site["site_id"]}: {len(jobs)} repeated windows retained',flush=True)
        json_write(self.out/'continuation_runs.json',summaries)
        csv_write(self.out/'continuation_runs.csv',summaries)
        for m in tracs.values():m.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','diagnostics','candidates','continuations'])
    parser.add_argument('--root',default='.')
    args=parser.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    study=Study(args.root)
    getattr(study,args.stage)()


if __name__=='__main__':main()
