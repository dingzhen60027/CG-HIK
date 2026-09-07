"""Residual/configuration intervention entry point, never a versioned runtime.

Stages are exclusive writers. Reuse all old identities and data; only projected
current candidates and the uniform-refinement treatment are newly executed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import numpy as np
import scipy

from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import digest, json_write, csv_write
from .study import Study
from .observation import metrics
from .disentangle_math import (MATCH_POSITION, MATCH_ORIENTATION, MIN_CONFIGURATION_DISTANCE,
    PERTURBATIONS, candidate_features, matched_projection, next_target_demand, uniform_refine)

BASELINE = 'e192176f3182f86a4356a3d0003957ce170a736f'
FOLDER = 'residual_configuration_disentanglement'


class Disentanglement:
    def __init__(self, root='.'):
        self.study = Study(root)
        self.oldout = self.study.out
        self.out = self.oldout/FOLDER
        self.root = self.study.root
        self.pools = json.loads((self.oldout/'candidate_pools.json').read_text())
        self.oldruns = json.loads((self.oldout/'continuation_runs.json').read_text())
        self.sites = [b for b in self.pools if b['site']['frame']>0]
        assert len(self.pools)==18 and sum(len(b['candidates']) for b in self.pools)==95
        assert len(self.sites)==9

    def protect(self):
        self.study.check_seal()
        old=json.loads((self.oldout/'delivery_manifest.json').read_text())
        for name,h in old['files'].items():
            assert digest(self.root/name)==h, name
        if (self.out/'protocol.json').exists():
            p=json.loads((self.out/'protocol.json').read_text())
            for name,h in p['measurement_sources'].items():
                assert digest(self.root/name)==h, name
        return True

    def prepare(self):
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()==BASELINE
        if self.out.exists(): raise FileExistsError(self.out)
        self.protect()
        names=['src/confik/continuation_mechanism/disentangle.py',
               'src/confik/continuation_mechanism/disentangle_math.py',
               'tests/test_residual_configuration.py',
               'tmp/revision_dependencies/build/librevision_trac.so']
        protocol=dict(baseline=BASELINE,created_utc=datetime.now(timezone.utc).isoformat(),
            scope='Selected Panda mechanism states; no population rate, no total gate, no new algorithm',
            reused_sites=18,reused_candidates=95,intervention_sites=[b['site'] for b in self.sites],
            robot='panda',repeats=5,dt=.02,trac_budget_ms=5,
            matching_precision=dict(position_m=MATCH_POSITION,orientation_rad=MATCH_ORIENTATION),
            minimum_configuration_difference_rad=MIN_CONFIGURATION_DISTANCE,
            projection=dict(centers=['original_focal','refined_focal'],
                max_new_candidates_per_site=6,max_per_center=3,perturbations=PERTURBATIONS,
                nullspace='SVD of task-tolerance and joint-speed scaled Jacobian',
                projection='fix largest null coordinate; nonlinear least-squares of full achieved center pose in remaining six coordinates',
                max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12,
                retention='current verifier AND actual full-pose matching AND distinct; never continuation outcomes'),
            refinement=dict(objective='sum of squared current translation(m) and rotation-vector(rad) residuals, identical to old study',
                max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12,
                selection='use refined result only if current verifier accepts and same objective strictly decreases; otherwise retain original legal candidate',
                initial='old original/refined configurations reused; initial B refinement is timed as treatment costing and must reproduce an existing stored configuration',
                future='refine every TRAC-IK accepted candidate using actual previous_q; do not refine failed solver returns'),
            boundary='all new native calls/projections/refinements use unchanged representable_interior; original verifier unchanged',
            ordinary_A='reuse existing 5 ms interior-bound focal records; no rerun',
            current_only='reuse existing 5 ms interior-bound refined-center records',
            B='initial prescribed original candidate is refined, then accepted ordinary TRAC-IK candidates are refined every suffix frame',
            new_candidates='all new matched candidates receive ordinary 5 ms interior-bound TRAC-IK suffixes, five repeats',
            randomness='no TRAC-IK seed pairing/control claim; scheduling seed 960907 controls interleaving only',
            next_demand=dict(scaling='row scaling by separate public pose tolerances; columns by per-joint velocity*dt+tolerance',
                objective='minimize max absolute normalized joint increment over all seven variables',
                constraints='joint range; separate position and orientation L2 tolerance balls in linearized task space',
                solve='bounded linear least-squares start then SLSQP, ftol=1e-10, maxiter=300; report nonconvergence as unknown',
                validation='nonlinear FK and unchanged verifier on returned joint configuration, without clipping an over-speed demand',
                scope='offline reads next target; local-linear optimizer is not an infeasibility certificate or a deployed no-preview criterion'),
            timing='suffix outer call includes native solve and original diagnostics, refinement if applicable, final verification; candidate feature/next-target diagnostics and serialization excluded; initial prescribed-candidate refinement timed separately',
            evaluation_counts='actual refinement residual callbacks including finite differences, scipy.nfev and verifier calls reported separately; native TRAC-IK FEV and hence total FEV unavailable, never zero',
            historical_timing='A and current-only are reused historical measurements, not new contemporaneous speed trials',
            analysis='within-site candidate comparisons with unequal completion or median prefix; descriptive ranking concordance, ties half; unknown demands separately counted; no fitting, p-values or independent-repeat inference',
            figure=dict(archetype='quantitative grid',question='Does uniform refinement suffice, or do pose-matched configurations retain different continuation?',
                panels=['uniform vs current-only vs ordinary completion','full-pose matched configuration contrasts','within-site explanatory concordance','observed cost of uniform refinement'],
                backend='python',width_mm=183,height_mm=170,min_font_pt=5,formats=['pdf','svg','png'],
                integrity='all selected sites; no outcome-based candidate exclusion; source data and numerical repeats retained'),
            environment=dict(python=platform.python_version(),scipy=scipy.__version__,numpy=np.__version__,backend='URDFKinematics plus official TRAC-IK KDL'),
            parent_manifest_sha256=digest(self.oldout/'delivery_manifest.json'),
            measurement_sources={name:digest(self.root/name) for name in names})
        json_write(self.out/'protocol.json',protocol)
        print('Prepared fixed protocol: 9 intervention sites, at most 6 new matched candidates each.',flush=True)

    def project(self):
        self.protect(); blocks=[]; attempts=[]
        for block in self.sites:
            site=block['site']; query=self.study.query(site['uid'],site['frame'],site['previous_q'])
            new=[];centers=[]
            for source in ['original_focal','refined_focal']:
                center=next(c for c in block['candidates'] if source in c['aliases'])
                centers.append(dict(source=source,candidate_id=center['candidate_id'],q=center['q']))
                count=0
                for amp in PERTURBATIONS:
                    r=matched_projection(self.study.kin,self.study.verifier,query,center['q'],amp)
                    r.update(site_id=site['site_id'],center_id=center['candidate_id'],center_source=source)
                    old_same=next((c['candidate_id'] for c in block['candidates']
                        if np.max(np.abs(np.asarray(r['q'])-c['q']))<1e-10),None)
                    new_same=next((c['candidate_id'] for c in new
                        if np.max(np.abs(np.asarray(r['q'])-c['q']))<1e-10),None)
                    r.update(duplicate_old=old_same,duplicate_new=new_same,retained=False)
                    if r['usable'] and old_same is None and new_same is None:
                        cid=f'matched_{len(new):02d}'
                        new.append(dict(candidate_id=cid,q=r['q'],center_id=center['candidate_id'],center_source=source,
                                        projection=r))
                        r.update(retained=True,candidate_id=cid);count+=1
                    attempts.append(r)
                    if count==3:break
            assert len(new)<=6
            blocks.append(dict(site=site,centers=centers,candidates=new))
            print(f'{site["site_id"]}: {len(new)} matched candidates',flush=True)
        json_write(self.out/'projection_attempts.json',attempts)
        json_write(self.out/'matched_candidates.json',blocks)

    def diagnostics(self):
        self.protect(); records=[]
        new=json.loads((self.out/'matched_candidates.json').read_text())
        allblocks=[(b,'old') for b in self.pools]+[(b,'new_matched') for b in new]
        for block,source in allblocks:
            site=block['site']
            for c in block['candidates']:
                query=self.study.query(site['uid'],site['frame'],site['previous_q'])
                current=candidate_features(self.study.kin,self.study.verifier,query,c['q'])
                assert current['verifier_accepted']
                nxt=self.study.query(site['uid'],site['frame']+1,c['q'])
                demand=next_target_demand(self.study.kin,self.study.verifier,nxt)
                records.append(dict(site_id=site['site_id'],uid=site['uid'],role=site['role'],frame=site['frame'],
                    candidate_id=c['candidate_id'],source=source,query_hash=query_digest(query),
                    previous_q=site['previous_q'],target_position=query.target.position.tolist(),
                    target_rotation=query.target.rotation.tolist(),current=current,next_demand=demand))
            print(f'next-target diagnostics {site["site_id"]} {source}',flush=True)
        json_write(self.out/'candidate_diagnostics.json',records)

    def continuations(self):
        self.protect()
        new=json.loads((self.out/'matched_candidates.json').read_text())
        trac=self.study.trac(5); runs=[]; rng=np.random.default_rng(960907)
        path=self.out/'new_continuation_raw.jsonl.gz'
        if path.exists():raise FileExistsError(path)
        with gzip.open(path,'xt',encoding='utf8') as stream:
            for block in new:
                site=block['site']; old=next(b for b in self.sites if b['site']['site_id']==site['site_id'])
                focal=next(c for c in old['candidates'] if 'original_focal' in c['aliases'])
                refined=next(c for c in old['candidates'] if 'refined_focal' in c['aliases'])
                jobs=[(focal,'uniform_refinement',r) for r in range(5)]
                jobs += [(c,'ordinary_matched',r) for c in block['candidates'] for r in range(5)]
                for i in rng.permutation(len(jobs)):
                    c,variant,repeat=jobs[i];q=np.asarray(c['q']);t0=site['frame']
                    query0=self.study.query(site['uid'],t0,site['previous_q'])
                    initial_refinement=None
                    if variant=='uniform_refinement':
                        initial_refinement=uniform_refine(self.study.kin,self.study.verifier,query0,q)
                        q=np.asarray(initial_refinement['q'])
                        # This is treatment costing, not generation of a new pool.
                        expected=refined['q'] if initial_refinement['improved'] else focal['q']
                        assert np.max(np.abs(q-expected))<1e-10
                    assert self.study.verifier.check(q,query0).accepted
                    run_id=f'{site["site_id"]}_{c["candidate_id"]}_{variant}_r{repeat}'
                    initial=dict(frame=t0,previous_q=site['previous_q'],q=q.tolist(),query_hash=query_digest(query0),
                        target_position=query0.target.position.tolist(),target_rotation=query0.target.rotation.tolist(),
                        dt=.02,accepted=True,refinement=initial_refinement)
                    rows=[]
                    for t in range(t0+1,t0+site['continuation_frames']+1):
                        query=self.study.query(site['uid'],t,q);start=perf_counter_ns()
                        obs=trac.observe(query,interior=True)
                        ref=None; command=None
                        if obs['accepted']:
                            command=np.asarray(obs['q'])
                            if variant=='uniform_refinement':
                                ref=uniform_refine(self.study.kin,self.study.verifier,query,command)
                                command=np.asarray(ref['q'])
                            assert self.study.verifier.check(command,query).accepted
                        elapsed=perf_counter_ns()-start
                        if command is not None:q=command.copy()
                        row=dict(run_id=run_id,site_id=site['site_id'],uid=site['uid'],frame=t,repeat=repeat,
                            candidate_id=c['candidate_id'],variant=variant,query_hash=query_digest(query),
                            previous_q=query.previous_q.tolist(),target_position=query.target.position.tolist(),
                            target_rotation=query.target.rotation.tolist(),dt=.02,accepted=command is not None,
                            q=None if command is None else command.tolist(),accepted_state_q=q.tolist(),
                            solver_observation=obs,refinement=ref,total_latency_ns=elapsed,
                            refinement_residual_calls=ref['residual_calls'] if ref else 0,
                            native_trac_fev=None,total_fev=None,
                            failure_kind=obs['failure_kind'],
                            state_metrics=metrics(self.study.kin,self.study.verifier,query,q))
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    failures=[r for r in rows if not r['accepted']]
                    first=failures[0] if failures else None
                    summary=dict(run_id=run_id,site_id=site['site_id'],uid=site['uid'],role=site['role'],
                        candidate_id=c['candidate_id'],variant=variant,repeat=repeat,frame=t0,frames=len(rows),
                        complete=not failures,accepted_frames=sum(r['accepted'] for r in rows),
                        first_failure=first['frame'] if first else None,first_failure_kind=first['failure_kind'] if first else None,
                        contiguous_success_frames=first['frame']-t0-1 if first else len(rows),
                        total_latency_ns=sum(r['total_latency_ns'] for r in rows),
                        refinement_latency_ns=sum(r['refinement']['refinement_latency_ns'] for r in rows if r['refinement']),
                        refinement_residual_calls=sum(r['refinement_residual_calls'] for r in rows),
                        refinement_scipy_nfev=sum(r['refinement']['scipy_nfev'] for r in rows if r['refinement']),
                        initial_refinement_latency_ns=initial_refinement['refinement_latency_ns'] if initial_refinement else 0,
                        initial_refinement_residual_calls=initial_refinement['residual_calls'] if initial_refinement else 0,
                        initial=initial,native_trac_fev=None,total_fev=None)
                    if not failures:
                        wp=self.out/'successful_witnesses'/f'{run_id}.json'
                        json_write(wp,dict(run=summary,initial=initial,frames=rows,
                            scope='all commands including initial reachable candidate verified; fixed suffix only'))
                        summary['witness']=str(wp.relative_to(self.out))
                    runs.append(summary)
                stream.flush()
                print(f'completed {site["site_id"]}: {len(jobs)} new repeated windows',flush=True)
        trac.close()
        json_write(self.out/'new_continuation_runs.json',runs)
        csv_write(self.out/'new_continuation_runs.csv',[{k:v for k,v in r.items() if k!='initial'} for r in runs])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','project','diagnostics','continuations'])
    p.add_argument('--root',default='.')
    args=p.parse_args();getattr(Disentanglement(args.root),args.stage)()


if __name__=='__main__':main()
