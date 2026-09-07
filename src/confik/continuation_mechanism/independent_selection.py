"""Independent, current-information configuration selection experiment.

No new runtime, fitted rule, look-ahead selector, or change to frozen solvers.
Run prepare, sample, pools, continuations in order; all writers are exclusive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import numpy as np
import scipy

from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..geometry import pose_error
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import csv_write, digest, json_write
from ..revision_compute_allocation.data import trajectories
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..types import IKQuery, Pose
from .disentangle_math import (MATCH_ORIENTATION, MATCH_POSITION,
    MIN_CONFIGURATION_DISTANCE, PERTURBATIONS, matched_projection, scales)
from .observation import DiagnosticTracIK, metrics

BASELINE = 'fe486140748f40bfbad72c633cdb98b3bcd38b55'
OUT = 'outputs/continuation_mechanism_study/independent_configuration_selection'
FAMILIES = ('smooth', 'near_singular', 'joint_limit_return', 'high_curvature')
RULES = ('original', 'nearest', 'max_scaled_sigma_min', 'max_scaled_manipulability')
SEED_BASE = 970907000
INTERVENTION = 74
FRAMES = 150
HORIZON = 30
REPEATS = 5
DT = .02


def fixed_schedule():
    return [dict(site_id=f'state_{i:02d}', family=family, seed=SEED_BASE+i,
                 intervention_frame=INTERVENTION, frames=FRAMES, suffix_frames=HORIZON)
            for i, family in enumerate(f for f in FAMILIES for _ in range(10))]


def choose_index(values, *, maximize=False):
    """First candidate wins exact ties; never consume outcomes."""
    if not len(values):
        raise ValueError('empty candidate pool')
    a = np.asarray(values, dtype=float)
    if np.isnan(a).any():
        raise ValueError('NaN score')
    return int(np.argmax(a) if maximize else np.argmin(a))


def spectrum_scores(jacobian, task_scale, joint_scale):
    sv = np.linalg.svd(jacobian/task_scale[:, None]*joint_scale[None, :], compute_uv=False)
    log_volume = float(np.log(sv).sum()) if np.all(sv > 0) else None
    return sv, log_volume


def select_current(kin, verifier, query, candidates):
    """Measure each rule independently, including its required Jacobian/SVD.

    The inputs contain only the common current query and accepted configurations.
    Auxiliary unscaled metrics and pose checks are separately timed diagnostics.
    """
    task, step = scales(kin, verifier, query.dt)
    qs = [np.asarray(c['q']) for c in candidates]
    selections, timings = {}, {}
    t = perf_counter_ns()
    selections['original'] = candidates[0]['candidate_id']
    timings['original'] = perf_counter_ns()-t
    t = perf_counter_ns()
    distance = [float(np.linalg.norm(q-query.previous_q)) for q in qs]
    selections['nearest'] = candidates[choose_index(distance)]['candidate_id']
    timings['nearest'] = perf_counter_ns()-t
    t = perf_counter_ns()
    spectra = [spectrum_scores(kin.jacobian(q), task, step)[0] for q in qs]
    selections['max_scaled_sigma_min'] = candidates[choose_index([s[-1] for s in spectra], maximize=True)]['candidate_id']
    timings['max_scaled_sigma_min'] = perf_counter_ns()-t
    t = perf_counter_ns()
    volumes = [spectrum_scores(kin.jacobian(q), task, step)[1] for q in qs]
    selections['max_scaled_manipulability'] = candidates[choose_index(
        [-np.inf if v is None else v for v in volumes], maximize=True)]['candidate_id']
    timings['max_scaled_manipulability'] = perf_counter_ns()-t
    t = perf_counter_ns()
    auxiliary = []
    for c, q, d, sv, volume in zip(candidates, qs, distance, spectra, volumes):
        raw = np.linalg.svd(kin.jacobian(q), compute_uv=False)
        e = pose_error(query.target, kin.forward(q))
        auxiliary.append(dict(candidate_id=c['candidate_id'], distance_to_previous=d,
            scaled_singular_values=sv.tolist(), scaled_sigma_min=float(sv[-1]),
            scaled_log_manipulability=volume,
            unscaled_singular_values=raw.tolist(), unscaled_sigma_min=float(raw[-1]),
            unscaled_log_manipulability=float(np.log(raw).sum()) if np.all(raw > 0) else None,
            position_residual_vector=e[:3].tolist(), orientation_residual_vector=e[3:].tolist(),
            normalized_residual_norm=float(np.linalg.norm(e/task)),
            current=metrics(kin, verifier, query, q)))
    return dict(selected=selections, scoring_latency_ns=timings, auxiliary=auxiliary,
                auxiliary_latency_ns=perf_counter_ns()-t,
                task_scale=task.tolist(), joint_step_scale=step.tolist())


def construct_pool(kin, verifier, query, center):
    start = perf_counter_ns()
    if not verifier.check(np.asarray(center), query).accepted:
        raise ValueError('center must pass the original verifier')
    pool = [dict(candidate_id='candidate_00', q=np.asarray(center).tolist(), source='original_trac')]
    attempts = []
    for amplitude in PERTURBATIONS:
        attempt = matched_projection(kin, verifier, query, center, amplitude)
        duplicate = next((c['candidate_id'] for c in pool
                          if np.max(np.abs(np.asarray(attempt['q'])-c['q'])) < 1e-10), None)
        attempt.update(duplicate=duplicate, retained=False)
        if attempt['usable'] and duplicate is None:
            cid = f'candidate_{len(pool):02d}'
            pool.append(dict(candidate_id=cid, q=attempt['q'], source='pose_matched',
                             amplitude=amplitude, projection=attempt))
            attempt.update(retained=True, candidate_id=cid)
        attempts.append(attempt)
        if len(pool) == 6:
            break
    return dict(candidates=pool, projection_attempts=attempts,
                generation_latency_ns=perf_counter_ns()-start,
                projection_latency_ns=sum(a['latency_ns'] for a in attempts),
                projection_residual_calls=sum(a['residual_calls'] for a in attempts))


def metadata_keys(value, out=None):
    """Identity-only overlap audit, not loading old trajectory outcomes."""
    if out is None:
        out = dict(uids=set(), seeds=set(), hashes=set())
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('uid', 'trajectory_uid') and isinstance(item, str):
                out['uids'].add(item)
            elif key == 'seed' and isinstance(item, int):
                out['seeds'].add(item)
            elif key == 'query_hash' and isinstance(item, str):
                out['hashes'].add(item)
            elif key == 'query_hashes' and isinstance(item, list):
                out['hashes'].update(x for x in item if isinstance(x, str))
            metadata_keys(item, out)
    elif isinstance(value, list):
        for item in value:
            metadata_keys(item, out)
    return out


def frame_record(site, frame, query, obs, command, elapsed):
    return dict(site_id=site['site_id'], uid=site['uid'], family=site['family'], frame=frame,
        previous_q=query.previous_q.tolist(), target_position=query.target.position.tolist(),
        target_rotation=query.target.rotation.tolist(), dt=query.dt, query_hash=query_digest(query),
        accepted=command is not None, q=None if command is None else command.tolist(),
        accepted_state_q=(query.previous_q if command is None else command).tolist(),
        solver_observation=obs, total_latency_ns=elapsed, native_trac_fev=None,
        failure_kind=obs['failure_kind'])


def window_summary(rows):
    failed = next((r for r in rows if not r['accepted']), None)
    return dict(complete=failed is None, accepted_frames=sum(r['accepted'] for r in rows),
        prefix_frames=next((i for i,r in enumerate(rows) if not r['accepted']), len(rows)),
        first_failure_frame=failed['frame'] if failed else None,
        first_failure_kind=failed['failure_kind'] if failed else None,
        total_latency_ns=sum(r['total_latency_ns'] for r in rows))


class IndependentSelection:
    def __init__(self, root='.'):
        self.root = Path(root).resolve()
        self.out = self.root/OUT
        self.cfg = load_config(self.root/'configs/paper_v2.yaml')
        self.kin = load_robot(self.cfg, 'panda')
        self.verifier = SolutionVerifier(self.kin, VerifierConfig(**self.cfg['verifier']))

    def trac(self):
        return DiagnosticTracIK(self.root/'tmp/revision_dependencies/build/librevision_trac.so',
            resolve_path(self.cfg, self.cfg['robots']['panda']['urdf']), self.kin, self.verifier, 5)

    def protect(self):
        p = json.loads((self.out/'protocol.json').read_text())
        for name, h in p['protected_files'].items():
            assert digest(self.root/name) == h, name
        for name, h in p['measurement_sources'].items():
            assert digest(self.root/name) == h, name
        return p

    def prepare(self):
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip() == BASELINE
        if self.out.exists():
            raise FileExistsError(self.out)
        protected = {}
        for name in ['outputs/continuation_mechanism_study/delivery_manifest.json',
                     'outputs/continuation_mechanism_study/residual_configuration_disentanglement/delivery_manifest.json']:
            manifest = json.loads((self.root/name).read_text())
            for path, h in manifest['files'].items():
                assert digest(self.root/path) == h, path
                protected[path] = h
            protected[name] = digest(self.root/name)
        old_protocol = json.loads((self.root/'outputs/continuation_mechanism_study/protocol.json').read_text())
        for name, h in old_protocol['frozen_inputs'].items():
            assert digest(self.root/name) == h, name
            protected[name] = h
        paths = ['src/confik/continuation_mechanism/independent_selection.py',
                 'src/confik/continuation_mechanism/disentangle_math.py',
                 'src/confik/continuation_mechanism/observation.py',
                 'src/confik/revision_compute_allocation/data.py',
                 'src/confik/revision_compute_allocation/trac_ik.py',
                 'src/confik/revision_compute_allocation/native/trac_adapter.cpp',
                 'src/confik/solvers/verifier.py', 'src/confik/kinematics/urdf.py',
                 'src/confik/geometry.py', 'configs/paper_v2.yaml',
                 'tests/test_independent_configuration_selection.py',
                 'tmp/revision_dependencies/build/librevision_trac.so']
        urdf = resolve_path(self.cfg, self.cfg['robots']['panda']['urdf'])
        paths.append(str(urdf))
        protocol = dict(baseline=BASELINE, created_utc=datetime.now(timezone.utc).isoformat(),
            question='Do classical current-only configuration criteria suffice on independent Panda states?',
            schedule=fixed_schedule(), robot='panda', dt=DT, frames=FRAMES,
            intervention_frame_zero_based=INTERVENTION, suffix_frames=HORIZON, repeats=REPEATS,
            geometry='unchanged revision_compute_allocation.data.trajectories, one independently seeded path per call; all FK witness paths retained',
            intervention='same predeclared time index for every path; stop history at first failure, no replacement state/path or second center solve',
            initialization='known initial joint state from reference start; all subsequent history/current commands generated by TRAC-IK with actual previous_q',
            candidates=dict(max_total=6, max_alternatives=5, amplitudes=PERTURBATIONS,
                match_position_m=MATCH_POSITION, match_orientation_rad=MATCH_ORIENTATION,
                minimum_configuration_difference_rad=MIN_CONFIGURATION_DISTANCE,
                method='unchanged finite nullspace perturbation and achieved-full-FK nonlinear projection',
                no_preview=True, deduplication_max_difference_rad=1e-10),
            rules=RULES, tie='first candidate in fixed original/perturbation order; exact numerical ties',
            scaling='D=diag(public position tolerance x3, orientation tolerance x3); S=diag(joint velocity*dt + public velocity_tolerance)',
            trac=dict(budget_ms=5, epsilon=1e-5, solve_type='Speed', seed='actual previous_q',
                      boundary='unchanged representable_interior', random_seed_control=False),
            scheduling=dict(seed=970907901, order='all pools/selections sealed first; randomized site order and candidate/repeat order'),
            feedback='after failure hold previous accepted q, advance every target through the fixed 30 frames; completion requires no failure',
            statistics=dict(unit='one state from each of 40 independently seeded reference trajectories',
                repeat='five within-state search repeats, not independent trajectories',
                primary='all 40 scheduled states: unavailable contributes zero delivered-window success; conditional availability result separately labelled',
                prefix='mean within-state contiguous prefix; unavailable is 0 delivered suffix frames in all-scheduled summary, undefined IK window outcome',
                recovery='strictly higher completed-repeat count vs original; degradation strictly lower; stable recovery=0/5 vs 5/5',
                success_candidate='report both any repeat success and stable 5/5 success',
                hindsight='maximize completed-repeat count, then mean contiguous prefix, then fixed candidate order; descriptive upper bound only',
                groups='all candidates 5/5, some but not all repeat outcomes succeed, all candidates 0/5; post-hoc explanation only',
                uncertainty='descriptive state-level counts and paired mean differences; no total gate or significance threshold'),
            timing='history and center TRAC calls separately; candidate generation includes all projections and filtering; each rule independently scores full pool including its required FK/Jacobian/SVD; auxiliary diagnostics excluded and separately timed; suffix outer solve+verification time; no online acceleration claim excluding pool cost',
            fev='projection residual callbacks available; native TRAC-IK FEV unavailable (null)',
            environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                             cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                             load_average=list(os.getloadavg()),
                             thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                             timing_context='shared workstation, not an isolated latency benchmark',
                             backend='URDFKinematics public FK/Jacobian and KDL inside official TRAC-IK'),
            protected_files=protected, measurement_sources={name:digest(self.root/name) for name in paths})
        json_write(self.out/'protocol.json', protocol)
        print('Fixed 40 seeds and frame 74, before any new trajectory or TRAC-IK outcome.', flush=True)

    def sample(self):
        self.protect()
        if (self.out/'sampling_started.json').exists():
            raise FileExistsError('sampling already started')
        json_write(self.out/'sampling_started.json', dict(utc=datetime.now(timezone.utc).isoformat()))
        prior = dict(uids=set(), seeds=set(), hashes=set())
        metadata_files = {}
        for path in sorted((self.root/'outputs').rglob('*identit*.json')):
            if self.out in path.parents:
                continue
            metadata_keys(json.loads(path.read_text()), prior)
            metadata_files[str(path.relative_to(self.root))] = digest(path)
        arrays, identities = [], []
        for si, site in enumerate(fixed_schedule()):
            config = dict(trajectory_seeds={'panda':site['seed']}, _source=self.cfg,
                trajectories_per_family=1, trajectory_frames=FRAMES, dt=DT,
                trajectory_families=[site['family']])
            start = perf_counter_ns()
            ds, ids = trajectories(self.kin, config, 'panda')
            identity = dict(ids[0], **{k:v for k,v in site.items() if k not in ids[0]},
                            generation_latency_ns=perf_counter_ns()-start)
            assert site['seed'] not in prior['seeds'] and identity['uid'] not in prior['uids']
            assert not set(identity['query_hashes']) & prior['hashes']
            assert not set(identity['query_hashes']) & {h for i in identities for h in i['query_hashes']}
            ds.trajectory_id[:] = si
            arrays.append(ds); identities.append(identity)
            print(f'{site["site_id"]} {site["family"]}: reference feasible, seed {site["seed"]}', flush=True)
        joined = QueryDataset(**{k:np.concatenate([getattr(d,k) for d in arrays]) for k in QueryDataset.__dataclass_fields__})
        with (self.out/'reference_trajectories.npz').open('xb') as f:
            np.savez_compressed(f, **{k:getattr(joined,k) for k in QueryDataset.__dataclass_fields__})
        json_write(self.out/'trajectory_identities.json', identities)
        json_write(self.out/'identity_seal.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=digest(self.out/'protocol.json'),
            files={n:digest(self.out/n) for n in ['reference_trajectories.npz','trajectory_identities.json']},
            prior_identity_files=metadata_files, prior_counts={k:len(v) for k,v in prior.items()},
            overlap=dict(uid=0, seed=0, query_hash=0), solver_calls=0))

    def load_data(self):
        self.protect()
        seal = json.loads((self.out/'identity_seal.json').read_text())
        assert digest(self.out/'protocol.json') == seal['protocol_sha256']
        for n,h in seal['files'].items():
            assert digest(self.out/n) == h, n
        return QueryDataset.load(self.out/'reference_trajectories.npz'), json.loads((self.out/'trajectory_identities.json').read_text())

    @staticmethod
    def query(ds, index, frame, previous):
        row = index*FRAMES+frame
        return IKQuery(Pose(ds.target_position[row], ds.target_rotation[row]), np.asarray(previous), DT)

    def pools(self):
        ds, identities = self.load_data()
        path = self.out/'history_raw.jsonl.gz'
        if path.exists():
            raise FileExistsError(path)
        trac = self.trac()
        blocks = []
        try:
            with gzip.open(path, 'xt', encoding='utf8') as stream:
                for si, site in enumerate(identities):
                    previous = ds.previous_q[si*FRAMES].copy()
                    block = dict(site=site, initial_previous_q=previous.tolist(), status='not_reached',
                                 candidates=[], history=[], current=None, scoring=None, pool=None)
                    for t in range(INTERVENTION+1):
                        query = self.query(ds, si, t, previous)
                        start = perf_counter_ns()
                        obs = trac.observe(query, interior=True)
                        command = np.asarray(obs['q']) if obs['accepted'] else None
                        if command is not None:
                            assert self.verifier.check(command, query).accepted
                        row = frame_record(site, t, query, obs, command, perf_counter_ns()-start)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        if t < INTERVENTION:
                            block['history'].append(dict(frame=t, accepted=row['accepted'],
                                failure_kind=row['failure_kind'], total_latency_ns=row['total_latency_ns']))
                        else:
                            block['current'] = row
                        if command is None:
                            block.update(status='no_legal_center' if t == INTERVENTION else 'not_reached',
                                         first_failure_frame=t, first_failure_kind=obs['failure_kind'])
                            break
                        previous = command.copy()
                        if t == INTERVENTION:
                            pool = construct_pool(self.kin, self.verifier, query, command)
                            scoring = select_current(self.kin, self.verifier, query, pool['candidates'])
                            block.update(status='available', candidates=pool['candidates'], pool=pool,
                                         scoring=scoring, first_failure_frame=None, first_failure_kind=None)
                    blocks.append(block)
                    stream.flush()
                    print(f'{site["site_id"]}: {block["status"]}; {len(block["candidates"])} candidates', flush=True)
        finally:
            trac.close()
        json_write(self.out/'candidate_pools.json', blocks)
        csv_write(self.out/'availability.csv', [dict(site_id=b['site']['site_id'], uid=b['site']['uid'],
            family=b['site']['family'], seed=b['site']['seed'], status=b['status'],
            intervention_frame=INTERVENTION, candidate_count=len(b['candidates']),
            first_failure_frame=b['first_failure_frame'], first_failure_kind=b['first_failure_kind']) for b in blocks])
        json_write(self.out/'selection_seal.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            files={n:digest(self.out/n) for n in ['protocol.json','identity_seal.json','candidate_pools.json','availability.csv','history_raw.jsonl.gz']},
            selectors_frozen_before_suffix_calls=True, suffix_solver_calls=0))

    def continuations(self):
        ds, identities = self.load_data()
        seal = json.loads((self.out/'selection_seal.json').read_text())
        for n,h in seal['files'].items():
            assert digest(self.out/n) == h, n
        path = self.out/'continuation_raw.jsonl.gz'
        if path.exists():
            raise FileExistsError(path)
        blocks = json.loads((self.out/'candidate_pools.json').read_text())
        rng = np.random.default_rng(970907901)
        trac = self.trac()
        runs, witnesses = [], []
        try:
            with gzip.open(path, 'xt', encoding='utf8') as stream:
                for si in rng.permutation(len(blocks)):
                    b = blocks[si]
                    if b['status'] != 'available':
                        continue
                    site = b['site']
                    jobs = [(c, r) for c in b['candidates'] for r in range(REPEATS)]
                    for job in rng.permutation(len(jobs)):
                        c, repeat = jobs[job]
                        previous = np.asarray(c['q'])
                        query0 = self.query(ds, si, INTERVENTION, b['current']['previous_q'])
                        assert self.verifier.check(previous, query0).accepted
                        run_id = f'{site["site_id"]}_{c["candidate_id"]}_r{repeat}'
                        rows = []
                        for t in range(INTERVENTION+1, INTERVENTION+1+HORIZON):
                            query = self.query(ds, si, t, previous)
                            start = perf_counter_ns()
                            obs = trac.observe(query, interior=True)
                            command = np.asarray(obs['q']) if obs['accepted'] else None
                            if command is not None:
                                assert self.verifier.check(command, query).accepted
                            row = frame_record(site, t, query, obs, command, perf_counter_ns()-start)
                            row.update(run_id=run_id, candidate_id=c['candidate_id'], repeat=repeat)
                            if command is not None:
                                previous = command.copy()
                            rows.append(row)
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                        run = dict(run_id=run_id, site_id=site['site_id'], uid=site['uid'], family=site['family'],
                                   candidate_id=c['candidate_id'], repeat=repeat, **window_summary(rows))
                        runs.append(run)
                        if run['complete']:
                            witnesses.append(dict(run=run, initial_previous_q=query0.previous_q.tolist(),
                                initial_q=c['q'], initial_frame=INTERVENTION,
                                frames=[{k:r[k] for k in ['frame','previous_q','q','target_position','target_rotation','dt','accepted']} for r in rows]))
                    stream.flush()
                    print(f'{site["site_id"]}: {len(jobs)} fixed 30-frame windows completed', flush=True)
        finally:
            trac.close()
        json_write(self.out/'continuation_runs.json', runs)
        csv_write(self.out/'continuation_runs.csv', runs)
        with gzip.open(self.out/'successful_witnesses.jsonl.gz', 'xt', encoding='utf8') as f:
            for w in witnesses:
                f.write(json.dumps(w, allow_nan=False)+'\n')
        json_write(self.out/'run_receipt.json', dict(windows=len(runs), suffix_solver_calls=len(runs)*HORIZON,
            successful_witnesses=len(witnesses), completed_utc=datetime.now(timezone.utc).isoformat(),
            selection_seal_sha256=digest(self.out/'selection_seal.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','sample','pools','continuations'])
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    getattr(IndependentSelection(args.root), args.stage)()


if __name__ == '__main__':
    main()
