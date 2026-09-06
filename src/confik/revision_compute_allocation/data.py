"""Geometry-only fresh sampling with offline verifier witnesses; no solver access."""
from __future__ import annotations
import hashlib
import numpy as np
from ..data.datasets import QueryDataset
from ..latency_pilot_v3.benchmark import query_digest, query_from_dataset
from ..solvers.verifier import SolutionVerifier, VerifierConfig
from ..types import IKQuery, Pose


def near_singular_center(kin, rng, count=64):
    pool = np.array([kin.random_configuration(rng, margin=.05) for _ in range(count)])
    return pool[np.argmin([kin.min_singular_value(q) for q in pool])]


def _pack(rows, nq):
    return QueryDataset(**{key: np.asarray([r[key] for r in rows]) for key in QueryDataset.__dataclass_fields__})


def points(kin, config, robot, *, smoke=False):
    seed = config['smoke_seeds' if smoke else 'point_seeds'][robot]
    rng = np.random.default_rng(seed)
    verifier = SolutionVerifier(kin, VerifierConfig(**config['_source']['verifier']))
    rows, identities, oracle = [], [], []
    # A global position bound: every revolute URDF origin segment contributes at
    # most its length; these robot chains contain no prismatic joints.
    if any(j.kind == 'prismatic' for j in kin.chain):
        raise ValueError('position-bound witness requires a revolute/fixed chain')
    reach_bound = sum(np.linalg.norm(j.origin[:3,3]) for j in kin.chain)
    for family, full_count in config['point_counts'].items():
        count = min(full_count, 4) if smoke else full_count
        for j in range(count):
            previous = near_singular_center(kin, rng) if family == 'near_singular' else kin.random_configuration(rng,.07)
            if family == 'near_limit':
                axis = int(rng.integers(kin.nq))
                previous[axis] = kin.limits.upper[axis] - rng.uniform(.001,.008)*(kin.limits.upper[axis]-kin.limits.lower[axis])
            scale = rng.uniform(.75,.98) if family == 'hard_feasible' else rng.uniform(.1,.6)
            direction = rng.uniform(-1,1,kin.nq)
            if family == 'hard_feasible':
                direction = rng.choice([-1.,1.],kin.nq)*rng.uniform(.75,1.,kin.nq)
            reference = kin.clip(previous + scale*kin.limits.velocity*config['dt']*direction, margin=0.)
            pose = kin.forward(reference)
            feasible = family != 'constructed_inexecutable'
            if not feasible:
                unit = rng.normal(size=3); unit /= np.linalg.norm(unit)
                pose = Pose(unit*(reach_bound+1+rng.uniform(0,.5)),pose.rotation)
                reference = np.full(kin.nq,np.nan)
            query = IKQuery(pose, previous, dt=config['dt'])
            if feasible and not verifier.check(reference, query).accepted:
                raise AssertionError('geometric point witness violates the public contract')
            i = len(rows)
            uid = hashlib.sha256(f'revision_point:{robot}:{seed}:{family}:{j}'.encode()).hexdigest()
            rows.append(dict(previous_q=previous,target_position=pose.position,target_rotation=pose.rotation,
                             reference_q=reference,category=family,expected_reachable=feasible,continuity_feasible=feasible,
                             trajectory_id=i,time_index=0))
            identities.append(dict(uid=uid,seed=seed,family=family,query_hash=query_digest(query),witness=feasible,
                                   construction='FK of velocity-admissible reference' if feasible else 'target norm exceeds sum of all chain origin lengths by >=1m',
                                   position_reach_upper_bound_m=float(reach_bound),sigma_previous=float(kin.min_singular_value(previous))))
            if feasible and j < (min(2,config['oracle_counts'][family]) if smoke else config['oracle_counts'][family]):
                oracle.append(i)
    return _pack(rows,kin.nq), identities, np.asarray(oracle,int)


def trajectories(kin, config, robot, *, smoke=False):
    seed = config['smoke_seeds'][robot]+100 if smoke else config['trajectory_seeds'][robot]
    rng = np.random.default_rng(seed)
    verifier = SolutionVerifier(kin, VerifierConfig(**config['_source']['verifier']))
    count = 1 if smoke else config['trajectories_per_family']
    frames = 8 if smoke else config['trajectory_frames']
    rows, identities = [], []
    for family in config['trajectory_families']:
        for j in range(count):
            center = near_singular_center(kin,rng,128) if family == 'near_singular' else kin.random_configuration(rng,.22)
            u = np.linspace(0,1,frames+1)
            amplitude = rng.uniform(.1,.35,kin.nq)*rng.choice([-1.,1.],kin.nq)
            phase = rng.uniform(-np.pi,np.pi,kin.nq)
            if family == 'near_singular':
                offsets = np.cos(np.pi*u[:,None])*amplitude
            elif family == 'high_curvature':
                offsets = amplitude*(.55*np.sin(2*np.pi*u[:,None]+phase)+.35*np.sin(8*np.pi*u[:,None]+phase)+.1*np.sin(18*np.pi*u[:,None]))
            else:
                offsets = amplitude*np.sin(2*np.pi*u[:,None]+phase)
            limit_joint = None
            if family == 'joint_limit_return':
                limit_joint = int(rng.integers(kin.nq))
                center[limit_joint] = kin.limits.upper[limit_joint] - .002
                offsets[:,limit_joint] = -.30*(1-np.sin(np.pi*u)**2)
            # Reduce amplitudes only by explicit position and velocity conditions.
            bound_factor = 1.
            for k in range(kin.nq):
                if offsets[:,k].max()>0:
                    bound_factor=min(bound_factor,(kin.limits.upper[k]-center[k]-1e-6)/offsets[:,k].max())
                if offsets[:,k].min()<0:
                    bound_factor=min(bound_factor,(center[k]-kin.limits.lower[k]-1e-6)/(-offsets[:,k].min()))
            offsets *= min(1.,max(bound_factor,1e-6))
            utilization = np.max(np.abs(np.diff(offsets,axis=0))/(kin.limits.velocity*config['dt']))
            max_util = .92 if family == 'high_curvature' else .65
            offsets *= min(1., max_util/max(utilization,1e-12))
            reference = center+offsets
            uid = hashlib.sha256(f'revision_trajectory:{robot}:{seed}:{family}:{j}'.encode()).hexdigest()
            ti = len(identities)
            hashes=[]
            for t in range(frames):
                pose=kin.forward(reference[t+1]); query=IKQuery(pose,reference[t],dt=config['dt'])
                if not verifier.check(reference[t+1],query).accepted:
                    raise AssertionError('reference path violates the public verifier')
                rows.append(dict(previous_q=reference[t],target_position=pose.position,target_rotation=pose.rotation,
                                 reference_q=reference[t+1],category=family,expected_reachable=True,continuity_feasible=True,
                                 trajectory_id=ti,time_index=t))
                hashes.append(query_digest(query))
            identities.append(dict(uid=uid,seed=seed,family=family,frames=frames,query_hashes=hashes,
                                   max_velocity_utilization=float(np.max(np.abs(np.diff(reference,axis=0))/(kin.limits.velocity*config['dt']))),
                                   min_sigma=float(min(kin.min_singular_value(q) for q in reference)),
                                   min_limit_margin=float(np.min(kin.joint_margin(reference))),limit_joint=limit_joint,
                                   selection='geometry only; no method calls or outcomes'))
    return _pack(rows,kin.nq),identities
