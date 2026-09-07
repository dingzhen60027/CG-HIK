from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from confik.config import load_config,load_robot,resolve_path
from confik.continuation_mechanism.observation import representable_interior
from confik.continuation_mechanism.tolerance_comparison import run_summary,schedule,LIBRARY
from confik.continuation_mechanism.tolerance_solvers import (
    ToleranceSolver,QueryBoundedKinematics,cartesian_box,feedback,STRICT_EPS)
from confik.solvers.verifier import SolutionVerifier,VerifierConfig
from confik.types import IKQuery,Pose


def setup():
    cfg=load_config('configs/paper_v2.yaml');kin=load_robot(cfg,'panda')
    verifier=SolutionVerifier(kin,VerifierConfig(**cfg['verifier']))
    return cfg,kin,verifier


def native():
    if not Path(LIBRARY).exists():pytest.skip('build independent native adapter first')
    cfg,kin,v=setup()
    return ToleranceSolver('trac_task_5ms',kin,v,cfg,LIBRARY,resolve_path(cfg,cfg['robots']['panda']['urdf']))


def test_inscribed_box_is_not_the_full_public_ball():
    cfg,kin,v=setup();b=cartesian_box(v.config)
    assert np.all(b>STRICT_EPS)
    assert np.linalg.norm(b[:3])<=v.config.position_tolerance
    assert np.linalg.norm(b[3:])<=v.config.orientation_tolerance
    assert b[0]<.9*v.config.position_tolerance  # legitimate ball points are excluded


def test_upstream_residual_is_target_frame_rotation_vector_and_boxes_are_conservative():
    solver=native();rng=np.random.default_rng(970909800)
    try:
        for _ in range(100):
            target=Pose(rng.normal(size=3),Rotation.random(random_state=rng).as_matrix())
            e=(rng.uniform(-1,1,6)*.999)*solver.bounds
            actual=Pose(target.position+target.rotation@e[:3],target.rotation@Rotation.from_rotvec(e[3:]).as_matrix())
            out=np.empty(6)
            rc=solver.trac.lib.tolerance_relative_error(np.ascontiguousarray(target.position),
                np.ascontiguousarray(target.rotation),np.ascontiguousarray(actual.position),
                np.ascontiguousarray(actual.rotation),solver.bounds,STRICT_EPS,out)
            assert np.max(np.abs(out-e))<1e-12 and rc==1
            assert np.linalg.norm(out[:3])<solver.verifier.config.position_tolerance
            assert np.linalg.norm(out[3:])<solver.verifier.config.orientation_tolerance
        outside=target.position+target.rotation@np.array([solver.bounds[0]*1.1,0,0])
        assert solver.trac.lib.tolerance_relative_error(np.ascontiguousarray(target.position),
            np.ascontiguousarray(target.rotation),np.ascontiguousarray(outside),
            np.ascontiguousarray(target.rotation),solver.bounds,STRICT_EPS,np.empty(6))==0
    finally:solver.close()


def test_native_kinematics_joint_order_and_same_pose_solve():
    solver=native();rng=np.random.default_rng(970909801)
    try:
        for _ in range(20):
            q=solver.kin.random_configuration(rng);pose=solver.kin.forward(q)
            p,r=solver.trac.forward(q)
            assert np.max(np.abs(p-pose.position))<1e-12
            assert np.max(np.abs(r-pose.rotation))<1e-12
        result=solver.solve(pose.position,pose.rotation,q)
        assert result['accepted'] and result['internal_ok']
        assert result['total_latency_ns']>=result['solve_ns']+result['verification_ns']
    finally:solver.close()


def test_dls_variants_only_change_stopping_precision_and_share_actual_interval():
    cfg,kin,v=setup();a=ToleranceSolver('dls_strict',kin,v,cfg);b=ToleranceSolver('dls_task',kin,v,cfg)
    ca,cb=asdict(a.dls.config),asdict(b.dls.config)
    changed={k for k in ca if ca[k]!=cb[k]}
    assert changed=={'position_tolerance','orientation_tolerance'}
    q=(kin.limits.lower+kin.limits.upper)/2;target=kin.forward(q+.002)
    query=IKQuery(target,q,.02);lo,hi=representable_interior(kin,query,v)
    for solver in [a,b]:
        obs=solver.solve(target.position,target.rotation,q)
        assert np.array_equal(obs['lower'],lo) and np.array_equal(obs['upper'],hi)
        assert np.all(np.asarray(obs['q'])>=lo) and np.all(np.asarray(obs['q'])<=hi)
    bounded=QueryBoundedKinematics(kin);bounded.lower,bounded.upper=lo,hi
    assert np.array_equal(bounded.clip(q+2),hi)


def test_failed_dls_result_is_never_fed_forward():
    q=np.arange(7.)
    rejected=dict(accepted=False,q=(q+1).tolist())
    assert np.array_equal(feedback(q,rejected),q)
    assert np.array_equal(feedback(q,dict(accepted=True,q=(q+.01).tolist())),q+.01)


def test_deadline_completion_separate_from_geometric_completion_and_fixed_sample():
    rows=[dict(frame=i,accepted=True,returned_within_20ms=i!=2,total_latency_ns=21_000_000,
               failure_kind='accepted') for i in range(4)]
    r=run_summary(rows)
    assert r['complete'] and not r['complete_within_20ms']
    assert len(schedule())==40 and len({s['seed'] for s in schedule()})==40
    assert all(s['frames']==150 for s in schedule())
