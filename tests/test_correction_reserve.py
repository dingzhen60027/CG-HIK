"""Small algorithm tests. No frozen output files are written."""
import numpy as np
import pytest

from confik.config import load_config, load_robot, resolve_path
from confik.geometry import pose_error, axis_angle_matrix
from confik.types import Pose, IKQuery
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.correction_reserve.geometry import (predict_target, residual_linearization,
    correction_map, reserve_value, perturb_target, task_scale)
from confik.correction_reserve.runtime import CorrectionReserveIK
from confik.correction_reserve.native_geometry import NativeGeometry


@pytest.fixture(params=["panda","ur5e"])
def context(request):
    cfg=load_config("configs/paper_v2.yaml")
    kin=load_robot(cfg,request.param)
    v=SolutionVerifier(kin,VerifierConfig(**cfg["verifier"]))
    q=kin.random_configuration(np.random.default_rng(301),margin=.2)
    return cfg,kin,v,q,request.param


def test_prediction_only_observed_targets(context):
    _,kin,_,q,_=context
    a=kin.forward(q)
    delta=np.array([.01,-.006,.003])
    b=Pose(a.position+np.array([.002,.001,-.001]),
           a.rotation@axis_angle_matrix(delta,np.linalg.norm(delta)))
    p=predict_target(b,a)
    assert np.allclose(p.position,b.position+(b.position-a.position))
    assert np.allclose(p.rotation,b.rotation@axis_angle_matrix(delta,np.linalg.norm(delta)))
    assert np.allclose(predict_target(b,None).matrix,b.matrix)


def test_native_optimization_geometry_matches_unchanged_public_backend(context):
    cfg,kin,_,q,robot=context
    native=NativeGeometry(kin,resolve_path(cfg,cfg['robots'][robot]['urdf']))
    np.testing.assert_allclose(native.forward(q).matrix,kin.forward(q).matrix,atol=1e-12)
    np.testing.assert_allclose(native.jacobian(q),kin.jacobian(q),atol=1e-12)


def test_residual_jacobian_finite_difference(context):
    _,kin,v,q,_=context
    target=kin.forward(q+.004)
    scale=task_scale(v)
    _,J,_=residual_linearization(kin,target,q,scale)
    h=1e-7
    fd=np.column_stack([(pose_error(target,kin.forward(q+np.eye(kin.nq)[j]*h))-
                         pose_error(target,kin.forward(q-np.eye(kin.nq)[j]*h)))/(2*h)/scale
                        for j in range(kin.nq)])
    np.testing.assert_allclose(J,fd,atol=2e-5,rtol=2e-5)


def test_target_and_joint_map_cancel_at_first_order(context):
    _,kin,v,q,_=context
    target=kin.forward(q+.0001)
    scale=task_scale(v);step=kin.limits.velocity*.02+v.config.velocity_tolerance
    mapping=correction_map(kin,target,q,scale,step)
    assert mapping.full_rank
    direction=np.array([1.,-.3,.4,.5,-.2,.1]);direction/=np.linalg.norm(direction)
    eps=1e-4
    disturbed=perturb_target(target,direction,eps,scale)
    before=pose_error(target,kin.forward(q))/scale
    after=pose_error(disturbed,kin.forward(q+mapping.matrix@direction*eps))/scale
    assert np.linalg.norm(after-before)<1e-6


def test_row_norm_reserve_preserves_linear_joint_box(context):
    _,kin,v,q,_=context
    z=q+.002
    target=kin.forward(z)
    step=kin.limits.velocity*.02+v.config.velocity_tolerance
    gamma,mapping,slack=reserve_value(kin,target,q,z,task_scale(v),step)
    assert gamma>0 and mapping.full_rank and mapping.compensation_error<1e-7
    rng=np.random.default_rng(310)
    for _ in range(30):
        direction=rng.normal(size=6);direction/=np.linalg.norm(direction)
        corrected=z+mapping.matrix@direction*gamma
        assert np.max(np.abs(corrected-q)/step)<=1+1e-10
        assert np.all(corrected>=kin.limits.lower-1e-12)
        assert np.all(corrected<=kin.limits.upper+1e-12)


def test_rank_deficient_has_no_positive_guarantee(context):
    _,kin,v,q,_=context
    class RankDeficient:
        def forward(self,x):return kin.forward(x)
        def jacobian(self,x):
            J=kin.jacobian(x);J[-1]=0
            return J
    m=correction_map(RankDeficient(),kin.forward(q),q,task_scale(v),
                     kin.limits.velocity*.02+v.config.velocity_tolerance)
    assert not m.full_rank and m.matrix is None


@pytest.mark.parametrize("mode",["reserve","predictive","single","sigma"])
def test_online_runtime_only_returns_verified_commands(context,mode):
    cfg,kin,v,q,robot=context
    solver=CorrectionReserveIK(kin,v,cfg,"tmp/task_contract_build/libcontract_trac.so",
        resolve_path(cfg,cfg["robots"][robot]["urdf"]),mode=mode)
    try:
        previous=q.copy()
        for t in range(3):
            target=kin.forward(q+.001*t)
            r=solver.solve(target.position,target.rotation,previous,.02)
            if r["accepted"]:
                assert v.check(np.array(r["q"]),IKQuery(target,previous,.02)).accepted
                if r["nominal_next_verified"] and mode!="single":
                    pred=Pose(r["predicted_position"],r["predicted_rotation"])
                    assert v.check(np.array(r["nominal_next_q"]),IKQuery(pred,np.array(r["q"]),.02)).accepted
                previous=np.array(r["q"])
            assert r["total_latency_ns"]>=r["backup_ns"]
    finally:
        solver.close()
