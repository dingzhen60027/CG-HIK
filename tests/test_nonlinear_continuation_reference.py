import numpy as np
from scipy.optimize._numdiff import approx_derivative

from confik.config import load_config,load_robot
from confik.solvers.verifier import SolutionVerifier,VerifierConfig
from confik.types import IKQuery
from confik.continuation_mechanism.nonlinear_reference_math import (
    HorizonProblem,SearchBudget,pose_residual_jacobian,search_one,validate_path)


def setup():
    cfg=load_config('configs/paper_v2.yaml');kin=load_robot(cfg,'panda')
    v=SolutionVerifier(kin,VerifierConfig(**cfg['verifier']))
    q=(kin.limits.lower+kin.limits.upper)/2
    return cfg,kin,v,q


def test_pose_and_so3_jacobian_match_nonlinear_finite_differences():
    cfg,kin,v,q=setup()
    target=kin.forward(q+np.array([.03,-.02,.06,.04,-.05,.02,.01]))
    error,j=pose_residual_jacobian(kin,target,q)
    fd=approx_derivative(lambda x:pose_residual_jacobian(kin,target,x)[0],q,method='3-point')
    assert np.max(np.abs(j-fd))<1e-7


def test_full_horizon_constraints_keep_initial_q_fixed():
    cfg,kin,v,q=setup();targets=[kin.forward(q+.003*k) for k in [1,2,3]]
    problem=HorizonProblem(kin,v,q,targets)
    path=np.asarray([q+.002*k for k in [1,2,3]])
    x=problem.pack(path)
    fd=approx_derivative(problem.pose_constraints,x,method='3-point')
    assert np.max(np.abs(problem.pose_constraint_jacobian(x)-fd))<1e-5
    fdstep=approx_derivative(problem.step_constraints,x,method='3-point')
    assert np.max(np.abs(problem.step_constraint_jacobian(x)-fdstep))<1e-8
    assert len(x)==3*7+1 and np.array_equal(problem.q0,q)
    x[0]+=.1
    assert np.array_equal(problem.q0,q)


def test_known_witness_and_nonlinear_mismatch_are_distinguished():
    cfg,kin,v,q=setup();path=np.asarray([q+.001*k for k in [1,2,3]])
    targets=[kin.forward(x) for x in path]
    result=validate_path(kin,v,q,targets,path)
    assert result['found_legal_continuation'] and result['actual_rho']<1
    held=validate_path(kin,v,q,[kin.forward(q+.1)],np.asarray([q]))
    assert not held['found_legal_continuation']
    assert held['actual_rho']==0 and not held['pose_and_range_feasible']


def test_simple_nonlinear_search_returns_real_verified_path():
    cfg,kin,v,q=setup();target=kin.forward(q+.003)
    result=search_one(kin,v.config,q,[target],[],'hold',cfg,SearchBudget(max_iterations=60,max_seconds=10))
    assert result['fixed_q0_unchanged']
    assert result['found_legal_continuation']
    assert result['validation']['actual_rho']<=1
    assert result['counts']['total']['fk_calls']>0
    assert result['counts']['total']['geometric_jacobian_calls']>0
