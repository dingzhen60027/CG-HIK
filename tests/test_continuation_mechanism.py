"""Focused adapter checks; no supplementary experiment is rerun."""
import numpy as np
from confik.config import load_config, load_robot
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.types import IKQuery
from confik.continuation_mechanism.observation import metrics, failure_kind, representable_interior


def setup():
    cfg=load_config('configs/paper_v2.yaml');kin=load_robot(cfg,'panda')
    v=SolutionVerifier(kin,VerifierConfig(**cfg['verifier']))
    q=(kin.limits.lower+kin.limits.upper)/2
    return kin,v,IKQuery(kin.forward(q),q,.02)


def test_current_legality_and_missing_candidate():
    kin,v,query=setup()
    m=metrics(kin,v,query,query.previous_q)
    assert m['verifier_accepted'] and m['finite'] and m['velocity_margin_rad']>0
    assert metrics(kin,v,query,None)['finite'] is None


def test_failure_categories_are_separate():
    kin,v,query=setup()
    m=metrics(kin,v,query,query.previous_q)
    assert failure_kind(False,False,m)=='solver_failure'
    m.update(verification_reasons=['velocity_limit'])
    assert failure_kind(False,True,m)=='velocity_violation'
    m.update(verification_reasons=['position_tolerance','joint_limit'])
    assert failure_kind(False,True,m)=='pose_error+joint_limit'


def test_representable_interval_preserves_original_velocity_contract():
    kin,v,query=setup();rng=np.random.default_rng(960711)
    for _ in range(100):
        previous=kin.random_configuration(rng,.001)
        q=IKQuery(kin.forward(previous),previous,.02)
        lower,upper=representable_interior(kin,q,v)
        allowed=kin.limits.velocity*.02+v.config.velocity_tolerance
        assert np.all(np.abs(kin.difference(lower,previous))<=allowed)
        assert np.all(np.abs(kin.difference(upper,previous))<=allowed)
        assert np.all(lower>=kin.limits.lower) and np.all(upper<=kin.limits.upper)


def test_offline_reference_is_not_automatically_reachable():
    kin,v,query=setup();q=query.previous_q.copy();q[0]+=.2
    target=IKQuery(kin.forward(q),query.previous_q,.02)
    m=metrics(kin,v,target,q)
    assert m['position_error']<1e-12 and not m['verifier_accepted']
    assert 'velocity_limit' in m['verification_reasons']
