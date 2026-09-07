import numpy as np
import pytest

from confik.config import load_config, load_robot
from confik.continuation_mechanism.independent_selection import (
    FAMILIES, RULES, choose_index, construct_pool, fixed_schedule,
    select_current, spectrum_scores, window_summary)
from confik.continuation_mechanism.disentangle_math import scales
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.types import IKQuery


def setup():
    cfg = load_config('configs/paper_v2.yaml')
    kin = load_robot(cfg, 'panda')
    verifier = SolutionVerifier(kin, VerifierConfig(**cfg['verifier']))
    q = (kin.limits.lower+kin.limits.upper)/2
    return kin, verifier, q


def test_schedule_is_fixed_balanced_and_fresh():
    schedule = fixed_schedule()
    assert len(schedule) == 40 and len({s['seed'] for s in schedule}) == 40
    assert all(sum(s['family'] == f for s in schedule) == 10 for f in FAMILIES)
    assert all(s['intervention_frame'] == 74 and s['frames']-75 >= 30 for s in schedule)


def test_ties_are_candidate_order_not_outcome():
    assert choose_index([3,3,3], maximize=True) == 0
    assert choose_index([3,1,1]) == 1
    assert choose_index([-np.inf,-np.inf], maximize=True) == 0
    with pytest.raises(ValueError):
        choose_index([np.nan])


def test_scores_use_explicit_task_and_velocity_scaling():
    j = np.zeros((6,7)); j[:,:6] = np.diag([1,2,3,4,5,6])
    d = np.arange(1,7); s = np.arange(1,8)*.02
    sv, volume = spectrum_scores(j,d,s)
    expected = np.linalg.svd(np.diag(.02*np.arange(1,7)),compute_uv=False)
    assert np.allclose(sv,expected)
    assert np.isclose(volume,np.log(expected).sum())


def test_pool_reuses_achieved_fk_and_legal_common_previous():
    kin, v, q = setup()
    target = q.copy(); target[0] += .0001
    query = IKQuery(kin.forward(target),q,.02)
    pool = construct_pool(kin,v,query,q)
    assert 1 < len(pool['candidates']) <= 6
    assert len(pool['projection_attempts']) <= 6
    for c in pool['candidates']:
        assert v.check(np.asarray(c['q']),query).accepted
        if c['source'] == 'pose_matched':
            assert c['projection']['position_match_error'] <= 1e-8
            assert c['projection']['orientation_match_error'] <= 1e-8
    scores = select_current(kin,v,query,pool['candidates'])
    assert set(scores['selected']) == set(RULES)
    assert scores['selected']['nearest'] == 'candidate_00'
    task, step = scales(kin,v,.02)
    assert np.array_equal(scores['joint_step_scale'],step)
    for rule in RULES:
        assert scores['scoring_latency_ns'][rule] >= 0
    singleton = select_current(kin,v,query,pool['candidates'][:1])
    assert set(singleton['selected'].values()) == {'candidate_00'}


def test_full_window_is_not_a_recovered_later_frame():
    rows = [dict(accepted=ok,frame=75+i,failure_kind='accepted' if ok else 'solver_failure',total_latency_ns=10)
            for i,ok in enumerate([True,False,True])]
    summary = window_summary(rows)
    assert not summary['complete'] and summary['accepted_frames'] == 2
    assert summary['prefix_frames'] == 1 and summary['first_failure_frame'] == 76
