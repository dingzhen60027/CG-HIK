"""Post-measurement table checks: read-only and deliberately no IK execution."""
import json
from pathlib import Path
import pytest

ROOT=Path('outputs/task_contract_alignment/05_aggregate')


def read(name):
    return json.loads((ROOT/(name+'.json')).read_text())


def test_raw_replay_coverage_and_acceptance():
    r=read('verification')
    assert r['counts']==dict(zip(['02_point_study','03_contract_sensitivity',
        'panda_trajectory_replayed','ur5e_trajectory_replayed'],[120000,21000,84000,84000]))
    assert r['accepted_contract_violations']==0


def test_complete_fixed_populations_and_nested_repeats():
    assert len(read('point_main'))==32
    assert len(read('sensitivity_main'))==18
    assert len(read('trajectory_main'))==12
    assert len(read('trajectory_runs'))==1120
    for r in read('trajectory_main'):
        assert r['units']==40
        assert r['calls']==(6000 if r['method'].startswith('dls') else 18000)
    for r in read('point_main'):
        assert r['units']==(2500 if r['witness_feasible'] else 500)


def test_first_failure_merge_has_one_consistent_robot():
    rows=read('first_failures')
    assert {r['robot'] for r in rows}=={'panda','ur5e'}
    for r in rows:
        assert not r['accepted']
        assert r['source'].startswith('outputs/')


def test_panda_authoritative_values_and_retained_negative_result():
    t={(r['robot'],r['method']):r for r in read('trajectory_main')}
    assert t['panda','trac_strict_5ms']['completion_counts']==[34,34,34]
    assert t['panda','trac_task_5ms']['completion_counts']==[36,37,37]
    assert t['panda','trac_task_5ms']['cumulative_latency_ns_per_sweep']==2150280849.0
    assert t['panda','dls_task']['latency_p95_ms']==pytest.approx(1.45096235)
    assert t['ur5e','dls_strict']['completion_counts']==[35]
    assert t['ur5e','dls_task']['completion_counts']==[28]


def test_paired_independent_unit_counts():
    for name in ['point_paired','sensitivity_paired','trajectory_paired']:
        for r in read(name):
            assert r['units'] in [10,40,500,2500]
            assert len(set(r['gained_uids']))==len(r['gained_uids'])
            assert set(r['gained_uids']).isdisjoint(r['lost_uids'])
            assert r['latency_ratio'][1]<=r['latency_ratio'][2]
