from confik.continuation_mechanism.tolerance_reporting import compare_completion,dist


def test_comparison_counts_trajectory_fractions_not_repeats_as_new_trajectories():
    a=dict(method='trac',trajectories=2,repeats=3,completion_repeats_per_uid={'x':0,'y':3},
        mean_completion_count=1.,mean_cumulative_ms_per_sweep=20.)
    b=dict(method='dls',trajectories=2,repeats=1,completion_repeats_per_uid={'x':1,'y':0},
        mean_completion_count=1.,mean_cumulative_ms_per_sweep=10.)
    c=compare_completion(a,b,'comparison')
    assert c['stable_recovered_uids']==['x'] and c['stable_lost_uids']==['y']
    assert c['mean_completion_count_difference']==0 and c['cumulative_latency_ratio']==.5


def test_unknown_native_fev_is_not_zero_and_precision_quantiles_include_max():
    assert dist([None,None])=={'n':0}
    d=dist([1,2,3],1000)
    assert d['n']==3 and d['p50']==.002 and d['max']==.003
