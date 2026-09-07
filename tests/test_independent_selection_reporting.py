from confik.continuation_mechanism.independent_selection_reporting import (
    classify_candidates, hindsight_id, stable_classical_miss)


def pool(counts):
    return [dict(candidate_id=f'c{i}',completed_repeats=n,mean_prefix=10+i) for i,n in enumerate(counts)]


def test_posthoc_groups_and_oracle_are_not_selection_rules():
    assert classify_candidates(pool([5,5])) == 'all_success'
    assert classify_candidates(pool([0,0])) == 'all_failure'
    assert classify_candidates(pool([4,4])) == 'mixed'
    assert classify_candidates(pool([0,5])) == 'mixed'
    assert hindsight_id(pool([0,5,5])) == 'c2'


def test_stable_classical_miss_requires_full_repeated_contrast():
    chosen = dict(max_scaled_sigma_min='c0',max_scaled_manipulability='c1')
    assert stable_classical_miss(pool([0,0,5]),chosen)
    assert not stable_classical_miss(pool([0,1,5]),chosen)
    assert not stable_classical_miss(pool([0,0,4]),chosen)
