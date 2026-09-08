"""Postprocessing checks: do not run or change any solver."""
import pytest
from confik.correction_reserve.mechanism_contrasts import compare_summaries, compare_probes


def test_unavailable_site_is_not_dropped():
    base = dict(site_id='a', robot='panda', uid='u', family='smooth', current_accepted=False,
                predicted_gamma=None, probe_success=None, all_directions_radius=None)
    rows = [dict(base, method='two_step_predictive'), dict(base, method='cr_ik')]
    result = compare_summaries(rows)
    assert len(result) == 1 and result[0]['probe_success_difference'] is None


def test_direction_contrast_requires_identical_input_except_current_q():
    row = dict(amplitude=1., direction_id=0, repeat=0, common_previous_q=[0],
               current_position=[0, 0, 0], current_rotation=[1], probe_position=[1, 0, 0],
               probe_rotation=[1], dt=.02, direction=[-1, 0, 0, 0, 0, 0],
               current_q=[.01], continuation={'accepted': False})
    cr = dict(row, current_q=[.02], continuation={'accepted': True})
    result = compare_probes([row], [cr])
    assert result[0]['predictive_accepted'] == 0 and result[0]['crik_accepted'] == 1
    with pytest.raises(AssertionError):
        compare_probes([row], [dict(cr, common_previous_q=[.01])])
