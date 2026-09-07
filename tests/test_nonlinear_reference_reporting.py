from confik.continuation_mechanism.nonlinear_reference_reporting import best_record, distribution


def test_small_rho_on_infeasible_pose_is_not_a_reference_certificate():
    bad = dict(validation=dict(pose_and_range_feasible=False, actual_rho=.01))
    good = dict(validation=dict(pose_and_range_feasible=True, actual_rho=.8))
    assert best_record([bad]) is None
    assert best_record([bad, good]) is good


def test_actual_certificate_does_not_require_optimizer_success():
    valid = dict(optimizer=dict(success=False),
                 validation=dict(pose_and_range_feasible=True, actual_rho=.3))
    assert best_record([valid]) is valid
    assert distribution([1., 2., 3.])['total'] == 6.
