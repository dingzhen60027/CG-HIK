"""Bounded adapter smoke/unit checks; not trajectory-comparison evidence."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pink")
pytest.importorskip("pinocchio")
pytest.importorskip("qpsolvers")

from confik.config import load_config, load_robot, resolve_path
from confik.correction_reserve.pink_adapter import PinkAdapter, PINK_SETTINGS
from confik.task_contract_alignment.contract import verifier_for
from confik.types import IKQuery


def setup(robot):
    source = load_config("configs/paper_v2.yaml")
    kin = load_robot(source, robot)
    verifier = verifier_for(kin, source)
    adapter = PinkAdapter(kin, verifier, source, resolve_path(source, source["robots"][robot]["urdf"]))
    q = (kin.limits.lower + kin.limits.upper) / 2.0
    # A fixed nonzero wrist configuration avoids using an exactly singular UR
    # posture as the only smoke input; it is not selected by solver outcomes.
    q = q + np.linspace(-0.08, 0.08, kin.nq)
    return kin, verifier, adapter, q


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_exact_fk_joint_order_and_public_verifier(robot):
    kin, verifier, adapter, q = setup(robot)
    metadata = adapter.metadata()
    assert metadata["model_agreement"]["joint_order_equal"]
    for key in ["fk_max_position_component_error_m", "fk_max_rotation_component_error",
                "jacobian_max_component_error"]:
        assert metadata["model_agreement"][key] < 1e-11
    assert metadata["model_agreement"]["public_backend"] == "URDFKinematics"
    adapter.reset(q)
    target = kin.forward(q)
    result = adapter.solve(target.position, target.rotation, q)
    assert result["internal_ok"] and result["accepted"]
    assert verifier.check(result["q"], IKQuery(target, q, 0.02)).accepted
    assert result["total_latency_ns"] == sum(result["phase_times_ns"].values()) + result["accounting_remainder_ns"]
    assert result["accounting_remainder_ns"] >= 0
    assert result["public_verifier_calls"] == 1
    json.dumps(result, allow_nan=False)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_current_target_only_actual_previous_and_one_qp(robot):
    kin, verifier, adapter, q = setup(robot)
    target = kin.forward(q + 0.001)
    result = adapter.solve(target.position, target.rotation, q, last_target=object())
    assert result["accepted"]
    assert result["seed_q"] == q.tolist()
    assert np.linalg.norm(np.asarray(result["q"]) - q) > 0
    assert result["velocity_utilization"] <= 1.0
    # A stored state has no authority to replace the caller's accepted q.
    adapter.reset(q + 0.1)
    repeated = adapter.solve(target.position, target.rotation, q, last_target=None)
    assert np.allclose(repeated["q"], result["q"], rtol=0.0, atol=1e-10)
    assert adapter.settings["differential_steps_per_query"] == 1


@pytest.mark.parametrize("robot", ["panda", "ur5e"])
def test_representable_interval_includes_original_velocity_allowance(robot):
    kin, verifier, adapter, q = setup(robot)
    q[0] = kin.limits.upper[0] - 1e-10
    target = kin.forward(q)
    result = adapter.solve(target.position, target.rotation, q)
    lower, upper = np.asarray(result["lower"]), np.asarray(result["upper"])
    allowed = kin.limits.velocity * 0.02 + verifier.config.velocity_tolerance
    assert np.all(lower >= kin.limits.lower) and np.all(upper <= kin.limits.upper)
    assert np.all(np.abs(lower - q) <= allowed)
    assert np.all(np.abs(upper - q) <= allowed)
    assert np.allclose(adapter.model.velocityLimit,
                       kin.limits.velocity + verifier.config.velocity_tolerance / 0.02)
    if result["accepted"]:
        assert verifier.check(result["q"], IKQuery(target, q, 0.02)).accepted


def test_native_status_not_task_acceptance_and_rejected_state_not_committed(monkeypatch):
    kin, verifier, adapter, q = setup("panda")
    target = kin.forward(q)
    adapter.reset(q)
    info = SimpleNamespace(status="maximum iterations reached", status_val=7, iter=2000,
                           prim_res=0.0, dual_res=1.0, obj_val=0.0)
    fake = SimpleNamespace(found=False, x=np.zeros(kin.nq), extras={"info": info})
    monkeypatch.setattr(adapter.qpsolvers, "solve_problem", lambda *a, **kw: fake)
    result = adapter.solve(target.position, target.rotation, q)
    assert not result["internal_ok"] and result["accepted"]
    unreachable_position = target.position + np.array([10.0, 0.0, 0.0])
    result = adapter.solve(unreachable_position, target.rotation, q)
    assert not result["accepted"]
    assert "position_tolerance" in result["verification_reasons"]
    assert np.array_equal(adapter.previous_accepted_q, q)


def test_roundoff_box_reconstruction_is_bounded_and_retains_raw(monkeypatch):
    kin, verifier, adapter, q = setup("panda")
    dt = 0.02
    desired = q.copy()
    desired[0] += kin.limits.velocity[0] * dt + verifier.config.velocity_tolerance
    target = kin.forward(desired)
    info = SimpleNamespace(status="solved", status_val=1, iter=25,
                           prim_res=1e-12, dual_res=0.0, obj_val=0.0)
    delta = desired - q
    delta[0] += 1e-12
    fake = SimpleNamespace(found=True, x=delta, extras={"info": info})
    monkeypatch.setattr(adapter.qpsolvers, "solve_problem", lambda *a, **kw: fake)
    result = adapter.solve(target.position, target.rotation, q)
    assert result["accepted"]
    assert 0 < result["bound_roundoff_clip_rad"] <= PINK_SETTINGS["bound_roundoff_clip_max_rad"]
    assert result["raw_q"] != result["q"]
    assert not result["raw_verifier_accepted"]
    assert "velocity_limit" in result["raw_verification_reasons"]
    assert result["public_verifier_calls"] == 2
    fake.x[0] += 0.001
    result = adapter.solve(target.position, target.rotation, q)
    assert not result["accepted"]
    assert result["bound_roundoff_clip_rad"] == 0.0
    assert "velocity_limit" in result["verification_reasons"]
