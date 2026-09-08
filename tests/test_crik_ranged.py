"""Unit checks and opt-in bounded official RangedIK smoke checks (no dataset run)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from confik.config import load_config, load_robot, resolve_path
from confik.correction_reserve.ranged_adapter import RangedAdapter, range_mapping
import confik.correction_reserve.ranged_adapter as ranged_module
from confik.solvers.verifier import SolutionVerifier, VerifierConfig
from confik.types import IKQuery, Pose

ROOT = Path(__file__).resolve().parents[1]


def test_component_mapping_is_inscribed_and_modes_are_explicit():
    verifier = type("Verifier", (), {"config": VerifierConfig()})()
    original = range_mapping(verifier, "upstream")
    adapted = range_mapping(verifier, "positive_range")
    assert original["bounds"] == adapted["bounds"]
    b = np.array(adapted["bounds"])
    assert np.linalg.norm(b[:3]) <= verifier.config.position_tolerance
    assert np.linalg.norm(b[3:]) <= verifier.config.orientation_tolerance
    assert not any(original["ranged_loss_active"])
    assert all(adapted["ranged_loss_active"])
    assert "soft objectives" in adapted["conservativeness"]


def test_positive_range_patch_changes_only_two_cutoff_conditions():
    patch = (ROOT / "src/confik/correction_reserve/native/ranged_positive_range.patch").read_text()
    removed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert removed == ["-        if (bound <= 1e-2) {"] * 2
    assert added == ["+        if (bound <= 0.0) {"] * 2


def test_command_timing_stops_after_history_before_offline_packaging(monkeypatch):
    clock = {"ns": 0}

    def timestamp():
        clock["ns"] += 10
        return clock["ns"]

    def offline_velocity_difference(q, previous):
        clock["ns"] += 10_000
        return q - previous

    class Native:
        def crik_ranged_solve(self, *args):
            args[-2][0] = .001
            args[-1]._obj.solve_ns = 10
            return 0

    adapter = RangedAdapter.__new__(RangedAdapter)
    adapter.kin = SimpleNamespace(nq=1, limits=SimpleNamespace(velocity=np.ones(1)),
        difference=offline_velocity_difference)
    adapter.verifier = SimpleNamespace(config=VerifierConfig(), check=lambda q, query:
        SimpleNamespace(accepted=True, finite_ok=True, position_error=0.,
            orientation_error=0., joint_limit_ok=True, velocity_ok=True, reasons=()))
    adapter._history = np.zeros((4, 1))
    adapter.lib = Native()
    adapter.handle = 1
    adapter.native_bounds = np.zeros(6)
    adapter.range_mode = "positive_range"
    adapter.collision_objective = False
    adapter.time_limit_ms = 18.0
    adapter.max_duration_ns = 18_000_000
    monkeypatch.setattr(ranged_module, "perf_counter_ns", timestamp)
    monkeypatch.setattr(ranged_module, "representable_interior",
        lambda *args: (np.array([-.02]), np.array([.02])))
    result = adapter.solve(np.zeros(3), np.eye(3), np.zeros(1))
    assert result["total_latency_ns"] == 50
    assert result["history_update_latency_ns"] == 10
    assert result["offline_packaging_latency_ns"] >= 10_000
    np.testing.assert_array_equal(adapter._history[0], np.array([.001]))
    adapter.handle = None


def _models():
    source = load_config(ROOT / "configs/paper_v2.yaml")
    for robot in ("panda", "ur5e"):
        kin = load_robot(source, robot)
        verifier = SolutionVerifier(kin, VerifierConfig(**source["verifier"]))
        yield robot, source, kin, verifier, resolve_path(source, source["robots"][robot]["urdf"])


def _groove(x):
    return -np.exp(-(x * x) / (2 * .1 ** 2)) + 10 * x * x


def _ranged(x, bound):
    b = (-1 / np.log(.05)) ** (1 / 20)
    return (-np.exp(-x * x / (2 * (2 * bound) ** 2)) + .01 * x * x
            + 100 * (1 - np.exp(-((x / bound) / b) ** 20)))


@pytest.mark.skipif(os.environ.get("CRIK_RANGED_NATIVE_SMOKE") != "1", reason="opt-in native smoke, not an experiment")
def test_native_two_robot_fk_loss_activation_history_and_public_contract():
    """Four solves per mode/robot; retain records, including any rejected result."""
    records = {"scope": "adapter implementation smoke, not comparative evaluation", "robots": {}}
    for robot, source, kin, verifier, urdf in _models():
        initial = (kin.limits.lower + kin.limits.upper) / 2
        pose = kin.forward(initial)
        pair = {}
        for mode in ("upstream", "positive_range"):
            with RangedAdapter(kin, verifier, source, urdf, robot, range_mode=mode) as adapter:
                adapter.reset(initial)
                assert not adapter.collision_objective
                retained = 6 + kin.nq + 4
                native_collision_count = (kin.nq - 1) * (kin.nq - 2) // 2
                with RangedAdapter(kin, verifier, source, urdf, robot,
                        range_mode=mode, collision_objective=True) as official_objectives:
                    full_weights = official_objectives.active_objective_weights
                    assert len(full_weights) == retained + native_collision_count
                    assert adapter.active_objective_weights == full_weights[:retained]
                    assert full_weights[retained:] == [0.01] * native_collision_count
                b = adapter.native_bounds
                # Actual native objective calls, not a reimplemented solver.
                loss_pos_target = pose.position - pose.rotation @ np.array([.5 * b[0], 0., 0.])
                losses = adapter.native_component_losses(initial, loss_pos_target, pose.rotation)
                expected = _groove(.5 * b[0]) if mode == "upstream" else _ranged(.5 * b[0], b[0])
                assert np.isclose(losses[0], expected, atol=1e-10, rtol=1e-10)
                orientation = pose.rotation @ Rotation.from_rotvec(np.array([-.5 * b[3], 0., 0.])).as_matrix()
                rotation_losses = adapter.native_component_losses(initial, pose.position, orientation)
                expected_rot = _groove(.5 * b[3]) if mode == "upstream" else _ranged(.5 * b[3], b[3])
                assert np.isclose(rotation_losses[3], expected_rot, atol=1e-10, rtol=1e-10)
                calls = []
                previous = initial.copy()
                for offset in (0.0, 0.00025, 0.0005):
                    desired = kin.forward(initial + offset * np.ones(kin.nq))
                    result = adapter.solve(desired.position, desired.rotation, previous)
                    assert result["native_return_code"] != -999
                    assert result["history_q"][0] == previous.tolist()
                    assert result["total_latency_ns"] >= result["solve_latency_ns"]
                    assert result["history_update_latency_ns"] >= 0
                    assert result["offline_packaging_latency_ns"] >= 0
                    assert not result["collision_objective"]
                    if result["q"] is not None:
                        q = np.array(result["q"])
                        check = verifier.check(q, IKQuery(desired, previous, .02))
                        assert result["accepted"] == check.accepted
                        assert check.velocity_ok and check.joint_limit_ok
                        if check.accepted:
                            previous = q
                    calls.append(result)
                # An unreachable target is rejected; native history must hold the
                # actual command instead of adopting the unaccepted raw result.
                bad_position = pose.position + np.array([10., 0., 0.])
                failed = adapter.solve(bad_position, pose.rotation, previous)
                assert not failed["accepted"]
                np.testing.assert_array_equal(adapter._history[0], previous)
                calls.append(failed)
                pair[mode] = {"metadata": adapter.metadata, "calls": calls,
                    "native_position_half_bound_loss": losses[0],
                    "native_orientation_half_bound_loss": rotation_losses[3],
                    "expected_position_half_bound_loss": float(expected),
                    "expected_orientation_half_bound_loss": float(expected_rot)}
        assert pair["upstream"]["native_position_half_bound_loss"] != pair["positive_range"]["native_position_half_bound_loss"]
        records["robots"][robot] = pair
    # The initial with-collision smoke remains intact in smoke.json.
    destination = ROOT / "tmp/crik_dependencies/ranged_adapter_build/smoke_contract_only.json"
    destination.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
