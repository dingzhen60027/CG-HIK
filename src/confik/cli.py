from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config, load_robot
from .pipeline import (
    evaluate,
    generate_data,
    label_risk,
    run_all,
    run_repetitions,
    train_risk,
    train_seed,
)
from .pipeline_v2 import (
    aggregate_v2,
    evaluate_v2,
    generate_data_v2,
    label_risk_v2,
    run_all_v2,
    run_repetitions_v2,
    train_risk_v2,
    train_seed_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="confik", description="Confidence-gated hybrid IK experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-robot", help="parse a configured robot and print its limits")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--robot", required=True)

    for command in ("generate-data", "train-seed", "label-risk", "train-risk", "evaluate", "run-all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--robot", required=True)
        if command in {"generate-data", "label-risk", "evaluate", "run-all"}:
            child.add_argument("--force", action="store_true")
    repetitions = subparsers.add_parser("run-repetitions")
    repetitions.add_argument("--config", required=True)
    repetitions.add_argument("--robot", required=True)
    repetitions.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    repetitions.add_argument("--force", action="store_true")
    for command in (
        "generate-data-v2",
        "train-seed-v2",
        "label-risk-v2",
        "train-risk-v2",
        "evaluate-v2",
        "run-all-v2",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--robot", required=True)
        if command in {"generate-data-v2", "label-risk-v2", "evaluate-v2", "run-all-v2"}:
            child.add_argument("--force", action="store_true")
    repetitions_v2 = subparsers.add_parser("run-repetitions-v2")
    repetitions_v2.add_argument("--config", required=True)
    repetitions_v2.add_argument("--robot", required=True)
    repetitions_v2.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    repetitions_v2.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate-v2")
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--robots", nargs="+", default=["ur5e", "panda"])
    aggregate.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    config_path = Path(arguments.config)
    if arguments.command == "inspect-robot":
        robot = load_robot(load_config(config_path), arguments.robot)
        payload = {
            "name": robot.name,
            "nq": robot.nq,
            "joint_names": robot.joint_names,
            "lower": robot.limits.lower.tolist(),
            "upper": robot.limits.upper.tolist(),
            "velocity": robot.limits.velocity.tolist(),
            "zero_pose": robot.forward(np.zeros(robot.nq)).matrix.tolist(),
        }
    elif arguments.command == "generate-data":
        payload = generate_data(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "train-seed":
        payload = train_seed(config_path, arguments.robot)
    elif arguments.command == "label-risk":
        payload = label_risk(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "train-risk":
        payload = train_risk(config_path, arguments.robot)
    elif arguments.command == "evaluate":
        payload = evaluate(config_path, arguments.robot, force_test_data=arguments.force)
    elif arguments.command == "run-all":
        payload = run_all(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "run-repetitions":
        payload = run_repetitions(
            config_path,
            arguments.robot,
            arguments.seeds,
            force=arguments.force,
        )
    elif arguments.command == "generate-data-v2":
        payload = generate_data_v2(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "train-seed-v2":
        payload = train_seed_v2(config_path, arguments.robot)
    elif arguments.command == "label-risk-v2":
        payload = label_risk_v2(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "train-risk-v2":
        payload = train_risk_v2(config_path, arguments.robot)
    elif arguments.command == "evaluate-v2":
        payload = evaluate_v2(config_path, arguments.robot, force_test_data=arguments.force)
    elif arguments.command == "run-all-v2":
        payload = run_all_v2(config_path, arguments.robot, force=arguments.force)
    elif arguments.command == "run-repetitions-v2":
        payload = run_repetitions_v2(
            config_path,
            arguments.robot,
            arguments.seeds,
            force=arguments.force,
        )
    elif arguments.command == "aggregate-v2":
        payload = aggregate_v2(config_path, arguments.robots, arguments.seeds)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
