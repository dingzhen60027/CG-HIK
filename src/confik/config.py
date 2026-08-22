from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml

from .kinematics.urdf import URDFKinematics


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded
    config_dir = Path(config["_config_path"]).parent
    return (config_dir / expanded).resolve()


def load_robot(config: dict[str, Any], robot_name: str) -> URDFKinematics:
    robots = config.get("robots", {})
    if robot_name not in robots:
        raise KeyError(f"robot {robot_name!r} is not defined; available: {sorted(robots)}")
    robot = robots[robot_name]
    path = resolve_path(config, robot["urdf"])
    if not path.exists():
        raise FileNotFoundError(f"URDF does not exist: {path}")
    return URDFKinematics.from_file(
        path,
        base_link=robot.get("base_link"),
        end_link=robot.get("end_link"),
    )

