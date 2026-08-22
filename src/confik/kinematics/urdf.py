from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..geometry import axis_angle_matrix, rpy_matrix, transform
from ..types import FloatArray, Pose, RobotLimits
from .base import KinematicsModel


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    origin: FloatArray
    axis: FloatArray
    lower: float
    upper: float
    velocity: float

    @property
    def active(self) -> bool:
        return self.kind in {"revolute", "continuous", "prismatic"}


def _vector(element: ET.Element | None, attribute: str, default: str) -> FloatArray:
    value = default if element is None else element.attrib.get(attribute, default)
    return np.fromstring(value, sep=" ", dtype=np.float64)


class URDFKinematics(KinematicsModel):
    def __init__(self, name: str, chain: list[JointSpec], base_link: str, end_link: str):
        self.name = name
        self.chain = tuple(chain)
        self.base_link = base_link
        self.end_link = end_link
        active = [joint for joint in chain if joint.active]
        self.continuous_mask = np.array([joint.kind == "continuous" for joint in active], dtype=bool)
        self.joint_names = tuple(joint.name for joint in active)
        self.limits = RobotLimits(
            lower=np.array([joint.lower for joint in active], dtype=np.float64),
            upper=np.array([joint.upper for joint in active], dtype=np.float64),
            velocity=np.array([joint.velocity for joint in active], dtype=np.float64),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        base_link: str | None = None,
        end_link: str | None = None,
    ) -> "URDFKinematics":
        root = ET.parse(path).getroot()
        joints: list[JointSpec] = []
        for element in root.findall("joint"):
            kind = element.attrib.get("type")
            parent_element = element.find("parent")
            child_element = element.find("child")
            if kind is None or parent_element is None or child_element is None:
                continue
            origin_element = element.find("origin")
            xyz = _vector(origin_element, "xyz", "0 0 0")
            rpy = _vector(origin_element, "rpy", "0 0 0")
            origin = transform(rpy_matrix(rpy), xyz)
            axis = _vector(element.find("axis"), "xyz", "0 0 1")
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
            limit = element.find("limit")
            if kind == "continuous":
                lower, upper = -np.pi, np.pi
            elif kind in {"revolute", "prismatic"}:
                if limit is None:
                    raise ValueError(f"active joint {element.attrib['name']} has no limit")
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            else:
                lower, upper = -1.0, 1.0
            velocity = float(limit.attrib.get("velocity", "1.0")) if limit is not None else 1.0
            joints.append(
                JointSpec(
                    name=element.attrib["name"],
                    kind=kind,
                    parent=parent_element.attrib["link"],
                    child=child_element.attrib["link"],
                    origin=origin,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                    velocity=max(velocity, 1e-6),
                )
            )

        if not joints:
            raise ValueError(f"no kinematic joints found in {path}")
        by_child = {joint.child: joint for joint in joints}
        parents = {joint.parent for joint in joints}
        children = {joint.child for joint in joints}
        roots = sorted(parents - children)
        if base_link is None:
            if len(roots) != 1:
                raise ValueError(f"could not infer a unique base link; candidates: {roots}")
            base_link = roots[0]

        def path_to(candidate: str) -> list[JointSpec] | None:
            path_joints: list[JointSpec] = []
            link = candidate
            seen: set[str] = set()
            while link != base_link:
                if link in seen or link not in by_child:
                    return None
                seen.add(link)
                joint = by_child[link]
                path_joints.append(joint)
                link = joint.parent
            return list(reversed(path_joints))

        if end_link is None:
            leaves = sorted(children - parents)
            ranked: list[tuple[int, int, str, list[JointSpec]]] = []
            for leaf in leaves:
                candidate_path = path_to(leaf)
                if candidate_path is not None:
                    ranked.append(
                        (sum(joint.active for joint in candidate_path), len(candidate_path), leaf, candidate_path)
                    )
            if not ranked:
                raise ValueError("could not infer an end-effector chain")
            _, _, end_link, chain = max(ranked, key=lambda item: (item[0], item[1], item[2]))
        else:
            chain = path_to(end_link)
            if chain is None:
                raise ValueError(f"no chain from {base_link} to {end_link}")
        return cls(root.attrib.get("name", Path(path).stem), chain, base_link, end_link)

    def _evaluate(self, q: FloatArray) -> tuple[Pose, list[tuple[JointSpec, FloatArray, FloatArray]]]:
        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != (self.nq,):
            raise ValueError(f"expected {self.nq} joints, got {q_array.shape}")
        world = np.eye(4, dtype=np.float64)
        active_frames: list[tuple[JointSpec, FloatArray, FloatArray]] = []
        q_index = 0
        for joint in self.chain:
            world = world @ joint.origin
            if joint.active:
                axis_world = world[:3, :3] @ joint.axis
                point_world = world[:3, 3].copy()
                active_frames.append((joint, point_world, axis_world))
                value = float(q_array[q_index])
                q_index += 1
                if joint.kind in {"revolute", "continuous"}:
                    world = world @ transform(axis_angle_matrix(joint.axis, value))
                elif joint.kind == "prismatic":
                    world = world @ transform(translation=joint.axis * value)
        pose = Pose(world[:3, 3].copy(), world[:3, :3].copy())
        return pose, active_frames

    def forward(self, q: FloatArray) -> Pose:
        return self._evaluate(q)[0]

    def jacobian(self, q: FloatArray) -> FloatArray:
        pose, frames = self._evaluate(q)
        jacobian = np.zeros((6, self.nq), dtype=np.float64)
        for index, (joint, point, axis) in enumerate(frames):
            if joint.kind in {"revolute", "continuous"}:
                jacobian[:3, index] = np.cross(axis, pose.position - point)
                jacobian[3:, index] = axis
            elif joint.kind == "prismatic":
                jacobian[:3, index] = axis
        return jacobian

    def difference(self, q: FloatArray, reference: FloatArray) -> FloatArray:
        difference = np.asarray(q, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
        difference = difference.copy()
        difference[self.continuous_mask] = (
            difference[self.continuous_mask] + np.pi
        ) % (2.0 * np.pi) - np.pi
        return difference
