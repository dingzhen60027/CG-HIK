from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import FloatArray


@dataclass
class TransitionDataset:
    previous_q: FloatArray
    target_q: FloatArray
    target_position: FloatArray
    target_rotation: FloatArray
    trajectory_id: FloatArray

    def __post_init__(self) -> None:
        self.previous_q = np.asarray(self.previous_q, dtype=np.float64)
        self.target_q = np.asarray(self.target_q, dtype=np.float64)
        self.target_position = np.asarray(self.target_position, dtype=np.float64)
        self.target_rotation = np.asarray(self.target_rotation, dtype=np.float64)
        self.trajectory_id = np.asarray(self.trajectory_id, dtype=np.int64)
        count = self.previous_q.shape[0]
        if self.target_q.shape != self.previous_q.shape:
            raise ValueError("previous and target joints must have equal shapes")
        if self.target_position.shape != (count, 3):
            raise ValueError("target positions must have shape (samples, 3)")
        if self.target_rotation.shape != (count, 3, 3):
            raise ValueError("target rotations must have shape (samples, 3, 3)")
        if self.trajectory_id.shape != (count,):
            raise ValueError("trajectory IDs must have one entry per sample")

    def __len__(self) -> int:
        return self.previous_q.shape[0]

    def subset(self, indices: FloatArray) -> "TransitionDataset":
        idx = np.asarray(indices)
        return TransitionDataset(
            self.previous_q[idx],
            self.target_q[idx],
            self.target_position[idx],
            self.target_rotation[idx],
            self.trajectory_id[idx],
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            previous_q=self.previous_q,
            target_q=self.target_q,
            target_position=self.target_position,
            target_rotation=self.target_rotation,
            trajectory_id=self.trajectory_id,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionDataset":
        with np.load(path) as data:
            return cls(**{name: data[name] for name in data.files})


@dataclass
class RiskDataset:
    features: FloatArray
    labels: FloatArray
    iterations: FloatArray
    converged: FloatArray

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.iterations = np.asarray(self.iterations, dtype=np.int64)
        self.converged = np.asarray(self.converged, dtype=bool)
        count = self.features.shape[0]
        if not all(array.shape == (count,) for array in (self.labels, self.iterations, self.converged)):
            raise ValueError("risk arrays must have one row per feature vector")

    def __len__(self) -> int:
        return self.features.shape[0]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            iterations=self.iterations,
            converged=self.converged,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RiskDataset":
        with np.load(path) as data:
            return cls(**{name: data[name] for name in data.files})


@dataclass
class QueryDataset:
    previous_q: FloatArray
    target_position: FloatArray
    target_rotation: FloatArray
    reference_q: FloatArray
    category: np.ndarray
    expected_reachable: np.ndarray
    continuity_feasible: np.ndarray
    trajectory_id: np.ndarray
    time_index: np.ndarray

    def __post_init__(self) -> None:
        self.previous_q = np.asarray(self.previous_q, dtype=np.float64)
        self.target_position = np.asarray(self.target_position, dtype=np.float64)
        self.target_rotation = np.asarray(self.target_rotation, dtype=np.float64)
        self.reference_q = np.asarray(self.reference_q, dtype=np.float64)
        self.category = np.asarray(self.category, dtype=str)
        self.expected_reachable = np.asarray(self.expected_reachable, dtype=bool)
        self.continuity_feasible = np.asarray(self.continuity_feasible, dtype=bool)
        self.trajectory_id = np.asarray(self.trajectory_id, dtype=np.int64)
        self.time_index = np.asarray(self.time_index, dtype=np.int64)
        count = self.previous_q.shape[0]
        expected_shapes = {
            "target_position": (count, 3),
            "target_rotation": (count, 3, 3),
            "reference_q": self.previous_q.shape,
            "category": (count,),
            "expected_reachable": (count,),
            "continuity_feasible": (count,),
            "trajectory_id": (count,),
            "time_index": (count,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")

    def __len__(self) -> int:
        return self.previous_q.shape[0]

    @classmethod
    def from_transitions(cls, dataset: TransitionDataset, category: str = "id") -> "QueryDataset":
        count = len(dataset)
        return cls(
            previous_q=dataset.previous_q,
            target_position=dataset.target_position,
            target_rotation=dataset.target_rotation,
            reference_q=dataset.target_q,
            category=np.full(count, category),
            expected_reachable=np.ones(count, dtype=bool),
            continuity_feasible=np.ones(count, dtype=bool),
            trajectory_id=dataset.trajectory_id,
            time_index=np.arange(count, dtype=np.int64),
        )

    @classmethod
    def concatenate(cls, datasets: list["QueryDataset"]) -> "QueryDataset":
        if not datasets:
            raise ValueError("at least one query dataset is required")
        fields = cls.__dataclass_fields__.keys()
        return cls(**{name: np.concatenate([getattr(dataset, name) for dataset in datasets]) for name in fields})

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "QueryDataset":
        with np.load(path) as data:
            return cls(**{name: data[name] for name in data.files})
