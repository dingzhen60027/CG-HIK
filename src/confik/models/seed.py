from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..geometry import pose_distance, rotation_6d
from ..kinematics.base import KinematicsModel
from ..kinematics.urdf import URDFKinematics
from ..types import CandidateSet, FloatArray, IKQuery

if TYPE_CHECKING:
    from ..data.datasets import TransitionDataset


@dataclass(frozen=True)
class SeedTrainingConfig:
    members: int = 5
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    max_delta: float = 0.25
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    joint_loss_weight: float = 1.0
    fk_position_weight: float = 0.1
    fk_orientation_weight: float = 0.1
    seed: int = 17
    use_history: bool = True


class SeedMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...], max_delta: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.append(nn.Linear(previous, output_size))
        self.network = nn.Sequential(*layers)
        self.max_delta = max_delta

    def forward(self, inputs: Tensor) -> Tensor:
        return self.max_delta * torch.tanh(self.network(inputs))


class TorchURDFFK(nn.Module):
    """Differentiable batched FK for the serial subset parsed by URDFKinematics."""

    def __init__(self, kinematics: URDFKinematics):
        super().__init__()
        self.kinds = tuple(joint.kind for joint in kinematics.chain)
        self.active = tuple(joint.active for joint in kinematics.chain)
        self.register_buffer(
            "origins",
            torch.as_tensor(np.stack([joint.origin for joint in kinematics.chain]), dtype=torch.float32),
        )
        self.register_buffer(
            "axes",
            torch.as_tensor(np.stack([joint.axis for joint in kinematics.chain]), dtype=torch.float32),
        )

    @staticmethod
    def _skew(axis: Tensor) -> Tensor:
        x, y, z = axis.unbind()
        zero = torch.zeros_like(x)
        return torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero],
        ).reshape(3, 3)

    def forward(self, q: Tensor) -> tuple[Tensor, Tensor]:
        batch = q.shape[0]
        world = torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, 4, 4).clone()
        q_index = 0
        for index, (kind, active) in enumerate(zip(self.kinds, self.active, strict=True)):
            origin = self.origins[index].to(dtype=q.dtype, device=q.device)
            world = world @ origin
            if not active:
                continue
            value = q[:, q_index]
            q_index += 1
            motion = torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, 4, 4).clone()
            axis = self.axes[index].to(dtype=q.dtype, device=q.device)
            if kind in {"revolute", "continuous"}:
                cross = self._skew(axis)
                rotation = (
                    torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, 3, 3)
                    + torch.sin(value)[:, None, None] * cross
                    + (1.0 - torch.cos(value))[:, None, None] * (cross @ cross)
                )
                motion[:, :3, :3] = rotation
            elif kind == "prismatic":
                motion[:, :3, 3] = value[:, None] * axis
            world = world @ motion
        return world[:, :3, 3], world[:, :3, :3]


def encode_seed_inputs(
    kinematics: KinematicsModel,
    previous_q: FloatArray,
    target_positions: FloatArray,
    target_rotations: FloatArray,
    *,
    use_history: bool = True,
) -> FloatArray:
    previous = np.asarray(previous_q, dtype=np.float64)
    positions = np.asarray(target_positions, dtype=np.float64)
    rotations = np.asarray(target_rotations, dtype=np.float64)
    if previous.ndim == 1:
        previous = previous[None, :]
        positions = positions[None, :]
        rotations = rotations[None, :, :]
    normalized = np.stack([kinematics.normalize(q) for q in previous])
    if not use_history:
        normalized = np.zeros_like(normalized)
    rotation_features = np.stack([rotation_6d(rotation) for rotation in rotations])
    return np.concatenate([normalized, positions, rotation_features], axis=1)


class TorchSeedEnsemble:
    def __init__(
        self,
        kinematics: KinematicsModel,
        config: SeedTrainingConfig | None = None,
        *,
        device: str | None = None,
    ):
        self.kinematics = kinematics
        self.config = config or SeedTrainingConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        input_size = kinematics.nq + 3 + 6
        members: list[SeedMLP] = []
        for member_index in range(self.config.members):
            torch.manual_seed(self.config.seed + member_index)
            members.append(
                SeedMLP(
                    input_size,
                    kinematics.nq,
                    self.config.hidden_sizes,
                    self.config.max_delta,
                )
            )
        self.members = nn.ModuleList(
            members
        ).to(self.device)
        self.fk = TorchURDFFK(kinematics).to(self.device) if isinstance(kinematics, URDFKinematics) else None
        self.fitted = False
        self.training_history: list[dict[str, float]] = []

    def fit(self, dataset: "TransitionDataset") -> "TorchSeedEnsemble":
        inputs = encode_seed_inputs(
            self.kinematics,
            dataset.previous_q,
            dataset.target_position,
            dataset.target_rotation,
            use_history=self.config.use_history,
        ).astype(np.float32)
        target_delta = np.stack(
            [self.kinematics.difference(q, previous) for q, previous in zip(dataset.target_q, dataset.previous_q, strict=True)]
        ).astype(np.float32)
        target_q = dataset.target_q.astype(np.float32)
        target_position = dataset.target_position.astype(np.float32)
        target_rotation = dataset.target_rotation.astype(np.float32)
        sample_count = len(dataset)
        config = self.config
        self.training_history = []

        for member_index, model in enumerate(self.members):
            generator = np.random.default_rng(config.seed + member_index)
            bootstrap = generator.integers(0, sample_count, size=sample_count)
            tensors = TensorDataset(
                torch.from_numpy(inputs[bootstrap]),
                torch.from_numpy(target_delta[bootstrap]),
                torch.from_numpy(target_q[bootstrap]),
                torch.from_numpy(target_position[bootstrap]),
                torch.from_numpy(target_rotation[bootstrap]),
            )
            loader_generator = torch.Generator().manual_seed(config.seed + member_index)
            loader = DataLoader(
                tensors,
                batch_size=config.batch_size,
                shuffle=True,
                generator=loader_generator,
            )
            torch.manual_seed(config.seed + member_index)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            model.train()
            last_loss = 0.0
            for _ in range(config.epochs):
                total_loss = 0.0
                batches = 0
                for batch_inputs, batch_delta, batch_q, batch_position, batch_rotation in loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_delta = batch_delta.to(self.device)
                    batch_q = batch_q.to(self.device)
                    batch_position = batch_position.to(self.device)
                    batch_rotation = batch_rotation.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted_delta = model(batch_inputs)
                    joint_loss = F.huber_loss(predicted_delta, batch_delta)
                    loss = config.joint_loss_weight * joint_loss
                    if self.fk is not None and (
                        config.fk_position_weight > 0 or config.fk_orientation_weight > 0
                    ):
                        predicted_q = batch_q - batch_delta + predicted_delta
                        predicted_position, predicted_rotation = self.fk(predicted_q)
                        position_loss = torch.mean(torch.sum((predicted_position - batch_position) ** 2, dim=1))
                        relative = predicted_rotation.transpose(1, 2) @ batch_rotation
                        cosine = ((relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0).clamp(
                            -1.0 + 1e-6, 1.0 - 1e-6
                        )
                        orientation_loss = torch.mean(torch.acos(cosine) ** 2)
                        loss = (
                            loss
                            + config.fk_position_weight * position_loss
                            + config.fk_orientation_weight * orientation_loss
                        )
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1
                last_loss = total_loss / max(batches, 1)
            self.training_history.append({"member": float(member_index), "final_loss": last_loss})
            model.eval()
        self.fitted = True
        return self

    def predict_deltas(self, query: IKQuery) -> FloatArray:
        if not self.fitted:
            raise RuntimeError("seed ensemble must be fitted or loaded before prediction")
        features = encode_seed_inputs(
            self.kinematics,
            query.previous_q,
            query.target.position,
            query.target.rotation,
            use_history=self.config.use_history,
        ).astype(np.float32)
        tensor = torch.from_numpy(features).to(self.device)
        predictions: list[FloatArray] = []
        with torch.inference_mode():
            for model in self.members:
                predictions.append(model(tensor).cpu().numpy()[0].astype(np.float64))
        return np.stack(predictions)

    def predict_deltas_batch(
        self,
        previous_q: FloatArray,
        target_positions: FloatArray,
        target_rotations: FloatArray,
        *,
        batch_size: int = 8192,
    ) -> FloatArray:
        """Return predictions with shape ``(samples, members, joints)``."""
        if not self.fitted:
            raise RuntimeError("seed ensemble must be fitted or loaded before prediction")
        features = encode_seed_inputs(
            self.kinematics,
            previous_q,
            target_positions,
            target_rotations,
            use_history=self.config.use_history,
        ).astype(np.float32)
        per_member: list[np.ndarray] = []
        with torch.inference_mode():
            for model in self.members:
                chunks: list[np.ndarray] = []
                for start in range(0, len(features), batch_size):
                    tensor = torch.from_numpy(features[start : start + batch_size]).to(self.device)
                    chunks.append(model(tensor).cpu().numpy().astype(np.float64))
                per_member.append(np.concatenate(chunks, axis=0))
        return np.stack(per_member, axis=1)

    def candidates(self, query: IKQuery) -> CandidateSet:
        deltas = self.predict_deltas(query)
        joints = np.stack([self.kinematics.clip(query.previous_q + delta) for delta in deltas])
        variance = np.var(deltas, axis=0)
        scores = np.array([self._score(query, q) for q in joints], dtype=np.float64)
        order = np.argsort(scores)
        unique: list[int] = []
        for index in order:
            if all(
                np.max(np.abs(self.kinematics.difference(joints[index], joints[kept]))) >= 0.05
                for kept in unique
            ):
                unique.append(int(index))
        if not unique:
            unique = [int(order[0])]
        return CandidateSet(
            joints=joints[unique],
            scores=scores[unique],
            uncertainty_mean=float(np.mean(variance)),
            uncertainty_max=float(np.max(variance)),
            source=[f"learned:{index}" for index in unique],
        )

    def _score(self, query: IKQuery, q: FloatArray) -> float:
        position_error, orientation_error = pose_distance(query.target, self.kinematics.forward(q))
        smoothness = np.linalg.norm(self.kinematics.difference(q, query.previous_q)) / np.sqrt(
            self.kinematics.nq
        )
        margin = float(np.min(self.kinematics.joint_margin(q)))
        limit_penalty = max(0.0, 0.1 - margin) / 0.1
        return float(position_error / 0.01 + orientation_error / 0.1 + 0.25 * smoothness + 0.25 * limit_penalty)

    def save(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "state_dict": self.members.state_dict(),
            "fitted": self.fitted,
            "training_history": self.training_history,
            "joint_names": self.kinematics.joint_names,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        kinematics: KinematicsModel,
        *,
        device: str | None = None,
    ) -> "TorchSeedEnsemble":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload["joint_names"]) != kinematics.joint_names:
            raise ValueError("saved ensemble joint names do not match the robot")
        config_dict = dict(payload["config"])
        config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
        ensemble = cls(kinematics, SeedTrainingConfig(**config_dict), device=device)
        ensemble.members.load_state_dict(payload["state_dict"])
        ensemble.members.eval()
        ensemble.fitted = bool(payload["fitted"])
        ensemble.training_history = list(payload.get("training_history", []))
        return ensemble


class PreviousStateCandidates:
    """Deterministic candidate provider used by numerical baselines and smoke tests."""

    def candidates(self, query: IKQuery) -> CandidateSet:
        return CandidateSet(
            joints=query.previous_q[None, :],
            scores=np.zeros(1),
            uncertainty_mean=0.0,
            uncertainty_max=0.0,
            source=["previous"],
        )
