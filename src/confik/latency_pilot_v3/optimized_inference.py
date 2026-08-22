from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..geometry import pose_distance
from ..kinematics.base import KinematicsModel
from ..models.risk import RiskModel
from ..models.seed import TorchSeedEnsemble, encode_seed_inputs
from ..types import CalibratedRisk, CandidateSet, FloatArray, IKQuery, Pose


REQUIRED_TIMING_KEYS = (
    "feature_preparation_ns",
    "numpy_torch_conversion_ns",
    "learned_seed_inference_ns",
    "uncertainty_risk_inference_ns",
)


def empty_inference_timings() -> dict[str, int]:
    return {key: 0 for key in REQUIRED_TIMING_KEYS}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _BatchedLinear(nn.Module):
    """One linear transform for every ensemble member in one tensor operation."""

    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach().contiguous())
        self.register_buffer("bias", bias.detach().contiguous())

    def forward(self, values: Tensor) -> Tensor:
        return torch.matmul(values, self.weight.transpose(1, 2)) + self.bias[:, None, :]


class VectorizedSeedMLP(nn.Module):
    """Frozen ensemble represented as stacked member weights.

    The member dimension is evaluated by batched matrix multiplication.  The
    only Python/module loop is over network depth; there is no per-member loop
    on the inference path.
    """

    def __init__(self, layers: list[_BatchedLinear], members: int, max_delta: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.members = int(members)
        self.max_delta = float(max_delta)

    @classmethod
    def from_ensemble(
        cls,
        ensemble: TorchSeedEnsemble,
        *,
        device: str | torch.device = "cpu",
    ) -> "VectorizedSeedMLP":
        if not ensemble.fitted:
            raise RuntimeError("seed ensemble must be fitted before vectorization")
        linear_layers: list[list[nn.Linear]] = []
        for member in ensemble.members:
            linear_layers.append([layer for layer in member.network if isinstance(layer, nn.Linear)])
        if not linear_layers or not linear_layers[0]:
            raise ValueError("seed ensemble contains no linear layers")
        depth = len(linear_layers[0])
        if any(len(member_layers) != depth for member_layers in linear_layers):
            raise ValueError("all seed ensemble members must have the same depth")
        target_device = torch.device(device)
        stacked: list[_BatchedLinear] = []
        # This is a one-time export step, not an inference-time member loop.
        for layer_index in range(depth):
            weight = torch.stack(
                [member_layers[layer_index].weight.detach().to(target_device) for member_layers in linear_layers]
            )
            bias = torch.stack(
                [member_layers[layer_index].bias.detach().to(target_device) for member_layers in linear_layers]
            )
            stacked.append(_BatchedLinear(weight, bias))
        module = cls(stacked, len(linear_layers), ensemble.config.max_delta).to(target_device)
        module.eval()
        return module

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.unsqueeze(0).expand(self.members, -1, -1)
        final_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index != final_index:
                values = F.silu(values)
        return (self.max_delta * torch.tanh(values)).permute(1, 0, 2)


class ExactSingleCallSeedEnsemble(nn.Module):
    """One-call ensemble graph that retains each frozen member's arithmetic."""

    def __init__(self, ensemble: TorchSeedEnsemble):
        super().__init__()
        self.members = ensemble.members

    def forward(self, inputs: Tensor) -> Tensor:
        # This construction-time loop is unrolled by torch.jit.trace.  The
        # deployed inference graph contains no per-member Python dispatch.
        return torch.stack([member(inputs) for member in self.members], dim=1)


@dataclass
class PreparedCandidates:
    candidates: CandidateSet
    deltas: FloatArray
    timings_ns: dict[str, int]
    best_pose: Pose
    best_margin: float
    best_joint_step: float
    kinematics: KinematicsModel


class SeedEngine(Protocol):
    name: str

    def prepare(self, query: IKQuery) -> PreparedCandidates: ...


class _CandidatePostprocessor:
    def __init__(self, ensemble: TorchSeedEnsemble):
        self.kinematics = ensemble.kinematics
        self.lower = np.asarray(self.kinematics.limits.lower, dtype=np.float64)
        self.upper = np.asarray(self.kinematics.limits.upper, dtype=np.float64)

    def build(
        self,
        query: IKQuery,
        deltas: FloatArray,
        timings: dict[str, int],
    ) -> PreparedCandidates:
        risk_started = perf_counter_ns()
        variance = np.var(deltas, axis=0)
        uncertainty_mean = float(np.mean(variance))
        uncertainty_max = float(np.max(variance))
        timings["uncertainty_risk_inference_ns"] += perf_counter_ns() - risk_started

        feature_started = perf_counter_ns()
        joints = np.clip(query.previous_q[None, :] + deltas, self.lower, self.upper)
        scores = np.empty(len(joints), dtype=np.float64)
        poses: list[Pose] = []
        margins = np.empty(len(joints), dtype=np.float64)
        joint_steps = np.empty(len(joints), dtype=np.float64)
        for index, q in enumerate(joints):
            pose = self.kinematics.forward(q)
            poses.append(pose)
            position_error, orientation_error = pose_distance(query.target, pose)
            joint_step = float(
                np.linalg.norm(self.kinematics.difference(q, query.previous_q))
                / np.sqrt(self.kinematics.nq)
            )
            margin = float(np.min(self.kinematics.joint_margin(q)))
            limit_penalty = max(0.0, 0.1 - margin) / 0.1
            scores[index] = (
                position_error / 0.01
                + orientation_error / 0.1
                + 0.25 * joint_step
                + 0.25 * limit_penalty
            )
            margins[index] = margin
            joint_steps[index] = float(
                np.linalg.norm(self.kinematics.difference(q, query.previous_q))
            )
        order = np.argsort(scores)
        unique: list[int] = []
        for index in order:
            if all(
                np.max(
                    np.abs(
                        self.kinematics.difference(joints[index], joints[kept])
                    )
                )
                >= 0.05
                for kept in unique
            ):
                unique.append(int(index))
        if not unique:
            unique = [int(order[0])]
        best_index = unique[0]
        candidate_set = CandidateSet(
            joints=joints[unique],
            scores=scores[unique],
            uncertainty_mean=uncertainty_mean,
            uncertainty_max=uncertainty_max,
            source=[f"learned:{index}" for index in unique],
        )
        timings["feature_preparation_ns"] += perf_counter_ns() - feature_started
        return PreparedCandidates(
            candidates=candidate_set,
            deltas=deltas,
            timings_ns=timings,
            best_pose=poses[best_index],
            best_margin=float(margins[best_index]),
            best_joint_step=float(joint_steps[best_index]),
            kinematics=self.kinematics,
        )


class EagerSeedEngine:
    """Instrumented version of the original per-member eager call path."""

    name = "eager_original"

    def __init__(self, ensemble: TorchSeedEnsemble):
        self.ensemble = ensemble
        self.ensemble.members.eval()
        self.postprocessor = _CandidatePostprocessor(ensemble)

    def prepare(self, query: IKQuery) -> PreparedCandidates:
        timings = empty_inference_timings()
        started = perf_counter_ns()
        features = encode_seed_inputs(
            self.ensemble.kinematics,
            query.previous_q,
            query.target.position,
            query.target.rotation,
            use_history=self.ensemble.config.use_history,
        ).astype(np.float32)
        timings["feature_preparation_ns"] += perf_counter_ns() - started

        started = perf_counter_ns()
        tensor = torch.from_numpy(features).to(self.ensemble.device)
        # A CUDA copy may be asynchronous.  Synchronize inside the conversion
        # interval so transfer cost is not misattributed to the first member.
        _sync(self.ensemble.device)
        timings["numpy_torch_conversion_ns"] += perf_counter_ns() - started
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for model in self.ensemble.members:
                started = perf_counter_ns()
                output = model(tensor)
                _sync(self.ensemble.device)
                timings["learned_seed_inference_ns"] += perf_counter_ns() - started
                started = perf_counter_ns()
                predictions.append(output.detach().cpu().numpy()[0].astype(np.float64))
                timings["numpy_torch_conversion_ns"] += perf_counter_ns() - started
        deltas = np.stack(predictions)
        return self.postprocessor.build(query, deltas, timings)


class OptimizedSeedEngine:
    """Preallocated float32 input plus one vectorized ensemble forward."""

    def __init__(
        self,
        ensemble: TorchSeedEnsemble,
        module: nn.Module,
        *,
        name: str = "optimized_pytorch",
        device: str | torch.device = "cpu",
    ):
        self.name = name
        self.ensemble = ensemble
        self.kinematics = ensemble.kinematics
        self.device = torch.device(device)
        self.module = module.to(self.device)
        try:
            self.module.eval()
        except NotImplementedError:
            # torch.export's GraphModule is already frozen and intentionally
            # rejects train()/eval() state changes.
            pass
        self.postprocessor = _CandidatePostprocessor(ensemble)
        self.input_size = self.kinematics.nq + 9
        self._input_numpy = np.empty((1, self.input_size), dtype=np.float32, order="C")
        if self.device.type == "cpu":
            self._input_tensor = torch.from_numpy(self._input_numpy)
        else:
            self._input_tensor = torch.empty(
                (1, self.input_size), dtype=torch.float32, device=self.device
            )
        lower = np.asarray(self.kinematics.limits.lower, dtype=np.float64)
        upper = np.asarray(self.kinematics.limits.upper, dtype=np.float64)
        self._center = (lower + upper) / 2.0
        self._half_span = (upper - lower) / 2.0
        self._normalized_work = np.empty(self.kinematics.nq, dtype=np.float64)

    def _fill_features(self, query: IKQuery) -> None:
        row = self._input_numpy[0]
        nq = self.kinematics.nq
        np.subtract(query.previous_q, self._center, out=self._normalized_work)
        np.divide(self._normalized_work, self._half_span, out=self._normalized_work)
        row[:nq] = self._normalized_work
        if not self.ensemble.config.use_history:
            row[:nq] = 0.0
        row[nq : nq + 3] = query.target.position
        row[nq + 3 : nq + 6] = query.target.rotation[:, 0]
        row[nq + 6 : nq + 9] = query.target.rotation[:, 1]

    def prepare(self, query: IKQuery) -> PreparedCandidates:
        timings = empty_inference_timings()
        started = perf_counter_ns()
        self._fill_features(query)
        timings["feature_preparation_ns"] += perf_counter_ns() - started

        started = perf_counter_ns()
        if self.device.type != "cpu":
            self._input_tensor.copy_(torch.from_numpy(self._input_numpy))
        timings["numpy_torch_conversion_ns"] += perf_counter_ns() - started

        with torch.inference_mode():
            started = perf_counter_ns()
            output = self.module(self._input_tensor)
            _sync(self.device)
            timings["learned_seed_inference_ns"] += perf_counter_ns() - started
        started = perf_counter_ns()
        deltas = output.detach().cpu().numpy()[0].astype(np.float64, copy=True)
        timings["numpy_torch_conversion_ns"] += perf_counter_ns() - started
        return self.postprocessor.build(query, deltas, timings)


class RiskEngine(Protocol):
    name: str

    def predict(self, features: FloatArray) -> CalibratedRisk: ...


class EagerRiskEngine:
    name = "sklearn_eager"

    def __init__(self, model: RiskModel):
        self.model = model

    def predict(self, features: FloatArray) -> CalibratedRisk:
        return self.model.predict(features)


class VectorizedHGBRiskModel:
    """Frozen NumPy evaluator for a fitted sklearn HGB + isotonic model.

    Trees are traversed simultaneously along their depth dimension.  This
    removes sklearn's high batch-one dispatch cost while retaining float64 tree
    thresholds, leaf values, class softmax, and fitted isotonic mappings.
    """

    name = "vectorized_frozen_hgb"

    def __init__(self, model: RiskModel):
        from sklearn.ensemble import HistGradientBoostingClassifier

        if model.kind != "gradient_boosting" or not isinstance(
            model.estimator, HistGradientBoostingClassifier
        ):
            raise TypeError("vectorized risk inference requires a fitted HistGradientBoostingClassifier")
        estimator = model.estimator
        if model.calibrators is None:
            raise ValueError("risk model must retain its fitted isotonic calibrators")
        trees = [predictor for iteration in estimator._predictors for predictor in iteration]
        if not trees:
            raise ValueError("risk model contains no fitted trees")
        if any(np.any(tree.nodes["is_categorical"]) for tree in trees):
            raise ValueError("categorical HGB nodes are not supported by the frozen evaluator")
        self.n_classes = int(len(estimator.classes_))
        self.trees_per_iteration = int(len(estimator._predictors[0]))
        if self.n_classes != 4 or self.trees_per_iteration != 4:
            raise ValueError("latency pilot expects the locked four-action risk model")
        self.iterations = int(len(estimator._predictors))
        self.tree_count = len(trees)
        node_count = max(len(tree.nodes) for tree in trees)
        shape = (self.tree_count, node_count)
        self.feature_idx = np.zeros(shape, dtype=np.int64)
        self.threshold = np.zeros(shape, dtype=np.float64)
        self.missing_left = np.zeros(shape, dtype=bool)
        self.left = np.zeros(shape, dtype=np.int64)
        self.right = np.zeros(shape, dtype=np.int64)
        self.is_leaf = np.ones(shape, dtype=bool)
        self.value = np.zeros(shape, dtype=np.float64)
        self.max_depth = 0
        for tree_index, tree in enumerate(trees):
            nodes = tree.nodes
            count = len(nodes)
            self.feature_idx[tree_index, :count] = nodes["feature_idx"]
            self.threshold[tree_index, :count] = nodes["num_threshold"]
            self.missing_left[tree_index, :count] = nodes["missing_go_to_left"].astype(bool)
            self.left[tree_index, :count] = nodes["left"]
            self.right[tree_index, :count] = nodes["right"]
            self.is_leaf[tree_index, :count] = nodes["is_leaf"].astype(bool)
            self.value[tree_index, :count] = nodes["value"]
            self.max_depth = max(self.max_depth, int(tree.get_max_depth()))
        self.baseline = np.asarray(estimator._baseline_prediction[0], dtype=np.float64)
        self.calibration_x = [
            np.asarray(calibrator.X_thresholds_, dtype=np.float64) for calibrator in model.calibrators
        ]
        self.calibration_y = [
            np.asarray(calibrator.y_thresholds_, dtype=np.float64) for calibrator in model.calibrators
        ]
        self._tree_ids = np.arange(self.tree_count, dtype=np.int64)
        self._single_indices = np.zeros(self.tree_count, dtype=np.int64)

    def _raw_single(self, row: np.ndarray) -> np.ndarray:
        indices = self._single_indices
        indices.fill(0)
        for _ in range(self.max_depth + 1):
            leaf = self.is_leaf[self._tree_ids, indices]
            active = ~leaf
            if not np.any(active):
                break
            trees = self._tree_ids[active]
            nodes = indices[active]
            values = row[self.feature_idx[trees, nodes]]
            go_left = np.where(
                np.isnan(values),
                self.missing_left[trees, nodes],
                values <= self.threshold[trees, nodes],
            )
            indices[active] = np.where(
                go_left,
                self.left[trees, nodes],
                self.right[trees, nodes],
            )
        leaves = self.value[self._tree_ids, indices]
        return self.baseline + leaves.reshape(self.iterations, self.n_classes).sum(axis=0)

    def _calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.column_stack(
            [
                np.interp(probabilities[:, index], self.calibration_x[index], self.calibration_y[index])
                for index in range(self.n_classes)
            ]
        )
        totals = calibrated.sum(axis=1, keepdims=True)
        zero = totals[:, 0] <= 1e-12
        calibrated[zero] = probabilities[zero]
        totals = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.maximum(totals, 1e-12)

    def predict_proba(self, features: FloatArray) -> FloatArray:
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        raw = np.stack([self._raw_single(row) for row in rows])
        shifted = raw - np.max(raw, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / np.sum(exp, axis=1, keepdims=True)
        return self._calibrate(probabilities)

    def predict(self, features: FloatArray) -> CalibratedRisk:
        row = np.asarray(features, dtype=np.float64)
        raw = self._raw_single(row)
        shifted = raw - np.max(raw)
        exp = np.exp(shifted)
        probabilities = (exp / np.sum(exp))[None, :]
        return CalibratedRisk(self._calibrate(probabilities)[0])


def cached_risk_features(
    query: IKQuery,
    prepared: PreparedCandidates,
    *,
    reuse_best_pose: bool,
) -> FloatArray:
    """Build the locked nine features, optionally reusing candidate diagnostics."""

    kinematics = prepared.kinematics
    best = prepared.candidates.joints[0]
    best_pose = prepared.best_pose if reuse_best_pose else kinematics.forward(best)
    position_error, orientation_error = pose_distance(query.target, best_pose)
    current_pose = kinematics.forward(query.previous_q)
    step_position, step_orientation = pose_distance(query.target, current_pose)
    margin = prepared.best_margin if reuse_best_pose else float(np.min(kinematics.joint_margin(best)))
    joint_step = (
        prepared.best_joint_step
        if reuse_best_pose
        else float(np.linalg.norm(kinematics.difference(best, query.previous_q)))
    )
    return np.array(
        [
            position_error,
            orientation_error,
            prepared.candidates.uncertainty_mean,
            prepared.candidates.uncertainty_max,
            kinematics.min_singular_value(best),
            margin,
            joint_step,
            step_position,
            step_orientation,
        ],
        dtype=np.float64,
    )
