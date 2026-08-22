from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

from .config import load_config, load_robot, resolve_path
from .data.datasets import QueryDataset, RiskDataset, TransitionDataset
from .data.generate import (
    generate_cartesian_path_tests,
    generate_mixed_transitions,
    generate_point_test_set,
    generate_smooth_transitions,
    label_solver_risk,
)
from .experiments.baselines import (
    KDTreeCandidates,
    RandomCandidates,
    TRFOnlyMethod,
    fixed_hybrid,
)
from .experiments.ablations import FeatureMaskRiskProvider, SingleMemberCandidates
from .experiments.evaluate import evaluate_methods, summarize_records, write_summary
from .experiments.metrics import risk_metrics
from .experiments.reporting import generate_report
from .experiments.statistics import (
    holm_adjust,
    paired_bootstrap_difference,
    paired_records,
    paired_sign_flip_pvalue,
)
from .models.risk import RiskModel, select_risk_model
from .models.seed import PreviousStateCandidates, SeedTrainingConfig, TorchSeedEnsemble
from .runtime.gate import ConfidenceGate, GateConfig
from .runtime.hybrid import HybridIK
from .solvers.dls import AdaptiveDLS, DLSConfig
from .solvers.fallback import KDTreeSeedBank, TRFConfig, TRFFallbackSolver
from .solvers.verifier import SolutionVerifier, VerifierConfig


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    datasets: Path
    models: Path
    results: Path
    seed_train: Path
    seed_validation: Path
    risk_train_queries: Path
    risk_validation_queries: Path
    calibration_queries: Path
    policy_validation_queries: Path
    risk_test_queries: Path
    test_id: Path
    test_queries: Path
    seed_model: Path
    risk_model: Path
    no_history_seed_model: Path
    uncalibrated_risk_model: Path
    no_uncertainty_risk_model: Path
    seed_bank: Path
    solver_metadata: Path
    risk_train: Path
    risk_validation: Path
    calibration: Path
    policy_validation: Path
    risk_test: Path

    @classmethod
    def build(cls, root: Path) -> "ArtifactPaths":
        datasets = root / "datasets"
        models = root / "models"
        results = root / "results"
        return cls(
            root=root,
            datasets=datasets,
            models=models,
            results=results,
            seed_train=datasets / "seed_train.npz",
            seed_validation=datasets / "seed_validation.npz",
            risk_train_queries=datasets / "risk_train_queries.npz",
            risk_validation_queries=datasets / "risk_validation_queries.npz",
            calibration_queries=datasets / "calibration_queries.npz",
            policy_validation_queries=datasets / "policy_validation_queries.npz",
            risk_test_queries=datasets / "risk_test_queries.npz",
            test_id=datasets / "test_id.npz",
            test_queries=datasets / "test_queries.npz",
            seed_model=models / "seed_ensemble.pt",
            risk_model=models / "risk_model.joblib",
            no_history_seed_model=models / "seed_no_history.pt",
            uncalibrated_risk_model=models / "risk_uncalibrated.joblib",
            no_uncertainty_risk_model=models / "risk_no_uncertainty.joblib",
            seed_bank=models / "seed_bank.npz",
            solver_metadata=models / "solver_metadata.json",
            risk_train=datasets / "risk_train.npz",
            risk_validation=datasets / "risk_validation.npz",
            calibration=datasets / "calibration.npz",
            policy_validation=datasets / "policy_validation.npz",
            risk_test=datasets / "risk_test.npz",
        )


def _paths(config: dict[str, Any], robot_name: str) -> ArtifactPaths:
    output_root = resolve_path(config, config.get("output_root", "../outputs"))
    experiment_name = str(config.get("experiment_name", "paper"))
    paths = ArtifactPaths.build(output_root / experiment_name / robot_name)
    for directory in (paths.root, paths.datasets, paths.models, paths.results):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _dls_from_metadata(kinematics: object, config: dict[str, Any], paths: ArtifactPaths) -> AdaptiveDLS:
    solver = dict(config.get("solver", {}))
    if paths.solver_metadata.exists():
        metadata = json.loads(paths.solver_metadata.read_text(encoding="utf-8"))
        solver["sigma_threshold"] = metadata["sigma_threshold"]
    allowed = DLSConfig.__dataclass_fields__.keys()
    return AdaptiveDLS(kinematics, DLSConfig(**{key: value for key, value in solver.items() if key in allowed}))  # type: ignore[arg-type]


def generate_data(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, str]:
    config = load_config(config_path)
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    data_config = config["data"]

    smooth_specs = {
        paths.seed_train: data_config["seed_train"],
        paths.seed_validation: data_config["seed_validation"],
        paths.test_id: data_config["test_id"],
    }
    for path, spec in smooth_specs.items():
        if force or not path.exists():
            dataset = generate_smooth_transitions(
                kinematics,
                trajectories=int(spec["trajectories"]),
                steps_per_trajectory=int(spec["steps_per_trajectory"]),
                seed=int(spec["seed"]),
                dt=float(data_config.get("dt", 0.02)),
                margin=float(spec.get("margin", 0.1)),
            )
            dataset.save(path)

    mixed_specs = {
        paths.risk_train_queries: data_config["risk_train"],
        paths.risk_validation_queries: data_config["risk_validation"],
        paths.calibration_queries: data_config["calibration"],
        paths.risk_test_queries: data_config["risk_test"],
    }
    for path, spec in mixed_specs.items():
        if force or not path.exists():
            dataset = generate_mixed_transitions(
                kinematics,
                samples=int(spec["samples"]),
                seed=int(spec["seed"]),
                challenge_fraction=float(spec.get("challenge_fraction", 0.5)),
                dt=float(data_config.get("dt", 0.02)),
            )
            dataset.save(path)

    seed_train = TransitionDataset.load(paths.seed_train)
    sigma_samples = min(int(data_config.get("sigma_samples", 5000)), len(seed_train))
    rng = np.random.default_rng(int(config.get("seed", 17)))
    indices = rng.choice(len(seed_train), size=sigma_samples, replace=False)
    sigmas = np.array([kinematics.min_singular_value(seed_train.target_q[index]) for index in indices])
    metadata = {
        "sigma_threshold": float(np.percentile(sigmas, 10)),
        "sigma_sample_count": sigma_samples,
        "robot": robot_name,
        "joint_names": kinematics.joint_names,
    }
    _write_json(paths.solver_metadata, metadata)
    return {"root": str(paths.root), "solver_metadata": str(paths.solver_metadata)}


def train_seed(config_path: str | Path, robot_name: str) -> dict[str, object]:
    config = load_config(config_path)
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    dataset = TransitionDataset.load(paths.seed_train)
    model_config = dict(config.get("seed_model", {}))
    if "hidden_sizes" in model_config:
        model_config["hidden_sizes"] = tuple(model_config["hidden_sizes"])
    training_config = SeedTrainingConfig(**model_config)
    ensemble = TorchSeedEnsemble(
        kinematics,
        training_config,
        device=config.get("device"),
    )
    ensemble.fit(dataset)
    ensemble.save(paths.seed_model)
    payload = {
        "model": str(paths.seed_model),
        "training_history": ensemble.training_history,
        "config": asdict(training_config),
    }
    _write_json(paths.results / "seed_training.json", payload)
    if bool(config.get("ablations", {}).get("enabled", False)):
        no_history_config = SeedTrainingConfig(
            **{**asdict(training_config), "hidden_sizes": training_config.hidden_sizes, "use_history": False}
        )
        no_history = TorchSeedEnsemble(
            kinematics,
            no_history_config,
            device=config.get("device"),
        ).fit(dataset)
        no_history.save(paths.no_history_seed_model)
        payload["no_history_model"] = str(paths.no_history_seed_model)
    return payload


def label_risk(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, str]:
    config = load_config(config_path)
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    ensemble = TorchSeedEnsemble.load(paths.seed_model, kinematics, device=config.get("device"))
    dls = _dls_from_metadata(kinematics, config, paths)
    pairs = {
        paths.risk_train_queries: paths.risk_train,
        paths.risk_validation_queries: paths.risk_validation,
        paths.calibration_queries: paths.calibration,
        paths.risk_test_queries: paths.risk_test,
    }
    for query_path, output_path in pairs.items():
        if force or not output_path.exists():
            transition_dataset = TransitionDataset.load(query_path)
            risk_dataset = label_solver_risk(
                kinematics,
                ensemble,
                dls,
                transition_dataset,
                max_iterations=int(config.get("risk", {}).get("label_budget", 50)),
                dt=float(config["data"].get("dt", 0.02)),
            )
            risk_dataset.save(output_path)
    return {name: str(path) for name, path in {
        "train": paths.risk_train,
        "validation": paths.risk_validation,
        "calibration": paths.calibration,
        "test": paths.risk_test,
    }.items()}


def train_risk(config_path: str | Path, robot_name: str) -> dict[str, object]:
    config = load_config(config_path)
    paths = _paths(config, robot_name)
    train = RiskDataset.load(paths.risk_train)
    validation = RiskDataset.load(paths.risk_validation)
    calibration = RiskDataset.load(paths.calibration)
    test = RiskDataset.load(paths.risk_test)
    model, selection_scores = select_risk_model(
        train.features,
        train.labels,
        validation.features,
        validation.labels,
        seed=int(config.get("seed", 17)),
    )
    model.save(paths.uncalibrated_risk_model)
    model.calibrate(calibration.features, calibration.labels)
    model.save(paths.risk_model)
    metrics = risk_metrics(model, test)
    payload: dict[str, object] = {
        "selected_model": model.kind,
        "selection_nll": selection_scores,
        "test_metrics": metrics,
        "class_counts": np.bincount(test.labels, minlength=4).tolist(),
    }
    if bool(config.get("ablations", {}).get("enabled", False)):
        keep = (0, 1, 4, 5, 6, 7, 8)
        no_uncertainty, no_uncertainty_scores = select_risk_model(
            train.features[:, keep],
            train.labels,
            validation.features[:, keep],
            validation.labels,
            seed=int(config.get("seed", 17)),
        )
        no_uncertainty.calibrate(calibration.features[:, keep], calibration.labels)
        no_uncertainty.save(paths.no_uncertainty_risk_model)
        payload["no_uncertainty_selection_nll"] = no_uncertainty_scores
    _write_json(paths.results / "risk_metrics.json", payload)
    return payload


def _build_seed_bank(config: dict[str, Any], kinematics: object, paths: ArtifactPaths) -> KDTreeSeedBank:
    train = TransitionDataset.load(paths.seed_train)
    bank_size = min(int(config.get("evaluation", {}).get("seed_bank_size", 100_000)), len(train))
    if paths.seed_bank.exists():
        with np.load(paths.seed_bank) as data:
            joints = data["joints"]
    else:
        rng = np.random.default_rng(int(config.get("seed", 17)))
        indices = rng.choice(len(train), size=bank_size, replace=False)
        joints = train.target_q[indices]
        np.savez_compressed(paths.seed_bank, joints=joints)
    return KDTreeSeedBank(kinematics).fit(joints)  # type: ignore[arg-type]


def _prepare_test_queries(
    config: dict[str, Any],
    robot_name: str,
    kinematics: object,
    paths: ArtifactPaths,
    dls: AdaptiveDLS,
    *,
    force: bool,
) -> QueryDataset:
    if paths.test_queries.exists() and not force:
        return QueryDataset.load(paths.test_queries)
    evaluation = config["evaluation"]
    point = generate_point_test_set(
        kinematics,  # type: ignore[arg-type]
        TransitionDataset.load(paths.test_id),
        per_category=int(evaluation["per_stress_category"]),
        id_count=int(evaluation["id_count"]),
        seed=int(evaluation.get("test_seed", 9001)),
        dt=float(config["data"].get("dt", 0.02)),
    )
    paths_config = evaluation.get("paths", {})
    if int(paths_config.get("paths_per_type", 0)) > 0:
        trajectory = generate_cartesian_path_tests(
            kinematics,  # type: ignore[arg-type]
            dls,
            paths_per_type=int(paths_config["paths_per_type"]),
            steps=int(paths_config["steps"]),
            seed=int(paths_config.get("seed", 9002)),
            dt=float(config["data"].get("dt", 0.02)),
            amplitude=float(paths_config.get("amplitude", 0.03)),
        )
        point = QueryDataset.concatenate([point, trajectory])
    point.save(paths.test_queries)
    return point


def evaluate(config_path: str | Path, robot_name: str, *, force_test_data: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    try:
        import torch

        torch.set_num_threads(int(config.get("evaluation", {}).get("cpu_threads", 1)))
    except ImportError:  # pragma: no cover
        pass
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    ensemble = TorchSeedEnsemble.load(paths.seed_model, kinematics, device=config.get("device"))
    risk_model = RiskModel.load(paths.risk_model)
    dls = _dls_from_metadata(kinematics, config, paths)
    verifier_config = dict(config.get("verifier", {}))
    verifier = SolutionVerifier(kinematics, VerifierConfig(**verifier_config))
    trf_config = dict(config.get("fallback", {}))
    allowed_trf = TRFConfig.__dataclass_fields__.keys()
    trf = TRFFallbackSolver(
        kinematics,
        TRFConfig(**{key: value for key, value in trf_config.items() if key in allowed_trf}),
    )
    seed_bank = _build_seed_bank(config, kinematics, paths)
    query_dataset = _prepare_test_queries(
        config,
        robot_name,
        kinematics,
        paths,
        dls,
        force=force_test_data,
    )

    methods = {
        "dls_previous_1x50": fixed_hybrid(
            kinematics, PreviousStateCandidates(), dls, verifier, candidate_count=1, iterations=50
        ),
        "random_multistart_5x25": fixed_hybrid(
            kinematics, RandomCandidates(kinematics, 5, int(config.get("seed", 17))), dls, verifier,
            candidate_count=5, iterations=25,
        ),
        "kdtree_3x25": fixed_hybrid(
            kinematics, KDTreeCandidates(seed_bank, 3), dls, verifier, candidate_count=3, iterations=25
        ),
        "learned_1x25": fixed_hybrid(
            kinematics, ensemble, dls, verifier, candidate_count=1, iterations=25
        ),
        "learned_3x15": fixed_hybrid(
            kinematics, ensemble, dls, verifier, candidate_count=3, iterations=15
        ),
        "trf_previous": TRFOnlyMethod(trf, verifier),
        "proposed": HybridIK(
            kinematics,
            ensemble,
            risk_model,
            dls,
            verifier,
            gate=ConfidenceGate(GateConfig(**config.get("gate", {}))),
            seed_bank=seed_bank,
            fallback=trf,
            fallback_seed_count=int(trf_config.get("seed_count", 3)),
        ),
    }
    if bool(config.get("ablations", {}).get("enabled", False)):
        no_history = TorchSeedEnsemble.load(
            paths.no_history_seed_model,
            kinematics,
            device=config.get("device"),
        )
        uncalibrated = RiskModel.load(paths.uncalibrated_risk_model)
        no_uncertainty = RiskModel.load(paths.no_uncertainty_risk_model)
        keep = (0, 1, 4, 5, 6, 7, 8)
        fixed_dls_config = asdict(dls.config)
        fixed_lambda = float(config.get("ablations", {}).get("fixed_damping", 0.01))
        fixed_dls_config["lambda_min"] = fixed_lambda
        fixed_dls_config["lambda_max"] = fixed_lambda
        fixed_dls = AdaptiveDLS(kinematics, DLSConfig(**fixed_dls_config))
        methods.update(
            {
                "ablation_no_history": HybridIK(
                    kinematics, no_history, risk_model, dls, verifier, gate=ConfidenceGate(),
                    seed_bank=seed_bank, fallback=trf,
                ),
                "ablation_single_member": HybridIK(
                    kinematics, SingleMemberCandidates(ensemble), risk_model, dls, verifier,
                    gate=ConfidenceGate(), seed_bank=seed_bank, fallback=trf,
                ),
                "ablation_no_uncertainty": HybridIK(
                    kinematics, ensemble, FeatureMaskRiskProvider(no_uncertainty, keep), dls, verifier,
                    gate=ConfidenceGate(), seed_bank=seed_bank, fallback=trf,
                ),
                "ablation_uncalibrated": HybridIK(
                    kinematics, ensemble, uncalibrated, dls, verifier, gate=ConfidenceGate(),
                    seed_bank=seed_bank, fallback=trf,
                ),
                "ablation_no_fallback": HybridIK(
                    kinematics, ensemble, risk_model, dls, verifier, gate=ConfidenceGate(),
                ),
                "ablation_fixed_damping": HybridIK(
                    kinematics, ensemble, risk_model, fixed_dls, verifier, gate=ConfidenceGate(),
                    seed_bank=seed_bank, fallback=trf,
                ),
            }
        )
    result_path = paths.results / "query_results.jsonl"
    records = evaluate_methods(
        methods,
        query_dataset,
        dt=float(config["data"].get("dt", 0.02)),
        output_jsonl=result_path,
        warmup_iterations=int(config.get("evaluation", {}).get("warmup_iterations", 0)),
        timing_repeats=int(config.get("evaluation", {}).get("timing_repeats", 1)),
    )
    summary = summarize_records(records)
    write_summary(summary, paths.results / "summary.json")
    statistics: dict[str, object] = {}
    raw_p_values: dict[str, float] = {}
    for field in ("accepted", "function_evaluations", "latency_seconds"):
        baseline, proposed = paired_records(records, "learned_3x15", "proposed", field)
        statistics[field] = paired_bootstrap_difference(
            baseline,
            proposed,
            samples=int(config.get("evaluation", {}).get("bootstrap_samples", 10_000)),
            seed=int(config.get("seed", 17)),
        )
        raw_p_values[field] = paired_sign_flip_pvalue(
            baseline,
            proposed,
            samples=int(config.get("evaluation", {}).get("bootstrap_samples", 10_000)),
            seed=int(config.get("seed", 17)),
        )
    adjusted = holm_adjust(raw_p_values)
    for field in raw_p_values:
        statistics[field]["p_value"] = raw_p_values[field]  # type: ignore[index]
        statistics[field]["holm_adjusted_p"] = adjusted[field]  # type: ignore[index]
    _write_json(paths.results / "paired_bootstrap.json", statistics)
    risk_payload = json.loads((paths.results / "risk_metrics.json").read_text(encoding="utf-8"))
    claim_gate = generate_report(
        records,
        summary,
        risk_model,
        RiskDataset.load(paths.risk_test),
        risk_payload,
        paths.results / "figures",
    )
    return {
        "results": str(result_path),
        "summary": summary,
        "paired_bootstrap": statistics,
        "claim_gate": claim_gate,
        "query_count": len(query_dataset),
    }


def run_all(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, object]:
    stages: dict[str, object] = {}
    stages["data"] = generate_data(config_path, robot_name, force=force)
    stages["seed"] = train_seed(config_path, robot_name)
    stages["risk_labels"] = label_risk(config_path, robot_name, force=force)
    stages["risk"] = train_risk(config_path, robot_name)
    stages["evaluation"] = evaluate(config_path, robot_name, force_test_data=force)
    return stages


def run_repetitions(
    config_path: str | Path,
    robot_name: str,
    seeds: list[int],
    *,
    force: bool = False,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one repetition seed is required")
    source = load_config(config_path)
    source.pop("_config_path", None)
    original = load_config(config_path)
    source["output_root"] = str(resolve_path(original, source.get("output_root", "../outputs")))
    for robot in source.get("robots", {}).values():
        robot["urdf"] = str(resolve_path(original, robot["urdf"]))
    base_name = str(source.get("experiment_name", "paper"))
    repetitions: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="confik_repetitions_") as temporary:
        for seed in seeds:
            repeated = json.loads(json.dumps(source))
            repeated["experiment_name"] = f"{base_name}_seed{seed}"
            repeated["seed"] = int(seed)
            repeated.setdefault("seed_model", {})["seed"] = int(seed)
            temporary_config = Path(temporary) / f"seed_{seed}.yaml"
            temporary_config.write_text(yaml.safe_dump(repeated, sort_keys=False), encoding="utf-8")
            result = run_all(temporary_config, robot_name, force=force)
            repetitions[str(seed)] = {
                "root": result["data"]["root"],  # type: ignore[index]
                "risk": result["risk"],
                "query_count": result["evaluation"]["query_count"],  # type: ignore[index]
            }
    return {"robot": robot_name, "seeds": seeds, "repetitions": repetitions}
