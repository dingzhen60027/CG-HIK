from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

from .config import load_config, load_robot, resolve_path
from .data.datasets import QueryDataset, RiskDataset, TransitionDataset
from .data.generate import generate_point_test_set, generate_smooth_transitions
from .data.generate_v2 import (
    generate_hard_valid_queries,
    generate_mixed_query_set,
    generate_reference_trajectory_tests,
    label_cascade_actions,
)
from .experiments.ablations import FeatureMaskRiskProvider, SingleMemberCandidates
from .experiments.baselines import TRFOnlyMethod, fixed_hybrid
from .experiments.baselines_v2 import ThresholdGuardRiskProvider
from .experiments.evaluate import evaluate_methods, summarize_records, write_summary
from .experiments.policy_selection import (
    action_policy_metrics,
    action_predictions,
    tune_action_gate,
    tune_threshold_guard,
)
from .experiments.provenance import ensure_protocol_manifest, environment_payload
from .experiments.reporting_v2 import write_claim_gate_v2
from .experiments.statistics import (
    holm_adjust,
    paired_cluster_bootstrap_difference,
    paired_cluster_records,
    paired_cluster_sign_flip_pvalue,
)
from .experiments.validation import seed_validation_metrics
from .models.risk import ConstantRiskProvider, RiskModel
from .models.seed import PreviousStateCandidates, TorchSeedEnsemble
from .pipeline import (
    _build_seed_bank,
    _dls_from_metadata,
    _paths,
    _write_json,
    train_risk,
    train_seed,
)
from .runtime.cascade import (
    ActionGateConfig,
    CalibratedActionGate,
    CascadeConfig,
    CascadedHybridIK,
    EntryAction,
    FixedEntryGate,
)
from .solvers.dls import AdaptiveDLS, DLSConfig
from .solvers.fallback import TRFConfig, TRFFallbackSolver
from .solvers.verifier import SolutionVerifier, VerifierConfig


def _cascade_config(config: dict[str, Any]) -> CascadeConfig:
    values = dict(config.get("cascade", {}))
    allowed = CascadeConfig.__dataclass_fields__.keys()
    return CascadeConfig(**{key: value for key, value in values.items() if key in allowed})


def _ensure_v2_manifest(config: dict[str, Any], robot_name: str, paths: object) -> dict[str, Any]:
    return ensure_protocol_manifest(paths.root / "protocol_manifest_v2.json", config, robot_name)  # type: ignore[attr-defined]


def _verifier(config: dict[str, Any], kinematics: object) -> SolutionVerifier:
    return SolutionVerifier(kinematics, VerifierConfig(**dict(config.get("verifier", {}))))  # type: ignore[arg-type]


def _fallback(config: dict[str, Any], kinematics: object) -> TRFFallbackSolver:
    values = dict(config.get("fallback", {}))
    allowed = TRFConfig.__dataclass_fields__.keys()
    return TRFFallbackSolver(
        kinematics,  # type: ignore[arg-type]
        TRFConfig(**{key: value for key, value in values.items() if key in allowed}),
    )


def generate_data_v2(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, str]:
    config = load_config(config_path)
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    _ensure_v2_manifest(config, robot_name, paths)
    data = config["data"]
    for path, name in (
        (paths.seed_train, "seed_train"),
        (paths.seed_validation, "seed_validation"),
        (paths.test_id, "test_id"),
    ):
        if force or not path.exists():
            spec = data[name]
            generate_smooth_transitions(
                kinematics,
                trajectories=int(spec["trajectories"]),
                steps_per_trajectory=int(spec["steps_per_trajectory"]),
                seed=int(spec["seed"]),
                dt=float(data.get("dt", 0.02)),
                margin=float(spec.get("margin", 0.1)),
            ).save(path)

    seed_train = TransitionDataset.load(paths.seed_train)
    sample_count = min(int(data.get("sigma_samples", 5000)), len(seed_train))
    rng = np.random.default_rng(int(config.get("seed", 17)))
    indices = rng.choice(len(seed_train), size=sample_count, replace=False)
    sigmas = np.array([kinematics.min_singular_value(seed_train.target_q[index]) for index in indices])
    _write_json(
        paths.solver_metadata,
        {
            "protocol_version": 2,
            "sigma_threshold": float(np.percentile(sigmas, 10)),
            "sigma_sample_count": sample_count,
            "robot": robot_name,
            "joint_names": kinematics.joint_names,
        },
    )
    screening_dls = _dls_from_metadata(kinematics, config, paths)
    screening_verifier = _verifier(config, kinematics)
    for path, name in (
        (paths.risk_train_queries, "risk_train"),
        (paths.risk_validation_queries, "risk_validation"),
        (paths.calibration_queries, "calibration"),
        (paths.policy_validation_queries, "policy_validation"),
        (paths.risk_test_queries, "risk_test"),
    ):
        if force or not path.exists():
            spec = data[name]
            total = int(spec["samples"])
            hard_count = int(round(total * float(spec.get("hard_valid_fraction", 0.10))))
            mixed_count = total - hard_count
            datasets = [
                generate_mixed_query_set(
                    kinematics,
                    samples=mixed_count,
                    seed=int(spec["seed"]),
                    challenge_fraction=float(spec.get("challenge_fraction", 0.6)),
                    dt=float(data.get("dt", 0.02)),
                )
            ]
            if hard_count > 0:
                datasets.append(
                    generate_hard_valid_queries(
                        kinematics,
                        screening_dls,
                        screening_verifier,
                        count=hard_count,
                        seed=int(spec["seed"]) + 50_000,
                        dt=float(data.get("dt", 0.02)),
                        easy_iterations=_cascade_config(config).easy_iterations,
                        hard_iterations=_cascade_config(config).hard_iterations,
                    )
                )
            QueryDataset.concatenate(datasets).save(path)
    return {"root": str(paths.root), "solver_metadata": str(paths.solver_metadata)}


def train_seed_v2(config_path: str | Path, robot_name: str) -> dict[str, object]:
    config = load_config(config_path)
    paths = _paths(config, robot_name)
    _ensure_v2_manifest(config, robot_name, paths)
    payload = train_seed(config_path, robot_name)
    kinematics = load_robot(config, robot_name)
    ensemble = TorchSeedEnsemble.load(paths.seed_model, kinematics, device=config.get("device"))
    metrics = seed_validation_metrics(
        ensemble,
        TransitionDataset.load(paths.seed_validation),
        pose_samples=int(config.get("seed_validation", {}).get("pose_samples", 2000)),
    )
    payload["validation_metrics"] = metrics
    if paths.no_history_seed_model.exists():
        payload["no_history_model"] = str(paths.no_history_seed_model)
    _write_json(paths.results / "seed_training.json", payload)
    return payload


def train_risk_v2(config_path: str | Path, robot_name: str) -> dict[str, object]:
    config = load_config(config_path)
    paths = _paths(config, robot_name)
    _ensure_v2_manifest(config, robot_name, paths)
    return train_risk(config_path, robot_name)


def _oracle_cascade(
    config: dict[str, Any],
    kinematics: object,
    paths: object,
    ensemble: TorchSeedEnsemble,
) -> CascadedHybridIK:
    dls = _dls_from_metadata(kinematics, config, paths)  # type: ignore[arg-type]
    verifier = _verifier(config, kinematics)
    seed_bank = _build_seed_bank(config, kinematics, paths)  # type: ignore[arg-type]
    return CascadedHybridIK(
        kinematics,  # type: ignore[arg-type]
        ensemble,
        ConstantRiskProvider(np.array([1.0, 0.0, 0.0, 0.0])),
        dls,
        verifier,
        gate=FixedEntryGate(EntryAction.EASY),
        seed_bank=seed_bank,
        fallback=_fallback(config, kinematics),
        config=_cascade_config(config),
    )


def label_risk_v2(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, str]:
    config = load_config(config_path)
    paths = _paths(config, robot_name)
    _ensure_v2_manifest(config, robot_name, paths)
    kinematics = load_robot(config, robot_name)
    ensemble = TorchSeedEnsemble.load(paths.seed_model, kinematics, device=config.get("device"))
    cascade = _oracle_cascade(config, kinematics, paths, ensemble)
    pairs = {
        paths.risk_train_queries: paths.risk_train,
        paths.risk_validation_queries: paths.risk_validation,
        paths.calibration_queries: paths.calibration,
        paths.policy_validation_queries: paths.policy_validation,
        paths.risk_test_queries: paths.risk_test,
    }
    distributions: dict[str, list[int]] = {}
    for query_path, output_path in pairs.items():
        if force or not output_path.exists():
            labels = label_cascade_actions(
                kinematics,
                cascade,
                QueryDataset.load(query_path),
                dt=float(config["data"].get("dt", 0.02)),
            )
            labels.save(output_path)
        dataset = RiskDataset.load(output_path)
        distributions[output_path.stem] = np.bincount(dataset.labels, minlength=4).tolist()
    _write_json(paths.results / "action_label_distributions.json", distributions)
    return {output.stem: str(output) for output in pairs.values()}


def _test_queries_v2(
    config: dict[str, Any],
    kinematics: object,
    paths: object,
    dls: AdaptiveDLS,
    verifier: SolutionVerifier,
    *,
    force: bool,
) -> QueryDataset:
    if paths.test_queries.exists() and not force:  # type: ignore[attr-defined]
        return QueryDataset.load(paths.test_queries)  # type: ignore[attr-defined]
    evaluation = config["evaluation"]
    point = generate_point_test_set(
        kinematics,  # type: ignore[arg-type]
        TransitionDataset.load(paths.test_id),  # type: ignore[attr-defined]
        per_category=int(evaluation["per_stress_category"]),
        id_count=int(evaluation["id_count"]),
        seed=int(evaluation.get("test_seed", 9001)),
        dt=float(config["data"].get("dt", 0.02)),
    )
    hard_count = int(evaluation.get("hard_valid_count", 0))
    datasets = [point]
    if hard_count > 0:
        datasets.append(
            generate_hard_valid_queries(
                kinematics,  # type: ignore[arg-type]
                dls,
                verifier,
                count=hard_count,
                seed=int(evaluation.get("hard_valid_seed", 9010)),
                dt=float(config["data"].get("dt", 0.02)),
                easy_iterations=_cascade_config(config).easy_iterations,
                hard_iterations=_cascade_config(config).hard_iterations,
            )
        )
    trajectory = evaluation.get("reference_trajectories", {})
    if int(trajectory.get("paths_per_type", 0)) > 0:
        datasets.append(
            generate_reference_trajectory_tests(
                kinematics,  # type: ignore[arg-type]
                paths_per_type=int(trajectory["paths_per_type"]),
                steps=int(trajectory["steps"]),
                seed=int(trajectory.get("seed", 9020)),
                dt=float(config["data"].get("dt", 0.02)),
            )
        )
    result = QueryDataset.concatenate(datasets)
    result.save(paths.test_queries)  # type: ignore[attr-defined]
    return result


def evaluate_v2(config_path: str | Path, robot_name: str, *, force_test_data: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    try:
        import torch

        torch.set_num_threads(int(config.get("evaluation", {}).get("cpu_threads", 1)))
    except ImportError:  # pragma: no cover
        pass
    kinematics = load_robot(config, robot_name)
    paths = _paths(config, robot_name)
    manifest = _ensure_v2_manifest(config, robot_name, paths)
    _write_json(paths.results / "environment_v2.json", environment_payload())
    _write_json(paths.results / "protocol_manifest_v2.json", manifest)
    ensemble = TorchSeedEnsemble.load(paths.seed_model, kinematics, device=config.get("device"))
    risk = RiskModel.load(paths.risk_model)
    dls = _dls_from_metadata(kinematics, config, paths)
    verifier = _verifier(config, kinematics)
    fallback = _fallback(config, kinematics)
    seed_bank = _build_seed_bank(config, kinematics, paths)
    cascade_config = _cascade_config(config)
    policy_config = dict(config.get("policy_selection", {}))
    max_false_reject = float(policy_config.get("max_false_reject_rate", 0.01))
    min_reject_recall = float(policy_config.get("min_reject_recall", 0.95))
    policy_validation = RiskDataset.load(paths.policy_validation)
    gate_config, learned_policy_report = tune_action_gate(
        risk,
        policy_validation,
        easy_grid=policy_config.get("easy_probability_grid", [0.60, 0.70, 0.80]),
        hard_grid=policy_config.get("hard_probability_grid", [0.35, 0.45, 0.55]),
        reject_grid=policy_config.get("reject_probability_grid", [0.80, 0.85, 0.90, 0.95, 0.99]),
        max_false_reject_rate=max_false_reject,
        min_reject_recall=min_reject_recall,
    )
    fixed_risk = ConstantRiskProvider(np.array([1.0, 0.0, 0.0, 0.0]))
    threshold_guard, threshold_policy_report = tune_threshold_guard(
        RiskDataset.load(paths.risk_train),
        policy_validation,
        quantiles=config.get("threshold_guard", {}).get(
            "quantile_grid", [0.99, 0.995, 0.9975, 0.999, 0.9995, 1.0]
        ),
        max_false_reject_rate=max_false_reject,
        min_reject_recall=min_reject_recall,
    )
    _write_json(paths.results / "threshold_guard.json", asdict(threshold_guard.config))
    risk_test = RiskDataset.load(paths.risk_test)
    learned_test_actions = action_predictions(risk.predict_proba(risk_test.features), gate_config)
    threshold_test_actions = threshold_guard.predict_actions(risk_test.features)
    policy_report = {
        "learned_gate": {
            **learned_policy_report,
            "test_metrics": action_policy_metrics(risk_test.labels, learned_test_actions),
        },
        "threshold_guard": {
            **threshold_policy_report,
            "test_metrics": action_policy_metrics(risk_test.labels, threshold_test_actions),
        },
    }
    _write_json(paths.results / "policy_selection_v2.json", policy_report)
    methods: dict[str, object] = {
        "dls_previous_1x50": fixed_hybrid(
            kinematics, PreviousStateCandidates(), dls, verifier, candidate_count=1, iterations=50
        ),
        "learned_1x25": fixed_hybrid(
            kinematics, ensemble, dls, verifier, candidate_count=1, iterations=25
        ),
        "fixed_robust_cascade": CascadedHybridIK(
            kinematics,
            ensemble,
            fixed_risk,
            dls,
            verifier,
            gate=FixedEntryGate(EntryAction.EASY),
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade_config,
        ),
        "threshold_guard_cascade": CascadedHybridIK(
            kinematics,
            ensemble,
            threshold_guard,
            dls,
            verifier,
            gate=CalibratedActionGate(gate_config),
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade_config,
        ),
        "trf_previous": TRFOnlyMethod(fallback, verifier),
        "proposed_v2": CascadedHybridIK(
            kinematics,
            ensemble,
            risk,
            dls,
            verifier,
            gate=CalibratedActionGate(gate_config),
            seed_bank=seed_bank,
            fallback=fallback,
            config=cascade_config,
        ),
    }
    if bool(config.get("ablations", {}).get("enabled", False)):
        no_history = TorchSeedEnsemble.load(paths.no_history_seed_model, kinematics, device=config.get("device"))
        uncalibrated = RiskModel.load(paths.uncalibrated_risk_model)
        no_uncertainty = RiskModel.load(paths.no_uncertainty_risk_model)
        no_reject_values = asdict(gate_config)
        no_reject_values["reject_probability"] = 1.1
        fixed_values = asdict(dls.config)
        fixed_lambda = float(config.get("ablations", {}).get("fixed_damping", 0.01))
        fixed_values["lambda_min"] = fixed_lambda
        fixed_values["lambda_max"] = fixed_lambda
        fixed_dls = AdaptiveDLS(kinematics, DLSConfig(**fixed_values))

        def cascade(candidate: object, risk_provider: object, *, gate: object | None = None, solver: AdaptiveDLS = dls, use_fallback: bool = True) -> CascadedHybridIK:
            return CascadedHybridIK(
                kinematics,
                candidate,  # type: ignore[arg-type]
                risk_provider,  # type: ignore[arg-type]
                solver,
                verifier,
                gate=gate or CalibratedActionGate(gate_config),  # type: ignore[arg-type]
                seed_bank=seed_bank if use_fallback else None,
                fallback=fallback if use_fallback else None,
                config=cascade_config,
            )

        methods.update(
            {
                "ablation_no_history": cascade(no_history, risk),
                "ablation_single_member": cascade(SingleMemberCandidates(ensemble), risk),
                "ablation_no_uncertainty": cascade(
                    ensemble,
                    FeatureMaskRiskProvider(no_uncertainty, (0, 1, 4, 5, 6, 7, 8)),
                ),
                "ablation_uncalibrated": cascade(ensemble, uncalibrated),
                "ablation_no_reject": cascade(
                    ensemble,
                    risk,
                    gate=CalibratedActionGate(ActionGateConfig(**no_reject_values)),
                ),
                "ablation_no_fallback": cascade(ensemble, risk, use_fallback=False),
                "ablation_fixed_damping": cascade(ensemble, risk, solver=fixed_dls),
            }
        )

    queries = _test_queries_v2(config, kinematics, paths, dls, verifier, force=force_test_data)
    evaluation = config["evaluation"]
    result_path = paths.results / "query_results_v2.jsonl"
    records = evaluate_methods(
        methods,  # type: ignore[arg-type]
        queries,
        dt=float(config["data"].get("dt", 0.02)),
        output_jsonl=result_path,
        warmup_iterations=int(evaluation.get("warmup_iterations", 0)),
        timing_repeats=int(evaluation.get("timing_repeats", 1)),
        method_order_seed=int(evaluation.get("method_order_seed", 9100)),
        synchronize_cuda=bool(evaluation.get("synchronize_cuda", True)),
    )
    summary = summarize_records(records)
    write_summary(summary, paths.results / "summary_v2.json")
    bootstrap_samples = int(evaluation.get("bootstrap_samples", 10_000))
    statistics: dict[str, object] = {}
    raw_p: dict[str, float] = {}
    comparisons = {
        "point_feasible_success": ("accepted", "point_feasible", True),
        "point_feasible_function_evaluations": (
            "function_evaluations", "point_feasible", True
        ),
        "point_rejectable_acceptance": ("accepted", "point_rejectable", True),
        "point_feasible_latency": ("latency_seconds", "point_feasible", False),
        "point_rejectable_function_evaluations": (
            "function_evaluations", "point_rejectable", False
        ),
        "trajectory_frame_success": ("accepted", "trajectory", False),
    }
    for name, (field, subset, primary) in comparisons.items():
        baseline, proposed, clusters = paired_cluster_records(
            records,
            "fixed_robust_cascade",
            "proposed_v2",
            field,
            subset=subset,
        )
        statistics[name] = paired_cluster_bootstrap_difference(
            baseline,
            proposed,
            clusters,
            samples=bootstrap_samples,
            seed=int(config.get("seed", 17)),
        )
        if primary:
            raw_p[name] = paired_cluster_sign_flip_pvalue(
                baseline,
                proposed,
                clusters,
                samples=bootstrap_samples,
                seed=int(config.get("seed", 17)),
            )
    adjusted = holm_adjust(raw_p)
    for name in raw_p:
        statistics[name]["p_value"] = raw_p[name]  # type: ignore[index]
        statistics[name]["holm_adjusted_p"] = adjusted[name]  # type: ignore[index]
    _write_json(paths.results / "cluster_statistics_v2.json", statistics)
    risk_payload = json.loads((paths.results / "risk_metrics.json").read_text(encoding="utf-8"))
    claim_gate = write_claim_gate_v2(records, risk_payload, paths.results, policy_report=policy_report)
    return {
        "results": str(result_path),
        "query_count": len(queries),
        "method_count": len(methods),
        "claim_gate": claim_gate,
        "statistics": statistics,
    }


def run_all_v2(config_path: str | Path, robot_name: str, *, force: bool = False) -> dict[str, object]:
    stages: dict[str, object] = {}
    stages["data"] = generate_data_v2(config_path, robot_name, force=force)
    stages["seed"] = train_seed_v2(config_path, robot_name)
    stages["action_labels"] = label_risk_v2(config_path, robot_name, force=force)
    stages["risk"] = train_risk_v2(config_path, robot_name)
    stages["evaluation"] = evaluate_v2(config_path, robot_name, force_test_data=force)
    return stages


def run_repetitions_v2(
    config_path: str | Path,
    robot_name: str,
    seeds: list[int],
    *,
    force: bool = False,
) -> dict[str, object]:
    source = load_config(config_path)
    source.pop("_config_path", None)
    original = load_config(config_path)
    source["output_root"] = str(resolve_path(original, source.get("output_root", "../outputs")))
    for robot in source.get("robots", {}).values():
        robot["urdf"] = str(resolve_path(original, robot["urdf"]))
    base_name = str(source.get("experiment_name", "v2"))
    repetitions: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="confik_v2_repetitions_") as temporary:
        for seed in seeds:
            repeated = json.loads(json.dumps(source))
            repeated["experiment_name"] = f"{base_name}_seed{seed}"
            repeated["seed"] = int(seed)
            repeated.setdefault("seed_model", {})["seed"] = int(seed)
            path = Path(temporary) / f"seed_{seed}.yaml"
            path.write_text(yaml.safe_dump(repeated, sort_keys=False), encoding="utf-8")
            result = run_all_v2(path, robot_name, force=force)
            repetitions[str(seed)] = {
                "root": result["data"]["root"],  # type: ignore[index]
                "claim_gate": result["evaluation"]["claim_gate"],  # type: ignore[index]
            }
    return {"robot": robot_name, "seeds": seeds, "repetitions": repetitions}


def aggregate_v2(
    config_path: str | Path,
    robots: list[str],
    seeds: list[int],
) -> dict[str, object]:
    """Aggregate locked runs without treating training seeds as new test samples."""
    if not robots or not seeds:
        raise ValueError("aggregate-v2 requires at least one robot and one seed")
    config = load_config(config_path)
    output_root = resolve_path(config, config.get("output_root", "../outputs"))
    base_name = str(config.get("experiment_name", "paper_v2"))
    runs: list[dict[str, object]] = []
    missing: list[str] = []
    for robot in robots:
        for seed in seeds:
            path = (
                output_root
                / f"{base_name}_seed{seed}"
                / robot
                / "results"
                / "claim_gate_v2.json"
            )
            if not path.exists():
                missing.append(str(path))
                continue
            claim = json.loads(path.read_text(encoding="utf-8"))
            runs.append({"robot": robot, "seed": seed, "path": str(path), "claim": claim})

    metric_names = (
        "point_feasible_success_gap",
        "point_feasible_evaluation_reduction",
        "point_feasible_p95_latency_reduction",
        "point_rejectable_evaluation_reduction",
        "point_rejectable_p95_latency_reduction",
        "trajectory_completion_gap",
        "threshold_guard_point_evaluation_reduction",
    )
    robot_summary: dict[str, object] = {}
    for robot in robots:
        robot_runs = [run for run in runs if run["robot"] == robot]
        summaries: dict[str, object] = {}
        for metric in metric_names:
            values = np.asarray(
                [float(run["claim"][metric]) for run in robot_runs],  # type: ignore[index]
                dtype=np.float64,
            )
            summaries[metric] = {
                "values": values.tolist(),
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "min": float(np.min(values)) if values.size else float("nan"),
                "max": float(np.max(values)) if values.size else float("nan"),
            }
        robot_summary[robot] = {
            "run_count": len(robot_runs),
            "metrics": summaries,
            "all_run_gates_pass": bool(
                robot_runs and all(bool(run["claim"]["pilot_gate_pass"]) for run in robot_runs)  # type: ignore[index]
            ),
        }

    expected_count = len(robots) * len(seeds)
    complete = len(runs) == expected_count and not missing
    all_run_gates = bool(
        complete and all(bool(run["claim"]["pilot_gate_pass"]) for run in runs)  # type: ignore[index]
    )
    direction_consistent = bool(
        complete
        and all(
            float(run["claim"]["point_feasible_success_gap"]) >= -0.01  # type: ignore[index]
            and float(run["claim"]["point_feasible_evaluation_reduction"]) > 0.0  # type: ignore[index]
            and float(run["claim"]["point_rejectable_evaluation_reduction"]) > 0.0  # type: ignore[index]
            and float(run["claim"]["threshold_guard_point_evaluation_reduction"]) > 0.0  # type: ignore[index]
            for run in runs
        )
    )
    payload: dict[str, object] = {
        "protocol": "v2 separate-estimand confirmatory aggregation",
        "robots": robots,
        "training_seeds": seeds,
        "expected_run_count": expected_count,
        "observed_run_count": len(runs),
        "missing": missing,
        "runs": [
            {
                "robot": run["robot"],
                "seed": run["seed"],
                "path": run["path"],
                "pilot_gate_pass": run["claim"]["pilot_gate_pass"],  # type: ignore[index]
            }
            for run in runs
        ],
        "by_robot": robot_summary,
        "all_run_gates_pass": all_run_gates,
        "effect_direction_consistent": direction_consistent,
        "paper_gate_pass": bool(complete and all_run_gates and direction_consistent),
        "statistical_note": (
            "Training seeds share each robot's locked query set and are sensitivity replicates, "
            "not independent query samples. Query-cluster intervals remain in each run artifact."
        ),
    }
    aggregate_dir = output_root / f"{base_name}_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(aggregate_dir / "paper_gate_v2.json", payload)
    return payload
