#!/usr/bin/env python3
"""Generate final-paper evidence from frozen CG-HIK artifacts only.

The development routing diagnostic replays an already-sealed predictor, but
this script never trains a model or calls an IK solver/verifier.  It never
writes below ``outputs/``.
"""

from __future__ import annotations

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
OUTPUTS = ROOT / "outputs"
SOURCE_DATA = PAPER_DIR / "source_data"
GENERATED = PAPER_DIR / "generated"

ROBOTS = ("panda", "ur5e")
ROBOT_LABELS = {"panda": "Panda", "ur5e": "UR5e"}
ROBOT_MACRO_PREFIXES = {"panda": "Panda", "ur5e": "UR"}
DEVELOPMENT_ROLES = (
    "risk_train_queries",
    "calibration_queries",
    "policy_validation_queries",
)
ENTRIES = ("easy", "medium", "hard")
FINAL_ACTIONS = (*ENTRIES, "reject", "defer")
POINT_METHODS = (
    "fixed_robust_cascade",
    "proposed_v2",
    "threshold_guard_cascade",
    "learned_1x25",
    "dls_previous_1x50",
    "trf_previous",
    "proposed_v4",
)
TRAJECTORY_METHODS = (
    "fixed_robust_cascade",
    "always_hard",
    "counterfactual_cghik_v4",
)
METHOD_LABELS = {
    "fixed_robust_cascade": "Fixed robust cascade",
    "always_hard": "Fixed hard-entry cascade",
    "counterfactual_cghik_v4": "CG-HIK",
    "proposed_v4": "CG-HIK",
    "proposed_v2": "Categorical routing baseline",
    "threshold_guard_cascade": "Cartesian-step threshold baseline",
    "learned_1x25": "Learned seed + fixed refinement",
    "dls_previous_1x50": "Previous-state DLS",
    "trf_previous": "Previous-state trust-region reflective",
}
FAMILY_LABELS = {
    "smooth_fast_orientation_smooth": "Smooth--orientation--smooth",
    "regular_near_singular_regular": "Near-singular transition",
    "central_joint_limit_skim_return": "Joint-limit skim",
    "slow_high_curvature_high_speed_slow": "High-curvature/high-speed",
}
FAMILY_TOKENS = {
    "smooth_fast_orientation_smooth": "Smooth",
    "regular_near_singular_regular": "NearSingular",
    "central_joint_limit_skim_return": "JointLimit",
    "slow_high_curvature_high_speed_slow": "HighCurvature",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty source data: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float, digits: int = 2) -> str:
    """Format a self-contained percentage for TeX macros and table cells."""

    return f"{100.0 * float(value):.{digits}f}\\%"


def number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def tex_text(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


class EvidenceRegistry:
    """Record every consumed frozen file and compact source-group digests."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, Any]] = {}

    def register(self, path: Path, *, role: str) -> str:
        resolved = path.resolve()
        if OUTPUTS.resolve() not in resolved.parents:
            raise RuntimeError(f"evidence source is outside frozen outputs: {path}")
        relative = str(path.relative_to(ROOT))
        digest = file_sha256(path)
        size_bytes = path.stat().st_size
        previous = self.files.get(relative)
        if previous is not None:
            if (
                previous["sha256"] != digest
                or previous["size_bytes"] != size_bytes
            ):
                raise RuntimeError(f"source changed during extraction: {relative}")
            previous["roles"] = sorted({*previous["roles"], role})
        else:
            self.files[relative] = {
                "sha256": digest,
                "size_bytes": size_bytes,
                "roles": [role],
            }
        return relative

    def group(self, name: str, paths: Iterable[Path], *, role: str) -> None:
        members = sorted({self.register(path, role=role) for path in paths})
        digest = sha256()
        for relative in members:
            metadata = self.files[relative]
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(metadata["sha256"].encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata["size_bytes"]).encode("ascii"))
            digest.update(b"\n")
        self.groups[name] = {
            "role": role,
            "member_count": len(members),
            "members": members,
            "sha256": digest.hexdigest(),
        }


def load_development_role(
    registry: EvidenceRegistry, robot: str, role: str
) -> tuple[dict[str, np.ndarray], list[Path]]:
    role_root = OUTPUTS / "counterfactual_v4_bulk" / robot / "seed17" / role
    selection = role_root / "selection.npz"
    selection_manifest = role_root / "selection_manifest.json"
    source_paths = [selection, selection_manifest]
    registry.register(selection, role="development identity selection")
    registry.register(selection_manifest, role="development identity manifest")
    fields = {
        name: []
        for name in (
            "features",
            "query_sha256",
            "category",
            "expected_reachable",
            "continuity_feasible",
            "verified_success",
            "latency_samples_ns",
            "function_evaluations",
        )
    }
    chunks = sorted((role_root / "chunks").glob("chunk_*"))
    if not chunks:
        raise FileNotFoundError(f"no development chunks: {role_root}")
    for chunk in chunks:
        labels_path = chunk / "counterfactual_labels.npz"
        manifest_path = chunk / "chunk_manifest.json"
        source_paths.extend((labels_path, manifest_path))
        registry.register(labels_path, role="action-complete development labels")
        registry.register(manifest_path, role="action-complete chunk manifest")
        manifest = load_json(manifest_path)
        if bool(manifest.get("test_data_loaded", True)):
            raise RuntimeError(f"development chunk accessed test data: {manifest_path}")
        with np.load(labels_path, allow_pickle=False) as labels:
            if tuple(str(value) for value in labels["action_names"][:3]) != ENTRIES:
                raise RuntimeError(f"entry schema mismatch: {labels_path}")
            if labels["latency_samples_ns"].shape[2] != 5:
                raise RuntimeError(f"development timing repeat mismatch: {labels_path}")
            for name in fields:
                fields[name].append(np.asarray(labels[name]).copy())
    result = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    with np.load(selection, allow_pickle=False) as selected:
        if not np.array_equal(result["query_sha256"], selected["query_sha256"]):
            raise RuntimeError(f"chunk order differs from selection: {role_root}")
        if not np.array_equal(result["category"], selected["category"]):
            raise RuntimeError(f"chunk categories differ from selection: {role_root}")
    return result, source_paths


def route_actions(prediction: Any, config: Mapping[str, Any]) -> np.ndarray:
    """Replay the exact frozen vectorized V4 policy decision rule."""

    success = np.asarray(prediction.verified_success_probability, dtype=np.float64)
    p95 = np.asarray(prediction.latency_p95_ms, dtype=np.float64)
    is_ood = np.asarray(prediction.is_ood, dtype=bool)
    eligible = (
        (success >= float(config["minimum_success_probability"]))
        & (p95 <= float(config["deadline_ms"]))
    )
    has_eligible = np.any(eligible, axis=1)
    selected = np.full(success.shape[0], 4, dtype=np.int64)  # defer
    reject = (
        ~is_ood
        & ~has_eligible
        & (
            np.asarray(prediction.fail_all_probability, dtype=np.float64)
            >= float(config["reject_probability"])
        )
    )
    selected[reject] = 3
    selectable = ~is_ood & has_eligible
    if np.any(selectable):
        masked = np.where(eligible, p95, np.inf)
        fastest = np.argmin(masked, axis=1)
        conservative = np.argmax(eligible, axis=1)
        rows = np.arange(success.shape[0])
        improvement = p95[rows, conservative] - p95[rows, fastest]
        chosen = np.where(
            improvement < float(config["latency_tie_margin_ms"]),
            conservative,
            fastest,
        )
        selected[selectable] = chosen[selectable]
    return selected


def development_evidence(
    registry: EvidenceRegistry,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    try:
        import torch
        from confik.counterfactual_v4.model import CounterfactualV4Predictor
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "Activate the project environment with torch and set PYTHONPATH=src. "
            "The script only replays the sealed development predictor; it never "
            "trains a model."
        ) from error
    torch.set_num_threads(1)

    tables: dict[str, list[dict[str, Any]]] = {
        "development_oracle_distribution.csv": [],
        "development_family_oracle.csv": [],
        "development_routing_metrics.csv": [],
        "development_predicted_observed_p95.csv": [],
    }
    snapshot: dict[str, Any] = {}
    source_paths: list[Path] = []
    training_metrics_path = OUTPUTS / "release_v4_candidate" / "training_metrics.json"
    registry.register(training_metrics_path, role="development prediction diagnostics")
    source_paths.append(training_metrics_path)
    training_metrics = load_json(training_metrics_path)

    for robot in ROBOTS:
        roles: dict[str, dict[str, np.ndarray]] = {}
        for role in DEVELOPMENT_ROLES:
            roles[role], paths = load_development_role(registry, robot, role)
            source_paths.extend(paths)
        combined = {
            name: np.concatenate([roles[role][name] for role in DEVELOPMENT_ROLES], axis=0)
            for name in roles[DEVELOPMENT_ROLES[0]]
        }
        query_count = int(combined["query_sha256"].size)
        if query_count != 20_000:
            raise RuntimeError(f"unexpected development count for {robot}")
        if np.unique(combined["query_sha256"]).size != query_count:
            raise RuntimeError(f"development roles overlap for {robot}")
        success_matrix = np.asarray(combined["verified_success"][:, :3], dtype=bool)
        if not np.all(success_matrix == success_matrix[:, :1]):
            raise RuntimeError(f"semantic entry success differs for {robot}")
        successful = success_matrix[:, 0]
        samples_ms = np.asarray(combined["latency_samples_ns"][:, :3], dtype=np.float64) / 1e6
        empirical_p50 = np.quantile(samples_ms, 0.50, axis=2)
        empirical_p95 = np.quantile(samples_ms, 0.95, axis=2)
        oracle = np.argmin(empirical_p95, axis=1)
        oracle_p95 = np.min(empirical_p95, axis=1)
        categories = np.asarray(combined["category"], dtype=str)
        success_count = int(np.sum(successful))

        oracle_entry: dict[str, dict[str, float | int]] = {}
        for index, entry in enumerate(ENTRIES):
            count = int(np.sum(oracle[successful] == index))
            row = {"robot": robot, "entry": entry, "count": count, "rate": count / success_count}
            tables["development_oracle_distribution.csv"].append(row)
            oracle_entry[entry] = {"count": count, "rate": row["rate"]}
        for family in sorted(np.unique(categories)):
            mask = successful & (categories == family)
            family_count = int(np.sum(mask))
            if not family_count:
                continue
            for index, entry in enumerate(ENTRIES):
                count = int(np.sum(oracle[mask] == index))
                tables["development_family_oracle.csv"].append(
                    {
                        "robot": robot,
                        "family": family,
                        "entry": entry,
                        "count": count,
                        "rate": count / family_count,
                        "family_success_count": family_count,
                    }
                )

        easy_gap = empirical_p95[:, 0] - oracle_p95
        best_fixed_index = int(np.argmin(np.mean(empirical_p95[successful], axis=0)))
        best_fixed_gap = empirical_p95[:, best_fixed_index] - oracle_p95
        fev = np.asarray(combined["function_evaluations"][:, :3], dtype=np.float64)
        oracle_fev = fev[np.arange(fev.shape[0]), oracle]
        p50_winner = np.argmin(empirical_p50, axis=1)
        hard_valid = successful & (categories == "hard_valid")
        metrics: dict[str, float | int | str] = {
            "query_count": query_count,
            "verified_success_count": success_count,
            "easy_to_oracle_gap_mean_ms": float(np.mean(easy_gap[successful])),
            "easy_to_oracle_gap_p95_ms": float(np.quantile(easy_gap[successful], 0.95)),
            "positive_easy_to_oracle_gap_rate": float(np.mean(easy_gap[successful] > 0.0)),
            "best_fixed_entry": ENTRIES[best_fixed_index],
            "best_fixed_to_oracle_gap_mean_ms": float(np.mean(best_fixed_gap[successful])),
            "easy_to_oracle_fev_mean": float(np.mean((fev[:, 0] - oracle_fev)[successful])),
            "p50_p95_winner_agreement_rate": float(
                np.mean(p50_winner[successful] == oracle[successful])
            ),
            "hard_valid_success_count": int(np.sum(hard_valid)),
            "hard_valid_oracle_hard_rate": float(np.mean(oracle[hard_valid] == 2)),
            "hard_valid_easy_to_oracle_gap_mean_ms": float(np.mean(easy_gap[hard_valid])),
        }

        policy = roles["policy_validation_queries"]
        model_path = OUTPUTS / "release_v4_candidate" / "models" / f"{robot}_seed17_predictor.pt"
        policy_path = OUTPUTS / "release_v4_locked" / robot / "v4_policy.json"
        registry.register(model_path, role="frozen development predictor")
        registry.register(policy_path, role="sealed V4 policy")
        source_paths.extend((model_path, policy_path))
        predictor = CounterfactualV4Predictor.load(model_path, device="cpu")
        prediction = predictor.predict(np.asarray(policy["features"], dtype=np.float64))
        policy_json = load_json(policy_path)
        selected = route_actions(prediction, policy_json["policy_config"])
        observed_counts = {
            action: int(np.sum(selected == index)) for index, action in enumerate(FINAL_ACTIONS)
        }
        if observed_counts != policy_json["selection_metrics"]["route_counts"]:
            raise RuntimeError(f"sealed route-count replay mismatch for {robot}")
        policy_success = np.asarray(policy["verified_success"][:, 0], dtype=bool)
        observed_p95 = np.quantile(
            np.asarray(policy["latency_samples_ns"][:, :3], dtype=np.float64) / 1e6,
            0.95,
            axis=2,
        )
        policy_oracle = np.argmin(observed_p95, axis=1)
        routed_success = policy_success & (selected < 3)
        row_index = np.arange(selected.size)
        regret = (
            observed_p95[row_index, np.minimum(selected, 2)]
            - observed_p95[row_index, policy_oracle]
        )[routed_success]
        metrics.update(
            {
                "policy_successful_non_abstained_count": int(np.sum(routed_success)),
                "oracle_agreement_rate": float(
                    np.mean(selected[routed_success] == policy_oracle[routed_success])
                ),
                "regret_within_0_15ms_rate": float(np.mean(regret <= 0.15)),
                "routing_regret_mean_ms": float(np.mean(regret)),
                "routing_regret_median_ms": float(np.median(regret)),
                "routing_regret_p95_ms": float(np.quantile(regret, 0.95)),
                **{f"route_count_{name}": count for name, count in observed_counts.items()},
            }
        )
        latency_report = training_metrics["robots"][robot]["latency_policy_validation"]
        for action_index, entry in enumerate(ENTRIES):
            metrics[f"raw_sample_p95_coverage_{entry}"] = float(
                latency_report[entry]["raw_sample_p95_coverage"]
            )
            metrics[f"predicted_p95_median_{entry}_ms"] = float(
                latency_report[entry]["predicted_p95_median_ms"]
            )
            metrics[f"observed_empirical_p95_median_{entry}_ms"] = float(
                np.median(observed_p95[:, action_index])
            )
            for query_index in range(selected.size):
                tables["development_predicted_observed_p95.csv"].append(
                    {
                        "robot": robot,
                        "query_sha256": str(policy["query_sha256"][query_index]),
                        "family": str(policy["category"][query_index]),
                        "entry": entry,
                        "predicted_p95_ms": float(
                            prediction.latency_p95_ms[query_index, action_index]
                        ),
                        "observed_empirical_p95_ms": float(
                            observed_p95[query_index, action_index]
                        ),
                        "selected_action": FINAL_ACTIONS[int(selected[query_index])],
                        "semantic_success": bool(policy_success[query_index]),
                    }
                )
        count_keys = {
            "query_count",
            "verified_success_count",
            "hard_valid_success_count",
            "policy_successful_non_abstained_count",
        }
        for key, value in metrics.items():
            unit = (
                "queries" if key in count_keys or key.startswith("route_count_")
                else "entry" if key == "best_fixed_entry"
                else "ms" if key.endswith("_ms")
                else "fraction" if key.endswith("_rate")
                else "FEV" if key.endswith("_mean")
                else "value"
            )
            policy_scope = key.startswith(
                ("policy_", "oracle_agreement", "regret_", "routing_", "route_count_", "raw_sample_", "predicted_", "observed_")
            )
            tables["development_routing_metrics.csv"].append(
                {
                    "robot": robot,
                    "metric": key,
                    "value": value,
                    "unit": unit,
                    "scope": (
                        "policy_validation_development"
                        if policy_scope
                        else "all_action_complete_development_roles"
                    ),
                }
            )
        snapshot[robot] = {
            "all_roles": {
                "query_count": query_count,
                "verified_success_count": success_count,
                "timing_repeats": int(samples_ms.shape[2]),
                "oracle_entry": oracle_entry,
                "diagnostics": metrics,
            },
            "policy_validation": {
                "query_count": int(selected.size),
                "route_counts": observed_counts,
                "frozen_model_sha256": file_sha256(model_path),
                "frozen_policy_sha256": file_sha256(policy_path),
            },
        }
    registry.group(
        "action_complete_development",
        source_paths,
        role="development-only action-complete labels and sealed policy replay",
    )
    return snapshot, tables


def iter_formal_point_records(
    registry: EvidenceRegistry, robot: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    root = OUTPUTS / ".test_v4_seed17.incomplete" / robot / "checkpoints"
    files = sorted(root.glob("id_points/query_*/records.jsonl.gz"))
    files.extend(sorted(root.glob("ood_points/query_*/records.jsonl.gz")))
    if not files:
        raise FileNotFoundError(f"missing frozen point records for {robot}")
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in files:
        manifest = path.with_name("checkpoint_manifest.json")
        registry.register(path, role="primary fresh point-query raw records")
        registry.register(manifest, role="point checkpoint manifest")
        sources.extend((path, manifest))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows, sources


def summarize_point_records(
    rows: Sequence[Mapping[str, Any]], *, feasible: bool
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if bool(row["expected_reachable"] and row["continuity_feasible"]) is feasible
    ]
    output: list[dict[str, Any]] = []
    for method in POINT_METHODS:
        method_rows = [row for row in selected if row["method"] == method]
        if not method_rows:
            raise RuntimeError(f"formal point method is absent: {method}")
        latencies = np.asarray(
            [
                sample / 1e6
                for row in method_rows
                for sample in row["latency_repeats_ns"]
            ],
            dtype=np.float64,
        )
        base = {
            "robot": str(method_rows[0]["robot"]),
            "method": method,
            "method_label": METHOD_LABELS[method],
            "query_count": len(method_rows),
            "verified_success_rate": float(
                np.mean([row["verified_success"] for row in method_rows])
            ),
            "mean_fev": float(
                np.mean([row["function_evaluations"] for row in method_rows])
            ),
            "p50_ms": float(np.quantile(latencies, 0.50)),
            "p95_ms": float(np.quantile(latencies, 0.95)),
            "p99_ms": float(np.quantile(latencies, 0.99)),
            "fallback_rate": float(
                np.mean([row["fallback_used"] for row in method_rows])
            ),
            "accepted_contract_violation_count": int(
                sum(
                    bool(row["accepted"]) and bool(row["contract_violations"])
                    for row in method_rows
                )
            ),
        }
        if feasible:
            output.append(base)
        else:
            output.append(
                {
                    **base,
                    "command_reject_rate": float(
                        np.mean(
                            [row.get("decision_action") == "reject" for row in method_rows]
                        )
                    ),
                    "total_fev": int(
                        sum(int(row["function_evaluations"]) for row in method_rows)
                    ),
                }
            )
    return output


def point_evidence(
    registry: EvidenceRegistry,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    aggregate_path = (
        OUTPUTS / "test_v4_aggregate_repair_v1" / "aggregate_summary_v4.json"
    )
    gate_path = OUTPUTS / "test_v4_aggregate_repair_v1" / "paper_gate_v4.json"
    holm_path = OUTPUTS / "test_v4_aggregate_repair_v1" / "joint_holm_v4.json"
    attestation_path = (
        OUTPUTS
        / "test_v4_aggregate_repair_v1_attestation_v1"
        / "attestation_final_manifest.json"
    )
    fixed_sources = [aggregate_path, gate_path, holm_path, attestation_path]
    for path, role in (
        (aggregate_path, "authoritative repaired V4 point-query aggregate"),
        (gate_path, "formal point-query gate"),
        (holm_path, "joint confirmatory Holm family"),
        (attestation_path, "independent aggregation-repair attestation"),
    ):
        registry.register(path, role=role)
    aggregate = load_json(aggregate_path)
    gate = load_json(gate_path)
    point_rows: list[dict[str, Any]] = []
    rejectable_rows: list[dict[str, Any]] = []
    sources = list(fixed_sources)
    snapshot: dict[str, Any] = {
        "primary_training_seed": int(gate["primary_training_seed"]),
        "sensitivity_training_seeds": [
            int(seed) for seed in gate["sensitivity_training_seeds"]
        ],
        "test_set_retuning_performed": bool(gate["test_set_retuning_performed"]),
        "joint_holm_gate_pass": bool(gate["joint_holm_gate_pass"]),
        "paper_gate_pass": bool(gate["both_robot_gates_pass"]),
        "robots": {},
    }
    for robot in ROBOTS:
        raw, raw_sources = iter_formal_point_records(registry, robot)
        sources.extend(raw_sources)
        feasible = summarize_point_records(raw, feasible=True)
        rejectable = summarize_point_records(raw, feasible=False)
        point_rows.extend(feasible)
        rejectable_rows.extend(rejectable)
        claim = aggregate["primary"][robot]["claim_gate"]
        for method, aggregate_key in (
            ("fixed_robust_cascade", "fixed"),
            ("proposed_v4", "proposed_v4"),
        ):
            row = next(item for item in feasible if item["method"] == method)
            expected_latency = claim["feasible_latency_ms"][aggregate_key]
            expected_metrics = claim[f"{aggregate_key}_metrics"]
            strata = ("id_points_feasible", "ood_points_feasible")
            count = sum(int(expected_metrics[stratum]["count"]) for stratum in strata)
            expected_success = sum(
                int(expected_metrics[stratum]["count"])
                * (1.0 - float(expected_metrics[stratum]["verified_failure_rate"]))
                for stratum in strata
            ) / count
            expected_fev = sum(
                int(expected_metrics[stratum]["count"])
                * float(expected_metrics[stratum]["function_evaluations"]["mean"])
                for stratum in strata
            ) / count
            checks = {
                "query_count": row["query_count"] == count,
                "success": np.isclose(row["verified_success_rate"], expected_success),
                "mean_fev": np.isclose(row["mean_fev"], expected_fev),
                "p50": np.isclose(row["p50_ms"], expected_latency["p50"]),
                "p95": np.isclose(row["p95_ms"], expected_latency["p95"]),
                "p99": np.isclose(row["p99_ms"], expected_latency["p99"]),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"raw/aggregate point mismatch: {robot}/{method}: {checks}"
                )
        fixed = next(
            item for item in feasible if item["method"] == "fixed_robust_cascade"
        )
        proposed = next(item for item in feasible if item["method"] == "proposed_v4")
        reject_fixed = next(
            item
            for item in rejectable
            if item["method"] == "fixed_robust_cascade"
        )
        reject_proposed = next(
            item for item in rejectable if item["method"] == "proposed_v4"
        )
        abstention = claim["ood_and_abstention"]
        fixed_feasible_raw = [
            row
            for row in raw
            if row["method"] == "fixed_robust_cascade"
            and bool(row["expected_reachable"] and row["continuity_feasible"])
        ]
        id_feasible_count = sum(
            str(row["role"]) == "id_points" for row in fixed_feasible_raw
        )
        ood_feasible_count = sum(
            str(row["role"]) == "ood_points" for row in fixed_feasible_raw
        )
        primary_repeat_counts = {
            len(row["latency_repeats_ns"])
            for row in raw
            if row["method"] in {"fixed_robust_cascade", "proposed_v4"}
        }
        if len(primary_repeat_counts) != 1:
            raise RuntimeError(f"point timing repeats differ for {robot}")
        primary_timing_repeats = primary_repeat_counts.pop()
        for row in rejectable:
            row["fev_avoided_fraction_vs_fixed"] = float(
                1.0 - row["total_fev"] / reject_fixed["total_fev"]
            )
            is_cghik = row["method"] == "proposed_v4"
            row["formal_reject_recall"] = (
                float(abstention["infeasible_command_reject_recall"])
                if is_cghik
                else ""
            )
            row["ood_auroc"] = float(abstention["ood_auroc"]) if is_cghik else ""
            row["ood_auprc"] = float(abstention["ood_auprc"]) if is_cghik else ""
            row["defer_recovery_success_rate"] = (
                float(abstention["defer_recovery_success_rate"])
                if is_cghik
                else ""
            )
        snapshot["robots"][robot] = {
            "feasible_query_count": int(fixed["query_count"]),
            "id_feasible_query_count": int(id_feasible_count),
            "ood_feasible_query_count": int(ood_feasible_count),
            "infeasible_query_count": int(reject_fixed["query_count"]),
            "timing_repeats_primary_methods": int(primary_timing_repeats),
            "fixed_robust_cascade": fixed,
            "cghik": proposed,
            "ratios_vs_fixed": {
                "mean_fev": proposed["mean_fev"] / fixed["mean_fev"],
                "p50_latency": proposed["p50_ms"] / fixed["p50_ms"],
                "p95_latency": float(claim["feasible_p95_latency_ratio"]),
                "p99_latency": float(claim["feasible_p99_latency_ratio"]),
                "fallback": proposed["fallback_rate"] / fixed["fallback_rate"],
            },
            "rejectable": {
                "fixed": reject_fixed,
                "cghik": reject_proposed,
                "reject_recall": float(
                    abstention["infeasible_command_reject_recall"]
                ),
                "fev_avoided_fraction": float(
                    abstention[
                        "infeasible_function_evaluations_avoided_fraction_vs_fixed"
                    ]
                ),
            },
            "ood_and_defer": {
                "ood_auroc": float(abstention["ood_auroc"]),
                "ood_auprc": float(abstention["ood_auprc"]),
                "defer_count_points": int(abstention["defer_count_points"]),
                "defer_semantic_match_rate": float(
                    abstention["defer_fixed_semantic_match_rate"]
                ),
                "defer_recovery_success_rate": float(
                    abstention["defer_recovery_success_rate"]
                ),
            },
            "formal_robot_gate_pass": bool(claim["formal_gate_pass"]),
        }
    registry.group(
        "fresh_point_query_formal",
        sources,
        role=(
            "sealed fresh point-query evidence: primary raw records, authoritative "
            "aggregate, and independent attestation"
        ),
    )
    return snapshot, {
        "point_formal_results.csv": point_rows,
        "point_rejectable_results.csv": rejectable_rows,
    }


def trajectory_evidence(
    registry: EvidenceRegistry,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = OUTPUTS / "fresh_transition_v4_test"
    paths = {
        name: root / name
        for name in (
            "main_table.json",
            "family_table.json",
            "trajectory_table.json",
            "completion_uids.json",
            "final_gate.json",
            "run_manifest.json",
            "preregistration_seal.json",
            "fresh_transition_v4_test.yaml",
        )
    }
    for name, path in paths.items():
        registry.register(path, role=f"fresh transition-rich {name}")
    main_source = load_json(paths["main_table.json"])
    family_source = load_json(paths["family_table.json"])
    gate = load_json(paths["final_gate.json"])
    completion = load_json(paths["completion_uids.json"])
    trajectory_config = yaml.safe_load(
        paths["fresh_transition_v4_test.yaml"].read_text(encoding="utf-8")
    )
    fresh_data = trajectory_config["fresh_data"]
    trajectory_count_per_robot = int(fresh_data["trajectories_per_robot"])
    trajectories_per_family = int(fresh_data["trajectories_per_family"])
    frames_per_trajectory = int(fresh_data["frames_per_trajectory"])
    dt_seconds = float(fresh_data["dt"])
    main_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    snapshot: dict[str, Any] = {
        "protocol": gate["protocol"],
        "status": gate["status"],
        "results_used_for_tuning": False,
        "trajectory_count_per_robot": trajectory_count_per_robot,
        "trajectories_per_family": trajectories_per_family,
        "frames_per_trajectory": frames_per_trajectory,
        "dt_seconds": dt_seconds,
        "gate_thresholds": {
            "aggregate_cumulative_latency_ratio_max": float(
                trajectory_config["final_gate"][
                    "aggregate_cumulative_latency_ratio_max"
                ]
            ),
            "mean_fev_ratio_max": float(
                trajectory_config["final_gate"]["mean_fev_ratio_max"]
            ),
            "p95_latency_ratio_max": float(
                trajectory_config["final_gate"]["p95_latency_ratio_max"]
            ),
            "p99_latency_ratio_max": float(
                trajectory_config["final_gate"]["p99_latency_ratio_max"]
            ),
        },
        "robots": {},
    }
    for robot in ROBOTS:
        hard = next(
            row
            for row in main_source
            if row["robot"] == robot and row["method"] == "always_hard"
        )
        proposed = next(
            row
            for row in main_source
            if row["robot"] == robot
            and row["method"] == "counterfactual_cghik_v4"
        )
        if any(
            int(row["trajectory_count"]) != trajectory_count_per_robot
            or int(row["frame_count"])
            != trajectory_count_per_robot * frames_per_trajectory
            for row in main_source
            if row["robot"] == robot
        ):
            raise RuntimeError(f"trajectory protocol count mismatch for {robot}")
        if any(
            int(row["trajectory_count"]) != trajectories_per_family
            for row in family_source
            if row["robot"] == robot
        ):
            raise RuntimeError(f"trajectory family count mismatch for {robot}")
        for source in [row for row in main_source if row["robot"] == robot]:
            main_rows.append(
                {
                    "robot": robot,
                    "method": source["method"],
                    "method_label": METHOD_LABELS[source["method"]],
                    "trajectory_count": int(source["trajectory_count"]),
                    "whole_trajectory_completion_count": int(
                        source["whole_trajectory_completion_count"]
                    ),
                    "whole_trajectory_completion_rate": float(
                        source["whole_trajectory_completion_rate"]
                    ),
                    "frame_count": int(source["frame_count"]),
                    "frame_verified_success_rate": float(
                        source["frame_verified_success_rate"]
                    ),
                    "total_cumulative_latency_seconds": float(
                        source["total_cumulative_latency_seconds"]
                    ),
                    "trajectory_cumulative_latency_mean_ms": float(
                        source["trajectory_cumulative_latency_mean_ms"]
                    ),
                    "trajectory_cumulative_latency_median_ms": float(
                        source["trajectory_cumulative_latency_median_ms"]
                    ),
                    "trajectory_cumulative_latency_p95_ms": float(
                        source["trajectory_cumulative_latency_p95_ms"]
                    ),
                    "frame_p50_latency_ms": float(source["frame_p50_latency_ms"]),
                    "frame_p95_latency_ms": float(source["frame_p95_latency_ms"]),
                    "frame_p99_latency_ms": float(source["frame_p99_latency_ms"]),
                    "mean_fev": float(source["mean_fev"]),
                    "fallback_rate": float(source["fallback_rate"]),
                    "accepted_contract_violation_count": int(
                        source["accepted_contract_violation_count"]
                    ),
                    "cumulative_latency_ratio_vs_hard": float(
                        source["total_cumulative_latency_seconds"]
                        / hard["total_cumulative_latency_seconds"]
                    ),
                    "mean_fev_ratio_vs_hard": float(
                        source["mean_fev"] / hard["mean_fev"]
                    ),
                    "p50_ratio_vs_hard": float(
                        source["frame_p50_latency_ms"]
                        / hard["frame_p50_latency_ms"]
                    ),
                    "p95_ratio_vs_hard": float(
                        source["frame_p95_latency_ms"]
                        / hard["frame_p95_latency_ms"]
                    ),
                    "p99_ratio_vs_hard": float(
                        source["frame_p99_latency_ms"]
                        / hard["frame_p99_latency_ms"]
                    ),
                }
            )
        gate_ratios = gate["robots"][robot]["ratios_vs_always_hard"]
        for key, field in (
            ("aggregate_cumulative_latency", "total_cumulative_latency_seconds"),
            ("mean_fev", "mean_fev"),
            ("p50_latency", "frame_p50_latency_ms"),
            ("p95_latency", "frame_p95_latency_ms"),
            ("p99_latency", "frame_p99_latency_ms"),
        ):
            observed = float(proposed[field]) / float(hard[field])
            if not np.isclose(observed, float(gate_ratios[key])):
                raise RuntimeError(f"trajectory gate ratio mismatch: {robot}/{key}")
        snapshot["robots"][robot] = {
            "always_hard": hard,
            "cghik": proposed,
            "ratios_vs_always_hard": gate_ratios,
            "reductions_vs_always_hard": {
                key: 1.0 - float(value) for key, value in gate_ratios.items()
            },
            "completion_comparison": {
                "lost_trajectory_uids": completion["robots"][robot][
                    "v4_lost_vs_always_hard_trajectory_uids"
                ],
                "gained_trajectory_uids": completion["robots"][robot][
                    "v4_gained_vs_always_hard_trajectory_uids"
                ],
            },
            "gate_pass": bool(gate["robots"][robot]["pass"]),
        }
        hard_families = {
            row["family"]: row
            for row in family_source
            if row["robot"] == robot and row["method"] == "always_hard"
        }
        for source in [row for row in family_source if row["robot"] == robot]:
            comparator = hard_families[source["family"]]
            latency_ratio = (
                source["total_cumulative_latency_seconds"]
                / comparator["total_cumulative_latency_seconds"]
            )
            fev_ratio = source["mean_fev"] / comparator["mean_fev"]
            family_rows.append(
                {
                    "robot": robot,
                    "family": source["family"],
                    "family_label": FAMILY_LABELS[source["family"]],
                    "method": source["method"],
                    "method_label": METHOD_LABELS[source["method"]],
                    "trajectory_count": int(source["trajectory_count"]),
                    "whole_trajectory_completion_count": int(
                        source["whole_trajectory_completion_count"]
                    ),
                    "whole_trajectory_completion_rate": float(
                        source["whole_trajectory_completion_rate"]
                    ),
                    "total_cumulative_latency_seconds": float(
                        source["total_cumulative_latency_seconds"]
                    ),
                    "mean_fev": float(source["mean_fev"]),
                    "fallback_rate": float(source["fallback_rate"]),
                    "frame_verified_success_rate": float(
                        source["frame_verified_success_rate"]
                    ),
                    "accepted_contract_violation_count": int(
                        source["accepted_contract_violation_count"]
                    ),
                    "completion_difference_vs_hard": int(
                        source["whole_trajectory_completion_count"]
                        - comparator["whole_trajectory_completion_count"]
                    ),
                    "cumulative_latency_ratio_vs_hard": float(latency_ratio),
                    "cumulative_latency_change_vs_hard": float(latency_ratio - 1.0),
                    "mean_fev_ratio_vs_hard": float(fev_ratio),
                    "mean_fev_change_vs_hard": float(fev_ratio - 1.0),
                }
            )
        snapshot["robots"][robot]["family_results"] = {
            family: {
                method: next(
                    row
                    for row in family_rows
                    if row["robot"] == robot
                    and row["family"] == family
                    and row["method"] == method
                )
                for method in TRAJECTORY_METHODS
            }
            for family in FAMILY_LABELS
        }

    # Deterministic illustrative case: first frozen-order trajectory gained by
    # CG-HIK versus hard entry. It is descriptive, not an inferential sample.
    representative_robot = "panda"
    raw_path = root / f"{representative_robot}_raw_records.npz"
    registry.register(raw_path, role="fresh representative trajectory time series")
    with np.load(raw_path, allow_pickle=False) as raw:
        method_names = [str(value) for value in raw["method_names"]]
        trajectory_order = [str(value) for value in raw["trajectory_order"]]
        gained = set(
            snapshot["robots"][representative_robot]["completion_comparison"][
                "gained_trajectory_uids"
            ]
        )
        candidates = [uid for uid in trajectory_order if uid in gained]
        if candidates:
            representative_uid = candidates[0]
            rule = "first frozen-order trajectory gained by CG-HIK versus hard entry"
        else:
            representative_uid = next(
                uid
                for uid in trajectory_order
                if str(
                    raw["category"][
                        np.flatnonzero(raw["trajectory_uid"] == uid)[0]
                    ]
                )
                == "regular_near_singular_regular"
            )
            rule = "first frozen-order near-singular trajectory"
        frame_indices = np.flatnonzero(raw["trajectory_uid"] == representative_uid)
        timeseries_rows: list[dict[str, Any]] = []
        for method_index, method in enumerate(method_names):
            for row_index in frame_indices:
                timeseries_rows.append(
                    {
                        "robot": representative_robot,
                        "trajectory_uid": representative_uid,
                        "selection_rule": rule,
                        "family": str(raw["category"][row_index]),
                        "frame": int(raw["time_index"][row_index]),
                        "time_seconds": (
                            float(raw["time_index"][row_index]) * dt_seconds
                        ),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "route": str(raw["entry_action"][row_index, method_index]),
                        "latency_ms": float(
                            raw["latency_ns"][row_index, method_index]
                        )
                        / 1e6,
                        "function_evaluations": int(
                            raw["function_evaluations"][row_index, method_index]
                        ),
                        "accepted": bool(raw["accepted"][row_index, method_index]),
                        "fallback_used": bool(
                            raw["fallback_used"][row_index, method_index]
                        ),
                    }
                )
    snapshot["representative_trajectory"] = {
        "robot": representative_robot,
        "trajectory_uid": representative_uid,
        "selection_rule": rule,
        "family": timeseries_rows[0]["family"],
    }
    registry.group(
        "fresh_transition_trajectory",
        [*paths.values(), raw_path],
        role="one-shot fresh transition-rich evaluation",
    )
    return snapshot, {
        "trajectory_main_results.csv": main_rows,
        "trajectory_family_results.csv": family_rows,
        "trajectory_representative_timeseries.csv": timeseries_rows,
    }


def solver_configuration(
    registry: EvidenceRegistry,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = OUTPUTS / "release_v3_locked" / "release_config.yaml"
    panda_spec_path = (
        OUTPUTS / "release_v4_locked" / "panda" / "v4_runtime_spec.json"
    )
    ur_spec_path = (
        OUTPUTS / "release_v4_locked" / "ur5e" / "v4_runtime_spec.json"
    )
    for path, role in (
        (config_path, "frozen shared solver configuration"),
        (panda_spec_path, "sealed Panda exact V4 runtime"),
        (ur_spec_path, "sealed UR5e exact V4 runtime"),
    ):
        registry.register(path, role=role)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "paper_v2_config_snapshot"
    ]
    panda = load_json(panda_spec_path)
    ur5e = load_json(ur_spec_path)
    for key in ("model_config", "input", "policy_config"):
        if key == "model_config":
            compared = ("hidden_sizes", "epochs")
        elif key == "input":
            compared = ("feature_names",)
        else:
            compared = (
                "deadline_ms",
                "latency_tie_margin_ms",
                "minimum_success_probability",
                "reject_probability",
            )
        for field in compared:
            if panda[key][field] != ur5e[key][field]:
                raise RuntimeError(f"robot runtime specification differs: {key}.{field}")
    values = {
        "dt_seconds": float(config["data"]["dt"]),
        "seed_ensemble_members": int(config["seed_model"]["members"]),
        "seed_model_hidden_layers": len(config["seed_model"]["hidden_sizes"]),
        "seed_model_hidden_width": int(config["seed_model"]["hidden_sizes"][0]),
        "easy_iterations": int(config["cascade"]["easy_iterations"]),
        "medium_iterations": int(config["cascade"]["medium_iterations"]),
        "hard_iterations": int(config["cascade"]["hard_iterations"]),
        "hard_learned_candidates": int(
            config["cascade"]["hard_learned_candidates"]
        ),
        "fallback_seeds": int(config["cascade"]["fallback_seed_count"]),
        "fallback_max_fev": int(config["fallback"]["max_function_evaluations"]),
        "position_tolerance_m": float(config["verifier"]["position_tolerance"]),
        "orientation_tolerance_rad": float(
            config["verifier"]["orientation_tolerance"]
        ),
        "joint_limit_tolerance_rad": float(
            config["verifier"]["joint_limit_tolerance"]
        ),
        "velocity_tolerance_rad": float(
            config["verifier"]["velocity_tolerance"]
        ),
        "model_input_features": len(panda["input"]["feature_names"]),
        "model_hidden_layers": len(panda["model_config"]["hidden_sizes"]),
        "model_hidden_width": int(panda["model_config"]["hidden_sizes"][0]),
        "model_hidden_sizes": "x".join(
            str(value) for value in panda["model_config"]["hidden_sizes"]
        ),
        "model_epochs": int(panda["model_config"]["epochs"]),
        "deadline_ms": float(panda["policy_config"]["deadline_ms"]),
        "minimum_success_probability": float(
            panda["policy_config"]["minimum_success_probability"]
        ),
        "reject_probability": float(panda["policy_config"]["reject_probability"]),
        "latency_tie_margin_ms": float(
            panda["policy_config"]["latency_tie_margin_ms"]
        ),
    }
    if len(set(config["seed_model"]["hidden_sizes"])) != 1:
        raise RuntimeError("seed proposal hidden widths unexpectedly differ")
    if len(set(panda["model_config"]["hidden_sizes"])) != 1:
        raise RuntimeError("routing predictor hidden widths unexpectedly differ")
    rows = [
        {
            "component": "Easy entry",
            "initialization": "Previous state",
            "budget_or_rule": f"{values['easy_iterations']} DLS iteration",
        },
        {
            "component": "Medium entry",
            "initialization": "Learned candidate",
            "budget_or_rule": f"{values['medium_iterations']} DLS iteration",
        },
        {
            "component": "Hard entry",
            "initialization": (
                f"{values['hard_learned_candidates']} learned candidate + "
                "previous state"
            ),
            "budget_or_rule": (
                f"{values['hard_iterations']} DLS iterations per seed"
            ),
        },
        {
            "component": "Robust fallback",
            "initialization": (
                "Previous + learned + up to "
                f"{values['fallback_seeds']} retrieved"
            ),
            "budget_or_rule": (
                f"TRF, max {values['fallback_max_fev']} FEV per seed"
            ),
        },
        {
            "component": "Verifier",
            "initialization": "Shared for every method",
            "budget_or_rule": (
                f"{values['position_tolerance_m']:.3g} m / "
                f"{values['orientation_tolerance_rad']:.4g} rad"
            ),
        },
        {
            "component": "Routing predictor",
            "initialization": (
                f"{values['model_input_features']} features; "
                f"{values['model_hidden_sizes']} MLP"
            ),
            "budget_or_rule": "shared success + action-specific P50/P95",
        },
    ]
    registry.group(
        "frozen_solver_and_policy_configuration",
        (config_path, panda_spec_path, ur_spec_path),
        role="shared solver portfolio and sealed exact V4 policy",
    )
    return values, rows


def macro(lines: list[str], name: str, value: str) -> None:
    if not name.isalpha():
        raise ValueError(f"TeX command names must contain letters only: {name}")
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def build_macros(
    development: Mapping[str, Any],
    points: Mapping[str, Any],
    trajectories: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
    solver: Mapping[str, Any],
) -> list[str]:
    lines = ["% Generated from frozen evidence; do not edit by hand."]
    if not np.isclose(float(trajectories["dt_seconds"]), solver["dt_seconds"]):
        raise RuntimeError("shared and fresh-trajectory control intervals differ")
    development_counts = [
        int(development[robot]["all_roles"]["query_count"]) for robot in ROBOTS
    ]
    development_timing_repeats = [
        int(development[robot]["all_roles"]["timing_repeats"])
        for robot in ROBOTS
    ]
    feasible_counts = [
        int(points["robots"][robot]["feasible_query_count"]) for robot in ROBOTS
    ]
    id_feasible_counts = [
        int(points["robots"][robot]["id_feasible_query_count"])
        for robot in ROBOTS
    ]
    ood_feasible_counts = [
        int(points["robots"][robot]["ood_feasible_query_count"])
        for robot in ROBOTS
    ]
    infeasible_counts = [
        int(points["robots"][robot]["infeasible_query_count"]) for robot in ROBOTS
    ]
    timing_repeats = [
        int(points["robots"][robot]["timing_repeats_primary_methods"])
        for robot in ROBOTS
    ]
    if not (
        len(set(development_counts))
        == len(set(development_timing_repeats))
        == len(set(feasible_counts))
        == len(set(id_feasible_counts))
        == len(set(ood_feasible_counts))
        == len(set(infeasible_counts))
        == len(set(timing_repeats))
        == 1
    ):
        raise RuntimeError("paper protocol counts unexpectedly differ by robot")
    macro(lines, "DevelopmentQueryCount", str(sum(development_counts)))
    macro(lines, "DevelopmentQueriesPerRobot", str(development_counts[0]))
    macro(lines, "DevelopmentTimingRepeats", str(development_timing_repeats[0]))
    macro(lines, "PointFeasibleQueriesPerRobot", str(feasible_counts[0]))
    macro(lines, "PointIDFeasibleQueriesPerRobot", str(id_feasible_counts[0]))
    macro(lines, "PointOODFeasibleQueriesPerRobot", str(ood_feasible_counts[0]))
    macro(lines, "PointInfeasibleQueriesPerRobot", str(infeasible_counts[0]))
    macro(lines, "PointTimingRepeats", str(timing_repeats[0]))
    macro(
        lines,
        "FreshTrajectoriesPerRobot",
        str(trajectories["trajectory_count_per_robot"]),
    )
    macro(
        lines,
        "FreshTrajectoriesPerFamily",
        str(trajectories["trajectories_per_family"]),
    )
    macro(
        lines,
        "FreshTrajectoryFrames",
        str(trajectories["frames_per_trajectory"]),
    )
    macro(
        lines,
        "ControlIntervalMilliseconds",
        number(1000 * solver["dt_seconds"], 0),
    )
    macro(lines, "SeedEnsembleMembers", str(solver["seed_ensemble_members"]))
    macro(lines, "SeedModelHiddenLayers", str(solver["seed_model_hidden_layers"]))
    macro(lines, "SeedModelHiddenWidth", str(solver["seed_model_hidden_width"]))
    macro(lines, "EasyIterations", str(solver["easy_iterations"]))
    macro(lines, "MediumIterations", str(solver["medium_iterations"]))
    macro(lines, "HardIterations", str(solver["hard_iterations"]))
    macro(lines, "HardLearnedCandidates", str(solver["hard_learned_candidates"]))
    macro(lines, "FallbackSeeds", str(solver["fallback_seeds"]))
    macro(lines, "FallbackMaximumFEV", str(solver["fallback_max_fev"]))
    macro(
        lines,
        "PositionTolerance",
        number(solver["position_tolerance_m"], 3),
    )
    macro(
        lines,
        "OrientationTolerance",
        number(solver["orientation_tolerance_rad"], 4),
    )
    macro(
        lines,
        "JointLimitTolerance",
        number(solver["joint_limit_tolerance_rad"], 9),
    )
    macro(
        lines,
        "VelocityTolerance",
        number(solver["velocity_tolerance_rad"], 4),
    )
    macro(lines, "RoutingFeatureCount", str(solver["model_input_features"]))
    macro(lines, "RoutingHiddenLayers", str(solver["model_hidden_layers"]))
    macro(lines, "RoutingHiddenWidth", str(solver["model_hidden_width"]))
    macro(lines, "RoutingHiddenSizes", tex_text(str(solver["model_hidden_sizes"])))
    macro(lines, "RoutingEpochs", str(solver["model_epochs"]))
    macro(
        lines,
        "RoutingDeadlineMilliseconds",
        number(solver["deadline_ms"], 0),
    )
    macro(
        lines,
        "RoutingSuccessThreshold",
        number(solver["minimum_success_probability"], 2),
    )
    macro(
        lines,
        "RoutingRejectThreshold",
        number(solver["reject_probability"], 2),
    )
    macro(
        lines,
        "RoutingTieMarginMilliseconds",
        number(solver["latency_tie_margin_ms"], 2),
    )
    macro(lines, "PointPrimaryTrainingSeed", str(points["primary_training_seed"]))
    macro(
        lines,
        "PointSensitivityTrainingSeeds",
        " and ".join(str(value) for value in points["sensitivity_training_seeds"]),
    )
    gate_thresholds = trajectories["gate_thresholds"]
    macro(
        lines,
        "TrajectoryCumulativeGateMaximum",
        number(gate_thresholds["aggregate_cumulative_latency_ratio_max"], 2),
    )
    macro(
        lines,
        "TrajectoryFEVGateMaximum",
        number(gate_thresholds["mean_fev_ratio_max"], 2),
    )
    macro(
        lines,
        "TrajectoryPNinetyFiveGateMaximum",
        number(gate_thresholds["p95_latency_ratio_max"], 2),
    )
    macro(
        lines,
        "TrajectoryPNinetyNineGateMaximum",
        number(gate_thresholds["p99_latency_ratio_max"], 2),
    )

    for robot in ROBOTS:
        prefix = ROBOT_MACRO_PREFIXES[robot]
        dev = development[robot]["all_roles"]
        diagnostic = dev["diagnostics"]
        macro(lines, f"{prefix}DevVerifiedQueries", str(dev["verified_success_count"]))
        for entry, token in zip(ENTRIES, ("Easy", "Medium", "Hard")):
            macro(
                lines,
                f"{prefix}DevOracle{token}",
                percent(dev["oracle_entry"][entry]["rate"]),
            )
        macro(
            lines,
            f"{prefix}DevEasyOracleGap",
            number(diagnostic["easy_to_oracle_gap_mean_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevEasyOracleGapMean",
            number(diagnostic["easy_to_oracle_gap_mean_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevBestFixedOracleGap",
            number(diagnostic["best_fixed_to_oracle_gap_mean_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevHardOracleGapMean",
            number(diagnostic["best_fixed_to_oracle_gap_mean_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevOracleAgreement",
            percent(diagnostic["oracle_agreement_rate"]),
        )
        macro(
            lines,
            f"{prefix}DevRoutingRegretMean",
            number(diagnostic["routing_regret_mean_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevRoutingRegretMedian",
            number(diagnostic["routing_regret_median_ms"]),
        )
        macro(
            lines,
            f"{prefix}DevRoutingRegretPNinetyFive",
            number(diagnostic["routing_regret_p95_ms"]),
        )

        point = points["robots"][robot]
        for label, values in (
            ("Fixed", point["fixed_robust_cascade"]),
            ("CG", point["cghik"]),
        ):
            macro(
                lines,
                f"{prefix}Point{label}Success",
                percent(values["verified_success_rate"]),
            )
            macro(lines, f"{prefix}Point{label}FEV", number(values["mean_fev"], 2))
            macro(lines, f"{prefix}Point{label}Median", number(values["p50_ms"]))
            macro(
                lines,
                f"{prefix}Point{label}PNinetyFive",
                number(values["p95_ms"]),
            )
            macro(
                lines,
                f"{prefix}Point{label}PNinetyNine",
                number(values["p99_ms"]),
            )
            macro(
                lines,
                f"{prefix}Point{label}Fallback",
                percent(values["fallback_rate"]),
            )
        ratios = point["ratios_vs_fixed"]
        macro(
            lines,
            f"{prefix}PointFEVReduction",
            percent(1.0 - ratios["mean_fev"]),
        )
        macro(
            lines,
            f"{prefix}PointMedianChange",
            percent(ratios["p50_latency"] - 1.0),
        )
        macro(
            lines,
            f"{prefix}PointPNinetyFiveReduction",
            percent(1.0 - ratios["p95_latency"]),
        )
        macro(
            lines,
            f"{prefix}PointPNinetyNineReduction",
            percent(1.0 - ratios["p99_latency"]),
        )
        macro(
            lines,
            f"{prefix}PointPNinetyFiveRatio",
            number(ratios["p95_latency"], 4),
        )
        macro(
            lines,
            f"{prefix}PointPNinetyNineRatio",
            number(ratios["p99_latency"], 4),
        )
        macro(
            lines,
            f"{prefix}RejectRecall",
            percent(point["rejectable"]["reject_recall"]),
        )
        macro(
            lines,
            f"{prefix}RejectFEVAvoided",
            percent(point["rejectable"]["fev_avoided_fraction"]),
        )
        macro(
            lines,
            f"{prefix}OODAUROC",
            number(point["ood_and_defer"]["ood_auroc"]),
        )
        macro(
            lines,
            f"{prefix}OODAUPRC",
            number(point["ood_and_defer"]["ood_auprc"]),
        )
        macro(
            lines,
            f"{prefix}DeferRecovery",
            percent(point["ood_and_defer"]["defer_recovery_success_rate"]),
        )

        trajectory = trajectories["robots"][robot]
        hard = trajectory["always_hard"]
        proposed = trajectory["cghik"]
        for label, values in (("Hard", hard), ("CG", proposed)):
            macro(
                lines,
                f"{prefix}Trajectory{label}CompletionCount",
                str(values["whole_trajectory_completion_count"]),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}CompletionRate",
                percent(values["whole_trajectory_completion_rate"]),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}CumulativeLatency",
                number(values["total_cumulative_latency_seconds"], 2),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}CumLatency",
                number(values["total_cumulative_latency_seconds"], 2),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}FEV",
                number(values["mean_fev"], 2),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}Median",
                number(values["frame_p50_latency_ms"]),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}PNinetyFive",
                number(values["frame_p95_latency_ms"]),
            )
            macro(
                lines,
                f"{prefix}Trajectory{label}PNinetyNine",
                number(values["frame_p99_latency_ms"]),
            )
        ratio = trajectory["ratios_vs_always_hard"]
        macro(
            lines,
            f"{prefix}TrajectoryCumulativeRatio",
            number(ratio["aggregate_cumulative_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryCumulativeReduction",
            percent(1.0 - ratio["aggregate_cumulative_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryCumReduction",
            percent(1.0 - ratio["aggregate_cumulative_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryFEVRatio",
            number(ratio["mean_fev"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryFEVReduction",
            percent(1.0 - ratio["mean_fev"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryMedianRatio",
            number(ratio["p50_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryMedianIncrease",
            percent(ratio["p50_latency"] - 1.0),
        )
        macro(
            lines,
            f"{prefix}TrajectoryMedianChange",
            percent(ratio["p50_latency"] - 1.0),
        )
        macro(
            lines,
            f"{prefix}TrajectoryPNinetyFiveRatio",
            number(ratio["p95_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryPNinetyNineRatio",
            number(ratio["p99_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryPNinetyFiveReduction",
            percent(1.0 - ratio["p95_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryPNinetyNineReduction",
            percent(1.0 - ratio["p99_latency"]),
        )
        macro(
            lines,
            f"{prefix}TrajectoryContractViolations",
            str(proposed["accepted_contract_violation_count"]),
        )

        for family, token in FAMILY_TOKENS.items():
            hard_family = next(
                row
                for row in family_rows
                if row["robot"] == robot
                and row["family"] == family
                and row["method"] == "always_hard"
            )
            cg_family = next(
                row
                for row in family_rows
                if row["robot"] == robot
                and row["family"] == family
                and row["method"] == "counterfactual_cghik_v4"
            )
            base = f"{prefix}{token}"
            macro(
                lines,
                f"{base}HardCompletion",
                str(hard_family["whole_trajectory_completion_count"]),
            )
            macro(
                lines,
                f"{base}CGCompletion",
                str(cg_family["whole_trajectory_completion_count"]),
            )
            macro(
                lines,
                f"{base}CumulativeChange",
                percent(cg_family["cumulative_latency_change_vs_hard"]),
            )
            macro(
                lines,
                f"{base}FEVChange",
                percent(cg_family["mean_fev_change_vs_hard"]),
            )
            macro(
                lines,
                f"{base}CumReduction",
                percent(1.0 - cg_family["cumulative_latency_ratio_vs_hard"]),
            )
            macro(
                lines,
                f"{base}FEVReduction",
                percent(1.0 - cg_family["mean_fev_ratio_vs_hard"]),
            )
    return lines


def build_table_rows(
    solver_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    solver_lines = ["% component & initialization & budget/rule"]
    for row in solver_rows:
        solver_lines.append(
            f"{tex_text(str(row['component']))} & "
            f"{tex_text(str(row['initialization']))} & "
            f"{tex_text(str(row['budget_or_rule']))} \\\\"
        )
    point_lines = [
        "% robot & method & success & mean FEV & P50 & P95 & P99 & fallback"
    ]
    for robot in ROBOTS:
        for method in ("fixed_robust_cascade", "proposed_v4"):
            row = next(
                item
                for item in point_rows
                if item["robot"] == robot and item["method"] == method
            )
            point_lines.append(
                f"{ROBOT_LABELS[robot]} & {tex_text(row['method_label'])} & "
                f"{percent(row['verified_success_rate'])} & "
                f"{number(row['mean_fev'], 2)} & {number(row['p50_ms'])} & "
                f"{number(row['p95_ms'])} & {number(row['p99_ms'])} & "
                f"{percent(row['fallback_rate'])} \\\\"
            )
    trajectory_lines = [
        "% robot & method & completion & cumulative seconds & mean FEV & P50 & P95 & P99 & violations"
    ]
    for robot in ROBOTS:
        for method in TRAJECTORY_METHODS:
            row = next(
                item
                for item in trajectory_rows
                if item["robot"] == robot and item["method"] == method
            )
            trajectory_lines.append(
                f"{ROBOT_LABELS[robot]} & {tex_text(row['method_label'])} & "
                f"{row['whole_trajectory_completion_count']}/{row['trajectory_count']} & "
                f"{number(row['total_cumulative_latency_seconds'], 2)} & "
                f"{number(row['mean_fev'], 2)} & "
                f"{number(row['frame_p50_latency_ms'])} & "
                f"{number(row['frame_p95_latency_ms'])} & "
                f"{number(row['frame_p99_latency_ms'])} & "
                f"{row['accepted_contract_violation_count']} \\\\"
            )
    return {
        "table_solver_rows.tex": "\n".join(solver_lines) + "\n",
        "table_point_rows.tex": "\n".join(point_lines) + "\n",
        "table_trajectory_rows.tex": "\n".join(trajectory_lines) + "\n",
    }


def add_headline_sources(snapshot: dict[str, Any]) -> None:
    descriptors = snapshot["source_descriptors"]
    point_sha = descriptors["fresh_point_query_formal"]["sha256"]
    trajectory_sha = descriptors["fresh_transition_trajectory"]["sha256"]
    development_sha = descriptors["action_complete_development"]["sha256"]
    source_manifest = snapshot["source_manifest"]

    def source_files(*relative_paths: str) -> dict[str, str]:
        return {
            path: source_manifest[path]["sha256"] for path in relative_paths
        }

    point_aggregate_path = (
        "outputs/test_v4_aggregate_repair_v1/aggregate_summary_v4.json"
    )
    trajectory_main_path = "outputs/fresh_transition_v4_test/main_table.json"
    trajectory_family_path = "outputs/fresh_transition_v4_test/family_table.json"
    trajectory_gate_path = "outputs/fresh_transition_v4_test/final_gate.json"
    rows: dict[str, Any] = {}
    for robot in ROBOTS:
        point = snapshot["fresh_point_query"]["robots"][robot]
        trajectory = snapshot["fresh_transition_trajectory"]["robots"][robot]
        development = snapshot["development"][robot]["all_roles"]
        rows[f"{robot}.development.oracle_entry_distribution"] = {
            "value": development["oracle_entry"],
            "source_group": "action_complete_development",
            "source_group_sha256": development_sha,
            "array_fields": ["verified_success", "latency_samples_ns", "category"],
            "source_path_patterns": [
                (
                    f"outputs/counterfactual_v4_bulk/{robot}/seed17/"
                    "{risk_train_queries,calibration_queries,policy_validation_queries}/"
                    "chunks/*/counterfactual_labels.npz"
                )
            ],
            "derivation": (
                "argmin empirical P95 over five observed repeats for each "
                "semantically successful query"
            ),
        }
        rows[f"{robot}.development.routing_regret_mean_ms"] = {
            "value": development["diagnostics"]["routing_regret_mean_ms"],
            "source_group": "action_complete_development",
            "source_group_sha256": development_sha,
            "array_fields": ["features", "verified_success", "latency_samples_ns"],
            "source_files": source_files(
                f"outputs/release_v4_candidate/models/{robot}_seed17_predictor.pt",
                f"outputs/release_v4_locked/{robot}/v4_policy.json",
            ),
            "source_path_patterns": [
                (
                    f"outputs/counterfactual_v4_bulk/{robot}/seed17/"
                    "policy_validation_queries/chunks/*/counterfactual_labels.npz"
                )
            ],
            "derivation": (
                "sealed predictor and sealed policy replay on policy-validation queries"
            ),
        }
        for method in ("fixed_robust_cascade", "cghik"):
            rows[f"{robot}.point.{method}"] = {
                "value": point[method],
                "source_group": "fresh_point_query_formal",
                "source_group_sha256": point_sha,
                "source_files": source_files(point_aggregate_path),
                "source_path_patterns": [
                    (
                        f"outputs/.test_v4_seed17.incomplete/{robot}/checkpoints/"
                        "{id_points,ood_points}/query_*/records.jsonl.gz"
                    )
                ],
                "json_path": f"primary.{robot}.claim_gate",
                "raw_fields": [
                    "verified_success",
                    "function_evaluations",
                    "latency_repeats_ns",
                    "fallback_used",
                    "contract_violations",
                ],
            }
        rows[f"{robot}.point.ood_and_defer"] = {
            "value": point["ood_and_defer"],
            "source_group": "fresh_point_query_formal",
            "source_group_sha256": point_sha,
            "source_files": source_files(point_aggregate_path),
            "json_path": f"primary.{robot}.claim_gate.ood_and_abstention",
        }
        rows[f"{robot}.point.rejectable"] = {
            "value": point["rejectable"],
            "source_group": "fresh_point_query_formal",
            "source_group_sha256": point_sha,
            "source_files": source_files(point_aggregate_path),
            "source_path_patterns": [
                (
                    f"outputs/.test_v4_seed17.incomplete/{robot}/checkpoints/"
                    "{id_points,ood_points}/query_*/records.jsonl.gz"
                )
            ],
            "json_path": f"primary.{robot}.claim_gate.ood_and_abstention",
            "raw_fields": [
                "decision_action",
                "function_evaluations",
                "expected_reachable",
                "continuity_feasible",
            ],
        }
        rows[f"{robot}.trajectory.cghik_vs_hard"] = {
            "value": {
                "hard_completion": trajectory["always_hard"][
                    "whole_trajectory_completion_count"
                ],
                "cghik_completion": trajectory["cghik"][
                    "whole_trajectory_completion_count"
                ],
                "ratios": trajectory["ratios_vs_always_hard"],
            },
            "source_group": "fresh_transition_trajectory",
            "source_group_sha256": trajectory_sha,
            "source_files": source_files(
                trajectory_main_path,
                trajectory_gate_path,
                "outputs/fresh_transition_v4_test/completion_uids.json",
            ),
            "json_paths": [
                f"main_table[robot={robot},method=always_hard]",
                f"main_table[robot={robot},method=counterfactual_cghik_v4]",
                f"final_gate.robots.{robot}.ratios_vs_always_hard",
            ],
        }
        for family in FAMILY_LABELS:
            family_values = trajectory["family_results"][family]
            rows[f"{robot}.trajectory.family.{family}"] = {
                "value": {
                    "always_hard": family_values["always_hard"],
                    "cghik": family_values["counterfactual_cghik_v4"],
                },
                "source_group": "fresh_transition_trajectory",
                "source_group_sha256": trajectory_sha,
                "source_files": source_files(trajectory_family_path),
                "json_paths": [
                    f"family_table[robot={robot},family={family},method=always_hard]",
                    (
                        f"family_table[robot={robot},family={family},"
                        "method=counterfactual_cghik_v4]"
                    ),
                ],
            }
    representative = snapshot["fresh_transition_trajectory"][
        "representative_trajectory"
    ]
    representative_raw_path = (
        f"outputs/fresh_transition_v4_test/{representative['robot']}_raw_records.npz"
    )
    rows["trajectory.representative_timeseries"] = {
        "value": representative,
        "source_group": "fresh_transition_trajectory",
        "source_group_sha256": trajectory_sha,
        "source_files": source_files(representative_raw_path),
        "array_fields": [
            "trajectory_uid",
            "trajectory_order",
            "category",
            "time_index",
            "method_names",
            "entry_action",
            "latency_ns",
            "function_evaluations",
            "accepted",
            "fallback_used",
        ],
        "derivation": representative["selection_rule"],
    }
    snapshot["headline_sources"] = rows


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    registry = EvidenceRegistry()

    solver, solver_rows = solver_configuration(registry)
    development, development_tables = development_evidence(registry)
    points, point_tables = point_evidence(registry)
    trajectories, trajectory_tables = trajectory_evidence(registry)
    source_tables: dict[str, list[dict[str, Any]]] = {
        "solver_portfolio_configuration.csv": solver_rows,
        **development_tables,
        **point_tables,
        **trajectory_tables,
    }
    for filename, rows in source_tables.items():
        write_csv(SOURCE_DATA / filename, rows)

    macros = build_macros(
        development,
        points,
        trajectories,
        trajectory_tables["trajectory_family_results.csv"],
        solver,
    )
    numbers_path = GENERATED / "paper_numbers.tex"
    numbers_path.write_text("\n".join(macros) + "\n", encoding="utf-8")
    table_payloads = build_table_rows(
        solver_rows,
        point_tables["point_formal_results.csv"],
        trajectory_tables["trajectory_main_results.csv"],
    )
    for filename, payload in table_payloads.items():
        (GENERATED / filename).write_text(payload, encoding="utf-8")

    generated_paths = [
        *(SOURCE_DATA / filename for filename in source_tables),
        numbers_path,
        *(GENERATED / filename for filename in table_payloads),
    ]
    snapshot: dict[str, Any] = {
        "schema_version": 2,
        "title": (
            "CG-HIK: Query-Adaptive Tail-Latency Routing for Kinematically "
            "Verified Online Inverse Kinematics"
        ),
        "central_claim": (
            "Learning allocates solver effort per query; numerical geometry "
            "generates joint commands; deterministic verification governs acceptance."
        ),
        "evidence_policy": {
            "new_experiments_run": False,
            "models_retrained": False,
            "solver_or_verifier_called": False,
            "formal_results_used_for_tuning": False,
            "v5_v6_v7_role": (
                "development-only mechanism analysis; excluded from the main "
                "method and formal evidence"
            ),
        },
        "solver_and_policy": solver,
        "development": development,
        "fresh_point_query": points,
        "fresh_transition_trajectory": trajectories,
        "source_descriptors": dict(sorted(registry.groups.items())),
        "source_manifest": dict(sorted(registry.files.items())),
        "generated_artifacts": {
            str(path.relative_to(ROOT)): {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(generated_paths)
        },
    }
    add_headline_sources(snapshot)
    (GENERATED / "evidence_snapshot.json").write_text(
        stable_json(snapshot), encoding="utf-8"
    )
    print(
        "Wrote final-paper evidence: "
        f"{len(source_tables)} CSV files, {len(macros) - 1} TeX macros, "
        f"{len(registry.files)} hashed frozen sources."
    )


if __name__ == "__main__":
    main()
