from __future__ import annotations

import numpy as np


def paired_bootstrap_difference(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    rng = np.random.default_rng(seed)
    differences = proposed - baseline
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = np.mean(differences[indices], axis=1)
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_lower": float(np.percentile(bootstrap, 2.5)),
        "ci_upper": float(np.percentile(bootstrap, 97.5)),
    }


def paired_sign_flip_pvalue(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> float:
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    difference = proposed - baseline
    observed = abs(float(np.mean(difference)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    batch_size = min(1000, samples)
    while completed < samples:
        current = min(batch_size, samples - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(difference)))
        null_statistics = np.abs(np.mean(signs * difference, axis=1))
        extreme += int(np.sum(null_statistics >= observed))
        completed += current
    return float((extreme + 1) / (samples + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def paired_records(
    records: list[dict[str, object]],
    baseline_method: str,
    proposed_method: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    indexed: dict[tuple[str, int], float] = {}
    for record in records:
        method = str(record["method"])
        if method in {baseline_method, proposed_method}:
            indexed[(method, int(record["query_index"]))] = float(record[field])
    query_ids = sorted(
        set(query for method, query in indexed if method == baseline_method)
        & set(query for method, query in indexed if method == proposed_method)
    )
    return (
        np.array([indexed[(baseline_method, query)] for query in query_ids]),
        np.array([indexed[(proposed_method, query)] for query in query_ids]),
    )


def paired_cluster_records(
    records: list[dict[str, object]],
    baseline_method: str,
    proposed_method: str,
    field: str,
    *,
    eligible_only: bool = False,
    subset: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_subsets = {"all", "point_feasible", "point_rejectable", "trajectory"}
    if subset not in valid_subsets:
        raise ValueError(f"subset must be one of {sorted(valid_subsets)}")

    def include(record: dict[str, object]) -> bool:
        feasible = bool(record["expected_reachable"]) and bool(record["continuity_feasible"])
        trajectory = bool(record.get("closed_loop", False))
        if eligible_only and not feasible:
            return False
        if subset == "point_feasible":
            return feasible and not trajectory
        if subset == "point_rejectable":
            return not feasible and not trajectory
        if subset == "trajectory":
            return trajectory
        return True

    indexed: dict[tuple[str, int], tuple[float, int]] = {}
    for record in records:
        if not include(record):
            continue
        method = str(record["method"])
        if method in {baseline_method, proposed_method}:
            indexed[(method, int(record["query_index"]))] = (
                float(record[field]),
                int(record["trajectory_id"]),
            )
    query_ids = sorted(
        set(query for method, query in indexed if method == baseline_method)
        & set(query for method, query in indexed if method == proposed_method)
    )
    baseline = np.array([indexed[(baseline_method, query)][0] for query in query_ids])
    proposed = np.array([indexed[(proposed_method, query)][0] for query in query_ids])
    clusters = np.array([indexed[(baseline_method, query)][1] for query in query_ids], dtype=np.int64)
    return baseline, proposed, clusters


def paired_cluster_bootstrap_difference(
    baseline: np.ndarray,
    proposed: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    clusters = np.asarray(clusters)
    if baseline.shape != proposed.shape or baseline.shape != clusters.shape or baseline.ndim != 1:
        raise ValueError("paired values and cluster IDs must be one-dimensional and equally sized")
    difference = proposed - baseline
    unique = np.unique(clusters)
    cluster_effects = np.array([np.mean(difference[clusters == cluster]) for cluster in unique])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(cluster_effects), size=len(cluster_effects))
        bootstrap[sample_index] = float(np.mean(cluster_effects[selected]))
    return {
        "mean_difference": float(np.mean(cluster_effects)),
        "ci_lower": float(np.percentile(bootstrap, 2.5)),
        "ci_upper": float(np.percentile(bootstrap, 97.5)),
        "cluster_count": float(len(unique)),
    }


def paired_cluster_sign_flip_pvalue(
    baseline: np.ndarray,
    proposed: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> float:
    difference = np.asarray(proposed, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    clusters = np.asarray(clusters)
    if difference.shape != clusters.shape or difference.ndim != 1:
        raise ValueError("paired values and cluster IDs must be one-dimensional and equally sized")
    unique = np.unique(clusters)
    cluster_means = np.array([np.mean(difference[clusters == cluster]) for cluster in unique])
    observed = abs(float(np.mean(cluster_means)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < samples:
        current = min(1000, samples - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(cluster_means)))
        null_statistics = np.abs(np.mean(signs * cluster_means, axis=1))
        extreme += int(np.sum(null_statistics >= observed))
        completed += current
    return float((extreme + 1) / (samples + 1))
