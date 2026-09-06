"""Task A: descriptive re-grouping of existing records; never calls a solver."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from ..release_v4_locked.artifacts import FrozenV4Policy, TorchScriptV4Inference, load_exact_v4_predictor, load_policy_config
from .common import csv_write, digest, json_write, load_npz, paired_interval

GROUPS = ('both_complete', 'only_cghik_complete', 'only_hard_complete', 'neither_complete')


def trajectory_decomposition(data, robot, hard='always_hard', full='counterfactual_cghik_v4'):
    names = list(data['method_names'])
    hi, ci = names.index(hard), names.index(full)
    rows = []
    for uid in sorted(set(data['trajectory_uid'])):
        ix = np.flatnonzero(data['trajectory_uid'] == uid)
        ix = ix[np.argsort(data['time_index'][ix])]
        hc, cc = bool(data['accepted'][ix, hi].all()), bool(data['accepted'][ix, ci].all())
        group = GROUPS[0 if hc and cc else 1 if cc else 2 if hc else 3]
        row = dict(robot=robot, trajectory_uid=str(uid), family=str(data['category'][ix[0]]), group=group, frames=len(ix))
        for label, mi in [('hard', hi), ('cghik', ci)]:
            failure = ix[~data['accepted'][ix, mi]]
            row.update({f'{label}_complete': bool(data['accepted'][ix, mi].all()),
                        f'{label}_total_latency_ns': int(data['latency_ns'][ix, mi].sum()),
                        f'{label}_total_fev': int(data['function_evaluations'][ix, mi].sum()),
                        f'{label}_reject_count': int((data['entry_action'][ix, mi] == 'reject').sum()),
                        f'{label}_defer_count': int((data['entry_action'][ix, mi] == 'defer').sum()),
                        f'{label}_fallback_count': int(data['fallback_used'][ix, mi].sum()),
                        f'{label}_first_failure_frame': int(data['time_index'][failure[0]]) if len(failure) else '',
                        f'{label}_first_failure_reason': str(data['reject_reason'][failure[0], mi]) if len(failure) else ''})
        row['saved_latency_ns'] = row['hard_total_latency_ns'] - row['cghik_total_latency_ns']
        rows.append(row)
    total_saved = sum(r['saved_latency_ns'] for r in rows)
    aggregates = []
    for group in GROUPS:
        subset = [r for r in rows if r['group'] == group]
        out = dict(robot=robot, group=group, trajectory_count=len(subset), analysis='post-hoc descriptive; not causal')
        for label in ['hard', 'cghik']:
            for key in ['total_latency_ns', 'total_fev', 'reject_count', 'defer_count', 'fallback_count']:
                out[f'{label}_{key}'] = sum(r[f'{label}_{key}'] for r in subset)
            out[f'{label}_median_trajectory_latency_ns'] = float(np.median([r[f'{label}_total_latency_ns'] for r in subset])) if subset else ''
            out[f'{label}_first_failures'] = json.dumps([{k: r[k] for k in ['trajectory_uid', f'{label}_first_failure_frame', f'{label}_first_failure_reason']} for r in subset])
        out['saved_latency_ns'] = sum(r['saved_latency_ns'] for r in subset)
        out['fraction_of_all_saved_latency'] = out['saved_latency_ns'] / total_saved if total_saved else ''
        ratio = paired_interval([r['cghik_total_latency_ns'] for r in subset], [r['hard_total_latency_ns'] for r in subset], [r['family'] for r in subset])
        out.update(zip(['latency_ratio', 'latency_ratio_ci_low', 'latency_ratio_ci_high'], ratio))
        aggregates.append(out)
    for label, mi in [('hard', hi), ('cghik', ci)]:
        assert sum(r[f'{label}_total_latency_ns'] for r in aggregates) == int(data['latency_ns'][:, mi].sum())
        assert sum(r[f'{label}_total_fev'] for r in aggregates) == int(data['function_evaluations'][:, mi].sum())
    return aggregates, rows


def development_labels(workspace, robot, role):
    root = Path(workspace) / f'outputs/counterfactual_v4_bulk/{robot}/seed17/{role}'
    files = sorted(root.glob('chunks/*/counterfactual_labels.npz'))
    if not files:
        raise FileNotFoundError(root)
    chunks = [load_npz(p) for p in files]
    keys = ['features', 'query_sha256', 'source_indices', 'category', 'verified_success', 'latency_samples_ns', 'function_evaluations']
    return {k: np.concatenate([c[k] for c in chunks]) for k in keys}, files


def load_policy(workspace, robot):
    root = Path(workspace) / f'outputs/release_v4_locked/{robot}'
    cfg, _ = load_policy_config(root / 'v4_policy.json')
    return FrozenV4Policy(TorchScriptV4Inference(load_exact_v4_predictor(root / 'exact_v4_predictor.ts', device='cpu')), cfg)


def same_population(workspace, robot):
    data, files = development_labels(workspace, robot, 'policy_validation_queries')
    policy = load_policy(workspace, robot)
    decisions = [policy.decide(f) for f in data['features']]
    # Successful and non-abstained is the previous router-regret population.
    mask = data['verified_success'][:, :3].all(axis=1) & np.array([d.action in ('easy', 'medium', 'hard') for d in decisions])
    ix = np.flatnonzero(mask)
    selected = np.array([('easy', 'medium', 'hard').index(decisions[i].action) for i in ix])
    label = data['latency_samples_ns'][ix, :3].astype(float)
    oracle = np.argmin(np.quantile(label, .95, axis=2), axis=1)
    # Stage telemetry supports numerical+verifier costs without invented gate costs.
    numerical = {}
    for p in files:
        with gzip.open(p.parent / 'counterfactual_records.jsonl.gz', 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r['entry_action'] in ('easy', 'medium', 'hard'):
                    t = r['timing_samples_ns']
                    numerical[(r['query_sha256'], r['entry_action'])] = np.array(t['numerical_solver_ns']) + np.array(t['verification_ns'])
    rows = []
    for name, chosen in [('fixed_easy', np.zeros(len(ix), int)), ('fixed_medium', np.ones(len(ix), int)), ('fixed_hard', np.full(len(ix), 2)), ('frozen_cghik', selected), ('empirical_oracle', oracle)]:
        costs = label[np.arange(len(ix)), chosen]
        path = np.array([numerical[(str(data['query_sha256'][i]), ('easy', 'medium', 'hard')[j])] for i,j in zip(ix, chosen)])
        gap = costs - label[np.arange(len(ix)), oracle]
        rows.append(dict(robot=robot, method=name, query_count=len(ix), population='same policy-validation successful non-abstained queries',
                         query_population_sha256=__import__('hashlib').sha256(''.join(data['query_sha256'][ix]).encode()).hexdigest(),
                         numerical_verifier_mean_ms=float(path.mean()/1e6),
                         measured_label_pipeline_mean_ms=float(costs.mean()/1e6),
                         measured_label_pipeline_p50_ms=float(np.quantile(costs,.5)/1e6),
                         measured_label_pipeline_p95_ms=float(np.quantile(costs,.95)/1e6),
                         mean_noisy_query_empirical_p95_ms=float(np.quantile(costs,.95,axis=1).mean()/1e6),
                         mean_gap_to_noisy_oracle_ms=float(gap.mean()/1e6),
                         measured_deployment_e2e_ms='',
                         e2e_status='not measured on this population; labels include candidate and constant-risk shell, not deployed router overhead',
                         oracle_status='five-repeat in-sample noisy diagnostic, not true optimum'))
    return rows


def run(workspace, output):
    output = Path(output) / '01_existing_result_decomposition'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    aggregate, individual, routing = [], [], []
    sources = {}
    for robot in ['panda', 'ur5e']:
        p = Path(workspace) / f'outputs/fresh_transition_v4_test/{robot}_raw_records.npz'
        a, b = trajectory_decomposition(load_npz(p), robot)
        aggregate.extend(a); individual.extend(b)
        routing.extend(same_population(workspace, robot))
        sources[str(p.relative_to(workspace))] = digest(p)
    csv_write(output / 'trajectory_savings_decomposition.csv', aggregate)
    csv_write(output / 'trajectory_details.csv', individual)
    csv_write(output / 'same_population_routing_comparison.csv', routing)
    json_write(output / 'manifest.json', dict(analysis='post-hoc descriptive analysis', solver_calls=0, sources=sources, exact_cost_reconstruction=True, deployment_e2e_missing='not imputed'))
    return aggregate
