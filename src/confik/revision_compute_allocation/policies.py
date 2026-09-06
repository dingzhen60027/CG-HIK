"""Controlled changes to entry selection, never to solver or acceptance semantics."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter_ns

import numpy as np

from ..counterfactual_v4.policy import DECISION_ENTRIES, V4Decision
from ..fresh_transition_v4_test.benchmark import _fixed_method, _v4_method
from ..latency_pilot_v3.benchmark import ProfiledOutcome
from ..release_v4_locked.artifacts import FrozenV4Policy
from ..runtime.cascade import EntryAction
from .common import csv_write, json_write
from .decomposition import development_labels, load_policy

INTERNAL_METHODS = ('always_hard', 'geometry_threshold', 'reject_only_hard', 'routing_only', 'p50_selection', 'full_cghik')


class ControlledPolicy(FrozenV4Policy):
    def __init__(self, original, variant, geometry=None):
        super().__init__(original.backend, original.config)
        self.variant, self.geometry = variant, geometry

    def decide(self, features):
        # One inference and one necessary decision only. In particular, a P50
        # control must not first pay for an unused P95 ranking.
        output = self.backend.infer(np.asarray(features, dtype=np.float64))
        self.last_output = output
        base = dict(ood_score=float(output.ood_score),is_ood=bool(output.is_ood),
                    predicted_success=tuple(float(x) for x in output.success_probabilities),
                    predicted_p50_ms=tuple(float(x) for x in output.latency_p50_ms),
                    predicted_p95_ms=tuple(float(x) for x in output.latency_p95_ms),
                    fail_all_probability=float(output.fail_all_probability))
        if output.is_ood:
            action,reason,eligible = 'defer','ood_defer',()
        else:
            indices=np.flatnonzero((output.success_probabilities>=self.config.minimum_success_probability)&(output.latency_p95_ms<=self.config.deadline_ms))
            eligible=tuple(DECISION_ENTRIES[int(i)] for i in indices)
            if not len(indices):
                if output.fail_all_probability>=self.config.reject_probability:
                    action,reason=('hard','no_early_reject_hard') if self.variant=='routing_only' else ('reject','high_confidence_fail_all')
                else:action,reason='defer','uncertain_no_eligible_action'
            elif self.variant=='reject_only_hard':
                action,reason='hard','reject_only_force_hard'
            elif self.variant=='geometry_threshold':
                p,r=self.geometry
                action='easy' if features[7]<=p and features[8]<=r else 'hard'
                reason='geometry_step_threshold'
            else:
                costs=output.latency_p50_ms if self.variant=='p50_selection' else output.latency_p95_ms
                fastest=int(indices[np.argmin(costs[indices])]);conservative=int(np.min(indices))
                chosen=conservative if costs[conservative]-costs[fastest]<self.config.latency_tie_margin_ms else fastest
                action=DECISION_ENTRIES[chosen]
                if chosen==conservative and chosen!=fastest:
                    reason='tie_margin_conservative_entry'
                else:
                    reason='minimum_predicted_p50' if self.variant=='p50_selection' else 'minimum_predicted_p95'
        decision=V4Decision(action=action,reason=reason,eligible_actions=eligible,**base)
        self.last_decision = decision
        return decision


class ForcedEntryRuntime:
    """Fixed entry with no unused diagnostic feature or model calls.

    Candidate preparation and unchanged cascade.run_stage generate all commands.
    This deployment-cost baseline intentionally drops the historical constant-risk
    profiling shell. It does not change any numerical stage, seed or verifier.
    """
    def __init__(self, base, entry):
        self.base, self.entry = base, entry

    def solve(self, query):
        start = perf_counter_ns()
        verifier = self.base._timed_verifier
        verifier.reset()
        prepared = self.base.seed_engine.prepare(query)
        timing = dict(prepared.timings_ns)
        traces, stages = [], []
        fallback, accepted, q, check = False, False, None, None
        numerical = 0
        for si in range(int(self.entry), int(EntryAction.HARD) + 1):
            stage_start = perf_counter_ns()
            stage = self.base._cascade.run_stage(query, prepared.candidates, EntryAction(si))
            numerical += perf_counter_ns() - stage_start
            stages.append(EntryAction(si).name.lower())
            traces.extend(stage.traces)
            fallback |= stage.fallback_used
            check = stage.verification
            if stage.accepted:
                accepted, q = True, np.asarray(stage.q).copy()
                break
        timing['numerical_solver_ns'] = numerical - verifier.elapsed_ns
        timing['verification_ns'] = verifier.elapsed_ns
        timing['uncertainty_risk_inference_ns'] = timing['routing_decision_ns'] = 0
        result = ProfiledOutcome(q, accepted, self.entry.name.lower(), tuple(stages), np.array([1.,0.,0.,0.]), 0.,
                                 sum(t.function_evaluations for t in traces), sum(t.iterations for t in traces),
                                 bool(fallback), tuple(check.reasons) if check else (), '' if accepted else 'all_cascade_stages_failed',
                                 len(prepared.candidates.joints), timing)
        timing['total_end_to_end_ns'] = perf_counter_ns() - start
        return result


def build_internal(workspace, config, robot, kin, geometry, names=INTERNAL_METHODS):
    args = dict(source_config=config, release_v3_root=workspace/'outputs/release_v3_locked', robot=robot,
                release_seed=17, kinematics=kin, device='cuda:0')
    methods = {}
    for name in names:
        if name.startswith('forced_') or name == 'always_hard':
            entry = name.removeprefix('forced_') if name.startswith('forced_') else 'hard'
            method = _fixed_method(name=name, action=EntryAction[entry.upper()], **args)
            method = replace(method, runtime=ForcedEntryRuntime(method.runtime, EntryAction[entry.upper()]))
        else:
            method = _v4_method(release_v4_root=workspace/'outputs/release_v4_locked', **args)
            method = replace(method, name=name)
            if name != 'full_cghik':
                engine = method.runtime.engine
                engine.policy = ControlledPolicy(engine.policy, name, geometry)
        methods[name] = method
    return methods


def select_geometry(workspace, config, output):
    selected, rows = {}, []
    for robot in config['robots']:
        data, _ = development_labels(workspace, robot, 'calibration_queries')
        policy = load_policy(workspace, robot)
        decisions = [policy.decide(f) for f in data['features']]
        candidates = []
        for p in config['geometry_position_grid_m']:
            for r in config['geometry_orientation_grid_rad']:
                entry = np.where((data['features'][:,7] <= p) & (data['features'][:,8] <= r), 0, 2)
                reject = np.array([d.action == 'reject' for d in decisions])
                entry[[d.action == 'defer' for d in decisions]] = 0
                latency = data['latency_samples_ns'][np.arange(len(entry)),entry].copy()
                latency[reject] = 0  # only for threshold ranking; shared unmeasured reject overhead cancels.
                success = data['verified_success'][np.arange(len(entry)),entry] & ~reject
                row = dict(robot=robot, position_m=p, orientation_rad=r, label_success=int(success.sum()),
                           label_p95_ns=float(np.quantile(latency,.95)), label_mean_ns=float(latency.mean()))
                rows.append(row)
                candidates.append(((-row['label_success'],row['label_p95_ns'],row['label_mean_ns'],p,r), (p,r)))
        selected[robot] = list(min(candidates)[1])
    csv_write(output/'geometry_calibration_candidates.csv', rows)
    json_write(output/'geometry_selection.json', dict(selected=selected, role='calibration_queries',
                  timing_scope='existing pathway labels only; shared gate/reject overhead unmeasured and not imputed for reporting'))
    return selected
