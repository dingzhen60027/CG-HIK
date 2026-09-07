"""Read-only aggregation and command verification of the independent study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

import numpy as np

from ..geometry import pose_error
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import csv_write, digest, json_write
from .independent_selection import (DT, FAMILIES, FRAMES, HORIZON, INTERVENTION,
    OUT, REPEATS, RULES, IndependentSelection, choose_index, spectrum_scores,
    window_summary)
from .disentangle_math import MATCH_POSITION, MATCH_ORIENTATION, scales


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def distribution(values, unit=1):
    a = np.asarray(values, dtype=float)/unit
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a,.95)), minimum=float(a.min()), maximum=float(a.max())) if len(a) else dict(n=0)


def candidate_summary(rows):
    assert len(rows) == REPEATS and {r['repeat'] for r in rows} == set(range(REPEATS))
    rows = sorted(rows,key=lambda r:r['repeat'])
    return dict(completed_repeats=sum(r['complete'] for r in rows),
        window_completion_rate=float(np.mean([r['complete'] for r in rows])),
        mean_prefix=float(np.mean([r['prefix_frames'] for r in rows])),
        median_prefix=float(np.median([r['prefix_frames'] for r in rows])),
        prefixes=[r['prefix_frames'] for r in rows],
        first_failure_frames=[r['first_failure_frame'] for r in rows],
        first_failure_kinds=dict(Counter(r['first_failure_kind'] for r in rows if not r['complete'])),
        cumulative_latency_ms=distribution([r['total_latency_ns'] for r in rows],1e6),
        run_ids=[r['run_id'] for r in rows])


def classify_candidates(candidates):
    counts = [c['completed_repeats'] for c in candidates]
    if all(n == REPEATS for n in counts):
        return 'all_success'
    if all(n == 0 for n in counts):
        return 'all_failure'
    return 'mixed'


def hindsight_id(candidates):
    # Diagnostics only; candidate insertion order is the final tie break.
    return max(candidates,key=lambda c:(c['completed_repeats'],c['mean_prefix']))['candidate_id']


def stable_classical_miss(candidates, selected):
    byid = {c['candidate_id']:c for c in candidates}
    return (any(c['completed_repeats'] == REPEATS for c in candidates)
            and all(byid[selected[r]]['completed_repeats'] == 0 for r in RULES[2:]))


def aggregate_states(states, rule):
    available = [s for s in states if s['status'] == 'available']
    records = [s['outcomes'][rule] for s in available]
    delivered = sum(c['completed_repeats'] for c in records)
    improved = [s['site_id'] for s in available if s['outcomes'][rule]['completed_repeats'] > s['outcomes']['original']['completed_repeats']]
    worsened = [s['site_id'] for s in available if s['outcomes'][rule]['completed_repeats'] < s['outcomes']['original']['completed_repeats']]
    stable_recovered = [s['site_id'] for s in available if s['outcomes'][rule]['completed_repeats'] == REPEATS and s['outcomes']['original']['completed_repeats'] == 0]
    stable_lost = [s['site_id'] for s in available if s['outcomes'][rule]['completed_repeats'] == 0 and s['outcomes']['original']['completed_repeats'] == REPEATS]
    any_success_pool = [s for s in available if s['pool_any_success']]
    stable_success_pool = [s for s in available if s['pool_stable_success']]
    return dict(rule=rule, scheduled_states=len(states), available_states=len(available),
        unavailable_states=len(states)-len(available), completed_repeat_windows=delivered,
        all_scheduled_delivered_window_rate=delivered/(len(states)*REPEATS) if states else None,
        conditional_available_window_rate=delivered/(len(available)*REPEATS) if available else None,
        stable_5_of_5_states=sum(c['completed_repeats'] == REPEATS for c in records),
        any_completed_states=sum(c['completed_repeats'] > 0 for c in records),
        all_scheduled_delivered_mean_prefix=sum(c['mean_prefix'] for c in records)/len(states) if states else None,
        available_mean_prefix=float(np.mean([c['mean_prefix'] for c in records])) if records else None,
        available_median_state_mean_prefix=float(np.median([c['mean_prefix'] for c in records])) if records else None,
        first_failure_kinds=dict(sum((Counter(c['first_failure_kinds']) for c in records),Counter())),
        unavailable_reasons=dict(Counter(s['status'] for s in states if s['status'] != 'available')),
        recovered_state_ids=improved, degraded_state_ids=worsened,
        stable_recovered_state_ids=stable_recovered, stable_degraded_state_ids=stable_lost,
        pool_any_success_states=len(any_success_pool),
        selected_any_success_when_available=sum(s['outcomes'][rule]['completed_repeats'] > 0 for s in any_success_pool),
        pool_stable_success_states=len(stable_success_pool),
        selected_stable_success_when_available=sum(s['outcomes'][rule]['completed_repeats'] == REPEATS for s in stable_success_pool),
        scoring_latency_ms=distribution([s['scoring_latency_ns'][rule] for s in available],1e6) if rule != 'hindsight' else None,
        intervention_pipeline_latency_ms=distribution([s['center_latency_ns'] +
            (0 if rule == 'original' else s['generation_latency_ns']) + s['scoring_latency_ns'][rule]
            for s in available],1e6) if rule != 'hindsight' else None)


def analyse(study):
    study.load_data()
    blocks = json.loads((study.out/'candidate_pools.json').read_text())
    runs = json.loads((study.out/'continuation_runs.json').read_text())
    bycandidate = defaultdict(list)
    for r in runs:
        bycandidate[(r['site_id'],r['candidate_id'])].append(r)
    states, candidates = [], []
    for b in blocks:
        s = dict(site_id=b['site']['site_id'],uid=b['site']['uid'],family=b['site']['family'],
            seed=b['site']['seed'],status=b['status'], candidate_count=len(b['candidates']),
            first_history_or_center_failure=b['first_failure_frame'],
            first_history_or_center_failure_kind=b['first_failure_kind'],
            history_latency_ns=sum(r['total_latency_ns'] for r in b['history']),
            center_latency_ns=b['current']['total_latency_ns'] if b['current'] else None)
        if b['status'] != 'available':
            states.append(s)
            continue
        local = []
        features = {f['candidate_id']:f for f in b['scoring']['auxiliary']}
        for c in b['candidates']:
            item = dict(site_id=s['site_id'],uid=s['uid'],family=s['family'],
                candidate_id=c['candidate_id'],source=c['source'],q=c['q'],
                **candidate_summary(bycandidate[(s['site_id'],c['candidate_id'])]),
                features=features[c['candidate_id']])
            local.append(item); candidates.append(item)
        selected = dict(b['scoring']['selected'],hindsight=hindsight_id(local))
        lookup = {c['candidate_id']:c for c in local}
        s.update(selected=selected, outcomes={r:lookup[cid] for r,cid in selected.items()},
            group=classify_candidates(local),
            pool_any_success=any(c['completed_repeats'] > 0 for c in local),
            pool_stable_success=any(c['completed_repeats'] == REPEATS for c in local),
            stable_classical_miss=stable_classical_miss(local, selected),
            stable_configuration_contrast=any(c['completed_repeats'] == 0 for c in local) and any(c['completed_repeats'] == REPEATS for c in local),
            generation_latency_ns=b['pool']['generation_latency_ns'],
            projection_latency_ns=b['pool']['projection_latency_ns'],
            projection_residual_calls=b['pool']['projection_residual_calls'],
            scoring_latency_ns=b['scoring']['scoring_latency_ns'],
            auxiliary_latency_ns=b['scoring']['auxiliary_latency_ns'])
        states.append(s)
    assert len(states) == 40 and len(candidates)*REPEATS == len(runs)
    main = [aggregate_states(states,r) for r in (*RULES,'hindsight')]
    family = [dict(family=f,**aggregate_states([s for s in states if s['family'] == f],r))
              for f in FAMILIES for r in (*RULES,'hindsight')]
    available = [s for s in states if s['status'] == 'available']
    group = [dict(group=g,**aggregate_states([s for s in available if s['group'] == g],r))
             for g in ['all_success','mixed','all_failure'] for r in (*RULES,'hindsight')]
    timing = dict(history_total_ms=distribution([s['history_latency_ns'] for s in states],1e6),
        center_solve_ms=distribution([s['center_latency_ns'] for s in states if s['center_latency_ns'] is not None],1e6),
        candidate_generation_ms=distribution([s['generation_latency_ns'] for s in available],1e6),
        projection_ms=distribution([s['projection_latency_ns'] for s in available],1e6),
        projection_residual_calls=distribution([s['projection_residual_calls'] for s in available]),
        auxiliary_diagnostics_ms=distribution([s['auxiliary_latency_ns'] for s in available],1e6),
        suffix_window_ms=distribution([r['total_latency_ns'] for r in runs],1e6),
        suffix_total_ms=sum(r['total_latency_ns'] for r in runs)/1e6,
        native_trac_fev=None,
        scope='timed actual Python pipeline; original solve includes native setup/solve, observation, deterministic verification; offline diagnostics and serialization excluded, separately costed; shared workstation')
    summary = dict(scheduled_states=40,availability=dict(Counter(s['status'] for s in states)),
        available_states=len(available),candidate_count=len(candidates),
        alternatives=sum(s['candidate_count']-1 for s in available),
        no_alternative_states=[s['site_id'] for s in available if s['candidate_count'] == 1],
        candidate_count_distribution=dict(Counter(s['candidate_count'] for s in states)),
        windows=len(runs),suffix_frames=len(runs)*HORIZON,
        descriptive_groups=dict(Counter(s['group'] for s in available)),
        stable_contrast_states=[s['site_id'] for s in available if s['stable_configuration_contrast']],
        stable_classical_miss_states=[s['site_id'] for s in available if s['stable_classical_miss']],
        main=main,timing=timing)
    json_write(study.out/'state_results.json',states)
    csv_write(study.out/'state_results.csv',[{k:v for k,v in s.items() if k != 'outcomes'} for s in states])
    json_write(study.out/'candidate_results.json',candidates)
    csv_write(study.out/'candidate_results.csv',[{k:v for k,v in c.items() if k != 'features'} for c in candidates])
    for name,rows in [('main_table',main),('family_table',family),('descriptive_group_table',group)]:
        json_write(study.out/f'{name}.json',rows)
        csv_write(study.out/f'{name}.csv',rows)
    json_write(study.out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def verify(study):
    ds, identities = study.load_data()
    seal = json.loads((study.out/'selection_seal.json').read_text())
    for name,h in seal['files'].items():
        assert digest(study.out/name) == h, name
    blocks = json.loads((study.out/'candidate_pools.json').read_text())
    site_index = {b['site']['site_id']:i for i,b in enumerate(blocks)}
    runs = {r['run_id']:r for r in json.loads((study.out/'continuation_runs.json').read_text())}
    reference_checks = 0
    for si in range(40):
        for t in range(FRAMES):
            idx = si*FRAMES+t
            query = study.query(ds,si,t,ds.previous_q[idx])
            assert study.verifier.check(ds.reference_q[idx],query).accepted
            assert query_digest(query) == identities[si]['query_hashes'][t]
            if t:
                assert np.array_equal(ds.previous_q[idx],ds.reference_q[idx-1])
            reference_checks += 1
    candidate_checks, match_checks = 0,0
    max_match = np.zeros(2)
    for si,b in enumerate(blocks):
        if b['status'] != 'available':
            assert not b['candidates']
            continue
        query = study.query(ds,si,INTERVENTION,b['current']['previous_q'])
        center_pose = study.kin.forward(np.asarray(b['candidates'][0]['q']))
        task,step = scales(study.kin,study.verifier,DT)
        distances, minima, volumes = [], [], []
        for c in b['candidates']:
            q = np.asarray(c['q'])
            assert study.verifier.check(q,query).accepted
            error = pose_error(center_pose,study.kin.forward(q))
            err = np.array([np.linalg.norm(error[:3]),np.linalg.norm(error[3:])])
            assert err[0] <= MATCH_POSITION and err[1] <= MATCH_ORIENTATION
            max_match = np.maximum(max_match,err)
            sv,volume = spectrum_scores(study.kin.jacobian(q),task,step)
            distances.append(float(np.linalg.norm(q-query.previous_q)))
            minima.append(sv[-1]);volumes.append(-np.inf if volume is None else volume)
            candidate_checks += 1
            match_checks += c['source'] == 'pose_matched'
        expected = dict(zip(RULES,[0,choose_index(distances),choose_index(minima,maximize=True),choose_index(volumes,maximize=True)]))
        assert b['scoring']['selected'] == {r:b['candidates'][i]['candidate_id'] for r,i in expected.items()}

    def check_row(row,previous,si):
        assert np.array_equal(row['previous_q'],previous)
        query = study.query(ds,si,row['frame'],previous)
        assert row['dt'] == DT and query_digest(query) == row['query_hash']
        assert np.array_equal(query.target.position,row['target_position'])
        assert np.array_equal(query.target.rotation,row['target_rotation'])
        if row['accepted']:
            assert row['solver_observation']['solver_ok']
            q = np.asarray(row['q'])
            assert study.verifier.check(q,query).accepted
        else:
            assert row['q'] is None
            q = np.asarray(previous)
        assert np.array_equal(row['accepted_state_q'],q)
        return q

    histories = defaultdict(list)
    history_accepted = 0
    for row in read_rows(study.out/'history_raw.jsonl.gz'):
        histories[row['site_id']].append(row)
    for si,b in enumerate(blocks):
        previous = ds.previous_q[si*FRAMES].copy()
        rows = histories[b['site']['site_id']]
        for t,row in enumerate(rows):
            assert row['frame'] == t
            previous = check_row(row,previous,si)
            history_accepted += row['accepted']
        assert len(rows) == (INTERVENTION+1 if b['status'] == 'available' else b['first_failure_frame']+1)
        if b['status'] == 'available':
            assert all(r['accepted'] for r in rows)
            assert np.array_equal(previous,b['candidates'][0]['q'])
        else:
            assert not rows[-1]['accepted'] and all(r['accepted'] for r in rows[:-1])
    suffix_accepted = 0
    stream_runs = defaultdict(list)
    for row in read_rows(study.out/'continuation_raw.jsonl.gz'):
        stream_runs[row['run_id']].append(row)
    assert set(stream_runs) == set(runs)
    for run_id,rows in stream_runs.items():
        run = runs[run_id]
        si = site_index[run['site_id']]; b = blocks[si]
        previous = np.asarray(next(c for c in b['candidates'] if c['candidate_id'] == run['candidate_id'])['q'])
        assert len(rows) == HORIZON
        for t,row in enumerate(rows,INTERVENTION+1):
            assert row['frame'] == t and row['candidate_id'] == run['candidate_id'] and row['repeat'] == run['repeat']
            previous = check_row(row,previous,si)
            suffix_accepted += row['accepted']
        for k,v in window_summary(rows).items():
            assert run[k] == v, (run_id,k)
    witness_count = 0
    for w in read_rows(study.out/'successful_witnesses.jsonl.gz'):
        rows = stream_runs[w['run']['run_id']]
        assert w['run']['complete'] and len(w['frames']) == HORIZON
        for row, saved in zip(rows,w['frames']):
            assert all(row[k] == v for k,v in saved.items())
        witness_count += 1
    assert witness_count == sum(r['complete'] for r in runs.values())
    result = dict(reference_commands_reverified=reference_checks, histories=len(histories),
        history_and_center_accepted_reverified=history_accepted,
        current_candidates_reverified=candidate_checks, matched_alternatives_reverified=match_checks,
        maximum_actual_pose_match_position_m=float(max_match[0]),
        maximum_actual_pose_match_orientation_rad=float(max_match[1]),
        selections_recomputed=True,suffix_windows_reverified=len(runs),
        accepted_suffix_commands_reverified=suffix_accepted, successful_witnesses=witness_count,
        original_contract_violations=0, old_artifacts_unchanged=True, verification_solver_calls=0)
    json_write(study.out/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


def details(study):
    """Additional exact lookup tables; no solver or selector changes."""
    study.load_data()
    states = json.loads((study.out/'state_results.json').read_text())
    candidates = json.loads((study.out/'candidate_results.json').read_text())
    runs = json.loads((study.out/'continuation_runs.json').read_text())
    byid = {r['run_id']:r for r in runs}
    flat, costs, failures = [], [], []
    for s in states:
        row = {k:s[k] for k in ['site_id','uid','family','seed','status','candidate_count',
            'first_history_or_center_failure','first_history_or_center_failure_kind']}
        for rule in (*RULES,'hindsight'):
            c = s['outcomes'][rule] if s['status'] == 'available' else None
            row[rule+'_candidate'] = c['candidate_id'] if c else None
            row[rule+'_completed_of_5'] = c['completed_repeats'] if c else None
            row[rule+'_mean_prefix'] = c['mean_prefix'] if c else None
        flat.append(row)
        if s['status'] != 'available':
            failures.append(dict(site_id=s['site_id'],uid=s['uid'],phase='history',rule=None,
                candidate_id=None,repeat=None,frame=s['first_history_or_center_failure'],
                failure_kind=s['first_history_or_center_failure_kind']))
            continue
        for rule in RULES:
            for rid in s['outcomes'][rule]['run_ids']:
                r = byid[rid]
                if not r['complete']:
                    failures.append(dict(site_id=s['site_id'],uid=s['uid'],phase='suffix',rule=rule,
                        candidate_id=r['candidate_id'],repeat=r['repeat'],frame=r['first_failure_frame'],
                        failure_kind=r['first_failure_kind']))
    for rule in RULES:
        available = [s for s in states if s['status'] == 'available']
        selected_runs = [byid[rid] for s in available for rid in s['outcomes'][rule]['run_ids']]
        costs.append(dict(rule=rule,available_states=len(available),
            current_center_solve_ms=distribution([s['center_latency_ns'] for s in available],1e6),
            generation_ms=distribution([0 if rule == 'original' else s['generation_latency_ns'] for s in available],1e6),
            scoring_ms=distribution([s['scoring_latency_ns'][rule] for s in available],1e6),
            suffix_window_ms=distribution([r['total_latency_ns'] for r in selected_runs],1e6),
            observed_suffix_total_ms=sum(r['total_latency_ns'] for r in selected_runs)/1e6,
            note='selected candidates reuse identical raw runs; shared selections are not extra solver executions'))
    for name,rows in [('all_state_table',flat),('rule_timing_table',costs),('first_failure_table',failures)]:
        json_write(study.out/f'{name}.json',rows)
        csv_write(study.out/f'{name}.csv',rows)
    miss_sites = {s['site_id'] for s in states if s.get('stable_classical_miss')}
    for sid in sorted(miss_sites):
        state = next(s for s in states if s['site_id'] == sid)
        json_write(study.out/f'{sid}_stable_counterexample.json',dict(state=state,
            candidates=[c for c in candidates if c['site_id'] == sid],
            interpretation='post-hoc identification of a prespecified independent state; no new candidates, repeats, score or threshold'))
        with gzip.open(study.out/f'{sid}_counterexample_raw.jsonl.gz','xt',encoding='utf8') as f:
            for row in read_rows(study.out/'continuation_raw.jsonl.gz'):
                if row['site_id'] == sid:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
    print(f'Exact lookup tables; stable-miss sites: {sorted(miss_sites)}',flush=True)


def finalize(study):
    """Final cross-file consistency checks and an immutable delivery inventory."""
    protocol = study.protect()
    pools = json.loads((study.out/'candidate_pools.json').read_text())
    candidates = json.loads((study.out/'candidate_results.json').read_text())
    summary = json.loads((study.out/'summary.json').read_text())
    verification = json.loads((study.out/'verification.json').read_text())
    bysite = {b['site']['site_id']:b for b in pools}
    bycandidate = {(b['site']['site_id'],c['candidate_id']):c for b in pools for c in b['candidates']}
    checked = 0
    for w in read_rows(study.out/'successful_witnesses.jsonl.gz'):
        run = w['run']; b = bysite[run['site_id']]
        assert w['initial_frame'] == INTERVENTION
        assert w['initial_previous_q'] == b['current']['previous_q']
        assert w['initial_q'] == bycandidate[(run['site_id'],run['candidate_id'])]['q']
        assert w['frames'][0]['previous_q'] == w['initial_q']
        checked += 1
    assert checked == verification['successful_witnesses']
    assert len(candidates) == summary['candidate_count']
    for r in summary['main']:
        if r['rule'] != 'hindsight':
            chosen = [next(c for c in candidates if c['site_id'] == b['site']['site_id'] and
                          c['candidate_id'] == b['scoring']['selected'][r['rule']])
                      for b in pools if b['scoring']]
            assert sum(c['completed_repeats'] for c in chosen) == r['completed_repeat_windows']
    doc = study.root/'docs/INDEPENDENT_CONFIGURATION_SELECTION_FINDINGS.md'
    body = doc.read_text()
    links = re.findall(r'\]\(([^)]+)\)',body)
    for link in links:
        if not link.startswith(('https://','http://')):
            assert (doc.parent/link).resolve().exists(), link
    qa = dict(successful_witness_initial_connections_checked=checked,
        main_tables_reconciled_to_candidate_runs=True, report_local_links_checked=len(links),
        independent_unit='40 scheduled trajectories/states, not 200 independent replicates',
        protected_file_hashes_checked=len(protocol['protected_files']),
        measurement_source_hashes_checked=len(protocol['measurement_sources']),
        report_audience='technical', delivery_surface='user-specified Markdown, no additional report app',
        report_structure=['one-page conclusion','full cohort and definitions','rules and family evidence',
                          'stable counterexample','actual costs','validation and limits'],
        table_contract='exact multi-metric and candidate-ID lookup; Markdown tables, neutral text, no color-only encoding',
        omitted_chart_reason='requested exact rule/candidate comparisons use multiple metrics and failure identities; no trend/visual inference needed',
        formal_test=False, old_paper_and_evidence_unchanged=True, solver_calls=0)
    json_write(study.out/'delivery_verification.json',qa)
    paths = sorted(p for p in study.out.rglob('*') if p.is_file())
    paths += [doc, study.root/'src/confik/continuation_mechanism/independent_selection.py',
              study.root/'src/confik/continuation_mechanism/independent_selection_reporting.py',
              study.root/'tests/test_independent_configuration_selection.py',
              study.root/'tests/test_independent_selection_reporting.py']
    json_write(study.out/'delivery_manifest.json',dict(
        baseline=protocol['baseline'],
        files={str(p.relative_to(study.root)):digest(p) for p in paths},
        note='Only new independent-selection artifacts. Historical results and paper unchanged.'))
    print(f'Finalized {len(paths)} file hashes; {checked} witness initial connections.',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['analyse','verify','details','finalize'])
    parser.add_argument('--root',default='.')
    args = parser.parse_args()
    globals()[args.stage](IndependentSelection(args.root))


if __name__ == '__main__':
    main()
