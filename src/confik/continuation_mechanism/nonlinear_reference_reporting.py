"""Read-only verification and aggregation of the fixed nonlinear reference.

No solver calls, candidate generation, policy selection, or evidence rewriting.
All products are new, exclusively created files in the reference subdirectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np

from ..revision_compute_allocation.common import csv_write, digest, json_write
from ..types import Pose
from .nonlinear_reference import NonlinearReference, INITIALIZATIONS
from .nonlinear_reference_math import validate_path


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), total=float(a.sum()), mean=float(a.mean()),
                median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max()))


def best_record(records):
    feasible = [r for r in records if r['validation']['pose_and_range_feasible']]
    return min(feasible, key=lambda r: r['validation']['actual_rho']) if feasible else None


def build(root='.'):
    study = NonlinearReference(root)
    protocol, cases = study.check()
    out = study.out
    completed = json.loads((out/'run_completed.json').read_text())
    assert completed['runs'] == 390
    runs = []
    initial_checks = []
    by_problem = {}
    compared_frames = 0
    witness_frames = 0
    witness_files = 0
    for case in cases:
        for horizon in case['horizons']:
            targets = [Pose(**t) for t in case['targets'][:horizon]]
            group = []
            for kind in INITIALIZATIONS:
                name = f'{case["case_id"]}_L{horizon}_{kind}.json'
                r = json.loads((out/'runs'/name).read_text())
                assert r['case_id'] == case['case_id'] and r['horizon'] == horizon
                assert r['fixed_q0_unchanged'] and r['initialization'] == kind
                for field in ['validation', 'last_iterate_validation']:
                    old = r[field]
                    assert np.array_equal(old['q0'], case['q0'])
                    assert np.asarray(old['path']).shape == (horizon, study.kin.nq)
                    checked = validate_path(study.kin, study.verifier, case['q0'], targets,
                                            np.asarray(old['path']), reported_rho=old['optimizer_reported_rho'])
                    assert checked == old, (name, field)
                    compared_frames += horizon
                assert r['found_legal_continuation'] == r['validation']['found_legal_continuation']
                if r['best_found_rho'] is not None:
                    assert r['validation']['pose_and_range_feasible']
                    assert r['best_found_rho'] == r['validation']['actual_rho']
                for key in ['fk_calls', 'geometric_jacobian_calls']:
                    assert r['counts']['total'][key] == sum(r['counts'][phase][key]
                        for phase in ['initialization', 'optimization', 'validation'])
                assert r['timing_ns']['total'] >= sum(r['timing_ns'][phase]
                    for phase in ['initialization', 'optimization', 'validation'])
                initial = validate_path(study.kin, study.verifier, case['q0'], targets,
                                        np.asarray(r['initial_path']))
                initial_checks.append(dict(case_id=case['case_id'], horizon=horizon, initialization=kind,
                    found_legal_continuation=initial['found_legal_continuation'],
                    pose_and_range_feasible=initial['pose_and_range_feasible'],
                    actual_rho=initial['actual_rho'],
                    max_position_constraint_excess=initial['max_position_constraint_excess'],
                    max_orientation_constraint_excess=initial['max_orientation_constraint_excess'],
                    verified_frames=sum(f['accepted'] for f in initial['frames']),
                    scope='post-run read-only initial-path check; not an additional solver run'))
                r['initial_path_verified'] = initial['found_legal_continuation']
                r['initial_path_actual_rho'] = initial['actual_rho']
                r['run_file'] = f'runs/{name}'
                if r['found_legal_continuation']:
                    w = json.loads((out/'successful_witnesses'/name).read_text())
                    assert w['validation'] == r['validation']
                    assert w['source_run'] == r['run_file']
                    assert len(w['validation']['frames']) == horizon
                    witness_files += 1
                    witness_frames += horizon
                runs.append(r)
                group.append(r)
            by_problem[case['case_id'], horizon] = group
    assert len(runs) == len(list((out/'runs').glob('*.json'))) == 390
    assert witness_files == len(list((out/'successful_witnesses').glob('*.json')))
    # The two frame-102 inputs share only the target, not the accepted frame-101 state.
    special = [c for c in cases if c['role'] == 'first_failure_input']
    assert special[0]['targets'] == special[1]['targets']
    assert not np.array_equal(special[0]['q0'], special[1]['q0'])
    for c in special:
        assert np.array_equal(c['q0'], c['old_next_frame']['previous_q'])

    detailed = []
    wide = []
    first_failure = []
    for case in cases:
        row = dict(case_id=case['case_id'], site_id=case['site_id'], candidate_id=case['candidate_id'],
                   uid=case['uid'], role=case['role'], ordinary_completed=case['ordinary_completed'],
                   ordinary_repeats=len(case['ordinary_run_ids']), ordinary_prefixes=case['ordinary_prefixes'])
        for horizon in case['horizons']:
            group = by_problem[case['case_id'], horizon]
            best = best_record(group)
            found = any(r['found_legal_continuation'] for r in group)
            rho = best['validation']['actual_rho'] if best else None
            row.update({f'L{horizon}_found': found, f'L{horizon}_rho_upper_bound': rho,
                        f'L{horizon}_initial_path_verified': any(r['initial_path_verified'] for r in group),
                        f'L{horizon}_selected_run': best['run_file'] if best else None})
            detail = dict(case_id=case['case_id'], horizon=horizon, found=found,
                best_found_rho_upper_bound=rho, successful_initializations=sum(r['found_legal_continuation'] for r in group),
                initial_path_verified=sum(r['initial_path_verified'] for r in group),
                selected_run=best['run_file'] if best else None,
                optimizer_statuses=[r['optimizer'] for r in group],
                all_three_starts_total_ms=sum(r['timing_ns']['total'] for r in group)/1e6,
                all_three_starts_fk_calls=sum(r['counts']['total']['fk_calls'] for r in group),
                all_three_starts_jacobian_calls=sum(r['counts']['total']['geometric_jacobian_calls'] for r in group),
                selected_max_position_excess=best['validation']['max_position_constraint_excess'] if best else None,
                selected_max_orientation_excess=best['validation']['max_orientation_constraint_excess'] if best else None,
                selected_minimum_joint_margin=best['validation']['minimum_joint_margin'] if best else None)
            detailed.append(detail)
            if case['role'] == 'first_failure_input':
                old = case['old_next_frame']
                first_failure.append(dict(case_id=case['case_id'], previous_frame=case['frame'], target_frame=102,
                    previous_q=case['q0'], target=case['targets'][0], old_record=old,
                    found=found, best_found_rho_upper_bound=rho,
                    witness=f'successful_witnesses/{best["run_file"].split("/")[-1]}' if found else None,
                    new_position_error=best['validation']['frames'][0]['position_error'] if best else None,
                    new_orientation_error=best['validation']['frames'][0]['orientation_error'] if best else None,
                    new_q=best['validation']['path'][0] if best else None,
                    all_three_starts_ms=detail['all_three_starts_total_ms'],
                    single_start_ms=[r['timing_ns']['total']/1e6 for r in group],
                    initial_path_verified=[r['initial_path_verified'] for r in group]))
        if case['role'] != 'first_failure_input':
            wide.append(row)
    assert len(wide) == 32 and len(detailed) == 130 and len(first_failure) == 2
    groups = []
    for sid in dict.fromkeys(r['site_id'] for r in wide):
        rows = [r for r in wide if r['site_id'] == sid]
        failed = [r for r in rows if r['ordinary_completed'] == 0]
        groups.append(dict(site_id=sid, candidates=len(rows),
            ordinary_zero_of_five=len(failed),
            zero_of_five_now_found_L30=sum(r['L30_found'] for r in failed),
            any_ordinary_failure=sum(r['ordinary_completed'] < 5 for r in rows),
            any_ordinary_failure_now_found_L30=sum(r['L30_found'] for r in rows if r['ordinary_completed'] < 5),
            **{f'L{L}_found': sum(r[f'L{L}_found'] for r in rows) for L in [1, 5, 10, 30]},
            any_initial_L30_verified=sum(r['L30_initial_path_verified'] for r in rows)))
    timing = []
    for L in [1, 5, 10, 30]:
        subset = [r for r in runs if r['horizon'] == L]
        totals = [r for r in detailed if r['horizon'] == L]
        timing.append(dict(horizon=L, starts=len(subset), problems=len(totals),
            single_start_total_ms=distribution([r['timing_ns']['total']/1e6 for r in subset]),
            all_three_starts_ms=distribution([r['all_three_starts_total_ms'] for r in totals]),
            initialization_ms=distribution([r['timing_ns']['initialization']/1e6 for r in subset]),
            optimization_ms=distribution([r['timing_ns']['optimization']/1e6 for r in subset]),
            verification_ms=distribution([r['timing_ns']['validation']/1e6 for r in subset]),
            setup_and_other_ms=distribution([(r['timing_ns']['total']-sum(r['timing_ns'][p]
                for p in ['initialization','optimization','validation']))/1e6 for r in subset]),
            fk_calls=sum(r['counts']['total']['fk_calls'] for r in subset),
            geometric_jacobian_calls=sum(r['counts']['total']['geometric_jacobian_calls'] for r in subset)))
    summary = dict(scope=protocol['scope'], groups=groups, timing=timing,
        runs=len(runs), verified_runs=sum(r['found_legal_continuation'] for r in runs),
        problems=len(detailed), verified_problems=sum(r['found'] for r in detailed),
        optimizer_status_counts=dict(Counter(str(r['optimizer']['code']) for r in runs)),
        verified_despite_non_success_status=sum(r['found_legal_continuation'] and not r['optimizer']['success'] for r in runs),
        total_ms=sum(r['timing_ns']['total'] for r in runs)/1e6,
        total_fk_calls=sum(r['counts']['total']['fk_calls'] for r in runs),
        total_geometric_jacobian_calls=sum(r['counts']['total']['geometric_jacobian_calls'] for r in runs),
        function_callbacks={key: sum(r['counts']['callbacks'][key] for r in runs)
                            for key in runs[0]['counts']['callbacks']},
        no_global_optimality_or_infeasibility_claim=True)
    for name, value in [('candidate_horizon_table', wide), ('horizon_search_details', detailed),
                        ('first_failure_input_table', first_failure), ('initialization_checks', initial_checks)]:
        json_write(out/f'{name}.json', value)
        csv_write(out/f'{name}.csv', value)
    json_write(out/'reference_summary.json', summary)
    csv_write(out/'group_summary.csv', groups)
    study.check()
    json_write(out/'verification.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        checked_runs=len(runs), rechecked_selected_and_last_iterate_frames=compared_frames,
        rechecked_initial_paths=len(initial_checks), successful_witness_files=witness_files,
        successful_witness_frames=witness_frames, accepted_contract_violations=0,
        fixed_q0_unchanged=True, same_targets_and_original_verifier=True,
        special_inputs_same_target_different_previous_q=True,
        protected_old_files_verified=len(protocol['protected_files']),
        historical_witness_checks=len(json.loads((out/'known_witness_selfcheck.json').read_text())['checks']),
        reporting_solver_calls=0,
        report_numerical_recheck_timing_excluded_from_original_solver_timing=True,
        seal_sha256=digest(out/'input_seal.json')))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--seal', action='store_true', help='Seal completed products without rerunning search or reporting')
    args = parser.parse_args()
    if args.seal:
        study = NonlinearReference(args.root)
        protocol, _ = study.check()
        validation = json.loads((study.out/'verification.json').read_text())
        assert validation['accepted_contract_violations'] == 0 and validation['checked_runs'] == 390
        extra = [
            'src/confik/continuation_mechanism/nonlinear_reference.py',
            'src/confik/continuation_mechanism/nonlinear_reference_math.py',
            'src/confik/continuation_mechanism/nonlinear_reference_reporting.py',
            'tests/test_nonlinear_continuation_reference.py',
            'tests/test_nonlinear_reference_reporting.py',
            'docs/NONLINEAR_CONTINUATION_REFERENCE_FINDINGS.md',
        ]
        files = sorted([p for p in study.out.rglob('*') if p.is_file()])
        files += [study.root/p for p in extra]
        destination = study.out/'delivery_manifest.json'
        if destination.exists():
            raise FileExistsError(destination)
        json_write(destination, dict(baseline=protocol['baseline'],
            created_utc=datetime.now(timezone.utc).isoformat(),
            scope=protocol['scope'], files={str(p.relative_to(study.root)): digest(p) for p in files},
            protected_old_file_count=len(protocol['protected_files']),
            input_seal_sha256=digest(study.out/'input_seal.json'),
            report='docs/NONLINEAR_CONTINUATION_REFERENCE_FINDINGS.md',
            exclusion='delivery manifest does not hash itself; old evidence is protected, not rewritten'))
        print(f'Sealed {len(files)} files at {destination}')
    else:
        build(args.root)


if __name__ == '__main__':
    main()
