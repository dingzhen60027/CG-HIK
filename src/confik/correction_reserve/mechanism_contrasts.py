"""Read-only paired contrasts of the already completed fixed-site probes.

No numerical IK is called. All sites, unavailable commands and direction outcomes
are retained; interesting individual directions are descriptive, not new tests.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .study import read_rows, sha, utc, write_json
from .reporting import csv_write


def compare_summaries(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row['method'] in ('cr_ik', 'two_step_predictive'):
            grouped[row['site_id']][row['method']] = row
    contrasts = []
    for site_id, pair in sorted(grouped.items()):
        base, cr = pair['two_step_predictive'], pair['cr_ik']
        assert all(base[k] == cr[k] for k in ('robot', 'uid', 'family'))
        row = {k: cr[k] for k in ('robot', 'uid', 'family')}
        row['site_id'] = site_id
        for prefix, result in (('predictive', base), ('crik', cr)):
            for field in ('current_accepted', 'predicted_gamma', 'probe_success', 'all_directions_radius'):
                row[f'{prefix}_{field}'] = result[field]
        for field in ('predicted_gamma', 'probe_success', 'all_directions_radius'):
            row[f'{field}_difference'] = (
                cr[field] - base[field] if cr[field] is not None and base[field] is not None else None)
        contrasts.append(row)
    return contrasts


def compare_probes(base, cr):
    key = lambda r: (r['amplitude'], r['direction_id'], r['repeat'])
    left, right = {key(r): r for r in base}, {key(r): r for r in cr}
    assert len(left) == len(base) and len(right) == len(cr) and left.keys() == right.keys()
    groups = defaultdict(list)
    for k in sorted(left):
        a, b = left[k], right[k]
        # Only the current returned configuration, and therefore the next query's
        # previous_q, may differ. No borrowed state or target substitutions.
        assert all(a[f] == b[f] for f in ('common_previous_q', 'current_position',
            'current_rotation', 'probe_position', 'probe_rotation', 'dt', 'direction'))
        groups[k[:2]].append((bool(a['continuation']['accepted']), bool(b['continuation']['accepted'])))
    return [dict(amplitude=k[0], direction_id=k[1], searches=len(v),
                 predictive_accepted=sum(a for a, _ in v), crik_accepted=sum(b for _, b in v))
            for k, v in sorted(groups.items())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='outputs/correction_reserve_ik/formal_mechanism')
    p.add_argument('--out', default='outputs/correction_reserve_ik/formal_mechanism_comparison')
    args = p.parse_args()
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contrasts = compare_summaries(json.loads((source / 'summaries.json').read_text()))
    directions, hashes = [], {str(source / 'summaries.json'): sha(source / 'summaries.json')}
    for row in contrasts:
        if not (row['predictive_current_accepted'] and row['crik_current_accepted']):
            continue
        paths = [source / 'raw' / f'{row["site_id"]}_{m}.jsonl.gz'
                 for m in ('two_step_predictive', 'cr_ik')]
        for path in paths:
            hashes[str(path)] = sha(path)
        directions.extend(dict(site_id=row['site_id'], robot=row['robot'], uid=row['uid'], **d)
                          for d in compare_probes(*(read_rows(path) for path in paths)))
    csv_write(out / 'all_site_contrasts.csv', contrasts)
    csv_write(out / 'all_direction_contrasts.csv', directions)
    write_json(out / 'source_data.json', dict(sites=contrasts, directions=directions))
    write_json(out / 'manifest.json', dict(utc=utc(), solver_calls=0, sources=hashes,
        generator_sha256=sha(__file__), sites=len(contrasts),
        independent_trajectories=len({r['uid'] for r in contrasts}),
        inference='descriptive contrasts within the preselected probe sample; no new independent units'))


if __name__ == '__main__':
    main()
