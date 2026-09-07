"""Resume only unwritten read-only tables after a duplicate-key aggregation error.

The original execution seal is retained. The sole post-measurement edit to a
sealed source is checked byte-for-byte here; no measured method or input changes.
"""
import csv
import hashlib
import json
import subprocess
from .study import Study, now
from .aggregate import trajectory_tables
from ..revision_compute_allocation.common import digest, json_write, csv_write


def main():
    s=Study();s.check(execution=False)
    seal=json.loads((s.out/'01_protocol/execution_seal.json').read_text())
    allowed='src/confik/task_contract_alignment/aggregate.py'
    old=subprocess.check_output(['git','show',seal['code_commit']+':'+allowed],cwd=s.root)
    assert hashlib.sha256(old).hexdigest()==seal['files'][allowed]
    replacement=old.replace(b"dict(robot=robot,source=run['source'],**bad)",
                            b"dict(bad,robot=robot,source=run['source'])")
    assert replacement!=old and (s.root/allowed).read_bytes()==replacement
    for path,h in seal['files'].items():
        if path!=allowed:assert digest(s.root/path)==h,path
    folder=s.out/'05_aggregate'
    existing={str(p.relative_to(s.root)):digest(p) for p in folder.iterdir() if p.is_file()}
    json_write(folder/'analysis_revision.json',dict(utc=now(),
        execution_seal_sha256=digest(s.out/'01_protocol/execution_seal.json'),
        reason='UR5e raw first-failure rows already contain robot; duplicate keyword aborted read-only aggregation before trajectory outputs were written.',
        scope='one dictionary merge; no solver calls, source/input/mapping/budget/measurement changes',
        source=allowed,original_sha256=seal['files'][allowed],analysis_sha256=digest(s.root/allowed),
        completed_tables_retained=existing))
    trajectory_tables(s)
    authority=s.root/'outputs/revision_compute_allocation/reports'
    tables={p.name:list(csv.DictReader(p.open())) for p in [authority/'point_paired_comparisons.csv',
        authority/'trajectory_paired_comparisons.csv',authority/'oracle_crossfit_summary.csv',
        s.root/'outputs/revision_compute_allocation/01_existing_result_decomposition/trajectory_savings_decomposition.csv']}
    json_write(folder/'allocation_boundary.json',dict(
        interpretation='read-only historical case study; NOT newly measured routing on aligned solvers',tables=tables))
    mappings=json.loads((s.out/'01_protocol/native_mappings.json').read_text())
    csv_write(folder/'native_mapping_table.csv',mappings)
    for p,h in existing.items():assert digest(s.root/p)==h,p
    json_write(folder/'completed.json',dict(utc=now(),analysis_only_resume=True))


if __name__=='__main__':main()
