"""Preserve an interrupted adapter attempt; never select or inspect its outcomes."""
from pathlib import Path
import json
import shutil
from .common import digest,json_write


def finalize():
    root=Path.cwd();out=root/'outputs/revision_compute_allocation'
    paths=[root/'src/confik/revision_compute_allocation'/name for name in ['policies.py','benchmark.py','data.py','runner.py','trac_ik.py']]
    paths+=list((root/'src/confik/revision_compute_allocation/native').rglob('*.cpp'))
    json_write(out/'final_execution_seal.json',dict(selection_seal_sha256=digest(out/'selection_and_identity_seal.json'),
        measurement_sources={str(p.relative_to(root)):digest(p) for p in paths},
        corrected_control='one frozen inference and only the required ranking; no discarded P95 decision',
        point_or_trajectory_parameters_changed=False,calibration_repeated=False,main_method_unchanged=True))


def main():
    root=Path.cwd();out=root/'outputs/revision_compute_allocation'
    archive=out/'diagnostic_adapter_attempt_01';archive.mkdir(exist_ok=False)
    raw=out/'02_point_mechanism_benchmark/panda_raw_records.jsonl.gz'
    raw_hash=digest(raw)
    shutil.move(str(raw),str(archive/'panda_raw_records.partial.jsonl.gz'))
    seal_path=out/'selection_and_identity_seal.json'
    original=json.loads(seal_path.read_text())
    original_hash=digest(seal_path)
    shutil.move(str(seal_path),str(archive/'original_selection_and_identity_seal.json'))
    original['implementation']={str(p.relative_to(root)):digest(p) for p in (root/'src/confik/revision_compute_allocation').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    original['adapter_correction']=dict(reason='Remove unused original P95 ranking from geometry, reject-only and P50 controls; full CG-HIK untouched',
        prior_attempt='interrupted before completion; all its measurements excluded wholesale, no outcomes inspected for decisions',
        preserved_raw_sha256=raw_hash,prior_seal_sha256=original_hash,models_changed=False,solver_changed=False,verifier_changed=False,
        parameters_changed=False,dataset_identities_changed=False,calibration_repeated=False)
    json_write(seal_path,original)
    json_write(archive/'status.json',dict(status='incomplete_implementation_attempt_not_an_evaluation_result',
        reason=original['adapter_correction'],progress_at_interruption='Panda progress last reported 1425/3000; UR5e and trajectories had not started'))


if __name__=='__main__':main()
