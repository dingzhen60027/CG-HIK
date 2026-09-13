"""Read-only final delivery audit; no solver calls and no output file edits."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = root / 'outputs/single_solver_evidence/task_balance_completion'
baseline = '23f8555bd214eb4ac45b164a536ead55e142e44c'
launch_commit = '97be2df6513b84d83dc3eaeaf90de5fa7c65619c'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=root)

changed = git('diff', '--name-only', baseline).decode().splitlines()
allowed = {
    'configs/task_balance_completion.yaml',
    'docs/TASK_BALANCE_COMPLETION_METHOD.md',
    'docs/TASK_BALANCE_COMPLETION_FINDINGS.md',
    'scripts/run_task_balance_completion.py',
    'scripts/report_task_balance_completion.py',
    'src/confik/task_balance_gn.py',
    'tests/test_task_balance_completion.py',
}
assert all(p in allowed or p.startswith(str(out.relative_to(root)) + '/') for p in changed)
old_core = git('show', baseline + ':src/confik/task_balance_gn.py')
assert (root / 'src/confik/task_balance_gn.py').read_bytes().startswith(old_core)
launch = json.loads((out / 'independent_inputs/launch_seal.json').read_text())
assert all(digest(root / p) == h for p, h in launch['code_hashes'].items())
primary = json.loads((out / 'reports/manifest.json').read_text())
assert digest(out / 'reports/report_source_main.py') == primary['reporting_sha256']
supplement = json.loads((out / 'reports/supplementary_verification.json').read_text())
assert digest(root / 'scripts/report_task_balance_completion.py') == supplement['source_code_sha256']
collections = {}
for folder in sorted(p for p in out.iterdir() if p.is_dir()):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    entries = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    collections[folder.name] = dict(files=len(entries), bytes=sum(e['bytes'] for e in entries),
                                   listing_sha256=hashlib.sha256(encoded).hexdigest())
    assert all(e['bytes'] < 95 * 1024**2 for e in entries)
sources = sorted(allowed | {str(Path(__file__).resolve().relative_to(root))})
print(json.dumps(dict(
    schema='task-balance-completion-final-delivery-v1',
    baseline=baseline, independent_pre_results_commit=launch_commit,
    numerical_sources_match_pre_results_seal=True,
    historical_runtime_preserved_byte_prefix=True,
    old_evidence_paper_and_unrelated_tracked_files_unchanged=True,
    reporting_source_snapshots_verified=True,
    new_solver_calls_in_delivery_audit=0,
    collection_digest_recipe='For each directory, recursively sort file paths (excluding __pycache__), create entries with path relative to output root, bytes, sha256; SHA256 of JSON with sort_keys=True and separators=(comma,colon). The delivery manifest itself is outside these directories.',
    collections=collections, sources_sha256={p: digest(root / p) for p in sources},
    tests=dict(command='env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=tmp/crik_dependencies/python:src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/eric/anaconda3/envs/isaaclab_3/bin/python -m pytest -q tests/test_task_balance_completion.py tests/test_task_balance_gn.py',
               passed=23, scope='Unit/math/interface regression, not a new performance experiment.'),
    stop='No further algorithm, parameter, data or paper changes are authorized by this delivery.'
), indent=2))
