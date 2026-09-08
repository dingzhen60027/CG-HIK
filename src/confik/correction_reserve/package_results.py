"""Read-only numerical-evidence checks and final artifact hashing; no IK calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .study import ROOT, code_hashes, sha, utc, write_json


def checked_json(command):
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--figure-skill', type=Path, required=True)
    args = p.parse_args()
    output = ROOT / 'outputs/correction_reserve_ik'
    seal = json.loads((output / 'formal_protocol/selection_seal.json').read_text())
    assert seal['configuration'] == yaml.safe_load((ROOT / 'configs/correction_reserve.yaml').read_text())
    assert seal['code_hashes'] == code_hashes()
    for name, digest in seal['files'].items():
        assert sha(output / 'formal_protocol' / name) == digest, name
    assert sha(ROOT / 'src/confik/correction_reserve/data.py') == seal['generation_source_sha256']
    changes = subprocess.check_output(['git', 'diff', '276f3d311aafc67786c916c5fec596dd860cc5dc',
        '--name-status', '--diff-filter=DMRT'], cwd=ROOT, text=True)
    assert not changes, changes
    for robot in ('panda', 'ur5e'):
        assert (output / 'formal' / robot / 'completed.json').exists()
    report = json.loads((output / 'findings_manifest.json').read_text())
    assert sha(ROOT / report['report']) == report['report_sha256']
    checks = output / 'delivery_checks'
    checks.mkdir(exist_ok=False)
    preflight = checked_json([sys.executable, str(args.figure_skill / 'scripts/validate_figure.py'),
        'src/confik/correction_reserve/figures.py', '--backend', 'python', '--json'])
    write_json(checks / 'figure_source_preflight.json', preflight)
    for pdf in sorted((output / 'formal_figures_final').glob('*.pdf')):
        result = checked_json([sys.executable, str(args.figure_skill / 'scripts/audit_pdf_text.py'),
            str(pdf), '--min-pt', '5', '--json'])
        assert result['auditable'] and result['below_minimum_count'] == 0
        write_json(checks / f'{pdf.stem}_glyph_audit.json', result)
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_crik_analysis.py'],
                          cwd=ROOT, text=True, capture_output=True, check=True)
    write_json(checks / 'postprocessing_tests.json', dict(command=test.args, returncode=test.returncode,
        stdout=test.stdout, stderr=test.stderr, solver_calls=0))
    files = [path for path in output.rglob('*') if path.is_file()]
    files += list((ROOT / 'docs').glob('CRIK_*.md'))
    files += [path for path in (ROOT / 'src/confik/correction_reserve').rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    files += list((ROOT / 'tests').glob('test_crik*.py'))
    files += [ROOT / 'tests/test_correction_reserve.py', ROOT / 'configs/correction_reserve.yaml',
              ROOT / 'scripts/run_correction_reserve.sh']
    files = sorted(set(files))
    too_large = [(str(path.relative_to(ROOT)), path.stat().st_size) for path in files
                 if path.stat().st_size >= 100 * 1024**2]
    assert not too_large, too_large
    write_json(output / 'delivery_manifest.json', dict(utc=utc(), numerical_solver_calls=0,
        pre_outcome_commit='e13f32f53a451eb19879292dbfa8e382edb4bc39', old_evidence_modifications=changes,
        state='complete; no further numerical experiments or parameter changes',
        files={str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files}))
    print(f'Packaged {len(files)} files; runtime, identities and old tracked evidence unchanged.')


if __name__ == '__main__':
    main()
