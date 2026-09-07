"""Final evidence-preservation manifest. No numerical solve or data replacement."""
import json
import subprocess
from .study import Study,now
from ..revision_compute_allocation.common import digest,json_write


def main():
    s=Study();s.check(False)
    assert (s.out/'05_aggregate/verification.json').exists()
    protected=json.loads((s.out/'01_protocol/protected_files.json').read_text())
    for p,h in protected.items():assert digest(s.root/p)==h,p
    reports=s.out/'reports'
    audits={}
    for p in sorted(reports.glob('*.pdf')):
        command=['/home/eric/anaconda3/envs/isaaclab_3/bin/python',
            '/home/eric/.codex/skills/nature-figure/scripts/audit_pdf_text.py',str(p),'--min-pt','5']
        run=subprocess.run(command,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        audits[p.name]=run.stdout
    json_write(reports/'figure_qa.json',dict(utc=now(),pdf_text_audits=audits,
        rendered_inspection='All six PNG exports inspected; no clipped labels or overlaps; all populations retained.',
        static_preflight_exceptions=[
            'Static whitelist does not recognize DejaVu Sans although this installed sans-serif is embedded in actual PDFs.',
            'Static parser misses loop-based PDF/SVG/PNG export; all eighteen actual files checked.',
            'PNG previews are 300 dpi; editable vector PDFs/SVGs are the publication outputs.'],
        version_note='Sealed reporting source retained; no result-driven panel removal or measurement change.'))
    tests=subprocess.run(['/home/eric/anaconda3/envs/isaaclab_3/bin/python','-m','pytest','-q',
        'tests/test_task_contract_analysis.py'],cwd=s.root,capture_output=True,text=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    json_write(s.out/'05_aggregate/analysis_tests.json',dict(utc=now(),output=tests.stdout,
        scope='five read-only aggregate tests; no numerical IK calls'))
    paths=[p for p in s.out.rglob('*') if p.is_file()]
    paths += [s.root/'docs/TASK_CONTRACT_ALIGNMENT_FINDINGS.md',s.root/'docs/TASK_CONTRACT_ALIGNMENT_PROTOCOL.md',
              s.root/'configs/task_contract_alignment.yaml',s.root/'scripts/run_task_contract_alignment.sh']
    paths += list((s.root/'src/confik/task_contract_alignment').rglob('*.py'))
    paths += list((s.root/'src/confik/task_contract_alignment/native').glob('*'))
    json_write(s.out/'delivery_manifest.json',dict(created_utc=now(),stage='all measurements complete; no more experiments',
        protected_old_files=len(protected),files={str(p.relative_to(s.root)):digest(p) for p in paths if p.is_file()},
        note='Panda remains authoritative; raw measurements and original seals immutable. Analysis-only merge fix and report formatting disclosed separately.'))


if __name__=='__main__':main()
