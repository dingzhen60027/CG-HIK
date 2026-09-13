"""Read-only final integrity check. Prints manifest; never solves or edits data."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASELINE='8e4aa63975d465f3de8a1eef2f7773889252c91d'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()

seal=json.loads((OUT/'inputs/seal.json').read_text())
for section in ('frozen_hashes','study_hashes'):
    for path,h in seal[section].items():assert sha(ROOT/path)==h
for name,h in seal['input_files'].items():assert sha(OUT/'inputs'/name)==h
selection=json.loads((OUT/'weight_selection.json').read_text());assert selection['theta']==.35 and selection['validation_solver_calls']==0
assert selection['input_seal_sha256']==sha(OUT/'inputs/seal.json')
allowed={'configs/task_set_boundary.yaml','docs/TASK_SET_BOUNDARY_PROTOCOL.md','docs/TASK_SET_BOUNDARY_FINDINGS.md',
    'scripts/run_task_set_boundary.py','scripts/report_task_set_boundary.py','src/confik/task_set_controls.py',
    'tests/test_task_set_boundary.py','tests/test_task_set_reporting.py'}
changed=git('diff','--name-only',BASELINE).splitlines()
assert all(p in allowed or p.startswith(str(OUT.relative_to(ROOT))+'/') for p in changed)
for name in ('src/confik/task_balance_gn.py','src/confik/bounded_gn.py','src/confik/solvers/verifier.py'):
    baseline_bytes=subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT)
    assert (ROOT/name).read_bytes()==baseline_bytes
report=json.loads((OUT/'reports/manifest.json').read_text())
assert sha(ROOT/'scripts/report_task_set_boundary.py')==report['reporting_sha256']
for name,h in report['files'].items():assert sha(OUT/'reports'/name)==h
calls=0;identities=[];half_check={}
for split in ('development','validation'):
    for robot in ('panda','ur5e'):
        folder=OUT/f'{split}_{robot}';manifest=json.loads((folder/'manifest.json').read_text())
        for name,h in manifest['files'].items():assert sha(folder/name)==h
        items=json.loads((OUT/f'inputs/{split}_{robot}.json').read_text())
        identities+=items;assert len(items)==(270 if split=='development' else 2160)
        assert len({i['anchor_uid'] for i in items})==(30 if split=='development' else 240)
        assert set(Counter((i['anchor_family'],i['displacement'],i['alpha']) for i in items).values())=={10 if split=='development' else 80}
        if split=='validation':
            assert manifest['commit']=='00e28a91102e6d71abe03896dbf0d77eb1b0eadc'
            assert manifest['selection_sha256']==sha(OUT/'weight_selection.json')
        else:
            rows=[json.loads(l) for l in gzip.open(folder/'records.jsonl.gz','rt')]
            ix={(r['uid'],r['method'],r['repeat']):r for r in rows};count=0
            for r in rows:
                if r['method']!='gn':continue
                c=ix[r['uid'],'weight_0.5',r['repeat']]
                assert r['accepted']==c['accepted'] and r['q']==c['q'];count+=1
            half_check[robot]=dict(same_input_calls=count,identical_decisions=True,max_joint_difference=0.)
        calls+=manifest['calls']
assert calls==108540
assert len({i['uid'] for i in identities})==len(identities)==4860
assert len({i['query_hash'] for i in identities})==4860
collections={}
for folder in sorted(p for p in OUT.iterdir() if p.is_dir()):
    entries=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    collections[folder.name]=dict(files=len(entries),bytes=sum(r['bytes'] for r in entries),
        listing_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest())
payload=dict(baseline=BASELINE,validation_pre_results_commit='00e28a91102e6d71abe03896dbf0d77eb1b0eadc',
    current_numerical_code_unchanged=True,old_evidence_and_paper_unchanged=True,
    validation_inputs_sealed_before_any_new_comparisons=True,selected_common_theta=.35,
    original_half_weight_equality=half_check,measured_calls=calls,geometric_witness_queries=4860,
    report_replay=report['verification'],core_tests_passed=22,reporting_tests_passed=3,
    tests_note='Regression tests and saved-command replay are not new benchmark samples.',
    code_and_docs_sha256={p:sha(ROOT/p) for p in sorted(allowed)},
    audit_script_sha256=sha(Path(__file__)),collections=collections,
    digest_recipe='Within each immediate subdirectory, sort recursive files excluding __pycache__; entries path(relative to result root),sha256,bytes; SHA256 of JSON sort_keys=True,separators=(comma,colon).',
    new_solver_calls_in_this_audit=0)
if len(sys.argv)>1:
    assert json.loads(Path(sys.argv[1]).read_text())==payload
    print('PASS: recorded collection hashes, sources, inputs, selection and old evidence protections match.')
else:print(json.dumps(payload,indent=2))
