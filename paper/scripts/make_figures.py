#!/usr/bin/env python3
"""Materialize verified frozen vector figures; no experiments or output mutation."""
import shutil
from build_evidence import ROOT,PAPER,EVIDENCE,digest,load,write
def main():
    snap=load(PAPER/'generated/evidence_snapshot.json')
    for p,h in snap['sources'].items():assert digest(ROOT/p)==h,p
    manifest=load(EVIDENCE/'delivery_manifest.json');dest=PAPER/'figures';dest.mkdir(exist_ok=True)
    names=['figure1_taxonomy','figure2_point_alignment','figure3_contract_sensitivity','figure4_trajectories','figure5_family_effects','supplement_dls_excess_iterations']
    files={}
    for name in names:
        for ext in ['pdf','svg','png']:
            source=EVIDENCE/'reports'/f'{name}.{ext}'
            assert digest(source)==manifest['files'][str(source.relative_to(ROOT))]
            target=dest/source.name;shutil.copyfile(source,target);files[str(target.relative_to(PAPER))]=digest(target)
    write(PAPER/'generated/figure_manifest.json',dict(files=files,source_data='source_data/task_*.csv; generated/evidence_snapshot.json',rendering_source='src/confik/task_contract_alignment/reporting.py',note='Exact copies of reviewed frozen editable PDF/SVG and PNG previews. No experiment or statistics rerun.'))
    print('Materialized six frozen figures in editable PDF/SVG and PNG.')
if __name__=='__main__':main()
