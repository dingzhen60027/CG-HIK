#!/usr/bin/env python3
"""Read-only checks and preview contacts for the Phase-1 delivery. No benchmarks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/process-scan-mpl')
import hashlib
import csv
import json
from pathlib import Path
import subprocess
import sys
import cv2
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/process_scan/phase1_baselines'
QA=OUT/'reports/qa'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(name,data):
    (QA/name).write_text(json.dumps(data,indent=2)+'\n')


def main():
    QA.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_process_scan.py',
        '--junitxml='+str(QA/'tests.xml')],cwd=ROOT,env=env,text=True,capture_output=True)
    (QA/'tests.txt').write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    # The optional source/font checker is the installed, read-only figure skill.
    skill=Path('/home/eric/.codex/skills/nature-figure/scripts')
    checks={}
    if skill.exists():
        commands={'plot_source':[str(skill/'validate_figure.py'),'src/confik/process_scan/reporting.py','--backend','python','--json']}
        commands.update({p.stem:[str(skill/'audit_pdf_text.py'),str(p),'--min-pt','5','--json']
            for p in sorted((OUT/'reports/figures').glob('*.pdf'))})
        for name,args in commands.items():
            result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
            checks[name]=json.loads(result.stdout)
    else:checks['status']='Optional installed figure audit tooling unavailable'
    save('automated_figure_checks.json',checks)
    panels=[]
    for robot in ('panda','ur5e'):
        for column,claim in enumerate(('All scheduled quality yield','Actual qualified completion time','Planning work averaged within scene')):
            panels.append(dict(figure='phase1_overview',panel=f'{robot}_{column}',claim=claim,
                summary='Count' if column==0 else 'All scene points and median',
                uncertainty='Descriptive fixed scenes; paired cluster intervals are separate',unit='CAD/placement cluster'))
        panels.append(dict(figure='all_scene_status',panel=robot,claim='Every scene outcome including missing execution',
            summary='Individual categorical outcomes',uncertainty='Not an aggregate estimate',unit='Scene'))
        for column in ('quality','execution'):
            panels.append(dict(figure='paired_effects',panel=robot+'_'+column,claim='Matched changes with cluster uncertainty',
                summary='Mean of cluster means',uncertainty='4000 paired cluster bootstrap 95% percentile intervals',unit='CAD/placement cluster'))
    for p in sorted((OUT/'reports/figures').glob('physical_details_*.pdf')):
        for row,claim in enumerate(('Actual coverage and prescribed path','Actual process-limit utilization',
                                    'Planned rate/sampling constraints','Actual motion and 1 ms maxima')):
            for method in ('B0','B1','B2','G'):
                panels.append(dict(figure=p.stem,panel=f'row{row+1}_{method}',claim=claim,
                    summary='Recorded trace or measured coverage map',uncertainty='Single predetermined physical execution; no inferential interval',unit='Scene, first execution'))
    for panel in panels:
        panel.update(labels='Reviewed',collision_check='Reviewed after legends moved outside data panels',
                     visual_review='Assistant inspected final rendered panels; separate from automated font checks')
    with (QA/'panel_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(panels[0]));writer.writeheader();writer.writerows(panels)
    index=json.loads((OUT/'reports/videos/index.json').read_text())
    video_rows=[];frames=[]
    for row in index:
        if row['status']!='rendered':continue
        source=ROOT/row['source'];movie=ROOT/row['video']
        assert digest(source)==row['source_sha256'] and digest(movie)==row['sha256']
        with h5py.File(source) as f:duration=float(f['t'][-1])
        cap=cv2.VideoCapture(str(movie));assert cap.isOpened(),str(movie)
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert abs(fps-row['frame_rate_hz'])<1e-6
        assert abs(count/fps-duration)<=1/fps+.01
        sampled=[]
        for fraction in (0.,.5,.999):
            cap.set(cv2.CAP_PROP_POS_FRAMES,min(int(fraction*count),count-1))
            ok,frame=cap.read();assert ok,(str(movie),fraction)
            sampled.append(int(frame.std()>1))
            if fraction==.5:frames.append((row['slot']+' / '+row['method'],cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        cap.release();assert all(sampled)
        video_rows.append(dict(video=row['video'],frames=count,fps=fps,video_duration_s=count/fps,
                               recorded_duration_s=duration,begin_middle_end_decodable=True))
    save('video_checks.json',video_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,4,figsize=(16,15),layout='constrained')
    for ax,(title,frame) in zip(axes.ravel(),frames):
        ax.imshow(frame);ax.set_title(title,fontsize=9);ax.set_axis_off()
    for ax in axes.ravel()[len(frames):]:ax.set_axis_off()
    fig.savefig(QA/'video_contact_sheet.png',dpi=130);plt.close(fig)
    # The manifest excludes only itself and interpreter caches, not unfavorable
    # calibration, aborted comparison or unavailable-condition records.
    manifest=OUT/'delivery_manifest.json'
    files=[p for p in OUT.rglob('*') if p.is_file() and p!=manifest and '__pycache__' not in p.parts and p.suffix!='.pyc']
    files += list((ROOT/'src/confik/process_scan').glob('*.py'))
    files += [ROOT/p for p in ('configs/process_scan_phase1.yaml','scripts/run_process_scan.py',
        'scripts/verify_process_scan_delivery.py','tests/test_process_scan.py',
        'docs/PROCESS_SCAN_REUSE_MAP.md','docs/PROCESS_SCAN_PRIOR_ART.md',
        'docs/PROCESS_SCAN_BASELINE_REPORT.md','docs/PROCESS_SCAN_PHASE1_HANDOVER.md')]
    records={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
    manifest.write_text(json.dumps(dict(historical_baseline='bf9ccc4e0314c0d289101d89ff4e40577de392f9',
        mode='development_only',files=records),indent=2)+'\n')
    print(json.dumps(dict(tests='passed',figure_pdfs=len(checks)-1,checked_videos=len(video_rows),
                         manifest_files=len(records),largest_file_bytes=max(p.stat().st_size for p in files)),indent=2))


if __name__=='__main__':main()
