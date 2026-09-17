#!/usr/bin/env python3
"""Read-only evidence checks; no IK, NLP, retiming, or physics integration."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import json
import subprocess
import numpy as np
import h5py
from scipy.interpolate import BSpline
from confik.process_scan.phase15 import OUT,OLD,code_hashes
from confik.process_scan.study import sha,write,FROZEN
from confik.process_scan.models import context
from confik.process_scan.execution import coverage
from confik.process_scan.task import ScanTask


def main():
    seal=json.loads((OUT/'inputs/seal.json').read_text());issues=[];executions=[];copies=[]
    assert seal['code_hashes']==code_hashes()
    assert len(seal['scenes'])==24
    for path,digest in seal['files'].items():assert sha(OUT/path)==digest,path
    for path in FROZEN+['paper/main.tex']:
        original=subprocess.check_output(['git','show','f85613a5a64c229de5d370bc9fbe4e38fc460988:'+path])
        assert (ROOT/path).read_bytes()==original,path
    subprocess.run(['git','diff','--exit-code','f85613a5a64c229de5d370bc9fbe4e38fc460988','--',
                    str(OLD.relative_to(ROOT))],cwd=ROOT,check=True,capture_output=True)
    for identity in seal['scenes']:
        original=json.loads((OLD/'inputs'/identity['slot']/'identity.json').read_text())
        assert identity['scene_uid']==original['scene_uid']
        task=ScanTask(**identity['task']);adapter,_=context(identity['robot'])
        for method in ('B0','B1','G','B2'):
            for repeat in range(3):
                root=OUT/'runs'/identity['slot']/f'{method}_r{repeat}'
                m=json.loads((root/'metrics.json').read_text())
                if method=='B2' and (root/'path_initialization.npz').exists():
                    shared=np.load(OUT/'shared_b1'/identity['slot']/'path.npz');given=np.load(root/'path_initialization.npz')
                    for key in ('coeff','knots','s','x','u','t'):np.testing.assert_array_equal(given[key],shared[key])
                    copies.append(dict(slot=identity['slot'],repeat=repeat,identical=True))
                if not m.get('physical_run'):continue
                assert repeat==0
                with h5py.File(root/'execution.h5') as f:
                    assert not f.attrs['qpos_runtime_overwrite']
                    assert f.attrs['source']=='torque_actuators_mj_step'
                    h={k:f[k][:] for k in f}
                np.testing.assert_allclose(np.diff(h['t']),.001,atol=1e-10,rtol=0)
                path=np.load(root/'planned_path.npz');basis=BSpline(path['knots'],np.eye(len(path['coeff'])),3)
                np.testing.assert_allclose(h['q'][0],basis(0)@path['coeff'],atol=1e-12)
                np.testing.assert_allclose(h['q'][0],identity['common_start'],atol=1e-12)
                vu=float(np.max(abs(h['dq'])/(.5*adapter.public.limits.velocity)))
                au=float(np.max(abs(h['qacc']))/2)
                np.testing.assert_allclose([vu,au],[m['velocity_utilization_max'],m['acceleration_utilization_max']],atol=1e-12)
                rays=np.load(root/'scan_samples.npz')
                points=rays['points'][rays['quality_valid']]
                cv,hole,mask=coverage(task,points)
                np.testing.assert_allclose([cv,hole],[m['valid_coverage_fraction'],m['max_hole_diameter_upper_bound_m']],atol=1e-12)
                np.testing.assert_array_equal(mask,rays['coverage_mask'])
                executions.append(dict(slot=identity['slot'],method=method,samples=len(h['t']),
                    dt_s=.001,actual_max_velocity_ratio=vu,actual_max_acceleration_ratio=au,
                    planned_max_velocity_ratio=float(np.max(abs(h['dqr'])/(.5*adapter.public.limits.velocity))),
                    planned_max_acceleration_ratio=float(np.max(abs(h['ddqr']))/2),coverage_recomputed=cv,hole_recomputed_m=hole))
    videos=[];video_index=OUT/'reports/videos/index.json'
    if video_index.exists():
        index=json.loads(video_index.read_text());assert len(index)==24
        for item in index:
            if item['status']!='rendered':continue
            source=ROOT/item['source'];movie=ROOT/item['video']
            assert sha(source)==item['source_sha256']
            assert sha(movie)==item['sha256']
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]))
            stream=next(s for s in probe['streams'] if s['codec_type']=='video')
            with h5py.File(source) as f:duration=float(f['t'][-1])
            assert (stream['width'],stream['height'])==(640,480)
            assert stream['r_frame_rate']=='10/1'
            assert abs(float(probe['format']['duration'])-duration)<.11
            videos.append(dict(video=item['video'],recorded_duration_s=duration,
                               video_duration_s=float(probe['format']['duration']),source_unchanged=True))
    write(OUT/'reports/qa/record_checks.json',dict(status='passed',issues=issues,executions=executions,b2_copies=copies,videos=videos,
        frozen_old_source_unchanged=True,all_24_identities_preserved=True,new_ik_or_physics_run=False))
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    write(OUT/'delivery_manifest.json',dict(files=files,complete_conditions=288,formal_scenes=0,old_evidence_overwritten=False))
    print('PASSED:',len(executions),'physical records;',len(copies),'identical B2 initial paths;',len(files),'files')


if __name__=='__main__':main()
