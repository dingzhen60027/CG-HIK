"""Preparation and sealed execution utilities. Never overwrite prior evidence."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from ..config import load_config, load_robot, resolve_path
from ..data.datasets import QueryDataset
from ..types import IKQuery, Pose
from ..latency_pilot_v3.benchmark import query_digest
from ..revision_compute_allocation.common import digest, json_write
from ..revision_compute_allocation.data import trajectories
from ..continuation_mechanism.independent_selection import metadata_keys
from .contract import METHODS, verifier_for, native_mapping
from .outcomes import ContractSolver

OUT=Path('outputs/task_contract_alignment')
CONFIG=Path('configs/task_contract_alignment.yaml')


def now():
    return datetime.now(timezone.utc).isoformat()


class Study:
    def __init__(self, root='.'):
        self.root=Path(root).resolve();self.out=self.root/OUT
        self.cfg=load_config(self.root/CONFIG)
        self.source=load_config(self.root/self.cfg['source_config'])

    def setup(self, robot, scale=1.):
        kin=load_robot(self.source,robot)
        return kin,verifier_for(kin,self.source,scale)

    def solvers(self, robot, methods, scale=1.):
        kin,v=self.setup(robot,scale)
        return {m:ContractSolver(m,kin,v,self.source,self.root/self.cfg['native_library'],
            resolve_path(self.source,self.source['robots'][robot]['urdf'])) for m in methods}

    def check(self, execution=True):
        p=json.loads((self.out/'01_protocol/input_seal.json').read_text())
        for name,h in p['files'].items():
            assert digest(self.root/name)==h,name
        if execution:
            e=json.loads((self.out/'01_protocol/execution_seal.json').read_text())
            for name,h in e['files'].items(): assert digest(self.root/name)==h,name
            assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
        return p

    def prepare(self):
        if self.out.exists():raise FileExistsError(self.out)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()==self.cfg['baseline']
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=self.root).decode().split('\0')
        protected={p:digest(self.root/p) for p in tracked if p and (self.root/p).is_file()}
        upstream=self.root/self.cfg['upstream']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==self.cfg['upstream_sha']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip()
        prior=dict(uids=set(),seeds=set(),hashes=set()); metadata={}
        for p in sorted((self.root/'outputs').rglob('*identit*.json')):
            metadata_keys(json.loads(p.read_text()),prior);metadata[str(p.relative_to(self.root))]=digest(p)
        identities={};mappings=[]; witness_counts={}
        for robot in self.cfg['robots']:
            kin,v=self.setup(robot); _,tight=self.setup(robot,.5)
            base=self.root/self.cfg['point_source']
            ds=QueryDataset.load(base/f'{robot}_queries.npz')
            ids=json.loads((base/f'{robot}_identities.json').read_text())['queries']
            assert len(ds)==len(ids)==3000 and sum(i['witness'] for i in ids)==2500
            valid=[]
            for i,identity in enumerate(ids):
                q=IKQuery(Pose(ds.target_position[i],ds.target_rotation[i]),ds.previous_q[i],self.cfg['dt'])
                assert query_digest(q)==identity['query_hash']
                assert identity['family']==ds.category[i]
                if identity['witness']:
                    assert v.check(ds.reference_q[i],q).accepted
                    if tight.check(ds.reference_q[i],q).accepted:valid.append(i)
                else:
                    assert identity['family']=='constructed_inexecutable'
                    reach=sum(np.linalg.norm(j.origin[:3,3]) for j in kin.chain)
                    assert all(j.kind!='prismatic' for j in kin.chain)
                    assert np.linalg.norm(q.target.position)>reach+2*v.config.position_tolerance
            selected=[]
            for family,n in self.cfg['sensitivity_counts'].items():
                pool=sorted((i for i in valid if ids[i]['family']==family),key=lambda i:ids[i]['uid'])
                assert len(pool)>=n
                selected+=pool[:n]
            identities[robot]=dict(point_source=str(base.relative_to(self.root)),point_indices=list(range(3000)),
                point_uids=[i['uid'] for i in ids],sensitivity_indices=selected,
                sensitivity_uids=[ids[i]['uid'] for i in selected],
                sensitivity_selection='fixed family counts; UID ascending among witnesses verified at 0.5x; no solver outcomes')
            witness_counts[robot]=dict(nominal=2500,tight_eligible=len(valid),sensitivity=len(selected))
            for scale in self.cfg['sensitivity_scales']:
                vv=verifier_for(kin,self.source,scale)
                for m in METHODS:
                    if m.startswith('trac'):mappings.append(dict(robot=robot,scale=scale,method=m,**native_mapping(m,vv)))
            for name in ['urdf']:
                path=resolve_path(self.source,self.source['robots'][robot][name]);protected[str(path)]=digest(path)
        arrays=[];ids=[];online=[];new_hashes=set()
        kin,v=self.setup('ur5e')
        for i,family in enumerate(f for f in self.cfg['trajectory_families'] for _ in range(10)):
            seed=self.cfg['trajectory_seed_base']+i
            cfg=dict(_source=self.source,trajectory_seeds={'ur5e':seed},trajectories_per_family=1,
                     trajectory_frames=150,dt=.02,trajectory_families=[family])
            ds,one=trajectories(kin,cfg,'ur5e');identity=dict(one[0],site_id=f'trajectory_{i:02d}')
            assert seed not in prior['seeds'] and identity['uid'] not in prior['uids']
            assert not set(identity['query_hashes'])&(prior['hashes']|new_hashes)
            new_hashes.update(identity['query_hashes']);ds.trajectory_id[:]=i
            arrays.append(ds);ids.append(identity)
            online.append(dict(site_id=identity['site_id'],uid=identity['uid'],family=family,seed=seed,dt=.02,
                initial_q=ds.previous_q[0].tolist(),target_position=ds.target_position.tolist(),
                target_rotation=ds.target_rotation.tolist()))
        folder=self.out/'01_protocol';folder.mkdir(parents=True)
        joined=QueryDataset(**{k:np.concatenate([getattr(d,k) for d in arrays]) for k in QueryDataset.__dataclass_fields__})
        with (folder/'ur5e_reference_trajectories.npz').open('xb') as f:
            np.savez_compressed(f,**{k:getattr(joined,k) for k in QueryDataset.__dataclass_fields__})
        json_write(folder/'point_identities.json',identities)
        json_write(folder/'ur5e_trajectory_identities.json',ids)
        json_write(folder/'ur5e_online_targets.json',online)
        json_write(folder/'native_mappings.json',mappings)
        json_write(folder/'protected_files.json',protected)
        json_write(folder/'protocol.json',dict(created_utc=now(),config=self.cfg,witness_validation=witness_counts,
            public_contract=asdict(v.config),backend='URDFKinematics; native TRAC-IK uses identical KDL chain',
            point_calls=120000,sensitivity_calls=21000,new_trajectory_calls=84000,
            historical_panda_calls=84000,scope='planned supplementary evaluation; points previously observed; no tuning or global gate',
            identity_scan=metadata,old_identity_counts={k:len(v) for k,v in prior.items()},
            ur5e_reference_verified_frames=6000,all_fixed_frames_retained=True))
        files={str(p.relative_to(self.root)):digest(p) for p in folder.iterdir() if p.is_file()}
        files.update({str(CONFIG):digest(self.root/CONFIG),'docs/TASK_CONTRACT_ALIGNMENT_PROTOCOL.md':digest(self.root/'docs/TASK_CONTRACT_ALIGNMENT_PROTOCOL.md')})
        json_write(folder/'input_seal.json',dict(created_utc=now(),files=files))
        print('Prepared 6000 fixed point identities, 1000 sensitivity identities and 40 UR5e reference paths; no study solver calls.',flush=True)

    def seal(self):
        self.check(False)
        assert (self.out/'01_protocol/smoke_and_tests.json').exists()
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        assert head!=self.cfg['baseline'],'commit protocol and code before smoke and execution'
        names=subprocess.check_output(['git','ls-files','src/confik/task_contract_alignment','tests/test_task_contract_alignment.py','configs/task_contract_alignment.yaml','scripts/run_task_contract_alignment.sh'],cwd=self.root,text=True).splitlines()
        files={p:digest(self.root/p) for p in names}
        files[self.cfg['native_library']]=digest(self.root/self.cfg['native_library'])
        oldlib='tmp/tolerance_solver_build/libtolerance_trac.so'
        old=json.loads((self.root/self.cfg['panda_authority']/'protocol.json').read_text())
        assert digest(self.root/oldlib)==old['measurement_sources'][oldlib]
        files[oldlib]=digest(self.root/oldlib)
        for p in ['src/confik/continuation_mechanism/tolerance_solvers.py',
                  'src/confik/continuation_mechanism/tolerance_comparison.py',
                  'src/confik/revision_compute_allocation/data.py']:
            assert digest(self.root/p)==old['measurement_sources'][p]
            files[p]=digest(self.root/p)
        for p in (self.root/self.cfg['upstream']/'trac_ik_lib').rglob('*'):
            if p.is_file():files[str(p.relative_to(self.root))]=digest(p)
        for p in ['src/confik/solvers/dls.py','src/confik/solvers/verifier.py','src/confik/continuation_mechanism/observation.py']:
            files[p]=digest(self.root/p)
        json_write(self.out/'01_protocol/execution_seal.json',dict(created_utc=now(),code_commit=head,files=files,
            environment=dict(python=platform.python_version(),numpy=np.__version__,
                threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                affinity=sorted(os.sched_getaffinity(0)),shared_workstation=True),
            note='all mapping, inputs and measurement code frozen before point outcomes; search random seed not controlled'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','seal']);a=p.parse_args()
    getattr(Study(),a.stage)()


if __name__=='__main__':main()
