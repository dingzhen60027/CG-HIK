"""Fixed-input interleaved search repetitions, with no previous-state feedback."""
import gzip
import json
import numpy as np
from ..data.datasets import QueryDataset
from ..revision_compute_allocation.common import json_write
from .study import Study, now
from .contract import METHODS


def run_points(study, folder, methods, scales, sensitivity=False):
    study.check()
    json_write(folder/'started.json',dict(utc=now()))
    selected=json.loads((study.out/'01_protocol/point_identities.json').read_text())
    for robot in study.cfg['robots']:
        base=study.root/study.cfg['point_source']
        ds=QueryDataset.load(base/f'{robot}_queries.npz')
        identities=json.loads((base/f'{robot}_identities.json').read_text())['queries']
        indices=selected[robot]['sensitivity_indices' if sensitivity else 'point_indices']
        total=0
        for scale in scales:
            solvers=study.solvers(robot,methods,scale)
            rng=np.random.default_rng(study.cfg['order_seed']+int(100*scale)+(1000 if sensitivity else 0))
            path=folder/f'{robot}_scale_{scale:g}_records.jsonl.gz'
            try:
                with gzip.open(path,'xt',encoding='utf8') as f:
                    for count,i in enumerate(rng.permutation(indices)):
                        identity=identities[i]
                        for repeat in range(3):
                            order=[m for m in methods if repeat==0 or m.startswith('trac')]
                            for j in rng.permutation(len(order)):
                                m=order[j]
                                row=solvers[m].solve(ds.target_position[i],ds.target_rotation[i],ds.previous_q[i],
                                    study.cfg['dt'],trace=(not sensitivity and m.startswith('dls')))
                                row.update(robot=robot,query_index=int(i),uid=identity['uid'],
                                    query_hash=identity['query_hash'],family=identity['family'],repeat=repeat,scale=scale,
                                    witness_available=identity['witness'],witness_source=f'{base.relative_to(study.root)}/{robot}_queries.npz:reference_q[{i}]',
                                    witness_confirmed_miss=bool(identity['witness'] and not row['accepted']))
                                f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');total+=1
                        if (count+1)%100==0:
                            f.flush();print(f'{robot} scale={scale:g}: {count+1}/{len(indices)} point clusters',flush=True)
            finally:
                for s in solvers.values():s.close()
        json_write(folder/f'{robot}_completed.json',dict(utc=now(),calls=total,queries=len(indices),scales=scales))
    json_write(folder/'completed.json',dict(utc=now()))


def main():
    s=Study();run_points(s,s.out/'02_point_study',METHODS,[1.])


if __name__=='__main__':main()
