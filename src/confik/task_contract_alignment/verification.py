"""Independent FK/verifier replay after measurement, never solver calls."""
import json
import numpy as np
from ..data.datasets import QueryDataset
from ..types import IKQuery,Pose
from ..revision_compute_allocation.common import json_write,digest
from .study import Study,now
from .aggregate import records
from .outcomes import taxonomy


def replay(row,query,verifier):
    q=np.asarray(row['q']) if row['q'] is not None else np.full(verifier.kinematics.nq,np.nan)
    c=verifier.check(q,query)
    assert c.accepted==row['accepted']
    assert list(c.reasons)==row['verification_reasons']
    assert c.finite_ok==row['finite']
    if c.finite_ok:
        assert abs(c.position_error-row['position_error'])<1e-12
        assert abs(c.orientation_error-row['orientation_error'])<1e-12
    assert row['returned_within_20ms']==(row['total_latency_ns']<=20_000_000)
    if 'accepted_within_20ms' in row:
        assert row['accepted_within_20ms']==(c.accepted and row['total_latency_ns']<=20_000_000)
        assert row['taxonomy']==taxonomy(row['internal_ok'],c.accepted)
        assert row['total_latency_ns']==sum(row[k] for k in ['conversion_ns','bounds_setup_ns','solve_ns','verification_ns','accounting_remainder_ns'])
    return q,c


def main():
    s=Study();s.check();counts={};witnesses=[]
    for sub in ['02_point_study','03_contract_sensitivity']:
        total=0
        for p in sorted((s.out/sub).glob('*_records.jsonl.gz')):
            robot=p.name.split('_')[0];scale=float(p.name.split('_')[2]);kin,v=s.setup(robot,scale)
            ds=QueryDataset.load(s.root/s.cfg['point_source']/f'{robot}_queries.npz');seen=set()
            for r in records(p):
                i=r['query_index'];key=(i,r['method'],r['repeat']);assert key not in seen;seen.add(key)
                query=IKQuery(Pose(ds.target_position[i],ds.target_rotation[i]),ds.previous_q[i],.02)
                replay(r,query,v)
                if r['witness_available']:assert v.check(ds.reference_q[i],query).accepted
                else:assert not r['accepted']
                total+=1
        counts[sub]=total
    assert counts['02_point_study']==120000 and counts['03_contract_sensitivity']==21000
    for robot,folder in [('panda',s.root/s.cfg['panda_authority']),('ur5e',s.out/'04_trajectory_study')]:
        kin,v=s.setup(robot);total=0;run_count=0
        for path in sorted((folder/'online_runs').glob('*.jsonl.gz')):
            rr=list(records(path));assert len(rr)==150
            previous=np.asarray(rr[0]['previous_q']);complete=True
            for t,r in enumerate(rr):
                assert r['frame']==t and np.array_equal(r['previous_q'],previous)
                query=IKQuery(Pose(r['target_position'],r['target_rotation']),previous,r['dt'])
                q,c=replay(r,query,v)
                if c.accepted:previous=q
                else:complete=False
                assert np.array_equal(r['accepted_state_q'],previous);total+=1
            run_count+=1
            if complete:witnesses.append(dict(robot=robot,uid=rr[0]['uid'],run_id=rr[0]['run_id'],
                source=str(path.relative_to(s.root)),frames=150,sha256=digest(path)))
        assert run_count==560 and total==84000
        counts[robot+'_trajectory_replayed']=total
    protected=json.loads((s.out/'01_protocol/protected_files.json').read_text())
    for name,h in protected.items():assert digest(s.root/name)==h,name
    json_write(s.out/'05_aggregate/successful_witness_index.json',witnesses)
    json_write(s.out/'05_aggregate/verification.json',dict(utc=now(),counts=counts,
        accepted_contract_violations=0,protected_files_unchanged=len(protected),successful_trajectory_witnesses=len(witnesses)))


if __name__=='__main__':main()
