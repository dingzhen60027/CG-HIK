"""UR5e counterpart to the frozen Panda trajectory protocol."""
import gzip
import json
import numpy as np
from ..config import resolve_path
from ..revision_compute_allocation.common import json_write
from ..continuation_mechanism.tolerance_solvers import ToleranceSolver,feedback
from ..continuation_mechanism.tolerance_comparison import run_summary
from .study import Study,now
from .contract import TRAJECTORY_METHODS


def main():
    s=Study();s.check()
    assert (s.out/'03_contract_sensitivity/completed.json').exists()
    folder=s.out/'04_trajectory_study';json_write(folder/'started.json',dict(utc=now()))
    inputs=json.loads((s.out/'01_protocol/ur5e_online_targets.json').read_text())
    kin,v=s.setup('ur5e')
    # Exact old Panda online wrapper and old frozen binary: no extra trace cost,
    # identical timing/feedback semantics; detailed point timing is a separate arm.
    solvers={m:ToleranceSolver(m,kin,v,s.source,s.root/'tmp/tolerance_solver_build/libtolerance_trac.so',
        resolve_path(s.source,s.source['robots']['ur5e']['urdf'])) for m in TRAJECTORY_METHODS}
    jobs=[(i,m,r) for i in inputs for m in TRAJECTORY_METHODS for r in range(3 if m.startswith('trac') else 1)]
    summaries=[];rng=np.random.default_rng(s.cfg['order_seed']+3000)
    (folder/'online_runs').mkdir()
    try:
        for index in rng.permutation(len(jobs)):
            i,m,repeat=jobs[index];rid=f'{i["site_id"]}_{m}_r{repeat}';previous=np.asarray(i['initial_q']);rows=[]
            for t,(p,r) in enumerate(zip(i['target_position'],i['target_rotation'])):
                result=solvers[m].solve(p,r,previous,i['dt'])
                row=dict(run_id=rid,robot='ur5e',site_id=i['site_id'],uid=i['uid'],family=i['family'],
                    repeat=repeat,frame=t,previous_q=previous.tolist(),target_position=p,target_rotation=r,dt=i['dt'],**result)
                previous=feedback(previous,result);row['accepted_state_q']=previous.tolist();rows.append(row)
            with gzip.open(folder/'online_runs'/f'{rid}.jsonl.gz','xt',encoding='utf8') as f:
                for row in rows:f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            summaries.append(dict(run_id=rid,robot='ur5e',site_id=i['site_id'],uid=i['uid'],family=i['family'],
                method=m,repeat=repeat,frames=150,initial_q=i['initial_q'],**run_summary(rows)))
            if len(summaries)%20==0:print(f'UR5e {len(summaries)}/{len(jobs)} complete 150-frame runs recorded',flush=True)
    finally:
        for solver in solvers.values():solver.close()
    json_write(folder/'online_run_summaries.json',summaries)
    json_write(folder/'completed.json',dict(utc=now(),runs=len(summaries),calls=len(summaries)*150))


if __name__=='__main__':main()
