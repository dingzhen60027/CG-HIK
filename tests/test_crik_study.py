"""Feedback, metadata and geometry tests, with no frozen-output writes."""
import numpy as np
import yaml
from pathlib import Path
from confik.correction_reserve.study import context,execute_trajectory
from confik.correction_reserve.data import reference_path
from confik.types import IKQuery


def test_rejection_holds_state_and_metadata_can_repeat_input():
    cfg=yaml.safe_load(Path('configs/correction_reserve.yaml').read_text())
    _,kin,v,_=context('panda',cfg)
    q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q)
    class Fake:
        def __init__(self):self.inputs=[]
        def solve(self,p,r,previous,dt):
            self.inputs.append(previous.copy());i=len(self.inputs)
            return dict(previous_q=previous.tolist(),method='fake',accepted=i!=2,
                q=(previous if i!=2 else previous+.5).tolist(),finite=True,
                joint_limit_ok=True,velocity_ok=i!=2,position_error=0.,orientation_error=0.,
                total_latency_ns=100,verification_reasons=[] if i!=2 else ['velocity'])
    solver=Fake()
    item=dict(robot='panda',uid='test',site_id='test',family='smooth',dt=.02,
        initial_q=q.tolist(),target_position=[pose.position.tolist()]*3,
        target_rotation=[pose.rotation.tolist()]*3)
    rows,summary=execute_trajectory(solver,kin,v,item,'fake',0)
    assert len(solver.inputs)==3 and np.array_equal(solver.inputs[2],q)
    assert not summary['completion'] and summary['successful_prefix']==1
    assert rows[2]['frame']==2


def test_geometry_paths_have_original_contract_witnesses():
    cfg=yaml.safe_load(Path('configs/correction_reserve.yaml').read_text())
    for robot in cfg['robots']:
        _,kin,v,_=context(robot,cfg)
        for i,family in enumerate(cfg['formal']['families']):
            reference,_=reference_path(kin,family,981009000+i,300,.02,i)
            for t,q in enumerate(reference):
                assert v.check(q,IKQuery(kin.forward(q),reference[max(0,t-1)],.02)).accepted
