"""Interface tests, not evidence of method performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('final_evidence_entry',ROOT/'scripts/run_task_balance_final_evidence.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
CFG=yaml.safe_load(entry.CONFIG.read_text())

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_paired_reconstruction_and_original_witness(robot):
    items=json.loads((entry.old.old.OUT/f'inputs/development_{robot}.json').read_text())
    _,kin,original,_=entry.context(robot,CFG)
    for condition in CFG['sensitivity']:
        _,_,v,_=entry.configured_context(robot,CFG,condition)
        for item in items:
            new=entry.reconstruct(item,kin,v,original,condition)
            np.testing.assert_array_equal(new['previous_q'],item['previous_q'])
            assert new['configuration_witness_check']['accepted']
            pose=kin.forward(np.array(item['q_witness']))
            ep=np.linalg.norm(np.array(new['target_position'])-pose.position)/v.config.position_tolerance
            er=Rotation.from_matrix(np.array(new['target_rotation'])@pose.rotation.T).magnitude()/v.config.orientation_tolerance
            assert ep==pytest.approx(item['alpha'],abs=1e-10)
            assert er==pytest.approx(item['alpha'],abs=1e-10)
            if condition=='nominal':
                assert new['target_position']==item['target_position']
                assert new['target_rotation']==item['target_rotation']

@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_instances_only_change_declared_settings(robot):
    nominal,_,v,_=entry.factory('relative',robot,CFG)
    strict,_,_,_=entry.factory('eta010',robot,CFG)
    loose,_,_,_=entry.factory('eta050',robot,CFG)
    assert entry.replace(strict.settings,forcing=.25)==nominal.settings
    assert entry.replace(loose.settings,forcing=.25)==nominal.settings
    for condition in ['position_half','orientation_half']:
        for method in ['relative','gn','direct_sqp']:
            s,kin,local,_=entry.factory(method,robot,CFG,condition)
            q=(kin.limits.lower+kin.limits.upper)/2;pose=kin.forward(q)
            row=s.solve(pose.position,pose.rotation,q,.02)
            assert row['accepted'];np.testing.assert_array_equal(row['q'],q)
            assert local.config.velocity_tolerance==v.config.velocity_tolerance
            if method=='relative':assert s.settings==nominal.settings
            if method=='direct_sqp':assert s.settings['constraint_interior_guard']==1e-7

def test_droid_raw_schema_pairing_and_period_are_not_observation_fk():
    selection=json.loads((entry.OUT/'source_replay/selection.json').read_text())
    assert len(selection['selected'])==30
    interface=[x for x in selection['selected'] if x['selection']=='interface']
    evaluation=[x for x in selection['selected'] if x['selection']=='evaluation']
    assert len(interface)==6 and len(evaluation)==24
    assert not ({x['session'] for x in interface}&{x['session'] for x in evaluation})
    assert [x['uid'] for x in selection['candidates']]==sorted(x['uid'] for x in selection['candidates'])
    # The helper never imports DROID's hardware-bearing modules.
    source=(ROOT/'src/confik/task_balance_replay.py').read_text()
    assert 'from droid.' not in source and 'import droid' not in source
