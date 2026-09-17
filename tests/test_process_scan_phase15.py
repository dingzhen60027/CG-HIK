import inspect
import json
from pathlib import Path
import numpy as np
from confik.process_scan.task import ScanTask
from confik.process_scan.planning import spline_basis
from confik.process_scan.phase15 import reference,core_one


def test_calibration_is_not_a_comparison_placement():
    c=ScanTask('plane',0,'u','panda',.003,.001,True)
    for placement in (0,1):
        old=ScanTask('plane',placement,'u','panda')
        assert not np.array_equal(c.center,old.center)
    assert c.rows==10
    assert ScanTask('plane',0,'v','panda',.003,.001).rows==14


def test_calibrated_route_is_c2_and_domain_unchanged():
    for family in ('plane','cylinder','saddle'):
        for direction in ('u','v'):
            task=ScanTask(family,0,direction,'panda',.003,.001)
            basis=spline_basis(task,96)
            assert basis.c.shape==(96,96)
            for i in range(len(task.polys)-1):
                for order in range(3):
                    np.testing.assert_allclose(task.polys[i](task.segments[i][4],nu=order),task.polys[i+1](0,nu=order),atol=1e-9)
            for s,p in zip(np.linspace(0,1,37),task.poses(np.linspace(0,1,37))):
                assert task.process_legal(p.position,p.rotation,s)


def test_reference_does_not_use_baseline_results():
    source=inspect.getsource(reference)
    for forbidden in ('shared_b1','planned_path','B1_r0','runs/'):
        assert forbidden not in source
    assert 'least_squares' in source and 'parents' in source


def test_current_run_has_no_b3_or_formal_entry():
    from confik.process_scan.phase15 import run_core
    source=inspect.getsource(run_core)
    assert "['B0','B1','G','B2']" in source
    assert 'B3' not in source and 'formal' not in source


def test_calibration_artifacts_demonstrate_noninitial_physical_improvement():
    root=Path('outputs/process_scan/phase15_baselines/b2_calibration')
    for family in ('plane','cylinder','saddle'):
        d=json.loads((root/family/'summary.json').read_text())
        assert d['status']=='validated'
        assert d['adopted_s']<d['initial_s']
        for label in ('initial','known_faster','adopted'):
            assert d['metrics'][label]['quality_completed']
        assert d['derivative_check']['constraints_scaled_relative_error']<1e-6


def test_three_original_missing_references_have_independent_witnesses():
    root=Path('outputs/process_scan/phase15_baselines/reference_audit_final')
    for slot in ('panda_cylinder_p0_u','panda_cylinder_p1_u','panda_cylinder_p1_v'):
        d=json.loads((root/slot/'selection.json').read_text())
        assert d['status']=='reference_available'
        assert d['attempts'][-1]['dense_feasible']
        assert d['attempts'][-1]['baseline_outputs_used'] is False
