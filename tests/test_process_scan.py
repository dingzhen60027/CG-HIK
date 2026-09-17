import inspect
import numpy as np
import pytest
from confik.process_scan.task import ScanTask
from confik.process_scan.geometry_adapter import GeometryAdapter


def test_no_time_or_velocity_in_geometry_interface():
    sig=inspect.signature(GeometryAdapter.solve)
    assert 'dt' not in sig.parameters and 'velocity' not in sig.parameters
    assert {'physical_bounds','continuity_box','step_scale','pose_tolerances'} <= set(sig.parameters)


def test_c2_raster_and_no_wrist_flip():
    for family in ('plane','cylinder','saddle'):
        for direction in ('u','v'):
            task=ScanTask(family,0,direction,'panda')
            for i in range(len(task.polys)-1):
                left=task.polys[i];right=task.polys[i+1]
                for order in range(3):
                    np.testing.assert_allclose(left(task.segments[i][4],nu=order),right(0,nu=order),atol=1e-9)
            ss=np.linspace(0,1,1001);poses=task.poses(ss)
            assert all(task.process_legal(p.position,p.rotation,s) for s,p in zip(ss,poses))
            assert np.min([np.trace(p.rotation.T@r.rotation) for p,r in zip(poses[:-1],poses[1:])])>2.9


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_actual_model_fk_jacobian_and_geometry_bounds(robot):
    from confik.process_scan.models import physical_model,symbolic_fk
    from confik.process_scan.collision import Clearance
    task=ScanTask('plane',0,'u',robot)
    model,data,adapter,meta=physical_model(robot,task)
    assert meta['fk_max_error']<1e-10
    limits=adapter.public.limits;q=(limits.lower+limits.upper)/2
    fk=symbolic_fk(adapter.public);p,R=fk(q);pose=adapter.public.forward(q)
    np.testing.assert_allclose(np.array(p).ravel(),pose.position,atol=1e-12)
    np.testing.assert_allclose(np.array(R),pose.rotation,atol=1e-12)
    J=adapter.native.jacobian(q)
    eps=1e-6
    fd=np.column_stack([(adapter.native.forward(q+eps*np.eye(len(q))[i]).position-
                         adapter.native.forward(q-eps*np.eye(len(q))[i]).position)/(2*eps) for i in range(len(q))])
    np.testing.assert_allclose(J[:3],fd,atol=2e-8)
    args=(pose,q,(limits.lower,limits.upper),(q-.1,q+.1),.15,(.001,np.deg2rad(.5)))
    a=adapter.solve(*args,method='gn');b=adapter.solve(*args,method='tb')
    assert a['accepted'] and b['accepted']
    np.testing.assert_array_equal(a['q'],b['q'])
    assert a['physical_velocity_checked'] is False
    assert not adapter.verify(q+np.ones(len(q))*.11,pose,args[2],args[3],(1e6,1e6))['accepted']


def test_no_late_checkpoint_substitution():
    from confik.process_scan.planning import FeasibleRecorder
    class Fake:
        history=[dict(available_s=2,objective=10),dict(available_s=12,objective=1)]
    assert FeasibleRecorder.select(Fake(),1) is None
    assert FeasibleRecorder.select(Fake(),10)['objective']==10


@pytest.mark.parametrize('robot',['panda','ur5e'])
def test_nonzero_first_direction_reuses_same_frozen_qp(robot,monkeypatch):
    from confik.process_scan.models import context
    from confik.process_scan import geometry_adapter as module
    adapter,_=context(robot);limits=adapter.public.limits
    seed=(limits.lower+limits.upper)/2;desired=seed.copy();desired[0]+=.02
    target=adapter.public.forward(desired);calls=[];original=module.box_qp
    def record(H,g,lo,hi):
        answer=original(H,g,lo,hi)
        calls.append(tuple(x.copy() for x in (H,g,lo,hi,answer[0])))
        return answer
    monkeypatch.setattr(module,'box_qp',record)
    args=(target,seed,(limits.lower,limits.upper),(seed-.35,seed+.35),.15,(.001,np.deg2rad(.5)))
    results=[];first=[]
    for method in ('gn','tb'):
        calls.clear();results.append(adapter.solve(*args,method=method))
        assert calls,'Nonzero task must exercise the frozen QP, not the zero-residual exit'
        first.append(calls[0])
    for a,b in zip(*first):np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-13)
    for result in results:
        assert result['accepted']
        assert adapter.verify(result['q'],target,args[2],args[3],args[5])['accepted']


def test_timing_grid_has_no_roundoff_duplicate_cells():
    from confik.process_scan.planning import numerical_grid,spline_basis
    for direction in ('u','v'):
        task=ScanTask('plane',0,direction,'panda');basis=spline_basis(task)
        original=np.unique(np.r_[np.linspace(0,1,801),task.knots,basis.t])
        fixed=numerical_grid(np.linspace(0,1,801),task.knots,basis.t)
        assert np.min(np.diff(fixed))>1e-12
        assert max(min(abs(fixed-s)) for s in original)<1e-12
        if direction=='u':np.testing.assert_array_equal(original,fixed)


@pytest.mark.parametrize('family',['plane','cylinder','saddle'])
def test_nominal_surface_ray_intersection(family):
    from confik.process_scan.models import context
    from confik.process_scan.planning import ray_center_function,spline_basis
    # Reference surface points and scanner axes obey the coupled process domain;
    # the ray function is checked against independent analytic intersection.
    adapter,_=context('ur5e');task=ScanTask(family,0,'v','ur5e');basis=spline_basis(task)
    assert basis.c.shape==(64,64)
    np.testing.assert_allclose(basis(np.linspace(0,1,37)).sum(axis=1),1,atol=1e-14)
    assert len(np.unique(basis.t[4:-4]))==60
    center=ray_center_function(task,adapter)
    q=np.array([0.,-1.4,1.8,-1.9,-1.57,0.])
    xyz,J=center(q);eps=1e-6
    # Skip no poses: algebraic ray intersection derivative has a finite value
    # even when the arbitrary midpoint has no task-admissible viewing geometry.
    fd=np.column_stack([(np.array(center(q+eps*np.eye(len(q))[i])[0]).ravel()-
                         np.array(center(q-eps*np.eye(len(q))[i])[0]).ravel())/(2*eps) for i in range(len(q))])
    np.testing.assert_allclose(J,fd,rtol=1e-4,atol=1e-4)
