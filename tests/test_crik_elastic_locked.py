"""Decision equivalence uses an identical supplied backup, not paired TRAC seeds."""
import copy
import inspect
import numpy as np
import pytest
from confik.config import load_config,load_robot,resolve_path
from confik.types import IKQuery
from confik.solvers.verifier import SolutionVerifier,VerifierConfig
from confik.correction_reserve.elastic_runtime import ElasticInterventionIK
from confik.correction_reserve.mu0_analytic import MuZeroAnalyticIK


def test_only_shortcut_and_truthful_labels_differ_from_frozen_solve():
    original=inspect.getsource(ElasticInterventionIK.solve)
    expected=original.replace("direct=bool(initial_legal and initial['demand_met'])","direct=bool(initial_legal)")
    expected=expected.replace("elif direct or (selected and met):decision='full_demand_satisfied'",
        "elif direct:decision='analytic_zero_intervention'\n        elif selected and met:decision='full_demand_satisfied'")
    expected=expected.replace('out.update(method=MU_METHODS[self.mu]',"out.update(method='cr_ik_mu0_analytic'")
    assert inspect.getsource(MuZeroAnalyticIK.solve)==expected


@pytest.fixture(params=['panda','ur5e'])
def pair(request):
    source=load_config('configs/paper_v2.yaml');kin=load_robot(source,request.param)
    v=SolutionVerifier(kin,VerifierConfig(**source['verifier']))
    q=kin.random_configuration(np.random.default_rng(98301),margin=.2)
    urdf=resolve_path(source,source['robots'][request.param]['urdf'])
    args=(kin,v,source,'tmp/task_contract_build/libcontract_trac.so',urdf)
    kw=dict(demand_parameters={'minimum':100.,'initial':100.})
    a=ElasticInterventionIK(*args,mu=0.,**kw);b=MuZeroAnalyticIK(*args,**kw)
    yield a,b,kin,v,q
    a.close();b.close()


def same_decision(a,b):
    for key in ('accepted','q','nominal_next_q','actual_objective','xi_actual','demand_met',
                'recovery_mode','demand','initial_pair_legal','command_unchanged','predicted_gamma'):
        assert a[key]==b[key],key


def test_legal_unmet_pair_returns_exact_same_command_and_nominal_history(pair):
    a,b,kin,v,q=pair;origin=q.copy();native=a.backup.solve
    for frame in range(6):
        pose=kin.forward(origin+.0001*frame)
        backup=native(pose.position,pose.rotation,q,.02)
        assert backup['accepted']
        a.backup.solve=lambda *args:copy.deepcopy(backup)
        b.backup.solve=lambda *args:copy.deepcopy(backup)
        ra=a.solve(pose.position,pose.rotation,q);rb=b.solve(pose.position,pose.rotation,q)
        same_decision(ra,rb)
        assert rb['direct_return'] and not rb['optimization_called'] and b.cone is None
        assert not rb['demand_met'] and rb['decision']=='analytic_zero_intervention'
        assert np.array_equal(rb['q'],backup['q'])
        assert v.check(np.array(rb['q']),IKQuery(pose,q,.02)).accepted
        np.testing.assert_array_equal(a.last_nominal,b.last_nominal)
        q=np.array(rb['q'])


@pytest.mark.parametrize('missing_backup',[False,True])
def test_illegal_initial_pair_uses_same_geometric_recovery(pair,monkeypatch,missing_backup):
    import confik.correction_reserve.elastic_runtime as er
    import confik.correction_reserve.mu0_analytic as ar
    a,b,kin,v,q=pair;pose=kin.forward(q)
    backup=a.backup.solve(pose.position,pose.rotation,q,.02)
    if missing_backup:
        backup.update(accepted=False,q=None,finite=False,internal_ok=False,verification_reasons=['injected_missing'])
    for s in (a,b):
        s.backup.solve=lambda *args:copy.deepcopy(backup)
        if not missing_backup:
            evaluate=s.evaluate;count=[0]
            def injected(*args,_evaluate=evaluate,_count=count,**kw):
                result=_evaluate(*args,**kw);_count[0]+=1
                if _count[0]==1:result.update(legal=False,objective=None,xi_actual=None)
                return result
            s.evaluate=injected
    calls=[]
    def fixed_recovery(*args):
        calls.append(args[9]);assert args[9]=='predictive'
        return np.zeros(2*kin.nq+1),[dict(stage='fixed_test_recovery',status='Solved',native_seconds=0.)]
    monkeypatch.setattr(er,'subproblem',fixed_recovery);monkeypatch.setattr(ar,'subproblem',fixed_recovery)
    ra=a.solve(pose.position,pose.rotation,q);rb=b.solve(pose.position,pose.rotation,q)
    same_decision(ra,rb)
    assert calls==['predictive','predictive'] and rb['decision']=='geometric_recovery'
    assert not rb['direct_return'] and rb['accepted']


def test_locked_design_and_jobs():
    from confik.correction_reserve.locked_study import configurations,jobs_for
    cfg,_,_=configurations()
    assert cfg['mu']==.25 and cfg['repeats']==3 and cfg['frames']==300
    assert cfg['trajectories_per_family']==40 and len(set(cfg['methods']))==6
    items=[dict(uid=str(i)) for i in range(160)]
    jobs=jobs_for(items,cfg)
    assert len(jobs)==2880 and sum(m=='pink_qp' for _,m,_ in jobs)==480
