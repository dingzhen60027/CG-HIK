"""Synthetic reporting regression; no IK experiments."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task_set_report_tests',ROOT/'scripts/report_task_set_boundary.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def synthetic(queries_per_anchor):
    items=[];units=[]
    for a in range(6):
        family=['ordinary','near_singular','near_limit'][a//2]
        for j in range(queries_per_anchor):
            item=dict(uid=f'{a}_{j}',anchor_uid=str(a),anchor_family=family,robot='panda',displacement='regular',alpha=0.)
            items.append(item)
            for method in ('relative','gn'):
                success=1. if method=='relative' else float(a%2)
                units.append(dict(item,method=method,accepted=success,timely=success,
                    latency_ns=float(10+a+(method=='gn')),success_by_repeat=[bool(success)]*3))
    return items,units

def test_anchor_not_derived_query_is_inference_unit():
    cfg=dict(anchor_families=['ordinary','near_singular','near_limit'],bootstrap_seed=414,bootstrap_samples=400)
    one,u1=synthetic(1);nine,u9=synthetic(9)
    p1,_=report.paired(u1,one,'all',cfg,'panda',['gn','relative'])
    p9,_=report.paired(u9,nine,'all',cfg,'panda',['gn','relative'])
    assert p1[0]['anchors']==p9[0]['anchors']==6
    assert p1[0]['queries']==6 and p9[0]['queries']==54
    for key in ('success_difference_ci','timely_difference_ci','latency_ratio_ci'):
        np.testing.assert_allclose(p1[0][key],p9[0][key],rtol=0,atol=1e-14)

def test_unknown_native_iterations_stay_unknown():
    assert report.mean_field([{'iterations':None},{}],'iterations') is None
    assert report.mean_field([{'iterations':2},{'iterations':None}],'iterations')==2.

def test_repetitions_averaged_before_query_comparison():
    i=dict(uid='u',anchor_uid='a',robot='panda',anchor_family='ordinary',displacement='regular',alpha=0.)
    rows=[dict(uid='u',method='gn',repeat=k,accepted=k==0,accepted_within_20ms=k==0,total_latency_ns=10+k) for k in range(3)]
    u=report.units(rows,[i])[0]
    assert u['accepted']==1/3 and u['latency_ns']==11 and u['success_by_repeat']==[True,False,False]
