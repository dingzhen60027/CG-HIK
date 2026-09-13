"""Run a selected-input mechanism check, NOT a robot benchmark.

The input is from a previously observed Panda failure. The local model is copied
from the user-provided validated development package. Original repository
verification remains required before treating the command as a server witness.
"""
import json, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
try:
    import numba
except (ImportError, AttributeError):
    stub = types.ModuleType('numba')
    stub.njit = lambda *args, **kwargs: (lambda function: function)
    sys.modules['numba'] = stub
from panda_model_local import LOW, HIGH, VEL, STEP, EPS, OFF, ANG, residual_jac
from bounded_gn_reference import BoundedGN, Settings
from task_excess_gn import solve_task_excess, projected_task_model
ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT/'selected_input_source.json').read_text())
i = source['input'];prev = np.array(i['previous_q']);pd = np.array(i['target_position']);Rd = np.array(i['target_rotation'])


def independent_fk(q):
    T = np.eye(4)
    for index in range(7):
        a = ANG[index];c, s = np.cos(a), np.sin(a)
        offset = np.eye(4);offset[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]];offset[:3,3] = OFF[index]
        c,s = np.cos(q[index]), np.sin(q[index])
        joint = np.eye(4);joint[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
        T = T @ offset @ joint
    end = np.eye(4);end[2,3] = .107
    return T @ end


def inspect(q):
    q = np.asarray(q,float);T = independent_fk(q)
    ep = float(np.linalg.norm(pd-T[:3,3]))
    er = float(Rotation.from_matrix(Rd @ T[:3,:3].T).magnitude())
    vel = float(np.max(np.abs(q-prev)/STEP))
    accepted = bool(ep <= .001 and er <= np.deg2rad(.5)
                    and np.all(q >= LOW-1e-9) and np.all(q <= HIGH+1e-9) and vel <= 1.)
    return dict(accepted=accepted, position_m=ep, orientation_rad=er,
                orientation_deg=float(np.rad2deg(er)), velocity_utilization=vel)


def run():
    derivative_errors = []
    rng = np.random.default_rng(2026091301)
    for _ in range(20):
        e = rng.normal(size=6)
        for start in (0,3):
            e[start:start+3] *= rng.choice([.4,1.6])/np.linalg.norm(e[start:start+3])
        f,v,W=projected_task_model(e)
        h=1e-6
        df=np.array([(projected_task_model(e+h*d)[0]-projected_task_model(e-h*d)[0])/(2*h) for d in np.eye(6)])
        dv=np.column_stack([(projected_task_model(e+h*d)[1]-projected_task_model(e-h*d)[1])/(2*h) for d in np.eye(6)])
        derivative_errors.append([float(np.max(np.abs(df-v))),float(np.max(np.abs(dv-W)))])
        assert np.linalg.eigvalsh(W).min() >= -1e-12
    assert np.max(derivative_errors) < 1e-7
    records=[]
    for kappa in (0.,1.):
        # Generous offline time allowance: this file measures numerical behavior,
        # not online server timing. Same outer numerical cap as the frozen solver.
        solver=BoundedGN(LOW,HIGH,VEL,settings=Settings(posture_weight=kappa,deadline_ms=10000.))
        r=solver.solve(prev,.02,lambda q:residual_jac(q,pd,Rd),lambda q:SimpleNamespace(accepted=inspect(q)['accepted']))
        records.append(dict(method='frozen_point_gn_k'+str(int(kappa)),q=r['q'].tolist(),
                            status=r['internal_status'],iterations=r['iterations'],**inspect(r['q'])))
    for point_control in (True,False):
        r=solve_task_excess(prev,.02,LOW,HIGH,VEL,lambda q:residual_jac(q,pd,Rd),
                           lambda q:inspect(q)['accepted'],point_loss_control=point_control)
        records.append(dict(method='same_loop_point_loss' if point_control else 'task_excess_generalized_gn',
                            q=r['q'],status=r['status'],iterations=r['iterations'],trace=r['trace'],**inspect(r['q'])))
    report=dict(scope='One selected previously observed Panda input; local calculation, not a fresh test, not original-server timing.',
                source_commit='c2a136b2bc0531998354f91c22d795fbe9d9e8c1',input=i,
                gradient_and_metric_max_errors=np.max(derivative_errors,axis=0).tolist(),
                local_model='Panda constants from the supplied single_solver_completed_results.zip',
                public_pose_checks=dict(position_m=.001,orientation_rad=float(np.deg2rad(.5))),
                internal_numerical_margin=1e-6,records=records)
    (ROOT/'selected_input_results.json').write_text(json.dumps(report,indent=2))
    for r in records:
        print(r['method'],r['accepted'],r['position_m']*1000,r['orientation_deg'],r['iterations'])
    # Algebraic linear task example: e_p=x, e_R=1.4-.5*x, two unit tolerances.
    xmin=.56
    algebra=dict(scope='scalar synthetic algebra check, not a robot experiment',
                 point_least_squares_minimizer=xmin,
                 point_min_residuals=[xmin,1.4-.5*xmin],
                 task_feasible_interval=[.8,1.],
                 feasible_example=.9,feasible_example_residuals=[.9,.95])
    assert 1.4-.5*xmin > 1 and all(abs(x)<=1 for x in algebra['feasible_example_residuals'])
    (ROOT/'algebra_example.json').write_text(json.dumps(algebra,indent=2))

if __name__=='__main__': run()
