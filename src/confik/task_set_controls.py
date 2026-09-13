"""Two controlled baselines; the frozen GN and completion files are not edited.

Fixed weights change only cost/H/g in an instance-local copy of the GN function.
Clarabel changes only the dual-direction provider in an instance-local copy of
the completion function. No global monkeypatch, altered task tests or fallback.
"""
import inspect
import textwrap
import types
from time import perf_counter_ns
import numpy as np
from .bounded_gn import BoundedGN
from .bounded_gn_adapter import SingleBoundedGN
from .task_balance_gn import CompletedTaskBalanceGN,solve_completion,task_value
from .task_balance_reference import EpigraphReference,objective


def weighted_cost(e,theta):
    if theta==.5:return .5*float(e@e)
    return theta*float(e[:3]@e[:3])+(1-theta)*float(e[3:]@e[3:])

def weighted_hessian(G,theta):
    if theta==.5:return G.T@G
    return 2*theta*(G[:3].T@G[:3])+2*(1-theta)*(G[3:].T@G[3:])

def weighted_gradient(G,e,theta):
    if theta==.5:return G.T@e
    return 2*theta*(G[:3].T@e[:3])+2*(1-theta)*(G[3:].T@e[3:])


# Preserve every original statement except the four cost sites and task H/g.
# Explicit source assertions make an upstream mismatch fail, not silently drift.
_source=textwrap.dedent(inspect.getsource(BoundedGN.solve))
_replacements={'.5*float(e@e)':'weighted_cost(e,self.theta)',
               '.5*float(er@er)':'weighted_cost(er,self.theta)',
               '.5*float(eq@eq)':'weighted_cost(eq,self.theta)',
               'G.T@G':'weighted_hessian(G,self.theta)',
               'G.T@e':'weighted_gradient(G,e,self.theta)'}
for _old,_new in _replacements.items():
    assert _source.count(_old)==1,('GN adapter mismatch',_old)
    _source=_source.replace(_old,_new)
_namespace=dict(BoundedGN.solve.__globals__,weighted_cost=weighted_cost,
                weighted_hessian=weighted_hessian,weighted_gradient=weighted_gradient)
exec(compile(_source,__file__+':fixed-weight-original-loop','exec'),_namespace)
_weighted_solve=_namespace['solve']


class FixedWeightGN(SingleBoundedGN):
    def __init__(self,kin,verifier,urdf,theta):
        if not 0<theta<1:raise ValueError('Fixed theta must be inside (0,1)')
        super().__init__(kin,verifier,urdf,posture_weight=1.)
        self.method='fixed_weight';self.engine.theta=float(theta)
        self.engine.solve=types.MethodType(_weighted_solve,self.engine)


class ClarabelBalance(CompletedTaskBalanceGN):
    """Exactly the frozen completion outer function, with a local conic backend."""
    def __init__(self,kin,verifier,urdf):
        super().__init__(kin,verifier,urdf,posture_weight=1.,forcing=None)
        self.method='clarabel';self.epigraph=EpigraphReference(kin.nq)
        kernel_globals=dict(solve_completion.__globals__,completion_dual_direction=self.direction)
        kernel=types.FunctionType(solve_completion.__code__,kernel_globals,
            solve_completion.__name__,solve_completion.__defaults__,solve_completion.__closure__)
        adapter=CompletedTaskBalanceGN.solve
        adapter_globals=dict(adapter.__globals__,solve_completion=kernel)
        bound=types.FunctionType(adapter.__code__,adapter_globals,adapter.__name__,adapter.__defaults__,adapter.__closure__)
        self.solve=types.MethodType(bound,self)

    def direction(self,e,G,lo,hi,lam,kappa,posture,w,box_qp,first,
                  acceptable,settings,deadline,trace=False):
        pzero=task_value(e)+.5*kappa*float(posture@posture)
        base=dict(reason='deadline',theta=.5,dual_updates=1,active_updates=int(first[3]),
                  upper=None,lower=None,gap=None,pzero=pzero,model_fraction=None,trace=[])
        if perf_counter_ns()>=deadline:return None,base
        d,q=self.epigraph.solve(e,G,lo,hi,lam,posture,w,kappa)
        base.update(conic=q,theta=q['theta'],dual_updates=2)
        if not np.isfinite(d).all() or q['feasible_box_violation']>1e-10:
            base['reason']='invalid_conic';return None,base
        if acceptable(d):base['reason']='task';return d,base
        # The same actual local objective, retaining the best available direction.
        u=objective(d,e,G,lam,posture,w,kappa)
        first_u=objective(first[2],e,G,lam,posture,w,kappa)
        if first_u<u:d=first[2];u=first_u
        lower=q['box_minorant_lower'];raw=u-lower
        valid=bool(np.isfinite([pzero,u,lower]).all() and raw>=-64*np.finfo(float).eps*max(1,abs(pzero),abs(u),abs(lower)))
        base.update(reason='conic_solved',upper=u,lower=lower,gap=max(0.,raw),raw_gap=raw,bounds_valid=valid)
        return d,base
