"""Endpoint graph built exclusively from the independent reference IK library.

The fine layers supply geometric edge witnesses, not baseline solutions. Both
costs use exactly the same nodes, witnesses and rejection mask. A graph path is
not an execution certificate: the fitted whole spline is independently checked.
"""
from time import perf_counter
import gzip
import json
import numpy as np
from .planning import numerical_grid
from .study import clean


def diverse_indices(q, required, span, cap=12, tolerance=.001):
    """Reference member first; deterministic farthest-point, lexicographic ties."""
    q=np.asarray(q);order=sorted(range(len(q)),key=lambda i:tuple(q[i]))
    unique=[int(required)]
    for i in order:
        if all(np.linalg.norm(q[i]-q[j])>tolerance for j in unique):unique.append(i)
    chosen=[int(required)];remaining=[i for i in unique if i!=required]
    while remaining and len(chosen)<cap:
        distances=[min(np.linalg.norm((q[i]-q[j])/span) for j in chosen) for i in remaining]
        i=remaining[int(np.argmax(distances))];chosen.append(i);remaining.remove(i)
    return chosen


def velocity_envelope(q, lower, upper, velocity, acceleration, rest=False):
    if rest:return np.zeros_like(q)
    # Any globally rest-to-rest bounded-acceleration motion must have room to
    # both acquire and remove its instantaneous speed inside the joint range.
    margin=np.maximum(0,np.minimum(q-lower,upper-q))
    return np.minimum(velocity,np.sqrt(2*acceleration*margin))


def displacement_capacity(T, v0, v1, vmax, acceleration):
    """Integral min(vmax, v0+a*t, v1+a*(T-t)); exact piecewise-linear area."""
    cuts=np.clip([0.,T,(vmax-v0)/acceleration,
                  T-(vmax-v1)/acceleration,(v1+acceleration*T-v0)/(2*acceleration)],0,T)
    cuts=np.unique(cuts);mid=(cuts[:-1]+cuts[1:])/2
    return float(np.sum(np.diff(cuts)*np.minimum(vmax,np.minimum(v0+acceleration*mid,v1+acceleration*(T-mid)))))


def time_lower_bound(q0,q1,lower,upper,velocity,acceleration,first=False,last=False):
    q0,q1=np.asarray(q0),np.asarray(q1);acceleration=np.broadcast_to(acceleration,q0.shape)
    v0=velocity_envelope(q0,lower,upper,velocity,acceleration,first)
    v1=velocity_envelope(q1,lower,upper,velocity,acceleration,last)
    per_joint=[]
    for delta,a,b,v,acc in zip(abs(q1-q0),v0,v1,velocity,acceleration):
        if delta==0:per_joint.append(0.);continue
        lo=0.;hi=float(delta/v+2*v/acc)
        for _ in range(48):
            mid=(lo+hi)/2
            if displacement_capacity(mid,a,b,v,acc)>=delta:hi=mid
            else:lo=mid
        # Use the lower bisection endpoint: never round a lower bound upwards.
        per_joint.append(lo)
    return dict(value_s=float(max(per_joint)),per_joint_s=per_joint,
                q0=q0,q1=q1,lower=lower,upper=upper,velocity=velocity,
                acceleration=acceleration,v0_upper=v0,v1_upper=v1,
                global_start_at_rest=first,global_end_at_rest=last,
                formula='max_i inf{T: |dq_i| <= integral_0^T min(v_i,v0_i+a_i*t,v1_i+a_i*(T-t))dt}',
                relaxation='Independent joints; boundary velocities bounded, not forced to stop at each line; ignores extra process/collision restrictions')


def load_library(identity,source_root,task,adapter,collision,cfg):
    start=perf_counter();root=source_root/identity['reference_source']
    raw=np.load(root/'reference.npz')
    with gzip.open(root/'candidate_graph.json.gz','rt') as f:source=json.load(f)
    grid=raw['s'];layers=[np.asarray(x,float) for x in source['layers']]
    endpoint_indices=[int(np.argmin(abs(grid-s))) for s in task.knots]
    assert np.max(abs(grid[endpoint_indices]-task.knots))<1e-12
    span=adapter.public.limits.upper-adapter.public.limits.lower
    endpoints=[];legal_log=[]
    for k in endpoint_indices:
        ref=int(np.argmin(np.linalg.norm(layers[k]-raw['q'][k],axis=1)))
        assert np.max(abs(layers[k][ref]-raw['q'][k]))<1e-10
        chosen=diverse_indices(layers[k],ref,span,cfg['endpoint_candidate_cap'],cfg['candidate_dedup_rad'])
        if k in (0,len(grid)-1):chosen=[ref]
        kept=[]
        for j in chosen:
            q=layers[k][j];p=adapter.native.forward(q);distance=collision.numeric(q)
            legal=bool(task.process_legal(p.position,p.rotation,grid[k]) and distance>=.005 and
                       np.all(q>=adapter.public.limits.lower) and np.all(q<=adapter.public.limits.upper))
            legal_log.append(dict(key_index=k,candidate_index=j,q=q,position=p.position,rotation=p.rotation,
                                  process=task.process_values(p.position,p.rotation,grid[k]),clearance=distance,legal=legal))
            if legal:kept.append(j)
        endpoints.append(kept)
    np.testing.assert_allclose(layers[0][endpoints[0][0]],identity['common_start'],atol=1e-12)
    np.testing.assert_allclose(layers[-1][endpoints[-1][0]],identity['common_end'],atol=1e-12)
    return dict(grid=grid,layers=layers,endpoint_indices=endpoint_indices,endpoints=endpoints,
                candidate_checks=legal_log,reference_q=raw['q'],reference_coeff=raw['coeff'],reference_knots=raw['knots'],
                elapsed_s=perf_counter()-start,source=str(root),baseline_paths_read=False)


def build_graph(library,task,adapter,collision,cfg):
    start=perf_counter();grid=library['grid'];layers=library['layers'];span=adapter.public.limits.upper-adapter.public.limits.lower
    fine=[];stats=dict(fine_possible=0,fine_discontinuous=0,fine_geometry_rejected=0,fine_collision_rejected=0,fine_retained=0)
    for k,(left,right) in enumerate(zip(layers[:-1],layers[1:])):
        delta=right[None,:,:]-left[:,None,:];cost=np.sum((delta/span)**2,axis=2)
        bad=np.max(abs(delta),axis=2)>cfg['fine_edge_max_joint_delta_rad'];cost[bad]=np.inf
        stats['fine_possible']+=cost.size;stats['fine_discontinuous']+=int(bad.sum())
        for a,b in zip(*np.where(~bad)):
            for alpha in np.linspace(0,1,cfg['fine_edge_interior_checks']+2)[1:-1]:
                q=(1-alpha)*left[a]+alpha*right[b];s=(1-alpha)*grid[k]+alpha*grid[k+1]
                pose=adapter.native.forward(q)
                if not task.process_legal(pose.position,pose.rotation,s):
                    cost[a,b]=np.inf;stats['fine_geometry_rejected']+=1;break
                if collision.numeric(q)<.005:
                    cost[a,b]=np.inf;stats['fine_collision_rejected']+=1;break
        stats['fine_retained']+=int(np.isfinite(cost).sum());fine.append(cost)
    edges=[];endidx=library['endpoint_indices'];sets=library['endpoints'];lim=adapter.public.limits
    for layer,(i0,i1) in enumerate(zip(endidx[:-1],endidx[1:])):
        segment=[]
        for a in sets[layer]:
            dist=np.full(len(layers[i0]),np.inf);dist[a]=0.;parents=[]
            for k in range(i0,i1):
                total=dist[:,None]+fine[k];parent=np.argmin(total,axis=0)
                dist=total[parent,np.arange(len(parent))];parents.append(parent)
            for b in sets[layer+1]:
                row=dict(layer=layer,source=a,target=b,retained=bool(np.isfinite(dist[b])))
                if row['retained']:
                    seq=[int(b)];j=b
                    for par in parents[::-1]:j=int(par[j]);seq.append(j)
                    seq=seq[::-1];q0,q1=layers[i0][a],layers[i1][b]
                    row.update(witness_indices=seq,fine_geometry_cost=float(dist[b]),
                               distance_cost=float(np.sum(((q1-q0)/span)**2)),
                               time_bound=time_lower_bound(q0,q1,lim.lower,lim.upper,.5*lim.velocity,
                                                          np.full(len(q0),2.),layer==0,layer==len(endidx)-2))
                else:row['reason']='no_continuous_process_collision_checked_fine_path'
                segment.append(row)
        edges.append(segment)
    stats.update(endpoint_nodes=sum(map(len,sets)),endpoint_layers=len(sets),
                 endpoint_possible=sum(map(len,edges)),endpoint_retained=sum(e['retained'] for es in edges for e in es))
    return dict(edges=edges,fine_costs=fine,statistics=stats,elapsed_s=perf_counter()-start)


def shortest_path(library,graph,cost_name):
    if cost_name not in ('distance','time'):raise ValueError(cost_name)
    start=perf_counter();dist={library['endpoints'][0][0]:0.};parents=[]
    for edges in graph['edges']:
        nxt={};par={}
        for e in edges:
            if not e['retained'] or e['source'] not in dist:continue
            cost=e['distance_cost'] if cost_name=='distance' else e['time_bound']['value_s']
            v=dist[e['source']]+cost
            if e['target'] not in nxt or v<nxt[e['target']]:nxt[e['target']]=v;par[e['target']]=e
        parents.append(par);dist=nxt
        if not dist:return dict(feasible=False,status='endpoint_graph_disconnected',elapsed_s=perf_counter()-start)
    last=library['endpoints'][-1][0]
    if last not in dist:return dict(feasible=False,status='global_endpoint_not_connected',elapsed_s=perf_counter()-start)
    sequence=[];j=last
    for par in parents[::-1]:e=par[j];sequence.append(e);j=e['source']
    sequence.reverse();indices=[]
    for k,e in enumerate(sequence):indices.extend(e['witness_indices'] if k==0 else e['witness_indices'][1:])
    q=np.array([layer[i] for layer,i in zip(library['layers'],indices)])
    return dict(feasible=True,status='graph_path_not_yet_spline_verified',cost=dist[last],cost_name=cost_name,
                selected_edges=sequence,indices=indices,s=library['grid'],q=q,elapsed_s=perf_counter()-start)


def transition_controls(task,basis,boundary_m=.020):
    intervals=[]
    for i,seg in enumerate(task.segments):
        if not seg[5]:intervals.append((max(0,task.knots[i]-boundary_m/task.length_parameter),
                                      min(1,task.knots[i+1]+boundary_m/task.length_parameter)))
    active=[]
    for j in range(len(basis.c)):
        if j in (0,len(basis.c)-1):continue
        if any(basis.t[j]>=a-1e-12 and basis.t[j+4]<=b+1e-12 for a,b in intervals):active.append(j)
    return np.array(active,int),intervals
