"""Common torque-actuated MuJoCo execution and actual-state profile returns."""
from time import perf_counter
import numpy as np
import mujoco as mj
from scipy.spatial import cKDTree, distance
from scipy.ndimage import label


class TimedPath:
    def __init__(self,basis,coeff,timing):
        self.basis=basis;self.coeff=coeff;self.s=timing['s'];self.x=timing['x'];self.t=timing['t'];self.u=timing['u']
        self.duration=float(self.t[-1])
        if np.any(np.diff(self.t)<=0):raise ValueError('Non-increasing time grid')
    def at(self,t):
        if t>=self.duration:return self.basis(1.)@self.coeff,np.zeros(self.coeff.shape[1]),np.zeros(self.coeff.shape[1]),1.
        k=min(np.searchsorted(self.t,t,side='right')-1,len(self.u)-1);k=max(0,k)
        dt=t-self.t[k];sd=np.sqrt(max(0.,self.x[k]));u=self.u[k]
        s=np.clip(self.s[k]+sd*dt+.5*u*dt*dt,0,1);sd=max(0.,sd+u*dt)
        qs=self.basis(s,nu=1)@self.coeff
        return self.basis(s)@self.coeff,qs*sd,(self.basis(s,nu=2)@self.coeff)*sd*sd+qs*u,float(s)


def scan_profile(model,data,task,site,s):
    position=data.site_xpos[site].copy();R=data.site_xmat[site].reshape(3,3).copy()
    offsets=np.linspace(-.02,.02,81)/.15
    rays=R[:,2][None,:]+offsets[:,None]*R[:,0][None,:];rays/=np.linalg.norm(rays,axis=1)[:,None]
    ids=np.full(81,-1,dtype=np.int32);ranges=np.full(81,-1.,float);normals=np.zeros((81,3))
    group=np.array([1,1,1,1,1,0],dtype=np.uint8)
    mj.mj_multiRay(model,data,position,rays.ravel(),group,True,int(model.site_bodyid[site]),ids,ranges,normals.ravel(),81,1.)
    points=position+ranges[:,None]*rays
    uvw=(points-task.center)@task.world_R
    roi=(abs(uvw[:,0])<=.12)&(abs(uvw[:,1])<=.08)
    workpiece=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'workpiece')
    raw=(ids==workpiece)&roi&(ranges>=0)
    c,n=task.surface(uvw[:,:2]);incidence=np.einsum('ij,ij->i',-rays,n)
    process=task.process_values(position,R,s);profile_legal=task.process_legal(position,R,s)
    valid=raw&(ranges>=.145)&(ranges<=.155)&(incidence>=np.cos(np.deg2rad(10)))&profile_legal
    return dict(points=points,raw_valid=raw,quality_valid=valid,ranges=ranges,geomids=ids,process=process,
                center_return=points[40] if raw[40] else np.full(3,np.nan),profile_legal=profile_legal)


def coverage(task,points):
    u,v=np.meshgrid(np.linspace(-.12,.12,241),np.linspace(-.08,.08,161))
    grid,n=task.surface(np.column_stack([u.ravel(),v.ravel()]))
    weights=1./np.maximum(n@task.world_R[:,2],1e-12)
    if len(points):covered=cKDTree(points).query(grid,workers=1)[0]<=.00075
    else:covered=np.zeros(len(grid),bool)
    mask=(~covered).reshape(u.shape);groups,count=label(mask,np.ones((3,3)))
    diameter=0.
    for i in range(1,count+1):
        pts=grid[(groups==i).ravel()]
        if len(pts)>1:
            # Conservative connected-hole diameter bound in 3-D; never
            # understates a hole and never silently drops a large component.
            diameter=max(diameter,float(np.linalg.norm(np.ptp(pts,axis=0))))
    return float(np.sum(weights*covered)/sum(weights)),diameter,covered.reshape(u.shape)


def execute(model,data,task,adapter,basis,coeff,timing,common_reference_duration,state_stride=5):
    started=perf_counter();plan=TimedPath(basis,coeff,timing);n=model.nv
    # ONLY reset writes the live generalized position/velocity.
    mj.mj_resetData(model,data);data.qpos[:]=plan.at(0)[0];data.qvel[:]=0.;mj.mj_forward(model,data)
    site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'scan_tcp')
    effort=model.actuator_ctrlrange[:,1];Kp=np.full(n,400.);Kd=np.full(n,40.)
    matrix=np.zeros((n,n));tau=np.zeros(n);pre=np.zeros(n)
    histories=[];profiles=[];settled=0.;collision_steps=0;motion_ok=True
    util=autil=tracking_max=torque_util=0.;velocity_exceed=acceleration_exceed=0
    saturation_steps=0;scan_steps=turn_steps=wait_steps=0
    timeout=3*common_reference_duration;finished=False
    for step in range(int(np.ceil(timeout/.001))+1):
        # mj_step integrates qpos but derived site/geom poses still describe its
        # evaluation state. Refresh before timestamped feedback/raycasting;
        # otherwise alternating 4/6 ms pose ages masquerade as 5 ms profiles.
        mj.mj_forward(model,data)
        t=float(data.time);qr,dqr,ddqr,s=plan.at(t)
        if step%2==0:
            mj.mj_fullM(model,data,matrix)
            # Computed-torque feedforward plus inertia-scaled PD: Kp [s^-2],
            # Kd [s^-1]. Fixed torque-space D=40 is unstable for wrist inertias
            # at a 2 ms sampled controller and is NOT used in the study.
            pre=matrix@(ddqr+Kp*(qr-data.qpos)+Kd*(dqr-data.qvel))+data.qfrc_bias-data.qfrc_passive
            tau=np.clip(pre,-effort,effort);data.ctrl[:]=tau
            # Log acceleration under the command actually applied, not the
            # reset-time gravity acceleration before any actuator command.
            mj.mj_forward(model,data)
        if step%5==0:
            _,_,_,_,tags=task.references([s]);profile=scan_profile(model,data,task,site,s) if tags[0] and t<=plan.duration else None
            if profile is not None:profiles.append(dict(t=t,s=s,**profile))
        if step%state_stride==0:
            histories.append(dict(t=t,s=s,q=data.qpos.copy(),dq=data.qvel.copy(),qacc=data.qacc.copy(),
                qr=qr,dqr=dqr,ddqr=ddqr,tau=tau.copy(),pre_tau=pre.copy(),tcp=data.site_xpos[site].copy(),
                rotation=data.site_xmat[site].copy(),ncon=int(data.ncon)))
        # Physical contact excludes welded/adjacent joints via MuJoCo's standard
        # filter; workpiece/support collisions always count.
        collision_steps+=int(data.ncon>0)
        lim=adapter.public.limits
        motion_ok &= bool(np.all(data.qpos>=lim.lower-1e-6)&np.all(data.qpos<=lim.upper+1e-6))
        vu=float(np.max(abs(data.qvel)/(.5*lim.velocity)));au=float(np.max(abs(data.qacc))/2)
        util=max(util,vu);autil=max(autil,au);velocity_exceed+=int(vu>1.001);acceleration_exceed+=int(au>1.001)
        tracking_max=max(tracking_max,float(np.max(abs(data.qpos-qr))))
        torque_util=max(torque_util,float(np.max(abs(pre)/effort)))
        saturation_steps+=int(np.any(abs(pre)>=.999*effort))
        if t>plan.duration:wait_steps+=1
        elif task.references([s])[4][0]:scan_steps+=1
        else:turn_steps+=1
        if t>=plan.duration and np.max(abs(data.qvel))<=.01 and np.max(abs(data.qpos-qr))<=.001:
            settled+=.001
            if settled>=.2:finished=True;break
        else:settled=0.
        if not np.isfinite(data.qpos).all():break
        mj.mj_step(model,data)
    valid=np.concatenate([p['points'][p['quality_valid']] for p in profiles]) if profiles else np.empty((0,3))
    raw=np.concatenate([p['points'][p['raw_valid']] for p in profiles]) if profiles else np.empty((0,3))
    cover,hole,coverage_mask=coverage(task,valid);rawcover,_,_=coverage(task,raw)
    actualq=np.array([h['q'] for h in histories]);dq=np.array([h['dq'] for h in histories]);acc=np.array([h['qacc'] for h in histories])
    gaps=[]
    for left,right in zip(profiles[:-1],profiles[1:]):
        # Do not turn the prescribed between-line transition into a within-line gap.
        ul,vl=task.uv([left['s'],right['s']])[0]
        same_line=abs((ul[1]-vl[1]) if task.direction=='u' else (ul[0]-vl[0]))<1e-8
        if same_line and np.isfinite([left['center_return'],right['center_return']]).all():
            gaps.append(np.linalg.norm(right['center_return']-left['center_return']))
    maxgap=float(max(gaps,default=np.inf))
    physical_complete=bool(finished and motion_ok and collision_steps==0 and util<=1.001 and autil<=1.001)
    metric=dict(execution_completed=physical_complete,
        quality_completed=bool(physical_complete and cover>=.99 and hole<=.002 and maxgap<=.001),
        simulated_completion_time_s=float(data.time) if physical_complete else None,
        elapsed_until_stop_s=float(data.time),simulation_wall_s=perf_counter()-started,
        valid_coverage_fraction=cover,raw_coverage_fraction=rawcover,max_hole_diameter_upper_bound_m=hole,
        max_along_scan_gap_m=maxgap,collision_steps=collision_steps,velocity_utilization_max=util,
        acceleration_utilization_max=autil,tracking_error_max_rad=tracking_max,
        torque_saturation_fraction=saturation_steps/(step+1),torque_requested_utilization_max=torque_util,
        velocity_exceed_duration_s=velocity_exceed*.001,acceleration_exceed_duration_s=acceleration_exceed*.001,
        scan_duration_s=scan_steps*.001,transition_duration_s=turn_steps*.001,settling_duration_s=wait_steps*.001,
        sample_state_dt_s=.001*state_stride,physical_extrema_dt_s=.001,
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(acc*acc))),
        jerk_rms_rad_s3=float(np.sqrt(np.mean((np.diff(acc,axis=0)/(.001*state_stride))**2))),
        hole_measurement='conservative_3d_component_bounding_box_diagonal')
    for key in ('standoff_m','center_error_m','incidence_rad','line_error_rad'):
        values=np.array([p['process'][key] for p in profiles])
        metric[key+'_p95']=float(np.quantile(values,.95)) if len(values) else None
        metric[key+'_max']=float(np.max(values)) if len(values) else None
        metric[key+'_min']=float(np.min(values)) if len(values) else None
    metric['profile_process_violation_duration_s']=.005*sum(not p['profile_legal'] for p in profiles)
    return metric,histories,profiles,coverage_mask
