"""Headless call of the official experiment-1 mathematical setup.

This checks upstream code/backend only; it is NOT a Panda/UR5e scan result.
ROS message types must be genuinely built, not replaced by local stubs.
"""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R
from bound_mpc.RobotModel import RobotModel
from bound_mpc.BoundMPC.BoundMPC import BoundMPC
from bound_mpc.utils import get_default_path,get_default_weights,integrate_joint


def main():
    q=np.zeros(7);q[1]=np.pi/3.5;q[3]=-np.pi/3.5;q[5]=-12.85714286*np.pi/180
    robot=RobotModel();dq=np.zeros(7);ddq=np.zeros(7);jerk=np.zeros(7)
    pose,J,_=robot.forward_kinematics(q,dq);p0=pose[:3];r0=R.from_rotvec(pose[3:])
    values=list(get_default_path(p0,r0,5))
    values[0]=[p0,p0+[-p0[0]*2,0,0],p0+[-p0[0],p0[0],0],p0+[-p0[0],-p0[0],0],p0]
    r1=R.from_euler('XYZ',[0,0,-np.pi])*r0;r2=R.from_euler('XYZ',[0,0,-np.pi/2])*r1
    r3=R.from_euler('XYZ',[0,np.pi/2,0])*R.from_euler('XYZ',[np.pi/1.001,0,0])*r2
    values[1]=[r.as_matrix() for r in (r0,r1,r2,r3,r0)]
    values[9]=[.5]*len(values[9]);values[5][0]=np.array([0.,1,0]);values[5][1]=np.array([0.,1,0])
    params=SimpleNamespace(n=10,dt=.1,weights=get_default_weights(),build=True,real_time=False,nr_segs=4)
    started=perf_counter();mpc=BoundMPC(*values,p0=pose,params=params);build=perf_counter()-started
    rows=[];v=np.zeros(6)
    for k in range(3):
        started=perf_counter()
        traj,ref,err,cpu,iters=mpc.step(q,dq,ddq,pose,v,np.array([mpc.phi_max[0],0,0]),jerk)
        rows.append(dict(step=k,outer_seconds=perf_counter()-started,reported_seconds=cpu,
                         iterations=iters,error_count=mpc.error_count,phi=mpc.phi_current.tolist()))
        jerk_path=traj['dddq'];stack=np.concatenate([jerk[:,None],jerk_path[:,:2]],axis=1)
        q,dq,ddq,pose,v,acc,jcart=integrate_joint(robot,stack,q,dq,ddq,.1);jerk=jerk_path[:,0]
    Path('official_example_check.json').write_text(json.dumps(dict(
        identity='Official experiment1 path/robot/weights, first 3 headless updates; no hardware/scan claim',
        linear_solver='Ipopt bundled MUMPS (MA57 not configured)',build_seconds=build,updates=rows),indent=2))


if __name__=='__main__':main()
