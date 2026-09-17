"""Offline replay of recorded physical states; never feeds a controller or solver."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import numpy as np
import h5py
import mujoco as mj
import imageio.v2 as imageio
import cv2
from .study import OUT,ROOT,write,sha
from .models import physical_model
from .task import ScanTask


def render():
    target=OUT/'reports/videos';target.mkdir(parents=True,exist_ok=True);index=[]
    # Fixed selection: placement 0, u direction, every family and robot, first
    # execution repeat. Do not select a successful run after seeing outcomes.
    for robot in ('panda','ur5e'):
        for family in ('plane','cylinder','saddle'):
            slot=f'{robot}_{family}_p0_u';task=ScanTask(family,0,'u',robot)
            model,_,_,_=physical_model(robot,task)
            camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=[.35,0,.32];camera.distance=1.4;camera.azimuth=125;camera.elevation=-28
            for method in ('B0','B1','B2','G'):
                root=OUT/'runs'/slot/f'{method}_r0';source=root/'execution.h5';movie=target/f'{slot}_{method}.mp4'
                row=dict(slot=slot,method=method,repeat=0,selection='predefined placement0/u/all families and robots',
                         source=str(source.relative_to(ROOT)),video=None,execution_not_rerun=True)
                if not source.exists():
                    row.update(status='not_executed',reason=json.loads((root/'metrics.json').read_text())['status']);index.append(row);continue
                if not movie.exists():
                    with h5py.File(source) as f:history={k:f[k][:] for k in ('t','q','tcp')}
                    rays=np.load(root/'scan_samples.npz');metrics=json.loads((root/'metrics.json').read_text())
                    # Separate renderer-only state: assigning qpos here replays
                    # an immutable measurement. The execution file is untouched.
                    render_data=mj.MjData(model);renderer=mj.Renderer(model,height=720,width=960)
                    with imageio.get_writer(movie,fps=30,codec='libx264',quality=7,macro_block_size=16) as writer:
                        for t in np.arange(0,history['t'][-1],1/30):
                            k=min(np.searchsorted(history['t'],t),len(history['t'])-1)
                            render_data.qpos[:]=history['q'][k];mj.mj_forward(model,render_data)
                            renderer.update_scene(render_data,camera=camera)
                            j=min(np.searchsorted(rays['t'],t),len(rays['t'])-1)
                            if abs(rays['t'][j]-t)<.01:
                                for ray in range(0,81,4):
                                    if not rays['raw_valid'][j,ray]:continue
                                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                                    rgba=[.05,.9,.35,.8] if rays['quality_valid'][j,ray] else [1.,.5,.1,.8]
                                    mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
                                    mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_LINE,1.,history['tcp'][k],rays['points'][j,ray])
                                    renderer.scene.ngeom+=1
                            frame=renderer.render().copy()
                            cv2.putText(frame,f'{robot.upper()} {family} {method}  actual t={t:.2f}s',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(250,250,250),2)
                            cv2.putText(frame,f'Quality complete: {metrics["quality_completed"]} | recorded torque-actuated simulation',(18,690),cv2.FONT_HERSHEY_SIMPLEX,.55,(250,250,250),1)
                            writer.append_data(frame)
                    renderer.close()
                row.update(status='rendered',video=str(movie.relative_to(ROOT)),sha256=sha(movie),
                           source_sha256=sha(source),frame_rate_hz=30,physics_dt_s=.001,not_hardware=True)
                index.append(row);print('VIDEO',slot,method,flush=True)
    write(target/'index.json',index)
