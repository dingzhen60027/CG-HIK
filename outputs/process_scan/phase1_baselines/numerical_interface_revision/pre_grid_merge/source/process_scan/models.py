"""Common model construction from the frozen URDF, never from environment names."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .geometry_adapter import GeometryAdapter
from ..types import Pose

ROOT=Path(__file__).resolve().parents[3]


def context(robot):
    cfg=yaml.safe_load((ROOT/'configs/paper_v2.yaml').read_text())['robots'][robot]
    adapter=GeometryAdapter(cfg['urdf'],cfg['base_link'],cfg['end_link'])
    adapter.public=ToolFrame(adapter.public,.08)
    adapter.native=ToolFrame(adapter.native,.08)
    return adapter,Path(cfg['urdf'])


class ToolFrame:
    """Explicit 80 mm flange-to-optical-origin rigid scanner mount."""
    def __init__(self,base,offset):self.base=base;self.tool_offset=offset
    def __getattr__(self,key):return getattr(self.base,key)
    def forward(self,q):
        p=self.base.forward(q);return Pose(p.position+self.tool_offset*p.rotation[:,2],p.rotation)
    def jacobian(self,q):
        p=self.base.forward(q);J=self.base.jacobian(q).copy();r=self.tool_offset*p.rotation[:,2]
        J[:3]+=np.cross(J[3:].T,r).T
        return J


def symbolic_fk(public):
    import casadi as ca
    q=ca.SX.sym('q',public.nq);T=ca.SX.eye(4);i=0;positions=[]
    for joint in public.chain:
        T=T@ca.DM(joint.origin)
        if joint.active:
            x,y,z=joint.axis
            K=ca.DM([[0,-z,y],[z,0,-x],[-y,x,0]])
            U=ca.SX.eye(4)
            if joint.kind=='prismatic':U[:3,3]=ca.DM(joint.axis)*q[i]
            else:U[:3,:3]=ca.SX.eye(3)+ca.sin(q[i])*K+(1-ca.cos(q[i]))*(K@K)
            T=T@U;i+=1
        positions.append(T[:3,3])
    p=T[:3,3]+getattr(public,'tool_offset',0.)*T[:3,2]
    fk=ca.Function('f',[q],[p,T[:3,:3]])
    center=(public.limits.lower+public.limits.upper)/2
    p,R=fk(center);target=public.forward(center)
    np.testing.assert_allclose(np.array(p).ravel(),target.position,atol=1e-12)
    np.testing.assert_allclose(np.array(R),target.rotation,atol=1e-12)
    return fk


def vec(a):return ' '.join(format(float(v),'.17g') for v in np.asarray(a).ravel())


def model_xml(robot, task=None):
    """URDF inertias/axes and collision meshes; torque actuators, no position servo.

    A new virtual scanner is the only attached body. URDF mesh packages are
    resolved explicitly, and missing assets are errors (not replaced capsules).
    """
    adapter,urdf=context(robot);public=adapter.public
    source=ET.parse(urdf).getroot();links={x.attrib['name']:x for x in source.findall('link')}
    root=ET.Element('mujoco',model='process_scan_'+robot)
    ET.SubElement(root,'compiler',angle='radian',eulerseq='XYZ',balanceinertia='false')
    ET.SubElement(root,'option',timestep='.001',integrator='implicitfast',gravity='0 0 -9.81')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='960',offheight='720')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator');contacts=ET.SubElement(root,'contact')
    ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .02',pos='0 0 -.04',rgba='.3 .35 .4 1')
    parent=ET.SubElement(world,'body',name=public.base_link)
    package_roots={
        'franka_description':Path('/home/eric/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description'),
        'ur_description':ROOT/'tmp/process_scan_dependencies/ur_description'}
    def geometry(body,link):
        spec=links[link];inertia=spec.find('inertial')
        if inertia is not None:
            origin=inertia.find('origin');pos=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            ii=inertia.find('inertia');vals={k:float(v) for k,v in ii.attrib.items()}
            mat=np.array([[vals['ixx'],vals['ixy'],vals['ixz']],[vals['ixy'],vals['iyy'],vals['iyz']],[vals['ixz'],vals['iyz'],vals['izz']]])
            R=Rotation.from_euler('xyz',rpy).as_matrix();mat=R@mat@R.T
            ET.SubElement(body,'inertial',pos=vec(pos),mass=inertia.find('mass').get('value'),
                fullinertia=vec([mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[0,2],mat[1,2]]))
        for j,coll in enumerate(spec.findall('collision')):
            origin=coll.find('origin');attrs={}
            if origin is not None:
                attrs['pos']=origin.get('xyz','0 0 0');attrs['euler']=origin.get('rpy','0 0 0')
            geo=coll.find('geometry');mesh=geo.find('mesh');name=f'{link}_collision_{j}'
            if mesh is None:raise ValueError('Unhandled non-mesh source collision')
            uri=mesh.get('filename');pkg,relative=uri[len('package://'):].split('/',1)
            path=package_roots[pkg]/relative
            if not path.is_file():raise FileNotFoundError(path)
            ET.SubElement(asset,'mesh',name=name,file=str(path))
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba='.65 .69 .76 1',**attrs)
    geometry(parent,public.base_link)
    efforts=[]
    joints_source={j.get('name'):j for j in source.findall('joint')}
    for joint in public.chain:
        quat=Rotation.from_matrix(joint.origin[:3,:3]).as_quat(scalar_first=True)
        parent=ET.SubElement(parent,'body',name=joint.child,pos=vec(joint.origin[:3,3]),quat=vec(quat))
        ET.SubElement(contacts,'exclude',body1=joint.parent,body2=joint.child)
        if joint.active:
            ET.SubElement(parent,'joint',name=joint.name,type='hinge',axis=vec(joint.axis),
                range=vec([joint.lower,joint.upper]),limited='true',damping='.1',armature='.01')
            effort=float(joints_source[joint.name].find('limit').get('effort'));efforts.append(effort)
            ET.SubElement(act,'motor',name=joint.name+'_torque',joint=joint.name,gear='1',
                ctrllimited='true',ctrlrange=vec([-effort,effort]))
        geometry(parent,joint.child)
    tool=ET.SubElement(parent,'body',name='scanner',pos='0 0 .08')
    ET.SubElement(tool,'inertial',pos='0 0 -.02',mass='.15',diaginertia='.000065 .000065 .00004')
    ET.SubElement(tool,'geom',name='scanner_body',type='box',pos='0 0 -.02',size='.03 .02 .02',rgba='.1 .2 .2 1')
    ET.SubElement(tool,'site',name='scan_tcp',pos='0 0 0',size='.003')
    hfield=None
    if task is not None:
        grid=np.linspace(-.18,.18,181);u,v=np.meshgrid(grid,grid)
        xyz,_=task.surface(np.column_stack([u.ravel(),v.ravel()]))
        h=xyz[:,2]-task.center[2];minimum=h.min();span=max(float(np.ptp(h)),1e-4)
        hfield=(h-minimum)/span
        ET.SubElement(asset,'hfield',name='surface',nrow='181',ncol='181',size=vec([.18,.18,span,.03]))
        quat=Rotation.from_matrix(task.world_R).as_quat(scalar_first=True)
        ET.SubElement(world,'geom',name='workpiece',type='hfield',hfield='surface',
            pos=vec(task.center+[0,0,minimum]),quat=vec(quat),rgba='.2 .55 .75 1')
        # A conservative envelope supplies a well-defined convex signed-distance
        # query for planning. Physical contacts and scan rays use the hfield.
        # The envelope is invisible/noncolliding and excluded from scan rays.
        ET.SubElement(world,'geom',name='workpiece_planning_envelope',type='box',
            size=vec([.18,.18,(span+.03)/2]),pos=vec(task.center+[0,0,minimum+(span-.03)/2]),
            quat=vec(quat),contype='0',conaffinity='0',group='5',rgba='0 0 0 0')
        ET.SubElement(world,'geom',name='support',type='box',size='.18 .18 .06',
            pos=vec(task.center+[0,0,minimum-.09]),quat=vec(quat),rgba='.25 .25 .25 1')
    return ET.tostring(root,encoding='unicode'),hfield,np.array(efforts),adapter


def physical_model(robot,task=None):
    import mujoco as mj
    xml,hfield,efforts,adapter=model_xml(robot,task)
    model=mj.MjModel.from_xml_string(xml)
    if hfield is not None:model.hfield_data[:]=hfield
    data=mj.MjData(model);site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'scan_tcp')
    # These qpos writes are calibration/reset only, NEVER execution.
    error=0.
    for frac in (0.,.03,-.03):
        q=(adapter.public.limits.lower+adapter.public.limits.upper)/2+frac*(adapter.public.limits.upper-adapter.public.limits.lower)
        data.qpos[:]=q;mj.mj_forward(model,data);pose=adapter.public.forward(q)
        error=max(error,np.max(abs(pose.position-data.site_xpos[site])),np.max(abs(pose.rotation-data.site_xmat[site].reshape(3,3))))
    if error>1e-9:raise ValueError(f'MJCF/URDF TCP mismatch: {error}')
    return model,data,adapter,dict(xml=xml,fk_max_error=float(error),torque_limits=efforts.tolist())
