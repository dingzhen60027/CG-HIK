"""Common physical-geometry clearance callback, not a manipulability proxy."""
import casadi as ca
import numpy as np
import mujoco as mj
import coal


class Clearance(ca.Callback):
    def __init__(self,model,name='clearance'):
        ca.Callback.__init__(self);self.model=model;self.data=mj.MjData(model)
        self.pairs=[];self.shapes=[];self.owners=[]
        for gid in range(model.ngeom):
            typ=model.geom_type[gid]
            if typ==mj.mjtGeom.mjGEOM_MESH:
                mid=model.geom_dataid[gid];va=model.mesh_vertadr[mid];vn=model.mesh_vertnum[mid]
                fa=model.mesh_faceadr[mid];fn=model.mesh_facenum[mid]
                vertices=coal.StdVec_Vec3s();vertices.extend(model.mesh_vert[va:va+vn].astype(float))
                faces=coal.StdVec_Triangle();faces.extend([coal.Triangle(*map(int,f)) for f in model.mesh_face[fa:fa+fn]])
                bv=coal.BVHModelOBBRSS();bv.beginModel(int(fn),int(vn));bv.addSubModel(vertices,faces);bv.endModel();bv.buildConvexRepresentation(True)
                self.owners.append(bv);self.shapes.append(bv.convex)
            elif typ==mj.mjtGeom.mjGEOM_BOX:self.shapes.append(coal.Box(*(2*model.geom_size[gid])))
            elif typ==mj.mjtGeom.mjGEOM_PLANE:self.shapes.append(coal.Halfspace(np.array([0.,0,1.]),0.))
            elif typ==mj.mjtGeom.mjGEOM_HFIELD:self.shapes.append(None)
            else:raise ValueError(f'Unhandled collision shape {typ}')
        for a in range(model.ngeom):
            for b in range(a+1,model.ngeom):
                if model.geom(a).name=='workpiece' or model.geom(b).name=='workpiece':continue
                ba=int(model.geom_bodyid[a]);bb=int(model.geom_bodyid[b])
                wa=int(model.body_weldid[ba]);wb=int(model.body_weldid[bb])
                if wa==wb:continue
                pa=int(model.body_weldid[model.body_parentid[wa]])
                pb=int(model.body_weldid[model.body_parentid[wb]])
                # Same filtering as adjacent attached link collisions; do not
                # exclude nonadjacent links or robot/workpiece pairs.
                if ba!=0 and bb!=0 and (wa==pb or wb==pa):continue
                self.pairs.append((a,b))
        self.calls=0
        self.request=coal.DistanceRequest();self.request.enable_signed_distance=True
        self.cached_q=None;self.children=[]
        self.construct(name,{})
    def get_n_in(self):return 1
    def get_n_out(self):return 1
    def get_sparsity_in(self,i):return ca.Sparsity.dense(self.model.nq,1)
    def get_sparsity_out(self,i):return ca.Sparsity.scalar()
    def numeric(self,q):
        q=np.asarray(q).ravel()
        if self.cached_q is not None and np.array_equal(q,self.cached_q):return self.cached_value
        # Separate planner data: execution data are NEVER written here.
        self.data.qpos[:]=q;mj.mj_kinematics(self.model,self.data);mj.mj_comPos(self.model,self.data)
        self.calls+=1
        transforms=[coal.Transform3s(R.reshape(3,3),p) for R,p in zip(self.data.geom_xmat,self.data.geom_xpos)]
        best=np.inf;winner=None
        for a,b in self.pairs:
            result=coal.DistanceResult();v=coal.distance(self.shapes[a],transforms[a],self.shapes[b],transforms[b],self.request,result)
            if v<best:best=v;winner=(a,b,result)
        a,b,result=winner;p1=result.getNearestPoint1();p2=result.getNearestPoint2()
        direction=np.asarray(p1)-np.asarray(p2)
        norm=np.linalg.norm(direction)
        if norm<1e-12:raise ArithmeticError('Undefined distance normal at zero separation')
        direction=direction/norm*(1 if best>0 else -1)
        J1=np.zeros((3,self.model.nv));J2=J1.copy()
        mj.mj_jac(self.model,self.data,J1,None,p1,int(self.model.geom_bodyid[a]))
        mj.mj_jac(self.model,self.data,J2,None,p2,int(self.model.geom_bodyid[b]))
        self.cached_gradient=direction@(J1-J2)
        self.cached_q=q.copy();self.cached_value=float(best)
        return self.cached_value
    def eval(self,arg):return [self.numeric(arg[0])]
    def has_jacobian(self):return True
    def get_jacobian(self,name,inames,onames,opts):
        derivative=ClearanceJacobian(self,name);self.children.append(derivative);return derivative


class ClearanceJacobian(ca.Callback):
    def __init__(self,parent,name):
        ca.Callback.__init__(self);self.parent=parent;self.construct(name,{})
    def get_n_in(self):return 2
    def get_n_out(self):return 1
    def get_sparsity_in(self,i):return ca.Sparsity.dense(self.parent.model.nq,1) if i==0 else ca.Sparsity.scalar()
    def get_sparsity_out(self,i):return ca.Sparsity.dense(1,self.parent.model.nq)
    def eval(self,args):self.parent.numeric(args[0]);return [self.parent.cached_gradient.reshape(1,-1)]
