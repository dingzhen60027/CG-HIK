"""Cached native analytic geometry for optimization, never for acceptance.

Same URDF, joint order and base/end frames as the frozen public URDFKinematics.
The public verifier retains its original backend. Construction checks both FK
and world geometric Jacobians before this acceleration can be used.
"""
import numpy as np
from ..types import Pose


class NativeGeometry:
    def __init__(self, public, urdf):
        import pinocchio as pin
        self.public,self.pin=public,pin
        self.model=pin.buildModelFromUrdf(str(urdf));self.data=self.model.createData()
        if self.model.nq!=public.nq or tuple(self.model.names[1:])!=tuple(public.joint_names):
            raise ValueError('native optimizer joint ordering differs from public model')
        self.base=self.model.getFrameId(public.base_link);self.end=self.model.getFrameId(public.end_link)
        self.cached_q=None
        error=0.
        center=(public.limits.lower+public.limits.upper)/2
        direction=np.where(np.arange(public.nq)%2,-1.,1.)
        for frac in (0.,.07,-.07,.19,-.19):
            q=center+frac*(public.limits.upper-public.limits.lower)*direction
            a=self.forward(q);b=public.forward(q)
            error=max(error,float(np.max(np.abs(a.matrix-b.matrix))),
                      float(np.max(np.abs(self.jacobian(q)-public.jacobian(q)))))
        if error>1e-11:raise ValueError('native optimizer FK/Jacobian differs from public model')
        self.agreement_error=error;self.cached_q=None

    def __getattr__(self,key):return getattr(self.public,key)

    def evaluate(self,q):
        if self.cached_q is not None and np.array_equal(q,self.cached_q):return
        self.pin.computeJointJacobians(self.model,self.data,q)
        self.pin.updateFramePlacements(self.model,self.data)
        base=self.data.oMf[self.base];end=base.actInv(self.data.oMf[self.end])
        J=np.array(self.pin.getFrameJacobian(self.model,self.data,self.end,
                                           self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        J[:3]=base.rotation.T@J[:3];J[3:]=base.rotation.T@J[3:]
        self.cached_pose=Pose(np.array(end.translation),np.array(end.rotation))
        self.cached_jacobian=J;self.cached_q=np.array(q,copy=True)

    def forward(self,q):self.evaluate(q);return self.cached_pose

    def jacobian(self,q):self.evaluate(q);return self.cached_jacobian
