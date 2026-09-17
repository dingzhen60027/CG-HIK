"""Fixed CAD/raster definition, independent of all solver outcomes."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from ..types import Pose


@dataclass
class ScanTask:
    family: str
    placement: int
    direction: str
    robot: str
    endpoint_extension: float = 0.
    overlap_margin: float = 0.
    calibration: bool = False

    def __post_init__(self):
        self.center=np.array([.50 if self.placement==0 else .53,
                              -.035 if self.placement==0 else .045, .24])
        self.world_R=Rotation.from_euler('z', 0 if self.placement==0 else 15, degrees=True).as_matrix()
        if self.calibration:
            # Separate, predefined calibration placement; never a comparison scene.
            self.center=np.array([.485,.005,.265])
            self.world_R=Rotation.from_euler('z',7.5,degrees=True).as_matrix()
        self.long=.24 if self.direction=='u' else .16
        self.short=.16 if self.direction=='u' else .24
        self.rows=9 if self.direction=='u' else 13
        if self.overlap_margin:
            self.rows=int(np.ceil(self.short/(.02-self.overlap_margin)))+1
        # Piecewise polynomials match first and second derivatives at every join.
        # Turns occupy a small extension outside the ROI; only straight lines scan.
        segments=[]; knots=[0.]
        L=self.long+2*self.endpoint_extension; turn=.04
        for row in range(self.rows):
            sign=1 if row%2==0 else -1
            pitch=self.short/(self.rows-1) if self.overlap_margin else .02
            y=-self.short/2+pitch*row
            a=np.array([-sign*L/2,y]); b=np.array([sign*L/2,y]); tangent=np.array([sign,0.])
            segments.append((a,b,tangent,tangent,L,True,row)); knots.append(knots[-1]+L)
            if row<self.rows-1:
                end=np.array([sign*L/2,y+pitch])
                segments.append((b,end,tangent,-tangent,turn,False,row));knots.append(knots[-1]+turn)
        self.length_parameter=knots[-1];self.knots=np.array(knots)/knots[-1]
        self.segments=segments
        self.polys=[BPoly.from_derivatives([0,seg[4]],[[seg[0],seg[2],np.zeros(2)],
                         [seg[1],seg[3],np.zeros(2)]]) for seg in segments]

    def uv(self,s):
        s=np.atleast_1d(np.clip(s,0,1)); idx=np.minimum(np.searchsorted(self.knots,s,side='right')-1,len(self.segments)-1)
        result=np.empty((len(s),2));tags=np.empty(len(s),bool)
        for i in np.unique(idx):
            mask=idx==i; result[mask]=self.polys[i]((s[mask]-self.knots[i])*self.length_parameter)
            tags[mask]=self.segments[i][5]
        if self.direction=='v':result=result[:,::-1]
        return result,tags

    def surface(self,uv):
        uv=np.atleast_2d(uv);u,v=uv.T
        if self.family=='plane':h=np.zeros_like(u);hu=hv=np.zeros_like(u)
        elif self.family=='cylinder':
            radius=.35;h=np.sqrt(radius*radius-u*u)-radius;hu=-u/np.sqrt(radius*radius-u*u);hv=np.zeros_like(u)
        elif self.family=='saddle':h=1.0*(u*u-v*v);hu=2*u;hv=-2*v
        else:raise ValueError(self.family)
        xyz=np.column_stack([u,v,h]);norm=np.column_stack([-hu,-hv,np.ones_like(u)])
        norm/=np.linalg.norm(norm,axis=1)[:,None]
        return xyz@self.world_R.T+self.center,norm@self.world_R.T

    def references(self,s):
        uv,tags=self.uv(s);c,n=self.surface(uv)
        nominal_line=self.world_R[:,1 if self.direction=='u' else 0]
        b=nominal_line[None,:]-n*(n@nominal_line)[:,None]
        b/=np.linalg.norm(b,axis=1)[:,None]
        a=-n;rot=np.stack([b,np.cross(a,b),a],axis=2)
        return c,n,b,rot,tags

    def poses(self,s):
        c,n,_,R,_=self.references(s)
        return [Pose(p+.15*nn,rr) for p,nn,rr in zip(c,n,R)]

    def process_values(self,position,rotation,s):
        c,n,b,_,_=self.references([s]);c,n,b=c[0],n[0],b[0]
        a=rotation[:,2];offset=c-position;axial=np.dot(offset,a)
        line_distance=np.linalg.norm(offset-axial*a)
        bp=rotation[:,0]-n*np.dot(n,rotation[:,0]); bp/=max(np.linalg.norm(bp),1e-15)
        return dict(standoff_m=float(axial),center_error_m=float(line_distance),
            incidence_rad=float(np.arccos(np.clip(-a@n,-1,1))),
            line_error_rad=float(np.arccos(np.clip(abs(bp@b),0,1))))

    def process_legal(self,position,rotation,s):
        v=self.process_values(position,rotation,s)
        return (.145<=v['standoff_m']<=.155 and v['center_error_m']<=.001 and
                v['incidence_rad']<=np.deg2rad(10) and v['line_error_rad']<=np.deg2rad(5))
