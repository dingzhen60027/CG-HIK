// Extend the existing ABI without changing it or the pinned upstream solvers.
#include "../../revision_compute_allocation/native/trac_adapter.cpp"

static KDL::Frame frame_from(const double* p, const double* r) {
  return KDL::Frame(KDL::Rotation(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]),
                    KDL::Vector(p[0],p[1],p[2]));
}
extern "C" {
int tolerance_trac_solve(void* handle, int n, const double* previous,
                        const double* position, const double* rotation,
                        const double* lower, const double* upper,
                        const double* cartesian_bounds, double* output) {
  try {
    auto& adapter=*(Adapter*)handle;
    KDL::JntArray init(n),lb(n),ub(n),out(n);
    for(int i=0;i<n;i++){init(i)=previous[i];lb(i)=lower[i];ub(i)=upper[i];}
    adapter.solver->setKDLLimits(lb,ub);
    KDL::Twist bounds(KDL::Vector(cartesian_bounds[0],cartesian_bounds[1],cartesian_bounds[2]),
                      KDL::Vector(cartesian_bounds[3],cartesian_bounds[4],cartesian_bounds[5]));
    int rc=adapter.solver->CartToJnt(init,frame_from(position,rotation),out,bounds);
    for(int i=0;i<n;i++)output[i]=out(i);
    return rc;
  } catch(const std::exception& e) {last_error=e.what();return -999;}
}
// Diagnostic only: exercise the exact installed upstream residual convention.
int tolerance_relative_error(const double* tp,const double* tr,const double* ap,
                             const double* ar,const double* bounds,double eps,double* output) {
  auto error=KDL::diffRelative(frame_from(tp,tr),frame_from(ap,ar));
  for(int i=0;i<6;i++) {output[i]=error[i];if(std::abs(error[i])<=std::abs(bounds[i]))error[i]=0.;}
  return KDL::Equal(error,KDL::Twist::Zero(),eps) ? 1 : 0;
}
}
