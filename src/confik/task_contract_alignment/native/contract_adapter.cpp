// Unchanged upstream numerical code. Timing exposes setup versus search only.
#include "../../continuation_mechanism/tolerance_native/tolerance_adapter.cpp"
#include <chrono>
extern "C" {
int contract_trac_solve(void* handle, int n, const double* previous,
                       const double* position, const double* rotation,
                       const double* lower, const double* upper,
                       const double* bounds_raw, double* output, long long* times) {
  using Clock=std::chrono::steady_clock;
  try {
    auto a=Clock::now();
    auto& adapter=*(Adapter*)handle;
    KDL::JntArray init(n),lb(n),ub(n),out(n);
    for(int i=0;i<n;i++){init(i)=previous[i];lb(i)=lower[i];ub(i)=upper[i];}
    auto target=frame_from(position,rotation);
    KDL::Twist bounds(KDL::Vector(bounds_raw[0],bounds_raw[1],bounds_raw[2]),
                      KDL::Vector(bounds_raw[3],bounds_raw[4],bounds_raw[5]));
    auto b=Clock::now();
    adapter.solver->setKDLLimits(lb,ub);
    auto c=Clock::now();
    int rc=adapter.solver->CartToJnt(init,target,out,bounds);
    auto d=Clock::now();
    for(int i=0;i<n;i++)output[i]=out(i);
    auto e=Clock::now();
    times[0]=std::chrono::duration_cast<std::chrono::nanoseconds>((b-a)+(e-d)).count();
    times[1]=std::chrono::duration_cast<std::chrono::nanoseconds>(c-b).count();
    times[2]=std::chrono::duration_cast<std::chrono::nanoseconds>(d-c).count();
    return rc;
  } catch(const std::exception& e){last_error=e.what();return -999;}
}
}
