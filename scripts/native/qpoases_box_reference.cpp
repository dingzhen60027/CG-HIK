// Thin benchmark bridge to the unmodified official qpOASES 3.2.1 sources.
// SQProblem's matrix-shift overload is essential when H changes between QPs.
#include <qpOASES.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdint>
using namespace qpOASES;
using Clock = std::chrono::steady_clock;
static long long ns(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}
struct Workspace {
    SQProblem solver;
    std::vector<double> h[2];
    int slot=0, n; bool initialized=false;
    Workspace(int size):solver(size,0,HST_POSDEF),n(size) {
        h[0].resize(n*n); h[1].resize(n*n);
        Options o; o.setToDefault(); o.printLevel=PL_NONE;
        solver.setOptions(o);
    }
};
extern "C" {
void* qpr_create(int n) { return new Workspace(n); }
void qpr_destroy(void* p) { delete static_cast<Workspace*>(p); }
void qpr_reset(void* p) {
    auto& w=*static_cast<Workspace*>(p); w.solver.reset(); w.initialized=false;
}
int qpr_solve(void* p, const double* H,const double* g,const double* lo,
              const double* hi,double* x,int* info,long long* times) {
    auto& w=*static_cast<Workspace*>(p);
    auto a=Clock::now();
    // Never overwrite the old Hessian before matrix-shift hotstart has used it.
    w.slot=1-w.slot; std::copy(H,H+w.n*w.n,w.h[w.slot].begin());
    auto b=Clock::now(); int_t nwsr=1000;
    bool initial=!w.initialized;
    returnValue status;
    if(initial) status=w.solver.init(w.h[w.slot].data(),g,nullptr,lo,hi,nullptr,nullptr,nwsr);
    else status=w.solver.hotstart(w.h[w.slot].data(),g,nullptr,lo,hi,nullptr,nullptr,nwsr);
    auto c=Clock::now();
    auto primal=w.solver.getPrimalSolution(x);
    // A failed call is recorded; no hidden retry or alternative QP solve.
    w.initialized=(w.solver.getStatus()==QPS_SOLVED);
    info[0]=static_cast<int>(status); info[1]=nwsr;
    info[2]=static_cast<int>(primal); info[3]=initial;
    times[0]=ns(a,b); times[1]=ns(b,c); times[2]=ns(c,Clock::now());
    return static_cast<int>(status);
}
}
