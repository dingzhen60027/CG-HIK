// C ABI only. The three official trac_ik_lib source files are compiled unmodified.
#include <trac_ik/trac_ik.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <memory>
#include <string>
#include <stdexcept>

struct Adapter {
  KDL::Chain chain;
  std::unique_ptr<TRAC_IK::TRAC_IK> solver;
  std::string names;
};
static thread_local std::string last_error;
extern "C" {
const char* revision_trac_error() { return last_error.c_str(); }
void* revision_trac_create(const char* path, const char* base, const char* tip,
                          const double* lower, const double* upper, int n,
                          double timeout, double epsilon) {
  try {
    auto ptr = std::make_unique<Adapter>();
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(path, tree) || !tree.getChain(base, tip, ptr->chain))
      throw std::runtime_error("URDF base/tool chain extraction failed");
    if ((int)ptr->chain.getNrOfJoints() != n) throw std::runtime_error("Joint count mismatch");
    KDL::JntArray lb(n), ub(n);
    for (int i=0;i<n;i++) {lb(i)=lower[i];ub(i)=upper[i];}
    for (auto& seg : ptr->chain.segments) {
      if (seg.getJoint().getType()!=KDL::Joint::Fixed) {
        if (!ptr->names.empty()) ptr->names += "|";
        ptr->names += seg.getJoint().getName();
      }
    }
    ptr->solver = std::make_unique<TRAC_IK::TRAC_IK>(ptr->chain,lb,ub,timeout,epsilon,TRAC_IK::Speed);
    return ptr.release();
  } catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
const char* revision_trac_names(void* handle) {return ((Adapter*)handle)->names.c_str();}
void revision_trac_destroy(void* handle) {delete (Adapter*)handle;}
void revision_trac_seed(unsigned seed) {srand(seed);}
int revision_trac_solve(void* handle, int n, const double* previous, const double* position,
                        const double* rotation, const double* lower, const double* upper, double* output) {
  try {
    auto& adapter=*(Adapter*)handle;
    KDL::JntArray init(n),lb(n),ub(n),out(n);
    for(int i=0;i<n;i++){init(i)=previous[i];lb(i)=lower[i];ub(i)=upper[i];}
    adapter.solver->setKDLLimits(lb,ub); // required per-query setup remains inside timed call
    KDL::Frame target(KDL::Rotation(rotation[0],rotation[1],rotation[2],rotation[3],rotation[4],rotation[5],rotation[6],rotation[7],rotation[8]),
                      KDL::Vector(position[0],position[1],position[2]));
    int rc=adapter.solver->CartToJnt(init,target,out);
    for(int i=0;i<n;i++)output[i]=out(i);
    return rc;
  } catch(const std::exception& e) {last_error=e.what();return -999;}
}
int revision_trac_fk(void* handle, int n, const double* q, double* output) {
  KDL::JntArray joints(n);for(int i=0;i<n;i++)joints(i)=q[i];
  KDL::Frame frame; KDL::ChainFkSolverPos_recursive fk(((Adapter*)handle)->chain);
  int rc=fk.JntToCart(joints,frame);
  for(int i=0;i<3;i++){output[i]=frame.p(i);for(int j=0;j<3;j++)output[3+3*i+j]=frame.M(i,j);}
  return rc;
}
}
