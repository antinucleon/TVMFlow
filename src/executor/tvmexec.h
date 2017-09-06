#ifndef TVMFLOW_EXECUTOR_TVMEXEC_H_
#define TVMFLOW_EXECUTOR_TVMEXEC_H_

#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/tuple.h>
#include <tvm/expr.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvmflow/base.h>
#include <numeric>
#include <string>
namespace tvmflow {

#define TVM_CCALL(func)                    \
  {                                        \
    int ret = (func);                      \
    CHECK_EQ(ret, 0) << TVMGetLastError(); \
  }

struct VarState {
  DLTensor* blob{nullptr};
  DLDataType dtype;
  inline bool initialized() const { return blob != nullptr; }
  inline void ResetSpace(TShape s, DLContext ctx, DLDataType d) {
    if (blob == nullptr || s != Shape() || !_SameContext(ctx) || !_SameType(d)) {
      if (initialized()) {
        TVM_CCALL(TVMArrayFree(blob));
        blob = nullptr;
      }
      if (blob == nullptr) {
        TVM_CCALL(TVMArrayAlloc(
            s.data(), s.ndim(), d.code, d.bits, d.lanes, ctx.device_type, ctx.device_id, &blob));
      }
    }
    /*
    if (initialized()) {
      TVM_CCALL(TVMArrayFree(blob));
    }
    TVM_CCALL(TVMArrayAlloc(
        s.data(), s.ndim(), d.code, d.bits, d.lanes, ctx.device_type, ctx.device_id, &blob));
  */
  }
  inline bool _SameType(DLDataType d) const {
    if (initialized()) {
      auto dtype = Dtype();
      return (dtype.code == d.code && dtype.bits == d.bits && dtype.lanes == d.lanes);
    }
    return false;
  }
  inline bool _SameContext(DLContext ctx) const {
    if (initialized()) {
      auto context = Context();
      return (context.device_type == ctx.device_type && context.device_id == ctx.device_id);
    }
    return false;
  }

  inline TShape Shape() const {
    CHECK_EQ(initialized(), true);
    return DLShapeToTShape(blob->shape, blob->ndim);
  }
  inline DLDataType Dtype() const {
    CHECK_EQ(initialized(), true);
    return blob->dtype;
  }
  inline DLContext Context() const {
    CHECK_EQ(initialized(), true);
    return blob->ctx;
  }
  ~VarState() {
    if (initialized()) {
      TVM_CCALL(TVMArrayFree(blob));
    }
  }
};

// shared variable map structure
using VarStateMap = std::unordered_map<std::string, std::shared_ptr<VarState> >;
// operator executor closures
using FOpExec = std::function<void()>;

constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

class TVMExecutor {
 public:
  // Init executor
  void Init(Symbol symbol, VarStateMap* states, DLContext ctx, DLDataType dtype);
  // Run with given input
  const std::vector<DLTensor*>& Run(const std::unordered_map<std::string, DLTensor>& inputs);
  // return corresponding internal symbol
  inline const Symbol& symbol() const { return symbol_; }
  // destractor
  ~TVMExecutor() {
    CleanOutput();
    CleanPool();
  }

 private:
  // setup the executor space.
  inline void SetupAuxiliaryMembers();
  inline void ClearAuxiliaryMembers();
  inline void Setup(const std::unordered_map<std::string, DLTensor>& inputs);
  inline void SetupShapeDType(const std::unordered_map<std::string, DLTensor>& inputs,
                              bool* need_redo_infer);
  inline void SetupStorage();
  inline void SetupOpExecs();
  FOpExec CreateTVMOp(const nnvm::NodeAttrs& attrs, std::vector<DLTensor> args, size_t num_inputs);
  // clean the executor space
  void CleanPool();
  void CleanOutput();
  // internal symbol and graph
  nnvm::Symbol symbol_;
  nnvm::Graph graph_;
  // variable states map.
  VarStateMap* var_states_;
  // shape vector in graph attribute
  const nnvm::ShapeVector* node_shape_{nullptr};
  // type vector in graph attribute
  const nnvm::DTypeVector* node_dtype_{nullptr};
  // executor context
  DLContext ctx_;
  // executor data type
  DLDataType dtype_;
  // node id of place holder ops
  std::vector<uint32_t> placeholder_nids_;
  // size of number of node, placeholder_tblobs_[nid].data != nullptr
  // if nid is a placeholder and the content is the corresponding TBlob to be copied in.
  std::vector<DLTensor> placeholder_tblobs_;
  // node id of variable that is assigned in this executor
  std::vector<uint32_t> assign_var_nids_;
  // node id of variable that is readed by this executor
  // can overlap with assign_var_nids_
  std::vector<uint32_t> read_var_nids_;
  // vector maps nid->state, nullptr for non variables.
  std::vector<VarState*> node_states_;
  // assign op name
  std::unordered_set<std::string> assign_op_name_;
  // placeholder op name
  std::unordered_set<std::string> placeholder_op_name_;
  // ----------------------------
  // execution information
  // data of each outputs
  std::vector<DLTensor> data_entry_;
  // whether data entry is variable.
  std::vector<bool> data_entry_is_var_;
  // internal storage space.
  std::vector<DLTensor*> storage_pool_;
  // operator executor closures
  std::vector<FOpExec> op_execs_;
  // tvm module states of each operator.
  tvm::runtime::Module module_;
  // The storage space to hold outputs.
  std::vector<DLTensor*> outputs_;
};  // class TVMExecutor

}  // namespace tvmflow
#endif
