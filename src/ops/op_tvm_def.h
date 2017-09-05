// Copyright (c) 2016 by Contributors
// implementation of common nn operators
#ifndef TVMFLOW_OPS_OP_TVM_DEF_
#define TVMFLOW_OPS_OP_TVM_DEF_

#include <nnvm/op_attr_types.h>
#include <tvmflow/base.h>
#include <utility>
#include "./op_attr_types.h"
#include "./op_util.h"

namespace tvmflow {

using namespace nnvm;
using tvm::Tensor;
using tvm::runtime::PackedFunc;

template <typename Attr>
inline bool EmptyAttr(const NodeAttrs& attrs, std::vector<Attr>* ishape,
                      std::vector<Attr>* oshape) {
  oshape->at(0) = Attr{0};
  return true;
}

Array<Tensor> ComputeAssign(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.assign");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeAdd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.add");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeExp(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.exp");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeNop(const NodeAttrs& attrs, const Array<Tensor>& inputs) { return {}; }

///////////////// TensorOp

Array<Tensor> ComputeZeros(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.zeros");
  CHECK_EQ(inputs.size(), 0U);
  const std::string& shape = attrs.dict.at("shape");
  Tensor ret = pf(shape);
  return {ret};
}

Array<Tensor> ComputeZerosLike(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.zeros_like");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeOnes(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.ones");
  CHECK_EQ(inputs.size(), 0U);
  const std::string& shape = attrs.dict.at("shape");
  Tensor ret = pf(shape);
  return {ret};
}

Array<Tensor> ComputeOnesLike(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.ones_like");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeNormal(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.normal");
  CHECK_EQ(inputs.size(), 0U);
  const std::string& shape = attrs.dict.at("shape");
  std::string loc = "0.0";
  std::string scale = "1.0";
  if (attrs.dict.find("loc") != attrs.dict.end()) {
    loc = attrs.dict.at("loc");
  }
  if (attrs.dict.find("scale") != attrs.dict.end()) {
    loc = attrs.dict.at("scale");
  }
  Tensor ret = pf(shape, loc, scale);
  return {ret};
}

Array<Tensor> ComputeEqual(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.equal");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

/**************************************************
 *   Schedule
 **************************************************/

Schedule ScheduleExtern(const NodeAttrs& attrs, const Array<Tensor>& outs,
                        const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.extern");
  return pf(outs, target);
}

Schedule ScheduleEWise(const NodeAttrs& attrs, const Array<Tensor>& outs,
                       const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.ewise");
  return pf(outs, target);
}

}  // namespace tvmflow
#endif  // TVMFLOW_OPS_OP_TVM_DEF_