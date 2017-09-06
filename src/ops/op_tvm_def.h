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

enum UFUNC : int { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3 };

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

Array<Tensor> ComputeSum(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc");
  CHECK_GE(inputs.size(), 1U);
  int op = kAdd;
  Tensor ret;
  if (inputs.size() == 1) {
    ret = inputs[0];
  } else {
    ret = pf(inputs[0], inputs[1], op);
    for (size_t i = 2; i < inputs.size(); ++i) {
      ret = pf(ret, inputs[i], op);
    }
  }
  return {ret};
}

Array<Tensor> ComputeAdd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc");
  CHECK_EQ(inputs.size(), 2U);
  int op = kAdd;
  Tensor ret = pf(inputs[0], inputs[1], op);
  return {ret};
}

Array<Tensor> ComputeAddScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc_scalar");
  CHECK_EQ(inputs.size(), 1U);
  int op = kAdd;
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar, op);
  return {ret};
}

Array<Tensor> ComputeSub(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc");
  CHECK_EQ(inputs.size(), 2U);
  int op = kSub;
  Tensor ret = pf(inputs[0], inputs[1], op);
  return {ret};
}

Array<Tensor> ComputeSubScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc_scalar");
  CHECK_EQ(inputs.size(), 1U);
  int op = kSub;
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar, op);
  return {ret};
}

Array<Tensor> ComputeRSubScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.rsub");
  CHECK_EQ(inputs.size(), 1U);
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar);
  return {ret};
}

Array<Tensor> ComputeMul(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc");
  CHECK_EQ(inputs.size(), 2U);
  int op = kMul;
  Tensor ret = pf(inputs[0], inputs[1], op);
  return {ret};
}

Array<Tensor> ComputeMulScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc_scalar");
  CHECK_EQ(inputs.size(), 1U);
  int op = kMul;
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar, op);
  return {ret};
}

Array<Tensor> ComputeDiv(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc");
  CHECK_EQ(inputs.size(), 2U);
  int op = kDiv;
  Tensor ret = pf(inputs[0], inputs[1], op);
  return {ret};
}

Array<Tensor> ComputeDivScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc_scalar");
  CHECK_EQ(inputs.size(), 1U);
  int op = kDiv;
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar, op);
  return {ret};
}

Array<Tensor> ComputeExp(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.exp");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeLog(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.log");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeSqrt(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.sqrt");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputePow(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.pow");
  CHECK_EQ(inputs.size(), 2U);
  LOG(FATAL) << "intrin_rule is not implemented";
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeRPowScalar(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.ufunc_scalar");
  CHECK_EQ(inputs.size(), 1U);
  LOG(FATAL) << "intrin_rule is not implemented";
  int op = kDiv;
  float scalar = atof(attrs.dict.at("scalar").c_str());
  Tensor ret = pf(inputs[0], scalar, op);
  return {ret};
}

Array<Tensor> ComputeMatmul(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.matmul");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeMatmulBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& mm = GetPackedFunc("tvm_graph.compute.matmul");
  static const PackedFunc& xt = GetPackedFunc("tvm_graph.compute.mat_trans");
  CHECK_EQ(inputs.size(), 3U);
  Tensor rhs = mm(xt(inputs[1]), inputs[0]);
  Tensor lhs = mm(inputs[0], xt(inputs[2]));
  return {lhs, rhs};
}

Array<Tensor> ComputeArgmax(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.tensor.argmax");
  const auto& axis = dmlc::get<ReduceParam>(attrs.parsed).reduction_indices;
  CHECK_EQ(axis.ndim(), 1);
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0], axis[0]);
  return {ret};
}

Array<Tensor> ComputeRedSum(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.red_sum");
  std::string axis = "[]";
  if (attrs.dict.find("reduction_indices") != attrs.dict.end()) {
    axis = attrs.dict.at("reduction_indices");
  }
  bool keepdims = false;
  if (attrs.dict.find("keepdims") != attrs.dict.end()) {
    LOG(INFO) << attrs.dict.at("keepdims");
    if (attrs.dict.at("keepdims") == "True") keepdims = true;
  }
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0], axis, keepdims);
  return {ret};
}

Array<Tensor> ComputeRedSumBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.red_sum_bwd");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeRedMean(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.red_mean");
  std::string axis = "[]";
  if (attrs.dict.find("reduction_indices") != attrs.dict.end()) {
    axis = attrs.dict.at("reduction_indices");
  }
  bool keepdims = false;
  if (attrs.dict.find("keepdims") != attrs.dict.end()) {
    if (attrs.dict.at("keepdims") == "True") keepdims = true;
  }
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0], axis, keepdims);
  return {ret};
}

Array<Tensor> ComputeRedMeanBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.red_mean_bwd");
  CHECK_EQ(inputs.size(), 2U);
  std::string axis = "[]";
  if (attrs.dict.find("reduction_indices") != attrs.dict.end()) {
    axis = attrs.dict.at("reduction_indices");
  }
  Tensor ret = pf(inputs[0], inputs[1], axis);
  return {ret};
}

Array<Tensor> ComputeSoftmax(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.softmax");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeSoftmaxBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.softmax_bwd");
  CHECK_EQ(inputs.size(), 2U);
  Tensor ret = pf(inputs[0], inputs[1]);
  return {ret};
}

Array<Tensor> ComputeReLU(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.relu");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeReLUBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.relu_bwd");
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

Schedule ScheduleMatmul(const NodeAttrs& attrs, const Array<Tensor>& outs,
                        const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.matmul");
  return pf(outs, target);
}

Schedule ScheduleReduction(const NodeAttrs& attrs, const Array<Tensor>& outs,
                           const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.reduction");
  return pf(outs, target);
}

Schedule ScheduleBroadcast(const NodeAttrs& attrs, const Array<Tensor>& outs,
                           const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.broadcast");
  return pf(outs, target);
}
}  // namespace tvmflow
#endif  // TVMFLOW_OPS_OP_TVM_DEF_