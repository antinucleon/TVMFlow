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

Array<Tensor> ComputeFlatten(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.flatten");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeFlattenBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.flatten_bwd");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeBias(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  CHECK_EQ(inputs.size(), 2);
  if (inputs[0].ndim() == 4) {
    static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bias4d");
    Tensor ret = pf(inputs[0], inputs[1]);
    return {ret};
  } else if (inputs[0].ndim() == 2) {
    static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bias2d");
    Tensor ret = pf(inputs[0], inputs[1]);
    return {ret};
  } else {
    LOG(FATAL) << "Not support dim";
    return {};
  }
}

Array<Tensor> ComputeBiasBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  CHECK_EQ(inputs.size(), 2);
  if (inputs[0].ndim() == 4) {
    static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bias4d_bwd");
    Tensor gb = pf(inputs[0], inputs[1]);
    static const PackedFunc& pf2 = GetPackedFunc("tvm_graph.compute.indentity");
    Tensor gd = pf2(inputs[0]);
    return {gd, gb};
  } else if (inputs[0].ndim() == 2) {
    static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bias2d_bwd");
    Tensor gb = pf(inputs[0], inputs[1]);
    static const PackedFunc& pf2 = GetPackedFunc("tvm_graph.compute.indentity");
    Tensor gd = pf2(inputs[0]);
    return {gd, gb};
  } else {
    LOG(FATAL) << "Not support dim";
    return {};
  }
}

Array<Tensor> ComputeBatchNorm(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bn_train");
  CHECK_EQ(inputs.size(), 3U);
  std::string eps;
  if (attrs.dict.find("eps") != attrs.dict.end()) {
    eps = attrs.dict.at("eps");
  } else {
    eps = "1e-5";
  }
  Tensor ret = pf(inputs[0], inputs[1], inputs[2], atof(eps.c_str()));
  return {ret};
}

Array<Tensor> ComputeBatchNormBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  CHECK_EQ(inputs.size(), 3U);
  std::string eps;
  if (attrs.dict.find("eps") != attrs.dict.end()) {
    eps = attrs.dict.at("eps");
  } else {
    eps = "1e-5";
  }
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.bn_bwd_data");
  Tensor gx = pf(inputs[0], inputs[1], inputs[2], atof(eps.c_str()));
  static const PackedFunc& pf2 = GetPackedFunc("tvm_graph.compute.bn_bwd_gamma");
  Tensor gg = pf2(inputs[0], inputs[1], inputs[2], atof(eps.c_str()));
  static const PackedFunc& pf3 = GetPackedFunc("tvm_graph.compute.bias4d_bwd");
  Tensor gb = pf3(inputs[0], inputs[2]);
  return {gx, gg, gb};
}

Array<Tensor> ComputeConv(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  CHECK_EQ(inputs.size(), 2U);
  std::string pad;
  std::string stride;
  if (attrs.dict.find("pad") != attrs.dict.end()) {
    pad = attrs.dict.at("pad");
  } else {
    LOG(INFO) << "Use default SAME pad.";
    pad = "SAME";
  }
  if (attrs.dict.find("stride") != attrs.dict.end()) {
    stride = attrs.dict.at("stride");
  } else {
    LOG(INFO) << "Use default stride 1.";
    stride = "1";
  }
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.conv2d");
  Tensor ret = pf(inputs[0], inputs[1], atoi(stride.c_str()), pad);
  return {ret};
}

Array<Tensor> ComputeConvBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  CHECK_EQ(inputs.size(), 3U);  // ograd, input, weight
  std::string pad;
  std::string stride;
  if (attrs.dict.find("pad") != attrs.dict.end()) {
    pad = attrs.dict.at("pad");
  } else {
    LOG(INFO) << "Use default SAME pad.";
    pad = "SAME";
  }
  if (attrs.dict.find("stride") != attrs.dict.end()) {
    stride = attrs.dict.at("stride");
  } else {
    LOG(INFO) << "Use default stride 1.";
    stride = "1";
  }
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.conv2d_bwd_data");
  Tensor gdata = pf(inputs[0], inputs[1], inputs[2], stride, pad);
  static const PackedFunc& pf2 = GetPackedFunc("tvm_graph.compute.conv2d_bwd_weight");
  Tensor gweight = pf2(inputs[0], inputs[1], inputs[2], stride, pad);
  return {gdata, gweight};
}

Array<Tensor> ComputeGlobalPool(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.global_pool");
  CHECK_EQ(inputs.size(), 1U);
  Tensor ret = pf(inputs[0]);
  return {ret};
}

Array<Tensor> ComputeGlobalPoolBwd(const NodeAttrs& attrs, const Array<Tensor>& inputs) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.compute.global_pool_bwd");
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

Schedule ScheduleSoftmaxBwd(const NodeAttrs& attrs, const Array<Tensor>& outs,
                            const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.softmax_bwd");
  return pf(outs, target);
}

Schedule ScheduleConv(const NodeAttrs& attrs, const Array<Tensor>& outs,
                      const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.conv");
  return pf(outs, target);
}

Schedule ScheduleReduction(const NodeAttrs& attrs, const Array<Tensor>& outs,
                           const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.reduction");
  return pf(outs, target);
}

Schedule ScheduleSoftmax(const NodeAttrs& attrs, const Array<Tensor>& outs,
                         const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.softmax");
  return pf(outs, target);
}

Schedule ScheduleBroadcast(const NodeAttrs& attrs, const Array<Tensor>& outs,
                           const std::string& target) {
  static const PackedFunc& pf = GetPackedFunc("tvm_graph.schedule.broadcast");
  return pf(outs, target);
}
}  // namespace tvmflow
#endif  // TVMFLOW_OPS_OP_TVM_DEF_