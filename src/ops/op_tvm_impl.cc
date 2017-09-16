/*!
 *  Copyright (c) 2017 by Contributors
 * \file Operator defintions in TVM.
 */
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <tvmflow/base.h>
#include "./op_attr_types.h"
#include "./op_tvm_def.h"

namespace tvmflow {

using namespace nnvm;
using tvm::Tensor;
using tvm::runtime::PackedFunc;

NNVM_REGISTER_OP_GROUP(ElementwiseOpAttr)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise)
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(placeholder)
    .describe("placeholder op")
    .set_num_inputs(0)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeNop);

NNVM_REGISTER_OP(_nop)
    .describe("no operation")
    .set_num_inputs(0)
    .set_num_outputs(1)
    .set_attr<FInferShape>("FInferShape", EmptyAttr<TShape>)
    .set_attr<FInferType>("FInferType", EmptyAttr<int>)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeNop);

NNVM_REGISTER_OP(assign)
    .describe("assign second to the first")
    .set_num_inputs(2)
    .set_attr<FMutateInputs>("FMutateInputs",
                             [](const NodeAttrs& attrs) { return std::vector<uint32_t>{0}; })
    .set_attr<FInferShape>("FInferShape", AssignShape)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn1Out0)
    .set_attr<FInferType>("FInferType", AssignType)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeAssign)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<int>("TOpPattern", kExtern);

// special no gradient op to report error when take
// gradient wrt non-differentiable inputs
NNVM_REGISTER_OP(_no_gradient).describe("Special op indicating no gradient").set_num_inputs(0);

////////////////////////// Tensor OP
NNVM_REGISTER_OP(zeros)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeZeros)
    .set_num_inputs(0)
    .set_num_outputs(1)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FInferType>("FInferType", NodeType)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", NodeShape);

NNVM_REGISTER_OP(zeros_like)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeZerosLike)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FInferType>("FInferType", SameType)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(ones)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeOnes)
    .set_num_inputs(0)
    .set_num_outputs(1)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FInferType>("FInferType", NodeType)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", NodeShape);

NNVM_REGISTER_OP(ones_like)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeOnesLike)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FInferType>("FInferType", SameType)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(normal)
    .describe("normal distribution")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeNormal)
    .set_num_inputs(0)
    .set_num_outputs(1)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FInferType>("FInferType", NodeType)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", NodeShape);

NNVM_REGISTER_OP(equal)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeEqual)
    .include("ElementwiseOpAttr");

NNVM_REGISTER_OP(__add_symbol__)
    .describe("add two data together")
    .set_num_inputs(2)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeAdd)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient",
                         [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
                           return std::vector<NodeEntry>{ograds[0], ograds[0]};
                         })
    .include("ElementwiseOpAttr");

NNVM_REGISTER_OP(__ewise_sum__)
    .describe("ewise sum")
    .set_num_inputs(nnvm::kVarg)
    .set_num_outputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSum)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>(n->num_inputs(), ograds[0]);
    });

NNVM_REGISTER_OP(__add_scalar__)
    .describe("add symbol with scalar")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeAddScalar)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{ograds[0]};
    });

NNVM_REGISTER_OP(__sub_symbol__)
    .describe("do subtract")
    .set_num_inputs(2)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSub)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__mul_scalar__", n->attrs.name + "_grad_0", {ograds[0]}, {{"scalar", "1"}}),
          MakeNode("__mul_scalar__", n->attrs.name + "_grad_1", {ograds[0]}, {{"scalar", "-1"}}),
      };
    });

NNVM_REGISTER_OP(__sub_scalar__)
    .describe("subtract symbol with scalar")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSubScalar)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{ograds[0]};
    });

NNVM_REGISTER_OP(__rsub_scalar__)
    .describe("subtract scalar with symbol")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRSubScalar)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__mul_scalar__", n->attrs.name + "_grad_1", {ograds[0]}, {{"scalar", "-1"}}),
      };
    });

NNVM_REGISTER_OP(mul)
    .add_alias("__mul_symbol__")
    .describe("add two data together")
    .set_num_inputs(2)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeMul)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("mul", n->attrs.name + "_grad_0", {ograds[0], n->inputs[1]}),
          MakeNode("mul", n->attrs.name + "_grad_1", {ograds[0], n->inputs[0]})};
    });

NNVM_REGISTER_OP(__mul_scalar__)
    .describe("Multiply symbol with scalar")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeMulScalar)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__mul_scalar__",
                   n->attrs.name + "_grad_0",
                   {ograds[0]},
                   {{"scalar", n->attrs.dict["scalar"]}}),
      };
    });

NNVM_REGISTER_OP(__div_symbol__)
    .add_alias("div")
    .describe("do division")
    .set_num_inputs(2)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeDiv)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      NodeEntry n1 = MakeNode("mul", n->attrs.name + "_grad_sub_0", {ograds[0], n->inputs[0]});
      NodeEntry n2 =
          MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1", {n1}, {{"scalar", "-1"}});
      NodeEntry n3 = MakeNode("mul", n->attrs.name + "_grad_sub_2", {n->inputs[1], n->inputs[1]});
      return std::vector<NodeEntry>{
          MakeNode("__div_symbol__", n->attrs.name + "_grad_0", {ograds[0], n->inputs[1]}),
          MakeNode("__div_symbol__", n->attrs.name + "_grad_1", {n1, n2})};
    });

NNVM_REGISTER_OP(__div_scalar__)
    .describe("division symbol with scalar")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeDivScalar)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__div_scalar__",
                   n->attrs.name + "_grad_0",
                   {ograds[0]},
                   {{"scalar", n->attrs.dict["scalar"]}}),
      };
    });

NNVM_REGISTER_OP(exp)
    .describe("take elemtnwise exponation")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeExp)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__mul_symbol__", n->attrs.name + "_grad_0", {ograds[0], NodeEntry{n, 0, 0}})};
    });

NNVM_REGISTER_OP(log)
    .describe("take elemtnwise logarithm")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeLog)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("__div_symbol__", n->attrs.name + "_grad_0", {ograds[0], n->inputs[0]})};
    });

NNVM_REGISTER_OP(sqrt)
    .describe("return square root of input")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSqrt)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>(
        // 1 / (2 * sqrt(x)) == 1 / (2 * y)
        "FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
          NodeEntry n1 = MakeNode("__mul_scalar__",
                                  n->attrs.name + "_grad_sub_1",
                                  {NodeEntry{n, 0, 0}},
                                  {{"scalar", "2"}});
          return std::vector<NodeEntry>{
              MakeNode("__div_symbol__", n->attrs.name + "_grad_0", {ograds[0], n1})};
        });

NNVM_REGISTER_OP(__pow_symbol__)
    .add_alias("pow")
    .describe("take elmtnwise power between two tensor")
    .set_num_inputs(2)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputePow)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      // lhs: b*pow(a, b-1), rhs: pow(a, b)*ln(a)
      NodeEntry n0 = MakeNode(
          "__add_scalar__", n->attrs.name + "_grad_sub_0", {n->inputs[1]}, {{"scalar", "-1"}});
      NodeEntry n1 = MakeNode("pow", n->attrs.name + "_grad_sub_1", {n->inputs[0], n0});
      NodeEntry d_lhs = MakeNode("mul", n->attrs.name + "_grad_sub_2", {n1, n->inputs[1]});
      NodeEntry n2 = MakeNode("log", n->attrs.name + "_grad_sub_3", {n->inputs[0]});
      NodeEntry d_rhs = MakeNode("mul", n->attrs.name + "_grad_sub_4", {NodeEntry{n, 0, 0}, n2});
      return std::vector<NodeEntry>{
          MakeNode("__mul_symbol__", n->attrs.name + "_grad_0", {ograds[0], d_lhs}),
          MakeNode("__mul_symbol__", n->attrs.name + "_grad_1", {ograds[0], d_rhs})};

    });

NNVM_REGISTER_OP(__rpow_scalar__)
    .describe("take elmtnwise power between a number and a tensor")
    .set_num_inputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRPowScalar)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      // pow(m, x) * ln(m)
      double num = std::stod(n->attrs.dict["scalar"]);
      NodeEntry n0 = MakeNode("__mul_scalar__",
                              n->attrs.name + "_grad_sub_4",
                              {NodeEntry{n, 0, 0}},
                              {{"scalar", std::to_string(std::log(num))}});
      return std::vector<NodeEntry>{
          MakeNode("__mul_symbol__", n->attrs.name + "_grad_0", {ograds[0], n0})};
    });

NNVM_REGISTER_OP(matmul)
    .describe("Matrix multiplication")
    .set_num_inputs(2)
    .set_attr<FInferShape>("FInferShape",
                           [](const NodeAttrs& attrs, std::vector<TShape>* ishape,
                              std::vector<TShape>* oshape) {
                             if (ishape->at(0).ndim() == 0) return false;
                             if (ishape->at(1).ndim() == 0) return false;
                             CHECK_EQ(ishape->at(0).ndim(), 2);
                             CHECK_EQ(ishape->at(1).ndim(), 2);
                             CHECK_EQ(ishape->at(0)[1], ishape->at(1)[0]);
                             TShape target{ishape->at(0)[0], ishape->at(1)[1]};
                             SHAPE_ASSIGN(oshape->at(0), target);
                             return true;
                           })
    .set_attr<FTVMCompute>("FTVMCompute", ComputeMatmul)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleMatmul)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_matmul_backward", n, {ograds[0], n->inputs[0], n->inputs[1]});
    });

NNVM_REGISTER_OP(_matmul_backward)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeMatmulBwd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleMatmul)
    .set_attr<int>("TOpPattern", kExtern)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true);

inline bool ReduceShape(const NodeAttrs& attrs, std::vector<TShape>* ishape,
                        std::vector<TShape>* oshape) {
  const auto& axis = dmlc::get<ReduceParam>(attrs.parsed).reduction_indices;
  if (ishape->at(0).ndim() == 0) return false;
  if (axis.ndim() == 0) {
    SHAPE_ASSIGN(oshape->at(0), TShape{1});
  } else {
    TShape tmp = ishape->at(0);
    for (uint32_t idx : axis) {
      tmp[idx] = 0;
    }
    std::vector<uint32_t> ret;
    for (uint32_t x : tmp) {
      if (x != 0) ret.push_back(x);
    }
    if (ret.size() == 0) ret.push_back(1);
    SHAPE_ASSIGN(oshape->at(0), TShape(ret.begin(), ret.end()));
  }
  return true;
}

DMLC_REGISTER_PARAMETER(ReduceParam);

NNVM_REGISTER_OP_GROUP(ReduceBackwardIndeAttr).set_attr<nnvm::TIsBackward>("TIsBackward", true);

NNVM_REGISTER_OP(_argmax)
    .set_attr_parser(ParamParser<ReduceParam>)
    .set_num_inputs(1)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeArgmax)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern)
    .set_attr<FInferShape>("FInferShape", ReduceShape);

NNVM_REGISTER_OP(reduce_sum)
    .describe("reduce sum")
    .set_attr_parser(ParamParser<ReduceParam>)
    .set_num_inputs(1)
    .set_attr<FInferShape>("FInferShape", ReduceShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRedSum)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_reduce_sum_backward", n, {ograds[0], n->inputs[0]}, n->attrs.dict);
    });

NNVM_REGISTER_OP(_reduce_sum_backward)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRedSumBwd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .include("ReduceBackwardIndeAttr");

NNVM_REGISTER_OP(reduce_mean)
    .describe("reduce mean")
    .set_attr_parser(ParamParser<ReduceParam>)
    .set_num_inputs(1)
    .set_attr<FInferShape>("FInferShape", ReduceShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRedMean)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads(
          "_reduce_mean_backward", n, {ograds[0], n->inputs[0]}, n->attrs.dict);
    });

NNVM_REGISTER_OP(_reduce_mean_backward)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeRedMeanBwd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .include("ReduceBackwardIndeAttr");

NNVM_REGISTER_OP(softmax)
    .describe("softmax")
    .set_num_inputs(1)
    .set_attr<TOpPattern>("TOpPattern", kComplex)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSoftmax)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("softmax_bwd", n->attrs.name + "_grad_0", {ograds[0], NodeEntry{n, 0, 0}})};
    });

NNVM_REGISTER_OP(softmax_bwd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<TOpPattern>("TOpPattern", kComplex)
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeSoftmaxBwd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true);

NNVM_REGISTER_OP(relu)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeReLU)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("relu_bwd", n->attrs.name + "_grad_0", {ograds[0], n->inputs[0]})};
    });

NNVM_REGISTER_OP(relu_bwd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeReLUBwd)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true);

NNVM_REGISTER_OP(flatten)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<FTVMCompute>("FTVMCompute", ComputeFlatten)
    .set_attr<nnvm::FInferShape>("FInferShape",
                                 [](const NodeAttrs& attrs, std::vector<TShape>* ishape,
                                    std::vector<TShape>* oshape) {
                                   CHECK_EQ(ishape->size(), 1);
                                   if (ishape->at(0).ndim() == 0) return false;
                                   TShape target{ishape->at(0)[0], ishape->at(0)[1]};
                                   SHAPE_ASSIGN(oshape->at(0), target);
                                   return true;
                                 })
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
          MakeNode("flatten_bwd", n->attrs.name + "_grad_0", {ograds[0]})};
    });

NNVM_REGISTER_OP(flatten_bwd)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .include("ElementwiseOpAttr")
    .set_attr<nnvm::FInferShape>("FInferShape",
                                 [](const NodeAttrs& attrs, std::vector<TShape>* ishape,
                                    std::vector<TShape>* oshape) {
                                   CHECK_EQ(ishape->size(), 1);
                                   if (ishape->at(0).ndim() == 0) return false;
                                   TShape target{ishape->at(0)[0], ishape->at(0)[1], 1, 1};
                                   SHAPE_ASSIGN(oshape->at(0), target);
                                   return true;
                                 })
    .set_attr<FTVMCompute>("FTVMCompute", ComputeFlattenBwd)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true);

NNVM_REGISTER_OP(batch_norm)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr<TOpPattern>("TOpPattern", kComplex)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeBatchNorm)

    .set_attr<nnvm::FInferShape>("FInferShape",
                                 [](const NodeAttrs& attrs, std::vector<TShape>* ishape,
                                    std::vector<TShape>* oshape) {
                                   CHECK_EQ(ishape->size(), 3);
                                   if (ishape->at(0).ndim() == 0) return false;
                                   CHECK_EQ(ishape->at(0).ndim(), 4);
                                   TShape pshape{1, ishape->at(0)[1], 1, 1};
                                   SHAPE_ASSIGN(ishape->at(1), pshape);
                                   SHAPE_ASSIGN(ishape->at(2), pshape);
                                   SHAPE_ASSIGN(oshape->at(0), ishape->at(0));
                                   return true;
                                 })
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return MakeBackwardGrads("_batch_norm_backward", n, {ograds[0], n->inputs[0], n->inputs[1]});
    });

NNVM_REGISTER_OP(_batch_norm_backward)
    .set_num_inputs(3)
    .set_num_outputs(3)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeBatchNormBwd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleReduction)
    .set_attr<int>("TOpPattern", kComplex)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true);

}  // namespace tvmflow
