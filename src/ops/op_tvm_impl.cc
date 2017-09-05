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

NNVM_REGISTER_OP(placeholder)
    .describe("placeholder op")
    .set_num_inputs(0)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeNop);

// NNVM_REGISTER_OP(__add_symbol__)
//     .set_attr<FTVMCompute>("FTVMCompute", ComputeAdd)
//    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise);

NNVM_REGISTER_OP(exp)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeExp)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise);

NNVM_REGISTER_OP(_nop)
    .describe("no operation")
    .set_num_inputs(0)
    .set_num_outputs(1)
    .set_attr<FInferShape>("FInferShape", EmptyAttr<TShape>)
    .set_attr<FInferType>("FInferType", EmptyAttr<int>)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeNop)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleExtern);

NNVM_REGISTER_OP(assign)
    .describe("assign second to the first")
    .set_num_inputs(2)
    .set_num_outputs(1)
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
    .set_attr<FInferShape>("FInferShape", SameShape)
    .set_attr<int>("TOpPattern", kElemWise)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise);

NNVM_REGISTER_OP(__add_symbol__)
    .describe("add two data together")
    .set_num_inputs(2)
    .set_attr<FTVMCompute>("FTVMCompute", ComputeAdd)
    .set_attr<FTVMSchedule>("FTVMSchedule", ScheduleEWise)
    .set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
    .set_attr<FGradient>("FGradient", [](const NodePtr& n, const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
    });
}  // namespace tvmflow
