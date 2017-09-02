/*!
 *  Copyright (c) 2017 by Contributors
 * \file Operator Declarations.
 */
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <tvmflow/base.h>
#include "./op_attr_types.h"

namespace tvmflow {

inline bool SameShape(const nnvm::NodeAttrs& attrs, std::vector<TShape>* ishape,
                      std::vector<TShape>* oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

NNVM_REGISTER_OP_GROUP(ElementwiseOpAttr)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)
    .set_attr<nnvm::FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(__add_symbol__)
    .describe("add two data together")
    .set_num_inputs(2)
    .include("ElementwiseOpAttr");

NNVM_REGISTER_OP(exp).describe("Take exp").set_num_inputs(1).include("ElementwiseOpAttr");

}  // namespace tvmflow
