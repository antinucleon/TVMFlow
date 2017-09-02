/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Basic data structures.
 */
#ifndef TVMFLOW_BASE_H_
#define TVMFLOW_BASE_H_

#include <dlpack/dlpack.h>
#include <nnvm/base.h>
#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <nnvm/symbolic.h>
#include <nnvm/tuple.h>
#include <string>
#include <vector>

namespace tvmflow {

using nnvm::Graph;
using nnvm::Node;
using nnvm::Op;
using nnvm::Symbol;
using nnvm::TShape;

/*!
 * \brief If registered and TBackwardNumNoGrad=k
 *  The last k inputs do not have gradient.
 * \note Register as TBackwardNumNoGradInputs
 */
using TBackwardNumNoGradInputs = int;

/*!
 * \brief Whether backward need weight.
 * \note Register as TBackwardNeedInputs
 */
using TBackwardNeedInputs = bool;

/*!
 * \brief Whether backward op need outputs.
 * \note Register as TBackwardNeedOutputs
 */
using TBackwardNeedOutputs = bool;

/*! \brief Executor of a graph */
class Session {
 public:
  /*!
   * \brief Run the given graph
   * \param g the graph to run.
   * \param inputs The input feed_dict mapping
   * \note The session hold the ownership of the outputs.
   *  The results are only valid before calling any functions of this session again.
   * \return The output tensors.
   */
  virtual const std::vector<DLTensor*>& Run(
      Symbol* g, const std::unordered_map<std::string, DLTensor>& inputs) = 0;
  /*! \brief virtual destructor */
  virtual ~Session() {}
  /*!
   * \brief create a new session of given type.
   * \param type The type of the session.
   * \return a new created session.
   */
  static Session* Create(const std::string& type);
};

inline TShape DLShapeToTShape(int64_t* shape, int ndim) { return TShape(shape, shape + ndim); }

struct TVMOpParam : public dmlc::Parameter<TVMOpParam> {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  bool flatten_data;
  DMLC_DECLARE_PARAMETER(TVMOpParam) {
    DMLC_DECLARE_FIELD(func_name);
    DMLC_DECLARE_FIELD(num_inputs).set_default(1);
    DMLC_DECLARE_FIELD(num_outputs).set_default(1);
    DMLC_DECLARE_FIELD(flatten_data).set_default(false);
  }
};

}  // namespace tvmflow

#endif  // TVMFLOW_BASE_H_