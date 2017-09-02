#include "tvmexec.h"

using DLTypeVector = std::vector<DLDataType>;

inline int GetNNVMType(DLDataType dtype) {
  if (dtype.code == kFloat && dtype.bits == 32 && dtype.lanes == 1) {
    return 0;
  }
  LOG(FATAL) << "unknown dtype=";
  return 0;
}

inline DLDataType GetDLType(int type_flag) {
  if (type_flag == 0) return tvm::Type2TVMType(tvm::Float(32));
  LOG(FATAL) << "unknown type_flag=" << type_flag;
  return tvm::Type2TVMType(tvm::Float(32));
}

inline void InitEmptyDLTensor(DLTensor& tensor) {
  tensor.data = nullptr;
  tensor.ndim = 0;
  tensor.shape = nullptr;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
}

namespace tvmflow {
void TVMExecutor::Init(Symbol symbol, VarStateMap* states, DLContext ctx, DLDataType dtype) {
  ctx_ = ctx;
  dtype_ = dtype;
  graph_.outputs = symbol.outputs;
  symbol_.outputs = graph_.outputs;
  var_states_ = states;
  std::string target;
  if (ctx_.device_type == kCPU) {
    target = "llvm";
    graph_.attrs["target"] = std::make_shared<dmlc::any>(std::move(target));
  } else if (ctx_.device_type == kMetal) {
    target = "metal";
    graph_.attrs["target"] = std::make_shared<dmlc::any>(std::move(target));
  } else if (ctx_.device_type == kOpenCL) {
    target = "opencl";
    graph_.attrs["target"] = std::make_shared<dmlc::any>(std::move(target));
  } else if (ctx_.device_type == kGPU) {
    target = "cuda";
    graph_.attrs["target"] = std::make_shared<dmlc::any>(std::move(target));
  }
  SetupAuxiliaryMembers();
}

void TVMExecutor::SetupAuxiliaryMembers() {
  // initialize all node auxiliary data structures.
  const Op* assign_op = Op::Get("assign");
  const Op* placeholder_op = Op::Get("placeholder");
  const auto& idx = graph_.indexed_graph();
  node_states_.resize(idx.num_nodes(), nullptr);

  std::vector<int> read_count(idx.num_nodes(), 0);
  std::vector<int> assign_count(idx.num_nodes(), 0);
  placeholder_tblobs_.resize(idx.num_nodes());

  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      const std::string& key = inode.source->attrs.name;
      if (var_states_->count(key) == 0) {
        (*var_states_)[key] = std::make_shared<VarState>();
      }
      node_states_[nid] = var_states_->at(key).get();
      if (read_count[nid] != 0 || assign_count[nid] == 0) {
        read_var_nids_.push_back(nid);
      }
      if (assign_count[nid] != 0) {
        assign_var_nids_.push_back(nid);
      }
    } else {
      const auto& name = inode.source->attrs.name;
      if (inode.source->op() == placeholder_op) {
        placeholder_op_name_.insert(name);
      }
      if (inode.source->op() == assign_op) {
        assign_op_name_.insert(name);
      }
      if (placeholder_op_name_.find(name) != placeholder_op_name_.end()) {
        placeholder_nids_.push_back(nid);
      } else if (assign_op_name_.find(name) != assign_op_name_.end()) {
        CHECK_EQ(inode.inputs.size(), 2);
        ++read_count[inode.inputs[1].node_id];
        ++assign_count[inode.inputs[0].node_id];
      } else {
        for (auto e : inode.inputs) {
          ++read_count[e.node_id];
        }
      }
    }
  }
}

void TVMExecutor::ClearAuxiliaryMembers() {
  placeholder_nids_.clear();
  placeholder_tblobs_.clear();
  assign_var_nids_.clear();
  read_var_nids_.clear();
  node_states_.clear();
}

const std::vector<DLTensor*>& TVMExecutor::Run(
    const std::unordered_map<std::string, DLTensor>& inputs) {
  Setup(inputs);
  {
    // execution
    const auto& idx = graph_.indexed_graph();

    for (size_t i = 0; i < op_execs_.size(); ++i) {
      // copy in place holder as demanded.
      if (placeholder_tblobs_[i].data != nullptr) {
        DLTensor src = placeholder_tblobs_[i];
        DLTensor dst = data_entry_[idx.entry_id(i, 0)];
        TVM_CCALL(TVMArrayCopyFromTo(&src, &dst, nullptr));
      }
      try {
        // TODO op_execs_[i].nil()?
        if (op_execs_[i]) {
          op_execs_[i]();
        }
      } catch (dmlc::Error e) {
        LOG(INFO) << "error catched in op " << idx[i].source->op()->name;
        throw e;
      }
    }
  }
  {
    // copy outputs
    // TODO: clear output everytime?
    const auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = idx.entry_id(idx.outputs()[i]);
      TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], outputs_[i], nullptr));
    }
  }
  return outputs_;
}

void TVMExecutor::Setup(const std::unordered_map<std::string, DLTensor>& inputs) {
  bool need_redo_infer;
  SetupShapeDType(inputs, &need_redo_infer);
  if (need_redo_infer) {
    SetupStorage();
    op_execs_.clear();
    SetupOpExecs();
  }
  {
    // copy inputs
    const auto& idx = graph_.indexed_graph();
    for (uint32_t nid : placeholder_nids_) {
      const std::string& key = idx[nid].source->attrs.name;
      const DLTensor& value = inputs.at(key);
      placeholder_tblobs_[nid] = value;
    }
  }
}

void TVMExecutor::SetupOpExecs() {
  static const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");
  const auto& idx = graph_.indexed_graph();
  op_execs_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (inode.source->op()->name == "placeholder") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(data_entry_[idx.entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      args.push_back(data_entry_[eid]);
    }
    CHECK_EQ(inode.source->op(), tvm_op) << "transform the graph to tvm op";
    op_execs_[nid] = CreateTVMOp(inode.source->attrs, args, inode.inputs.size());
  }
}

FOpExec TVMExecutor::CreateTVMOp(const nnvm::NodeAttrs& attrs, std::vector<DLTensor> args,
                                 size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  auto it = attrs.dict.find("func_name");
  CHECK(it != attrs.dict.end()) << "tvm_op must need func_name attr";
  bool flatten = (attrs.dict.at("flatten_data") == "1");
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (flatten) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (flatten) {
      int64_t s = 1;
      arg_ptr->shape_data[i] =
          std::accumulate(t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  // get compiled function from module.
  tvm::runtime::PackedFunc pf = module_.GetFunction(it->second, false);
  CHECK(pf != nullptr) << "no such function in module: " << it->second;
  auto fexec = [arg_ptr, pf]() {
    tvm::runtime::TVMRetValue rv;
    tvm::runtime::TVMArgs targs(arg_ptr->arg_values.data(),
                                arg_ptr->arg_tcodes.data(),
                                static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}
void TVMExecutor::CleanPool() {
  for (size_t i = 0; i < storage_pool_.size(); ++i) {
    TVM_CCALL(TVMArrayFree(storage_pool_[i]));
  }
}
void TVMExecutor::CleanOutput() {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    TVM_CCALL(TVMArrayFree(outputs_[i]));
  }
}
void TVMExecutor::SetupShapeDType(const std::unordered_map<std::string, DLTensor>& inputs,
                                  bool* p_need_redo_infer) {
  const auto& idx = graph_.indexed_graph();
  bool& need_redo_infer = *p_need_redo_infer;
  need_redo_infer = (node_shape_ == nullptr);

  // check the variable states
  if (!need_redo_infer) {
    CHECK(node_dtype_ != nullptr);
    for (uint32_t nid : read_var_nids_) {
      VarState* state = node_states_[nid];
      CHECK(state != nullptr);
      if (node_shape_->at(idx.entry_id(nid, 0)) != state->Shape()) {
        need_redo_infer = true;
        break;
      }
      if (node_dtype_->at(idx.entry_id(nid, 0)) != GetNNVMType(state->Dtype())) {
        need_redo_infer = true;
        break;
      }
    }
  }

  // check placeholder shapes.
  if (!need_redo_infer) {
    for (uint32_t nid : placeholder_nids_) {
      const std::string& key = idx[nid].source->attrs.name;
      CHECK(inputs.count(key)) << "Not enought placeholder argument to feed_dict";
      const DLTensor& value = inputs.at(key);
      auto value_shape = DLShapeToTShape(value.shape, value.ndim);
      if (node_shape_->at(idx.entry_id(nid, 0)) != value_shape) {
        need_redo_infer = true;
        break;
      }
      if (node_dtype_->at(idx.entry_id(nid, 0)) != GetNNVMType(value.dtype)) {
        need_redo_infer = true;
        break;
      }
    }
  }

  if (!need_redo_infer) return;
  // run shape inference.
  nnvm::ShapeVector new_shape(idx.num_node_entries(), TShape());
  nnvm::DTypeVector new_dtype(idx.num_node_entries(), -1);

  for (uint32_t nid : read_var_nids_) {
    VarState* state = node_states_[nid];
    // TODO more strict rule
    if (state->initialized()) {
      new_shape[idx.entry_id(nid, 0)] = state->Shape();
      new_dtype[idx.entry_id(nid, 0)] = GetNNVMType(state->Dtype());
    } else if (std::find(assign_var_nids_.cbegin(), assign_var_nids_.cend(), nid) ==
               assign_var_nids_.cend()) {
      CHECK(state->initialized()) << "Attempt to execute a graph un-initialized Variable";
    }
  }

  for (uint32_t nid : placeholder_nids_) {
    const std::string& key = idx[nid].source->attrs.name;
    const DLTensor& value = inputs.at(key);
    auto value_shape = DLShapeToTShape(value.shape, value.ndim);
    new_shape[idx.entry_id(nid, 0)] = value_shape;
    new_dtype[idx.entry_id(nid, 0)] = GetNNVMType(value.dtype);
  }

  graph_.attrs["shape"] = std::make_shared<dmlc::any>(std::move(new_shape));
  graph_.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(new_dtype));

  graph_ = ApplyPasses(std::move(graph_), {"InferShape", "InferType"});

  CHECK_EQ(graph_.GetAttr<size_t>("shape_num_unknown_nodes"), 0)
      << "Shape information in the graph is in-complete";
  CHECK_EQ(graph_.GetAttr<size_t>("dtype_num_unknown_nodes"), 0)
      << "Type information in the graph is in-complete";

  LOG(INFO) << "Compile the graph into execution code.";
  graph_ = ApplyPasses(std::move(graph_), {"GraphPartition", "GraphFuse"});
  auto func = tvm::runtime::Registry::Get("tvm_graph._get_module_from_graph");
  module_ = (*func)(&graph_);
  LOG(INFO) << "Compile finish.";
  node_shape_ = &(graph_.GetAttr<nnvm::ShapeVector>("shape"));
  node_dtype_ = &(graph_.GetAttr<nnvm::DTypeVector>("dtype"));
  ClearAuxiliaryMembers();
  SetupAuxiliaryMembers();

  const auto& new_idx = graph_.indexed_graph();
  for (uint32_t nid : assign_var_nids_) {
    node_states_[nid]->ResetSpace(node_shape_->at(new_idx.entry_id(nid, 0)),
                                  ctx_,
                                  GetDLType(node_dtype_->at(new_idx.entry_id(nid, 0))));
  }
}

void TVMExecutor::SetupStorage() {
  const auto& idx = graph_.indexed_graph();
  if (storage_pool_.size() == 0) {
    graph_ = nnvm::ApplyPass(std::move(graph_), "PlanMemory");
  }
  const auto& vstorage = graph_.GetAttr<nnvm::StorageVector>("storage_id");
  const auto& vshape = graph_.GetAttr<nnvm::ShapeVector>("shape");

  if (data_entry_.size() == 0) {
    data_entry_.resize(idx.num_node_entries());
    for (size_t i = 0; i < data_entry_.size(); ++i) {
      InitEmptyDLTensor(data_entry_[i]);
    }
    data_entry_is_var_.resize(idx.num_node_entries(), false);
    for (uint32_t nid : idx.input_nodes()) {
      CHECK(node_states_[nid] != nullptr);
      data_entry_[idx.entry_id(nid, 0)] = *node_states_[nid]->blob;
      data_entry_is_var_[idx.entry_id(nid, 0)] = true;
    }
  }

  // size of each storage pool entry
  std::vector<size_t> pool_entry_size;
  for (size_t i = 0; i < vshape.size(); ++i) {
    if (data_entry_is_var_[i]) continue;
    int storage_id = vstorage[i];
    size_t size = vshape[i].Size();
    DLDataType t = dtype_;
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size *= (bits / 8U);  // in bytes
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op yet";
    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_entry_size.size()) {
      pool_entry_size.resize(sid + 1, 0);
    }
    pool_entry_size[sid] = std::max(pool_entry_size[sid], size);
  }

  CleanPool();
  storage_pool_.clear();
  for (size_t i = 0; i < pool_entry_size.size(); ++i) {
    nnvm::TShape shape{static_cast<int64_t>(pool_entry_size[i] + 3) / 4};
    DLTensor* tensor;
    TVM_CCALL(
        TVMArrayAlloc(shape.data(), 1, kFloat, 32, 1, ctx_.device_type, ctx_.device_id, &tensor));
    storage_pool_.push_back(tensor);
  }

  // assign pooled data to entry
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    if (data_entry_is_var_[i]) continue;
    int storage_id = vstorage[i];
    data_entry_[i] = *storage_pool_[storage_id];
    data_entry_[i].shape = const_cast<int64_t*>(node_shape_->at(i).data());
    data_entry_[i].ndim = node_shape_->at(i).ndim();
    data_entry_[i].dtype = dtype_;  // TODO
    data_entry_[i].ctx = ctx_;
  }

  DLContext cpu_ctx;
  cpu_ctx.device_type = kCPU;
  cpu_ctx.device_id = 0;
  CleanOutput();
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    uint32_t eid = idx.entry_id(idx.outputs()[i]);
    DLTensor* tensor;
    TVM_CCALL(TVMArrayAlloc(const_cast<int64_t*>(node_shape_->at(eid).data()),
                            node_shape_->at(eid).ndim(),
                            dtype_.code,
                            dtype_.bits,
                            dtype_.lanes,
                            cpu_ctx.device_type,
                            cpu_ctx.device_id,
                            &tensor));
    outputs_.push_back(tensor);
  }
}

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
template <typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

DMLC_REGISTER_PARAMETER(TVMOpParam);

// ewise tvm op
NNVM_REGISTER_OP(tvm_op)
    .set_attr_parser(ParamParser<TVMOpParam>)
    .set_num_inputs([](const nnvm::NodeAttrs& attrs) {
      const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
      return param.num_inputs;
    })
    .set_num_outputs([](const nnvm::NodeAttrs& attrs) {
      const TVMOpParam& param = nnvm::get<TVMOpParam>(attrs.parsed);
      return param.num_outputs;
    });

}  // namespace tvmflow