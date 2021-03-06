/*!
 *  Copyright (c) 2017 by Contributors
 * \file Additional optimization pass of NNVM.
 */
#include <dmlc/json.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/tuple.h>
#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include "../ops/op_attr_types.h"

namespace tvmflow {

using nnvm::IndexedGraph;
using nnvm::any;
using namespace tvm;

// The single fuse rule.
enum class FuseRule { kUknown, kFuseToMaster, kRealize };

DLDataType GetDLType(int type_flag) {
  if (type_flag == 0) return Type2TVMType(Float(32));
  LOG(FATAL) << "unknown type_flag=" << type_flag;
  return Type2TVMType(Float(32));
}

// Partition the graph into segments
// Each segment will be compiled into one operator.
// Need also mark the property of the segment.
nnvm::Graph GraphPartition(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  // Transform to dltype
  // In future, directly fo type inference in dltype.
  DLTypeVector dltype_vec = DLTypeVector(dtype_vec.size());
  for (size_t i = 0; i < dtype_vec.size(); ++i) {
    dltype_vec[i] = GetDLType(dtype_vec[i]);
  }

  // Reference counter of each op node
  // For now, always store result when an op is referred more than once.
  std::vector<uint32_t> ref_count(idx.num_nodes(), 0);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      ++ref_count[e.node_id];
    }
  }
  for (const auto& e : idx.outputs()) {
    // this line will realize all the outputs
    ref_count[e.node_id] += 2;
  }
  // Pattern fo the subgraph
  std::vector<TOpPattern> pattern_vec(idx.num_nodes(), kExtern);
  // Whether node can be fused to parent.
  std::vector<FuseRule> fuse_vec(idx.num_nodes(), FuseRule::kUknown);
  // Master node id of fusion segment.
  std::vector<int> master_vec(idx.num_nodes(), -1);
  // Operator pattern
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      fuse_vec[nid] = FuseRule::kRealize;
      continue;
    }

    TOpPattern pt = op_pattern.get(inode.source->op(), kExtern);
    if (pt <= kBroadcast) {
      int chosen_master = -1;
      bool ewise = inode.source->num_outputs() == 1;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          TOpPattern ipt = pattern_vec[e.node_id];
          if (ipt != kElemWise) ewise = false;
          if (ipt <= kBroadcast) {
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else if (ipt == kComplex && chosen_master == -1 &&
                     shape_vec[idx.entry_id(nid, 0)] == shape_vec[idx.entry_id(e)]) {
            chosen_master = master_vec[e.node_id];
            fuse_vec[e.node_id] = FuseRule::kFuseToMaster;
          } else {
            fuse_vec[e.node_id] = FuseRule::kRealize;
          }
        }
        if (ewise) {
          if (shape_vec[idx.entry_id(nid, 0)] != shape_vec[idx.entry_id(e)]) {
            ewise = false;
          }
        }
      }
      master_vec[nid] = chosen_master;
      if (chosen_master != -1) {
        pt = kComplex;
      } else {
        pt = ewise ? kElemWise : kBroadcast;
      }
    } else {
      master_vec[nid] = nid;
      for (const auto& e : inode.inputs) {
        if (fuse_vec[e.node_id] == FuseRule::kUknown) {
          fuse_vec[e.node_id] = FuseRule::kRealize;
          if (master_vec[e.node_id] == -1) {
            master_vec[e.node_id] = e.node_id;
          }
        }
      }
    }

    pattern_vec[nid] = pt;
    if (ref_count[nid] > 1) {
      fuse_vec[nid] = FuseRule::kRealize;
      if (master_vec[nid] == -1) {
        master_vec[nid] = nid;
      }
    }
  }

  // point to the group root id of each node
  std::vector<int> group_vec(idx.num_nodes(), -1);
  for (uint32_t i = idx.num_nodes(); i != 0; --i) {
    uint32_t nid = i - 1;
    const auto& inode = idx[nid];
    if (group_vec[nid] == -1) {
      group_vec[nid] = nid;
    }
    // propagate the group id.
    for (const auto& e : inode.inputs) {
      if (fuse_vec[e.node_id] == FuseRule::kFuseToMaster) {
        CHECK(group_vec[e.node_id] == -1 || group_vec[e.node_id] == group_vec[nid]);
        group_vec[e.node_id] = group_vec[nid];
      }
    }
  }
  g.attrs["group_root"] = std::make_shared<any>(std::move(group_vec));
  g.attrs["group_master"] = std::make_shared<any>(std::move(master_vec));
  g.attrs["pattern"] = std::make_shared<any>(std::move(pattern_vec));
  g.attrs["dltype"] = std::make_shared<any>(std::move(dltype_vec));
  return g;
}

NNVM_REGISTER_PASS(GraphPartition)
    .set_body(GraphPartition)
    .depend_graph_attr("shape")
    .depend_graph_attr("dtype")
    .provide_graph_attr("dltype");

struct NodeEntryHash {
  size_t operator()(const IndexedGraph::NodeEntry& e) const { return e.node_id; }
};

struct NodeEntryEqual {
  size_t operator()(const IndexedGraph::NodeEntry& a, const IndexedGraph::NodeEntry& b) const {
    return a.node_id == b.node_id && a.index == b.index;
  }
};

// Auxiliary data structure for representing fused op.
struct FuseEntry {
  // The inputs
  std::vector<IndexedGraph::NodeEntry> inputs;
  // The input map
  std::unordered_map<IndexedGraph::NodeEntry, Tensor, NodeEntryHash, NodeEntryEqual> imap;
  // Output tensors
  Array<Tensor> outputs;
  // Placeholder for inputs
  Array<Tensor> placeholder;
  // Computing schedule
  Schedule schedule;
  // Function name
  std::string func_name;
};

// Fuse the partitioned graph into segments.
// Create a new graph with fused noded.
// Also inheritate attribute shape, dltype from previous graph.
nnvm::Graph GraphFuse(nnvm::Graph g) {
  // setup ref counter
  const IndexedGraph& idx = g.indexed_graph();
  // Get attributes from the graph
  const ShapeVector& shape_vec = g.GetAttr<ShapeVector>("shape");
  const DLTypeVector& dltype_vec = g.GetAttr<DLTypeVector>("dltype");
  const DTypeVector& dtype_vec = g.GetAttr<DTypeVector>("dtype");
  const std::vector<int>& group_vec = g.GetAttr<std::vector<int> >("group_root");
  const std::vector<int>& master_vec = g.GetAttr<std::vector<int> >("group_master");
  const std::vector<TOpPattern>& pattern_vec = g.GetAttr<std::vector<TOpPattern> >("pattern");
  std::string target = g.GetAttr<std::string>("target");
  std::vector<FuseEntry> fuse_vec(idx.num_nodes());
  // setup inputs and placeholder.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (inode.source->op()->name == "placeholder") continue;
    CHECK_GE(group_vec[nid], 0);
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    TOpPattern pt = pattern_vec[root_id];
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id && fe.imap.count(e) == 0) {
        Array<Expr> shape;
        if (pt == kElemWise) {
          // elementwise support flatten
          int64_t prod = 1;
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            prod *= x;
          }
          CHECK_LE(prod, static_cast<int64_t>(std::numeric_limits<int>::max()));
          shape.push_back(make_const(Int(32), prod));
        } else {
          for (int64_t x : shape_vec[idx.entry_id(e)]) {
            CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
            shape.push_back(make_const(Int(32), x));
          }
        }
        std::ostringstream os_name;
        os_name << "input" << fe.inputs.size();
        Tensor data = placeholder(shape, TVMType2Type(dltype_vec[idx.entry_id(e)]), os_name.str());
        fe.imap[e] = data;
        fe.inputs.push_back(e);
        fe.placeholder.push_back(data);
      }
    }
  }
  // Setup the Tensor
  std::vector<Tensor> tensor_vec(idx.num_node_entries());
  static auto& fcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");
  static auto& fschedule = nnvm::Op::GetAttr<FTVMSchedule>("FTVMSchedule");
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (inode.source->op()->name == "placeholder") continue;
    int root_id = group_vec[nid];
    FuseEntry& fe = fuse_vec[root_id];
    Array<Tensor> inputs;
    // input loading
    for (const auto& e : inode.inputs) {
      if (group_vec[e.node_id] != root_id) {
        auto it = fe.imap.find(e);
        CHECK(it != fe.imap.end());
        inputs.push_back(it->second);
      } else {
        Tensor t = tensor_vec[idx.entry_id(e)];
        CHECK(t.defined());
        inputs.push_back(t);
      }
    }

    Array<Tensor> out = fcompute[inode.source->op()](inode.source->attrs, inputs);
    CHECK_EQ(out.size(), inode.source->num_outputs());

    // schedule on root node, and use master's schedule
    if (nid != root_id) {
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        uint32_t eid = idx.entry_id(nid, index);
        tensor_vec[eid] = out[index];
      }
    } else {
      fe.outputs = out;
      int master = master_vec[root_id];
      CHECK_GE(master, 0);
      fe.schedule =
          fschedule[idx[master].source->op()](idx[master].source->attrs, fe.outputs, target);

      std::ostringstream os;
      os << idx[master].source->attrs.name + "_id" << nid;
      fe.func_name = os.str();
    }
  }
  static const PackedFunc& flower = GetPackedFunc("tvm_graph.lower");
  static const PackedFunc& fbuild = GetPackedFunc("tvm_graph.build_target");
  Array<tvm::LoweredFunc> funcs;
  for (const FuseEntry& fe : fuse_vec) {
    if (fe.schedule.defined()) {
      Array<tvm::Tensor> args = fe.placeholder;
      for (tvm::Tensor x : fe.outputs) {
        args.push_back(x);
      }
      Array<tvm::LoweredFunc> ret = flower(fe.schedule, args, fe.func_name);
      for (LoweredFunc x : ret) {
        funcs.push_back(x);
      }
    }
  }
  // only variable & placeholder
  tvm::runtime::Module module;
  if (funcs.size() > 0) {
    module = fbuild(funcs, target);
  } else {
  }
  // Final step: Remap the node, with given attribute
  const nnvm::Op* tvm_op = nnvm::Op::Get("tvm_op");

  std::unordered_map<uint32_t, nnvm::NodePtr> old_new;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable() || inode.source->op()->name == "placeholder") {
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs = inode.source->attrs;
      old_new[nid] = np;
    } else {
      int root_id = group_vec[nid];
      if (nid != root_id) continue;
      FuseEntry& fe = fuse_vec[root_id];
      nnvm::NodePtr np = nnvm::Node::Create();
      np->attrs.op = tvm_op;
      np->attrs.name = inode.source->attrs.name;
      np->attrs.dict["num_inputs"] = std::to_string(fe.inputs.size());
      np->attrs.dict["num_outputs"] = std::to_string(fe.outputs.size());
      np->attrs.dict["func_name"] = fuse_vec[nid].func_name;
      np->attrs.dict["flatten_data"] = std::to_string(pattern_vec[nid] == kElemWise);
      np->op()->attr_parser(&(np->attrs));
      for (const auto& e : fe.inputs) {
        auto it = old_new.find(e.node_id);
        CHECK(it != old_new.end()) << "cannot find node_id=" << e.node_id;
        np->inputs.emplace_back(nnvm::NodeEntry{it->second, e.index, e.version});
      }
      for (const uint32_t node_id : inode.control_deps) {
        auto it = old_new.find(node_id);
        CHECK(it != old_new.end());
        np->control_deps.emplace_back(it->second);
      }
      old_new[nid] = np;
    }
  }

  nnvm::Graph ret;
  for (const auto& e : idx.outputs()) {
    auto it = old_new.find(group_vec[e.node_id]);
    CHECK(it != old_new.end()) << "cannot find node_id=" << e.node_id;
    ret.outputs.emplace_back(nnvm::NodeEntry{it->second, e.index, e.version});
  }
  const IndexedGraph& new_idx = ret.indexed_graph();
  ShapeVector new_shape_vec = ShapeVector(new_idx.num_node_entries(), TShape());
  DTypeVector new_dtype_vec = DTypeVector(new_idx.num_node_entries());
  DLTypeVector new_dltype_vec = DLTypeVector(new_idx.num_node_entries());
  for (const auto& kv : old_new) {
    uint32_t nid = kv.first;
    const auto& inode = idx[nid];
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      uint32_t new_eid = new_idx.entry_id(new_idx.node_id(kv.second.get()), i);
      uint32_t old_eid = idx.entry_id(nid, i);
      new_shape_vec[new_eid] = shape_vec[old_eid];
      new_dtype_vec[new_eid] = dtype_vec[old_eid];
      new_dltype_vec[new_eid] = dltype_vec[old_eid];
    }
  }
  ret.attrs["shape"] = std::make_shared<any>(std::move(new_shape_vec));
  ret.attrs["dtype"] = std::make_shared<any>(std::move(new_dtype_vec));
  ret.attrs["dltype"] = std::make_shared<any>(std::move(new_dltype_vec));
  ret.attrs["module"] = std::make_shared<any>(std::move(module));
  ret = nnvm::ApplyPass(ret, "PlanMemory");
  return ret;
}

NNVM_REGISTER_PASS(GraphFuse).set_body(GraphFuse);

const TLayoutInfo& GetDefaultLayout() {
  static TLayoutInfo default_layout = "default";
  return default_layout;
}

nnvm::NodePtr CreateLayoutTransformNode(const std::string& src, const std::string& dst) {
  static const nnvm::Op* trans_op = nnvm::Op::Get("layout_transform");
  static int count = 0;
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = trans_op;
  n->attrs.name = src + "_to_" + dst + std::to_string(count++);
  n->attrs.dict["src_layout"] = src;
  n->attrs.dict["dst_layout"] = dst;
  n->op()->attr_parser(&(n->attrs));
  return n;
}

/*!
 * \brief A simple layout transform pass that will
 *  insert layout transform nodes automatically.
 */
nnvm::Graph LayoutTransform(nnvm::Graph src) {
  static auto& op_layout_request = nnvm::Op::GetAttr<FTVMLayoutRequest>("FTVMLayoutRequest");
  static auto& op_vecop = nnvm::Op::GetAttr<FTVMVectorizedOp>("FTVMVectorizedOp");
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  const std::vector<TLayoutInfo>& input_layouts = src.GetAttr<std::vector<TLayoutInfo> >("layout");

  const IndexedGraph& idx = src.indexed_graph();
  std::vector<TLayoutInfo> produce_vec(idx.num_node_entries(), GetDefaultLayout());
  std::vector<nnvm::NodePtr> mirror_vec(idx.num_nodes(), nullptr);

  // use op pattern to decide whether an op is map
  auto is_map_op = [&](size_t nid) {
    TOpPattern pt = op_pattern.get(idx[nid].source->op(), kExtern);
    bool is_map = (pt <= kBroadcast);
    if (pt == kBroadcast) {
      for (const auto& e : idx[nid].inputs) {
        if (shape_vec[idx.entry_id(nid, 0)] != shape_vec[idx.entry_id(e)]) {
          is_map = false;
          break;
        }
      }
    }
    return is_map;
  };

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    nnvm::NodePtr new_node = nnvm::Node::Create();
    *new_node = *(inode.source);
    if (new_node->is_variable()) {
      auto input_iter = std::find(idx.input_nodes().cbegin(), idx.input_nodes().cend(), nid);
      CHECK(input_iter != idx.input_nodes().cend());
      size_t input_id = std::distance(idx.input_nodes().cbegin(), input_iter);
      produce_vec[idx.entry_id(nid, 0)] = input_layouts[input_id];
      mirror_vec[nid] = new_node;
      continue;
    }

    if (op_vecop.count(inode.source->op())) {
      new_node = op_vecop[inode.source->op()](inode.source);
      new_node->inputs.resize(new_node->num_inputs());
    }

    // set up output and input layouts
    std::vector<TLayoutInfo> request_ilayouts(new_node->num_inputs(), GetDefaultLayout());
    if (op_layout_request.count(new_node->op())) {
      std::vector<TLayoutInfo> produce_olayouts(new_node->num_outputs(), GetDefaultLayout());
      CHECK(
          op_layout_request[new_node->op()](new_node->attrs, &request_ilayouts, &produce_olayouts))
          << "Layout request fail";

      CHECK_EQ(request_ilayouts.size(), new_node->num_inputs());
      CHECK_EQ(produce_olayouts.size(), new_node->num_outputs());
      for (size_t i = 0; i < new_node->num_outputs(); ++i) {
        produce_vec[idx.entry_id(nid, i)] = produce_olayouts[i];
      }
    }

    bool map_layout = is_map_op(nid);
    if (map_layout) {
      const TLayoutInfo& layout = produce_vec[idx.entry_id(inode.inputs[0])];
      for (const auto& e : inode.inputs) {
        if (produce_vec[idx.entry_id(e)] != layout) {
          map_layout = false;
          break;
        }
      }
      if (map_layout) {
        for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
          produce_vec[idx.entry_id(nid, i)] = layout;
        }
      }
    }

    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      const auto& e = inode.inputs[i];
      const nnvm::NodePtr& in = mirror_vec[e.node_id];
      new_node->inputs[i] = nnvm::NodeEntry{in, e.index, e.version};

      TLayoutInfo produce = produce_vec[idx.entry_id(e)];
      TLayoutInfo request = request_ilayouts[i];
      if (!map_layout && (produce != request)) {
        nnvm::NodePtr tnode = CreateLayoutTransformNode(produce, request);
        tnode->attrs.name = idx[e.node_id].source->attrs.name + "_" + request;
        tnode->inputs.emplace_back(new_node->inputs[i]);
        new_node->inputs[i] = nnvm::NodeEntry{tnode, 0, 0};
      }
    }
    mirror_vec[nid] = new_node;
  }

  std::vector<nnvm::NodeEntry> outputs;
  for (const auto& e : idx.outputs()) {
    TLayoutInfo produce = produce_vec[idx.entry_id(e)];
    if (produce != GetDefaultLayout()) {
      nnvm::NodePtr tnode = CreateLayoutTransformNode(produce, GetDefaultLayout());
      tnode->attrs.name = idx[e.node_id].source->attrs.name + "_default";
      tnode->inputs.emplace_back(nnvm::NodeEntry{mirror_vec[e.node_id], e.index, e.version});
      outputs.emplace_back(nnvm::NodeEntry{tnode, 0, 0});
    } else {
      outputs.emplace_back(nnvm::NodeEntry{mirror_vec[e.node_id], e.index, e.version});
    }
  }

  nnvm::Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(LayoutTransform).set_body(LayoutTransform);

DMLC_REGISTER_PARAMETER(LayoutTransformParam);

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

NNVM_REGISTER_OP(layout_transform)
    .set_attr_parser(ParamParser<LayoutTransformParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .add_argument("data", "NDArray-or-Symbol", "Input data")
    .add_arguments(LayoutTransformParam::__FIELDS__());

nnvm::Graph PruneGraph(nnvm::Graph src) {
  const auto& params = src.GetAttr<std::unordered_set<std::string> >("params");

  std::unordered_set<nnvm::Node*> pruned;
  nnvm::NodeEntryMap<nnvm::NodePtr> entry_var;
  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    bool can_be_pruned = true;
    if (n->is_variable()) {
      if (params.count(n->attrs.name)) {
        pruned.emplace(n.get());
      }
      can_be_pruned = false;
    }

    for (const auto& e : n->inputs) {
      if (!pruned.count(e.node.get())) {
        can_be_pruned = false;
      }
    }
    if (can_be_pruned) {
      pruned.emplace(n.get());
    } else {
      // scan again to find edge nodes, skip variables
      for (auto& e : n->inputs) {
        if (!e.node->is_variable() && pruned.count(e.node.get())) {
          if (!entry_var.count(e)) {
            nnvm::NodePtr var = nnvm::Node::Create();
            var->attrs.name = e.node->attrs.name + "_output" + std::to_string(e.index);
            entry_var.emplace(e, var);
          }
          e = nnvm::NodeEntry{entry_var.at(e), 0, 0};
        }
      }
    }
  });

  nnvm::Graph pre_graph;
  pre_graph.outputs.reserve(entry_var.size());
  std::vector<std::string> output_names;
  output_names.reserve(entry_var.size());
  for (auto kv : entry_var) {
    if (kv.first.node->is_variable()) continue;
    pre_graph.outputs.emplace_back(kv.first);
    output_names.emplace_back(kv.second->attrs.name);
  }

  pre_graph.attrs["pruned_params"] = std::make_shared<dmlc::any>(std::move(output_names));
  src.attrs["pre_graph"] = std::make_shared<dmlc::any>(std::move(pre_graph));
  return src;
}

NNVM_REGISTER_PASS(PruneGraph).set_body(PruneGraph);

}  // namespace tvmflow
