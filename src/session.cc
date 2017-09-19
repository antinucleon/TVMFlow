#include <tvm/operation.h>
#include <tvmflow/base.h>
#include "executor/tvmexec.h"

namespace tvmflow {

class TVMSession : public Session {
 public:
  explicit TVMSession(const std::string& option) {
    /*
    if (option.find("metal") != std::string::npos) {
      ctx_.device_type = kMetal;
      ctx_.device_id = 0;
    } else if (option.find("opencl") != std::string::npos) {
      ctx_.device_type = kOpenCL;
      ctx_.device_id = 0;
    } else if (option.find("cpu") != std::string::npos) {
      ctx_.device_type = kCPU;
      ctx_.device_id = 0;
    } else if (option.find("cuda") != std::string::npos) {
      ctx_.device_type = kGPU;
      ctx_.device_id = 0;
    } else {
      LOG(FATAL) << "unknown device";
    }
    // dtype set
    if (option.find("float32") != std::string::npos) {
      dtype_.code = kFloat;
      dtype_.bits = 32;
      dtype_.lanes = 1;
    } else {
      LOG(FATAL) << "Other format is not supported so far";
    }
    */
    std::istringstream iss(option);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>{}};
    LOG(INFO) << "Use Device: " << tokens[0] << "[" << tokens[1] << "]";
    if (tokens[0] == "metal") {
      ctx_.device_type = kMetal;
    } else if (tokens[0] == "opencl") {
      ctx_.device_type = kOpenCL;
    } else {
      ctx_.device_type = kCPU;
    }
    ctx_.device_id = atoi(tokens[1].c_str());
    dtype_.code = kFloat;
    dtype_.bits = 32;
    dtype_.lanes = 1;
  }
  const std::vector<DLTensor*>& Run(Symbol* g,
                                    const std::unordered_map<std::string, DLTensor>& inputs);

 private:
  struct ExecEntry {
    Symbol cached_symbol;
    std::shared_ptr<TVMExecutor> exec;
    size_t use_count{0};
  };
  std::unordered_map<uint64_t, ExecEntry> cached_execs_;
  VarStateMap states_;
  DLContext ctx_;
  DLDataType dtype_;
};  // TVMSession

Session* Session::Create(const std::string& option) { return new TVMSession(option); }

const std::vector<DLTensor*>& TVMSession::Run(
    Symbol* new_sym, const std::unordered_map<std::string, DLTensor>& inputs) {
  // compute the hash value
  uint64_t hash_value = new_sym->outputs.size();
  for (nnvm::NodeEntry& e : new_sym->outputs) {
    uint64_t value = reinterpret_cast<uint64_t>(e.node.get());
    hash_value ^= value + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
  }
  if (cached_execs_.count(hash_value) != 0) {
    auto& entry = cached_execs_.at(hash_value);
    const Symbol& old_sym = entry.cached_symbol;
    bool stale_exec = (old_sym.outputs.size() != new_sym->outputs.size());
    if (!stale_exec) {
      for (size_t i = 0; i < old_sym.outputs.size(); ++i) {
        if (old_sym.outputs[i].node.get() != new_sym->outputs[i].node.get() ||
            old_sym.outputs[i].index != new_sym->outputs[i].index ||
            old_sym.outputs[i].version != new_sym->outputs[i].version) {
          stale_exec = true;
          break;
        }
      }
    }
    if (!stale_exec) {
      ++entry.use_count;
      return entry.exec->Run(inputs);
    } else {
      cached_execs_.erase(hash_value);
    }
  }
  // dump technique, remove all previous executors
  // better strategy, LRU?
  cached_execs_.clear();
  ExecEntry e;
  e.cached_symbol = *new_sym;
  e.exec = std::make_shared<TVMExecutor>();
  e.exec->Init(*new_sym, &states_, ctx_, dtype_);
  cached_execs_[hash_value] = e;
  return e.exec->Run(inputs);
}

}  // namespace tvmflow