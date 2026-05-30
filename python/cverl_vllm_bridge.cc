#include <torch/extension.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cverl/text/hf_bpe_tokenizer.h"
#include "cverl/torch/causal_lm_policy.h"

namespace py = pybind11;

namespace {

std::string restore_hf_name(std::string registered_name) {
  std::replace(registered_name.begin(), registered_name.end(), '/', '.');
  return registered_name;
}

torch::Device parse_device(const std::string& device) {
  if (device == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (device == "cuda") {
    return torch::Device(torch::kCUDA, 0);
  }
  if (device.rfind("cuda:", 0) == 0) {
    return torch::Device(device);
  }
  throw std::invalid_argument("unsupported device: " + device);
}

std::string dtype_name(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return "float32";
    case torch::kFloat16:
      return "float16";
    case torch::kBFloat16:
      return "bfloat16";
    default:
      throw std::invalid_argument("unsupported dtype for vLLM weight sync");
  }
}

class QwenPolicyHandle {
 public:
  QwenPolicyHandle(const std::string& model_dir,
                   const std::string& tokenizer_path,
                   const std::string& device) {
    cverl::text::HfBpeTokenizerOptions tok_opts;
    tok_opts.tokenizer_json_path = tokenizer_path.empty()
        ? (std::filesystem::path(model_dir) / "tokenizer.json").string()
        : tokenizer_path;
    tokenizer_ = std::make_unique<cverl::text::HfBpeTokenizer>(tok_opts);

    cverl::torch_backend::CausalLmPolicyOptions opts;
    opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
    opts.qwen_model_dir = model_dir;
    opts.qwen_max_layers = -1;
    opts.pad_id = tokenizer_->pad_id() >= 0 ? tokenizer_->pad_id() : 0;
    policy_ = cverl::torch_backend::make_causal_lm_policy(opts);
    to_device(device);
  }

  void to_device(const std::string& device) {
    policy_->to_device(parse_device(device));
  }

  std::vector<std::pair<std::string, torch::Tensor>> named_parameters() {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    auto named = policy_->named_parameters(/*recurse=*/true);
    out.reserve(named.size());
    for (const auto& kv : named) {
      out.emplace_back(restore_hf_name(kv.key()), kv.value());
    }
    return out;
  }

  py::dict update_info(bool packed,
                       int64_t packed_buffer_size_bytes,
                       int64_t packed_num_buffers) {
    py::list names;
    py::list dtype_names;
    py::list shapes;
    for (const auto& item : named_parameters()) {
      names.append(item.first);
      dtype_names.append(dtype_name(item.second.scalar_type()));
      py::list shape;
      for (int64_t dim : item.second.sizes()) {
        shape.append(dim);
      }
      shapes.append(shape);
    }
    py::dict out;
    out["names"] = names;
    out["dtype_names"] = dtype_names;
    out["shapes"] = shapes;
    out["packed"] = packed;
    out["packed_buffer_size_bytes"] = packed_buffer_size_bytes;
    out["packed_num_buffers"] = packed_num_buffers;
    out["is_checkpoint_format"] = false;
    return out;
  }

 private:
  std::unique_ptr<cverl::text::HfBpeTokenizer> tokenizer_;
  std::shared_ptr<cverl::torch_backend::CausalLmPolicy> policy_;
};

}  // namespace

PYBIND11_MODULE(_cverl_vllm_bridge, m) {
  py::class_<QwenPolicyHandle>(m, "QwenPolicy")
      .def(py::init<const std::string&, const std::string&, const std::string&>(),
           py::arg("model_dir"),
           py::arg("tokenizer_path") = "",
           py::arg("device") = "cuda:0")
      .def("to_device", &QwenPolicyHandle::to_device)
      .def("named_parameters", &QwenPolicyHandle::named_parameters)
      .def("update_info", &QwenPolicyHandle::update_info,
           py::arg("packed") = true,
           py::arg("packed_buffer_size_bytes") = 1024LL * 1024LL * 1024LL,
           py::arg("packed_num_buffers") = 2);
}
