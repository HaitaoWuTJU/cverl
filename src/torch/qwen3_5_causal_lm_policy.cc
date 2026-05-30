#include "cverl/torch/qwen3_5_causal_lm_policy.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "cverl/model/hf_model_loader.h"
#include "cverl/model/safetensors_writer.h"

namespace cverl::torch_backend {

std::string Qwen3_5CausalLmPolicy::sanitize_name(const std::string& weight_name) {
  // nn::Module rejects parameter names containing '.' because they conflict
  // with the dotted state_dict path syntax. Substitute '/' so we still get a
  // unique, deterministic identifier we can map back to the underlying
  // Qwen35TextModel weight name.
  std::string out = weight_name;
  std::replace(out.begin(), out.end(), '.', '/');
  return out;
}

void Qwen3_5CausalLmPolicy::publish_overrides() {
  // Pull every parameter from this nn::Module by its registered name and
  // forward the (still-live) tensor handle into model_'s weight cache. This
  // is what makes `Qwen35TextModel::forward_logits` traverse parameters that
  // autograd will route gradients to. Re-fetching from named_parameters()
  // (rather than caching ourselves) keeps us correct after Module::to(...).
  auto named = named_parameters(/*recurse=*/false);
  for (size_t i = 0; i < weight_names_.size(); ++i) {
    const auto& reg = registered_names_[i];
    auto& tensor = named[reg];
    if (!tensor.defined()) {
      throw std::runtime_error("Qwen3_5CausalLmPolicy missing parameter: " + reg);
    }
    model_->set_weight_override(weight_names_[i], tensor);
  }
}

Qwen3_5CausalLmPolicy::Qwen3_5CausalLmPolicy(const std::string& model_dir,
                                             int32_t pad_id,
                                             int64_t max_layers)
    : model_dir_(model_dir),
      pad_id_(pad_id),
      max_layers_(max_layers) {
  HfModelLoader loader(model_dir_);
  model_ = std::make_unique<Qwen35TextModel>(std::move(loader));
  config_ = model_->config();

  // Materialize every weight the forward path will hit, cast to fp32, and
  // register as a trainable parameter on this nn::Module. We must NOT call
  // model_->to(...) afterwards — it would reallocate every entry in its
  // weights_ map and break the override link to our registered parameters.
  // Device migration goes through this->to_device(...).
  weight_names_ = model_->required_weight_names(max_layers_);
  registered_names_.reserve(weight_names_.size());
  for (const auto& name : weight_names_) {
    auto tensor = model_->loader().load_tensor(name).to(torch::kFloat32).contiguous();
    tensor.set_requires_grad(true);
    auto registered_name = sanitize_name(name);
    register_parameter(registered_name, tensor, /*requires_grad=*/true);
    registered_names_.push_back(std::move(registered_name));
  }
  publish_overrides();
}

void Qwen3_5CausalLmPolicy::to_device(torch::Device device) {
  // Move every registered parameter to `device`. nn::Module::to(device)
  // replaces parameter tensors in place, but our model_->weights_ map is
  // still pointing at the old storage. Re-publish so model_'s weight()
  // accessor sees the device-migrated parameters.
  this->to(device);
  publish_overrides();
}

torch::Tensor Qwen3_5CausalLmPolicy::forward(const torch::Tensor& prompt_ids,
                                             const torch::Tensor& response_ids) {
  TORCH_CHECK(prompt_ids.dim() == 2, "prompt_ids must be [B, P]");
  TORCH_CHECK(response_ids.dim() == 2, "response_ids must be [B, R]");
  TORCH_CHECK(prompt_ids.size(0) == response_ids.size(0), "batch mismatch");

  // Inputs may live on CPU; move them to the device of any registered
  // parameter (parameters share a single device after Module::to).
  auto named = named_parameters(/*recurse=*/false);
  TORCH_CHECK(!named.is_empty(), "Qwen3_5CausalLmPolicy has no registered parameters");
  torch::Device device = named.front().value().device();
  auto long_options = torch::TensorOptions().dtype(torch::kLong).device(device);

  auto p = prompt_ids.to(long_options);
  auto r = response_ids.to(long_options);
  const int64_t P = p.size(1);
  const int64_t R = r.size(1);
  TORCH_CHECK(P >= 1, "prompt length must be >= 1 to score response token 0");
  auto input_ids = torch::cat({p, r}, /*dim=*/1);

  auto logits = model_->forward_logits(input_ids, max_layers_);

  // logits[:, t, :] are the next-token predictions conditioned on
  // input_ids[:, :t+1]. To score response_ids[:, k] we need the logits at
  // position P + k - 1, i.e. slice [P - 1, P + R - 1).
  return logits.slice(/*dim=*/1, /*start=*/P - 1, /*end=*/P + R - 1);
}

std::shared_ptr<CausalLmPolicy> Qwen3_5CausalLmPolicy::clone_as_reference() const {
  auto ref = std::make_shared<Qwen3_5CausalLmPolicy>(model_dir_, pad_id_, max_layers_);
  // Copy parameters from `this` into ref, in registration order, then freeze.
  torch::NoGradGuard no_grad;
  auto src_named = const_cast<Qwen3_5CausalLmPolicy*>(this)->named_parameters(false);
  auto dst_named = ref->named_parameters(false);
  for (const auto& reg : registered_names_) {
    auto& src = src_named[reg];
    auto& dst = dst_named[reg];
    if (!src.defined() || !dst.defined()) {
      throw std::runtime_error("clone_as_reference: parameter map mismatch");
    }
    dst.copy_(src);
  }
  for (auto& kv : ref->named_parameters(false)) {
    kv.value().set_requires_grad(false);
  }
  ref->eval();
  ref->publish_overrides();
  return ref;
}

void Qwen3_5CausalLmPolicy::save_hf_checkpoint(const std::string& output_dir,
                                               const std::string& dtype) const {
  namespace fs = std::filesystem;
  if (max_layers_ >= 0 && max_layers_ < config_.num_hidden_layers) {
    throw std::runtime_error(
        "Qwen3_5CausalLmPolicy HF export requires all layers; rerun with --qwen-max-layers -1");
  }

  fs::path out_root(output_dir);
  fs::create_directories(out_root);
  for (const auto& entry : fs::directory_iterator(out_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path stale = entry.path();
    const std::string filename = stale.filename().string();
    if (stale.extension() == ".safetensors" || filename == "model.safetensors.index.json") {
      fs::remove(stale);
    }
  }
  for (const auto& entry : fs::directory_iterator(model_dir_)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path src = entry.path();
    const std::string filename = src.filename().string();
    if (src.extension() == ".safetensors" || filename == "model.safetensors.index.json") {
      continue;
    }
    fs::copy_file(src, out_root / filename, fs::copy_options::overwrite_existing);
  }

  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.reserve(weight_names_.size());
  auto named = const_cast<Qwen3_5CausalLmPolicy*>(this)->named_parameters(false);
  for (size_t i = 0; i < weight_names_.size(); ++i) {
    const auto& reg = registered_names_[i];
    auto& tensor = named[reg];
    if (!tensor.defined()) {
      throw std::runtime_error("save_hf_checkpoint missing parameter: " + reg);
    }
    tensors.emplace(weight_names_[i], tensor);
  }
  cverl::write_safetensors((out_root / "model.safetensors").string(), tensors, dtype);
}

}  // namespace cverl::torch_backend
