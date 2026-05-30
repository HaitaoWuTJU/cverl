// Verify the CausalLmPolicy + Qwen3.5 adapter:
//
//   1. forward(prompt, response) matches Qwen35TextModel::forward_logits on
//      the concatenated [prompt, response] sequence (this is what makes the
//      adapter "real Qwen" rather than a re-implementation).
//   2. Backward through the loss reaches every registered parameter -- if a
//      tensor isn't actually wired up (e.g. publish_overrides forgot it),
//      its grad stays None and we surface the regression here rather than
//      hiding it inside the bigger smoke binary.
//   3. clone_as_reference produces a frozen, decoupled copy.
//
// Inputs: env CVERL_QWEN_MODEL_DIR=/path/to/Qwen3.5-0.8B  (default points at
// the model checked into the test repo). The test runs on whichever device
// LibTorch + the env's CUDA build prefer.

#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"
#include "cverl/torch/qwen3_5_causal_lm_policy.h"

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_dir;
    if (argc > 1) {
      model_dir = argv[1];
    } else if (const char* env = std::getenv("CVERL_QWEN_MODEL_DIR")) {
      model_dir = env;
    } else {
      // Skip the test cleanly when no model is wired up. CI without the
      // model directory should still pass `make test`.
      std::cout << "test_qwen3_5_causal_lm_policy skipped: set "
                   "CVERL_QWEN_MODEL_DIR or pass model_dir as argv[1]\n";
      return 0;
    }

    torch::manual_seed(7);
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA, 0);
    }

    // Use a tiny prefix of the layer stack to keep the test fast on CPU and
    // compact on GPU. The same code paths run for full layers.
    const int64_t kLayers = 2;
    const int32_t pad_id = 0;

    cverl::torch_backend::Qwen3_5CausalLmPolicy policy(model_dir, pad_id, kLayers);
    policy.to_device(device);

    // Build a tiny batch: prompt [B, P] and response [B, R]. Choose ids
    // unlikely to be padding (avoid 0).
    auto prompt = torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}},
                                torch::TensorOptions().dtype(torch::kLong));
    auto response = torch::tensor({{10, 11, 12}, {13, 14, 15}},
                                  torch::TensorOptions().dtype(torch::kLong));

    // 1) Adapter forward must equal a hand-rolled forward_logits over the
    // concatenated sequence, sliced to the response window.
    auto adapter_logits = policy.forward(prompt, response);
    require(adapter_logits.dim() == 3, "adapter logits must be [B,R,V]");
    require(adapter_logits.size(0) == prompt.size(0), "batch dim");
    require(adapter_logits.size(1) == response.size(1), "response dim");
    require(adapter_logits.size(2) == policy.vocab_size(), "vocab dim");

    {
      // Reference: build a fresh Qwen35TextModel from the same dir, fp32, on
      // the same device, run forward_logits on concat, and slice. The two
      // models share the same config + weights but the reference does NOT go
      // through nn::Module overrides, which is the whole point of the cross-
      // check.
      torch::NoGradGuard no_grad;
      cverl::HfModelLoader ref_loader(model_dir);
      cverl::Qwen35TextModel reference(std::move(ref_loader));
      reference.to(device);
      auto p = prompt.to(device).to(torch::kLong);
      auto r = response.to(device).to(torch::kLong);
      auto cat = torch::cat({p, r}, 1);
      auto ref_logits = reference.forward_logits(cat, kLayers)
                            .slice(1, p.size(1) - 1, p.size(1) + r.size(1) - 1);
      auto diff = (adapter_logits.detach() - ref_logits).abs();
      double max_abs = diff.max().item<double>();
      double mean_abs = diff.mean().item<double>();
      std::cout << "adapter_vs_dense max_abs=" << max_abs
                << " mean_abs=" << mean_abs << "\n";
      require(max_abs < 1.0e-3, "adapter forward diverges from dense forward_logits");
    }

    // 2) Backward must populate grads on every registered parameter.
    auto target = response.to(device).clamp(0, policy.vocab_size() - 1);
    auto log_probs = torch::log_softmax(adapter_logits, -1)
                         .gather(-1, target.unsqueeze(-1))
                         .squeeze(-1);
    auto loss = -log_probs.mean();
    require(std::isfinite(loss.item<double>()), "loss must be finite");
    loss.backward();

    int64_t with_grad = 0;
    int64_t without_grad = 0;
    int64_t with_nonzero_grad = 0;
    for (const auto& kv : policy.named_parameters(false)) {
      if (kv.value().grad().defined()) {
        ++with_grad;
        if (kv.value().grad().abs().sum().item<double>() > 0.0) {
          ++with_nonzero_grad;
        }
      } else {
        ++without_grad;
      }
    }
    std::cout << "params=" << (with_grad + without_grad)
              << " with_grad=" << with_grad
              << " with_nonzero_grad=" << with_nonzero_grad << "\n";
    require(without_grad == 0, "every adapter parameter should receive a grad");
    require(with_nonzero_grad > 0, "at least one parameter should have a non-zero grad");

    // 3) clone_as_reference: frozen + decoupled.
    auto ref_policy_base = policy.clone_as_reference();
    auto* ref_policy = dynamic_cast<cverl::torch_backend::Qwen3_5CausalLmPolicy*>(ref_policy_base.get());
    require(ref_policy != nullptr, "clone returned wrong subtype");
    ref_policy->to_device(device);
    {
      torch::NoGradGuard no_grad;
      auto orig = policy.forward(prompt, response).detach();
      auto refl = ref_policy->forward(prompt, response).detach();
      auto diff = (orig - refl).abs().max().item<double>();
      require(diff < 1.0e-6, "reference policy initial logits must match");
    }
    for (const auto& kv : ref_policy->named_parameters(false)) {
      require(!kv.value().requires_grad(), "reference parameter must be frozen");
    }

    std::cout << "test_qwen3_5_causal_lm_policy passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_qwen3_5_causal_lm_policy failed: " << e.what() << "\n";
    return 1;
  }
}
