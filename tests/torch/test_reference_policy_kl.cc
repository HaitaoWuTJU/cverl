// Tests reference policy clone + KL penalty plumbing.
#include "cverl/text/byte_tokenizer.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"

#include <torch/torch.h>

#include <cmath>
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

int main() {
  try {
    torch::manual_seed(7);
    const int64_t hidden = 8;
    const int32_t pad_id = cverl::text::ByteTokenizer::kPadId;
    cverl::torch_backend::TinyCausalPolicy policy(
        cverl::text::ByteTokenizer::kVocabSize, hidden, pad_id);

    // Build a tiny dummy batch.
    torch::Tensor prompt_ids = torch::tensor({{4, 5, 6, 7, pad_id, pad_id},
                                              {pad_id, pad_id, 8, 9, 10, 11}},
                                             torch::kInt64);
    torch::Tensor response_ids = torch::tensor({{12, 13, 14, pad_id, pad_id},
                                                {15, 16, 17, 18, pad_id}},
                                               torch::kInt64);

    auto ref = cverl::torch_backend::clone_as_reference(policy);

    // Reference should have requires_grad=false on every parameter.
    for (const auto& p : ref->parameters()) {
      require(!p.requires_grad(), "ref param requires_grad=false");
    }

    // Reference logits == policy logits at clone time.
    auto policy_logits = policy->forward(prompt_ids, response_ids);
    auto ref_logits = ref->forward(prompt_ids, response_ids);
    auto diff = (policy_logits - ref_logits).abs().max().item<double>();
    require(diff < 1e-6, "ref logits match source after clone");

    // KL penalty between identical distributions is ~0 for K1/abs/K2/K3.
    auto policy_logp = cverl::torch_backend::response_log_probs(policy_logits, response_ids);
    auto ref_logp = cverl::torch_backend::response_log_probs(ref_logits, response_ids);
    for (auto mode : {CVERL_KL_K1, CVERL_KL_ABS, CVERL_KL_K2, CVERL_KL_K3}) {
      auto kl = cverl::torch_backend::kl_penalty(policy_logp, ref_logp, mode);
      require(kl.abs().max().item<double>() < 1e-5, "kl ~ 0 when policy==ref");
    }

    // After updating policy parameters, ref logits stay put.
    {
      torch::NoGradGuard ng;
      for (auto& p : policy->parameters()) {
        p.add_(torch::ones_like(p) * 0.1f);
      }
    }
    auto policy_logits_after = policy->forward(prompt_ids, response_ids);
    auto ref_logits_after = ref->forward(prompt_ids, response_ids);
    auto policy_diff = (policy_logits_after - policy_logits).abs().max().item<double>();
    auto ref_diff = (ref_logits_after - ref_logits).abs().max().item<double>();
    require(policy_diff > 1e-3, "policy moved");
    require(ref_diff < 1e-6, "ref unchanged after policy update");

    // KL gradient flows through `log_probs` only (ref is detached).
    {
      torch::Tensor logits = policy->forward(prompt_ids, response_ids);
      torch::Tensor log_probs = cverl::torch_backend::response_log_probs(logits, response_ids);
      torch::Tensor ref_logits_now = ref->forward(prompt_ids, response_ids);
      torch::Tensor ref_log_probs = cverl::torch_backend::response_log_probs(ref_logits_now, response_ids).detach();

      torch::Tensor kl_token = cverl::torch_backend::kl_penalty(log_probs, ref_log_probs, CVERL_KL_K2);
      torch::Tensor mask = torch::ones_like(response_ids, torch::kFloat32);
      torch::Tensor kl_loss = cverl::torch_backend::masked_mean(kl_token, mask);
      require(kl_loss.requires_grad(), "kl_loss carries grad");
      kl_loss.backward();
      double grad_norm = 0.0;
      for (const auto& p : policy->parameters()) {
        if (p.grad().defined()) {
          grad_norm += p.grad().abs().sum().item<double>();
        }
      }
      require(grad_norm > 0.0, "kl gradient propagates to policy");
    }

    std::cout << "reference policy + kl tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_reference_policy_kl failed: " << e.what() << "\n";
    return 1;
  }
}
