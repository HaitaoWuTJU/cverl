#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cverl::rollout {

// Generation request that the trainer hands to a rollout transport.
//
// Legacy/debug request type for CPU/shared-memory tests. The core efficient
// path uses `RolloutWorker` with GPU token tensors instead.
struct RolloutRequest {
  // Raw prompt strings.
  std::vector<std::string> prompts;

  // Optional pre-tokenized prompts. When non-empty its outer size must match
  // prompts.size() (or prompts may be empty if the transport works purely on
  // token ids).
  std::vector<std::vector<int32_t>> prompt_token_ids;

  // Number of samples to draw per prompt (e.g. GRPO group size).
  uint32_t n = 1;

  // Sampling parameters. -1 / 0 means "use server default".
  uint32_t max_tokens = 512;
  double temperature = 1.0;
  double top_p = 1.0;
  int32_t top_k = -1;
  std::vector<std::string> stop;
  bool return_token_ids = false;
  bool return_logprobs = false;
  uint64_t seed = 0;

  // Optional model id for legacy adapters.
  std::string model;

  // Free-form key/value parameters forwarded to the backend.
  std::map<std::string, std::string> extra_params;

  // Identifier propagated through to the response, useful for correlating
  // async / batched calls.
  uint64_t request_id = 0;
};

struct RolloutSequence {
  // Index into RolloutRequest::prompts that this sequence belongs to.
  uint32_t prompt_index = 0;
  // Index of this sample within the (prompt_index, n) group (0 .. n-1).
  uint32_t sample_index = 0;

  std::string text;
  std::vector<int32_t> token_ids;
  std::vector<float> logprobs;
  std::string finish_reason;  // "stop", "length", or backend-specific
};

struct RolloutResponse {
  uint64_t request_id = 0;
  std::vector<RolloutSequence> sequences;
  // Backend-specific stats (server-side latency, queue depth, etc.).
  std::map<std::string, std::string> metrics;
};

// Legacy/debug text transport. Efficient integrations should implement
// `RolloutWorker` instead.
class RolloutTransport {
 public:
  virtual ~RolloutTransport() = default;

  // Synchronous generate. Implementations must populate
  // RolloutResponse::sequences in (prompt_index, sample_index) order.
  virtual RolloutResponse generate(const RolloutRequest& request) = 0;

  // Human-readable transport name.
  virtual std::string name() const = 0;
};

// In-process transport that echoes the prompt back as the generation. Useful
// for unit tests and CI where no model server is available, and as a sanity
// reference for the rollout pipeline.
class LoopbackRolloutTransport : public RolloutTransport {
 public:
  struct Options {
    // If non-empty, this string is appended after the prompt to form the
    // generation. Use it to inject a rule-reward-friendly suffix.
    std::string completion_suffix;
    // If true, generated token_ids are populated as a hash of the prompt.
    bool synthesize_token_ids = false;
  };

  LoopbackRolloutTransport();
  explicit LoopbackRolloutTransport(Options options);
  RolloutResponse generate(const RolloutRequest& request) override;
  std::string name() const override { return "loopback"; }

 private:
  Options options_;
};

}  // namespace cverl::rollout
