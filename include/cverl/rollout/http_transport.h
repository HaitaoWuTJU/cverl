#pragma once

#include "cverl/rollout/transport.h"

#include <memory>
#include <string>

namespace cverl::rollout {

struct HttpRolloutOptions {
  // Base URL of an OpenAI-compatible server (vLLM / SGLang / OpenAI itself).
  // Example: "http://localhost:8000".
  std::string base_url;

  // API endpoint:
  //   "chat"        -> POST {base_url}/v1/chat/completions
  //   "completions" -> POST {base_url}/v1/completions
  // Default is "completions" because GRPO/PPO trainers usually want raw
  // continuations, not chat-style assistant messages.
  std::string endpoint = "completions";

  // Bearer token sent in the Authorization header. Optional; vLLM/SGLang
  // accept any value when --api-key is not configured.
  std::string api_key;

  // Default model id. Overridden per-request when RolloutRequest::model is
  // set.
  std::string model;

  // Network timeouts (seconds). 0 means "no timeout".
  long connect_timeout_seconds = 10;
  long total_timeout_seconds = 600;

  // Verbose libcurl logging (stderr). Useful when debugging serialization.
  bool verbose = false;

  // When endpoint == "chat", wrap each prompt as
  //   [{"role":"system","content":system_prompt},
  //    {"role":"user","content":prompt}]
  // System prompt is omitted when this is empty.
  std::string system_prompt;
};

// HTTP rollout client backed by libcurl. Talks to vLLM / SGLang / any
// OpenAI-compatible server. Each call is synchronous and blocks until the
// generation finishes.
class HttpRolloutTransport : public RolloutTransport {
 public:
  explicit HttpRolloutTransport(HttpRolloutOptions options);
  ~HttpRolloutTransport() override;

  HttpRolloutTransport(const HttpRolloutTransport&) = delete;
  HttpRolloutTransport& operator=(const HttpRolloutTransport&) = delete;

  RolloutResponse generate(const RolloutRequest& request) override;
  std::string name() const override;

  const HttpRolloutOptions& options() const { return options_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  HttpRolloutOptions options_;
};

}  // namespace cverl::rollout
