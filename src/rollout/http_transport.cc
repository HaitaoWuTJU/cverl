#include "cverl/rollout/http_transport.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace cverl::rollout {

namespace {

using json = nlohmann::json;

// Process-global libcurl init/cleanup. curl_global_init is required once per
// process before any easy handles are created.
class CurlGlobal {
 public:
  static CurlGlobal& instance() {
    static CurlGlobal global;
    return global;
  }

 private:
  CurlGlobal() {
    CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (rc != CURLE_OK) {
      throw std::runtime_error(std::string("curl_global_init failed: ") + curl_easy_strerror(rc));
    }
  }
  ~CurlGlobal() { curl_global_cleanup(); }
};

size_t curl_write_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* out = static_cast<std::string*>(userdata);
  size_t total = size * nmemb;
  out->append(ptr, total);
  return total;
}

std::string trim_trailing_slash(std::string url) {
  while (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  return url;
}

void apply_sampling(const RolloutRequest& request, json& body) {
  body["max_tokens"] = request.max_tokens;
  body["temperature"] = request.temperature;
  body["top_p"] = request.top_p;
  body["n"] = request.n;
  if (request.top_k > 0) {
    // OpenAI itself does not accept top_k, but vLLM/SGLang OpenAI-compat
    // endpoints do; servers that don't will reject it.
    body["top_k"] = request.top_k;
  }
  if (request.seed != 0) {
    body["seed"] = request.seed;
  }
  if (!request.stop.empty()) {
    body["stop"] = request.stop;
  }
  if (request.return_logprobs) {
    body["logprobs"] = true;
    body["top_logprobs"] = 0;
  }
  for (const auto& [k, v] : request.extra_params) {
    // Accept JSON-encoded extras (preferred for non-string values), fall
    // back to plain string when parsing fails.
    try {
      body[k] = json::parse(v);
    } catch (const std::exception&) {
      body[k] = v;
    }
  }
}

void parse_completions(const json& payload, RolloutResponse& out, uint32_t n) {
  if (!payload.contains("choices")) {
    throw std::runtime_error("rollout response missing 'choices'");
  }
  for (const auto& choice : payload["choices"]) {
    RolloutSequence seq;
    int index = choice.value("index", 0);
    seq.sample_index = static_cast<uint32_t>(index % static_cast<int>(std::max<uint32_t>(n, 1)));
    seq.prompt_index = static_cast<uint32_t>(index / static_cast<int>(std::max<uint32_t>(n, 1)));
    seq.text = choice.value("text", std::string{});
    seq.finish_reason = choice.value("finish_reason", std::string{});
    if (choice.contains("logprobs") && choice["logprobs"].is_object()) {
      const auto& lp = choice["logprobs"];
      if (lp.contains("token_logprobs") && lp["token_logprobs"].is_array()) {
        seq.logprobs.reserve(lp["token_logprobs"].size());
        for (const auto& v : lp["token_logprobs"]) {
          if (v.is_null()) {
            seq.logprobs.push_back(0.0f);
          } else {
            seq.logprobs.push_back(v.get<float>());
          }
        }
      }
      if (lp.contains("tokens") && lp["tokens"].is_array() && seq.token_ids.empty()) {
        // Some servers return string tokens here; we do not parse them as
        // ids. Caller should request token_ids via extra_params if the
        // server supports it.
      }
    }
    out.sequences.push_back(std::move(seq));
  }
}

void parse_chat(const json& payload, RolloutResponse& out, uint32_t n) {
  if (!payload.contains("choices")) {
    throw std::runtime_error("rollout response missing 'choices'");
  }
  for (const auto& choice : payload["choices"]) {
    RolloutSequence seq;
    int index = choice.value("index", 0);
    seq.sample_index = static_cast<uint32_t>(index % static_cast<int>(std::max<uint32_t>(n, 1)));
    seq.prompt_index = static_cast<uint32_t>(index / static_cast<int>(std::max<uint32_t>(n, 1)));
    if (choice.contains("message") && choice["message"].contains("content")) {
      seq.text = choice["message"]["content"].get<std::string>();
    }
    seq.finish_reason = choice.value("finish_reason", std::string{});
    out.sequences.push_back(std::move(seq));
  }
}

}  // namespace

struct HttpRolloutTransport::Impl {
  CURL* curl = nullptr;
  curl_slist* headers = nullptr;
  std::mutex mu;

  Impl() {
    CurlGlobal::instance();
    curl = curl_easy_init();
    if (curl == nullptr) {
      throw std::runtime_error("curl_easy_init failed");
    }
  }

  ~Impl() {
    if (headers != nullptr) {
      curl_slist_free_all(headers);
      headers = nullptr;
    }
    if (curl != nullptr) {
      curl_easy_cleanup(curl);
      curl = nullptr;
    }
  }
};

HttpRolloutTransport::HttpRolloutTransport(HttpRolloutOptions options)
    : impl_(std::make_unique<Impl>()), options_(std::move(options)) {
  options_.base_url = trim_trailing_slash(options_.base_url);
  if (options_.base_url.empty()) {
    throw std::invalid_argument("HttpRolloutTransport requires a base_url");
  }
  impl_->headers = curl_slist_append(impl_->headers, "Content-Type: application/json");
  impl_->headers = curl_slist_append(impl_->headers, "Accept: application/json");
  if (!options_.api_key.empty()) {
    std::string auth = "Authorization: Bearer " + options_.api_key;
    impl_->headers = curl_slist_append(impl_->headers, auth.c_str());
  }
}

HttpRolloutTransport::~HttpRolloutTransport() = default;

std::string HttpRolloutTransport::name() const {
  return "http(" + options_.endpoint + "@" + options_.base_url + ")";
}

RolloutResponse HttpRolloutTransport::generate(const RolloutRequest& request) {
  if (request.prompts.empty()) {
    return {};
  }

  const std::string model = request.model.empty() ? options_.model : request.model;
  const bool chat_endpoint = options_.endpoint == "chat";
  const std::string url = options_.base_url + (chat_endpoint ? "/v1/chat/completions" : "/v1/completions");

  RolloutResponse response;
  response.request_id = request.request_id;
  response.sequences.reserve(static_cast<size_t>(request.prompts.size()) * request.n);

  // Issue one HTTP call per prompt so prompt_index is unambiguous in the
  // response. Servers like vLLM accept a list of prompts on /v1/completions,
  // but the response indices are flattened, so using one call per prompt
  // keeps the routing simple and matches /v1/chat/completions behavior.
  for (size_t p = 0; p < request.prompts.size(); ++p) {
    json body;
    if (!model.empty()) {
      body["model"] = model;
    }
    if (chat_endpoint) {
      json messages = json::array();
      if (!options_.system_prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", options_.system_prompt}});
      }
      messages.push_back({{"role", "user"}, {"content", request.prompts[p]}});
      body["messages"] = std::move(messages);
    } else {
      body["prompt"] = request.prompts[p];
    }
    apply_sampling(request, body);

    std::string payload = body.dump();
    std::string out_buffer;
    long http_status = 0;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      curl_easy_reset(impl_->curl);
      curl_easy_setopt(impl_->curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(impl_->curl, CURLOPT_POST, 1L);
      curl_easy_setopt(impl_->curl, CURLOPT_HTTPHEADER, impl_->headers);
      curl_easy_setopt(impl_->curl, CURLOPT_POSTFIELDS, payload.data());
      curl_easy_setopt(impl_->curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
      curl_easy_setopt(impl_->curl, CURLOPT_WRITEFUNCTION, curl_write_string);
      curl_easy_setopt(impl_->curl, CURLOPT_WRITEDATA, &out_buffer);
      curl_easy_setopt(impl_->curl, CURLOPT_NOSIGNAL, 1L);
      curl_easy_setopt(impl_->curl, CURLOPT_CONNECTTIMEOUT, options_.connect_timeout_seconds);
      curl_easy_setopt(impl_->curl, CURLOPT_TIMEOUT, options_.total_timeout_seconds);
      curl_easy_setopt(impl_->curl, CURLOPT_VERBOSE, options_.verbose ? 1L : 0L);

      CURLcode rc = curl_easy_perform(impl_->curl);
      if (rc != CURLE_OK) {
        throw std::runtime_error(std::string("HTTP rollout transport error: ") + curl_easy_strerror(rc));
      }
      curl_easy_getinfo(impl_->curl, CURLINFO_RESPONSE_CODE, &http_status);
    }
    if (http_status < 200 || http_status >= 300) {
      std::ostringstream oss;
      oss << "HTTP rollout " << url << " returned status " << http_status << ": " << out_buffer;
      throw std::runtime_error(oss.str());
    }

    json parsed;
    try {
      parsed = json::parse(out_buffer);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("failed to parse rollout response: ") + e.what());
    }

    RolloutResponse per_prompt;
    if (chat_endpoint) {
      parse_chat(parsed, per_prompt, request.n);
    } else {
      parse_completions(parsed, per_prompt, request.n);
    }
    for (auto& seq : per_prompt.sequences) {
      seq.prompt_index = static_cast<uint32_t>(p);
      response.sequences.push_back(std::move(seq));
    }
    if (parsed.contains("usage") && parsed["usage"].is_object()) {
      for (const auto& [k, v] : parsed["usage"].items()) {
        response.metrics["usage." + k] = v.dump();
      }
    }
  }
  return response;
}

}  // namespace cverl::rollout
