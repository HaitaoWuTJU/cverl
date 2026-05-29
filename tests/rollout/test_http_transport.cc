// Tests HttpRolloutTransport against a tiny in-process HTTP server that
// emulates the OpenAI /v1/completions endpoint. The server writes back a
// canned response for two prompts so the client's per-prompt routing,
// sampling parameters, and JSON parsing can all be verified without
// depending on a real vLLM/SGLang process.
#include "cverl/rollout/http_transport.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

// Read the full HTTP request (headers + body) from `fd`. Returns the raw
// bytes; throws on socket error or premature close.
std::string read_http_request(int fd) {
  std::string buf;
  char chunk[4096];
  // First read headers.
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n < 0) {
      throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("client closed before headers");
    }
    buf.append(chunk, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
  }
  // Parse Content-Length to know how much body to read.
  size_t content_length = 0;
  {
    std::string lower;
    lower.reserve(buf.size());
    for (char c : buf.substr(0, header_end)) {
      lower.push_back(static_cast<char>(::tolower(static_cast<unsigned char>(c))));
    }
    auto pos = lower.find("content-length:");
    if (pos != std::string::npos) {
      pos += std::string("content-length:").size();
      while (pos < lower.size() && (lower[pos] == ' ' || lower[pos] == '\t')) ++pos;
      content_length = static_cast<size_t>(std::strtoull(lower.c_str() + pos, nullptr, 10));
    }
  }
  size_t body_start = header_end + 4;
  size_t have = buf.size() - body_start;
  while (have < content_length) {
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      throw std::runtime_error("client closed before body complete");
    }
    buf.append(chunk, static_cast<size_t>(n));
    have += static_cast<size_t>(n);
  }
  return buf;
}

std::string extract_body(const std::string& http) {
  auto pos = http.find("\r\n\r\n");
  if (pos == std::string::npos) {
    return {};
  }
  return http.substr(pos + 4);
}

void send_all(int fd, const std::string& data) {
  size_t offset = 0;
  while (offset < data.size()) {
    ssize_t n = ::send(fd, data.data() + offset, data.size() - offset, 0);
    if (n <= 0) {
      throw std::runtime_error("send failed");
    }
    offset += static_cast<size_t>(n);
  }
}

std::string make_http_response(const std::string& body) {
  std::ostringstream oss;
  oss << "HTTP/1.1 200 OK\r\n";
  oss << "Content-Type: application/json\r\n";
  oss << "Content-Length: " << body.size() << "\r\n";
  oss << "Connection: close\r\n\r\n";
  oss << body;
  return oss.str();
}

// Server thread: serves `expected_requests` responses then exits.
struct ServerStats {
  std::atomic<int> served{0};
  std::vector<json> received_bodies;
  std::mutex mu;
};

void run_server(int listen_fd, int expected_requests, ServerStats& stats) {
  for (int r = 0; r < expected_requests; ++r) {
    sockaddr_in client{};
    socklen_t client_len = sizeof(client);
    int conn = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&client), &client_len);
    if (conn < 0) {
      return;
    }
    try {
      std::string raw = read_http_request(conn);
      json body = json::parse(extract_body(raw));
      {
        std::lock_guard<std::mutex> lock(stats.mu);
        stats.received_bodies.push_back(body);
      }
      // Build a /v1/completions response. Emit `n` choices indexed 0..n-1.
      uint32_t n = body.value("n", 1u);
      json resp;
      resp["id"] = "cmpl-test";
      resp["object"] = "text_completion";
      resp["model"] = body.value("model", "test-model");
      resp["choices"] = json::array();
      for (uint32_t i = 0; i < n; ++i) {
        json choice;
        choice["index"] = static_cast<int>(i);
        choice["text"] = std::string("answer-") + std::to_string(stats.served.load()) + "-" +
                         std::to_string(i) + " #### " + std::to_string(i);
        choice["finish_reason"] = "stop";
        resp["choices"].push_back(choice);
      }
      resp["usage"] = {{"prompt_tokens", 7}, {"completion_tokens", 11}, {"total_tokens", 18}};
      send_all(conn, make_http_response(resp.dump()));
      stats.served.fetch_add(1);
    } catch (...) {
      // Best-effort: on any error close the socket and continue.
    }
    ::close(conn);
  }
}

int bind_loopback_socket() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("socket failed");
  }
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;  // ask kernel for a free port
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    throw std::runtime_error("bind failed");
  }
  if (::listen(fd, 8) != 0) {
    ::close(fd);
    throw std::runtime_error("listen failed");
  }
  return fd;
}

uint16_t socket_port(int fd) {
  sockaddr_in actual{};
  socklen_t len = sizeof(actual);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&actual), &len) != 0) {
    throw std::runtime_error("getsockname failed");
  }
  return ntohs(actual.sin_port);
}

}  // namespace

int main() {
  try {
    const int prompts = 2;
    const uint32_t n = 3;
    int listen_fd = bind_loopback_socket();
    uint16_t port = socket_port(listen_fd);

    ServerStats stats;
    std::thread server([&] { run_server(listen_fd, prompts, stats); });

    cverl::rollout::HttpRolloutOptions opts;
    opts.base_url = "http://127.0.0.1:" + std::to_string(port);
    opts.endpoint = "completions";
    opts.model = "test-model";
    opts.connect_timeout_seconds = 5;
    opts.total_timeout_seconds = 10;

    cverl::rollout::HttpRolloutTransport transport(std::move(opts));
    cverl::rollout::RolloutRequest req;
    req.prompts = {"hello", "world"};
    req.n = n;
    req.max_tokens = 32;
    req.temperature = 0.7;
    req.top_p = 0.9;
    req.seed = 11;
    req.stop = {"\n\n"};
    req.extra_params["repetition_penalty"] = "1.05";

    auto resp = transport.generate(req);

    server.join();
    ::close(listen_fd);

    require(stats.served.load() == prompts, "server served all prompts");
    require(resp.sequences.size() == static_cast<size_t>(prompts) * n, "response sequences count");
    require(resp.sequences[0].prompt_index == 0, "prompt_index 0");
    require(resp.sequences[n].prompt_index == 1, "prompt_index 1");
    require(resp.sequences[0].sample_index == 0, "sample_index 0");
    require(resp.sequences[2].sample_index == 2, "sample_index 2");
    require(resp.sequences[0].finish_reason == "stop", "finish_reason");
    require(resp.sequences[0].text.find("####") != std::string::npos, "text contains gsm8k marker");

    // Verify request body fields the client sent.
    {
      std::lock_guard<std::mutex> lock(stats.mu);
      require(stats.received_bodies.size() == prompts, "server received all bodies");
      const auto& first = stats.received_bodies[0];
      require(first.value("model", std::string{}) == "test-model", "model forwarded");
      require(first.value("n", 0u) == n, "n forwarded");
      require(first.value("max_tokens", 0u) == 32u, "max_tokens forwarded");
      require(std::abs(first.value("temperature", 0.0) - 0.7) < 1e-9, "temperature forwarded");
      require(std::abs(first.value("top_p", 0.0) - 0.9) < 1e-9, "top_p forwarded");
      require(first.value("seed", 0u) == 11u, "seed forwarded");
      require(first.contains("stop") && first["stop"].is_array() && first["stop"][0] == "\n\n",
              "stop forwarded");
      require(first.contains("repetition_penalty"), "extra_params forwarded");
      // extra_params parses JSON-shaped values, so "1.05" lands as a number.
      require(first["repetition_penalty"].is_number(), "extra_params parsed as JSON");
      require(first.value("prompt", std::string{}) == "hello", "prompt forwarded");
      require(stats.received_bodies[1].value("prompt", std::string{}) == "world", "second prompt routed");
    }

    require(resp.metrics.count("usage.total_tokens") == 1, "usage stats forwarded");

    std::cout << "http rollout transport tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_http_transport failed: " << e.what() << "\n";
    return 1;
  }
}
