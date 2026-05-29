#include "cverl/rollout/shared_memory_transport.h"

#include "cverl/rollout/shared_memory.h"

#include <fcntl.h>
#include <semaphore.h>
#include <sys/stat.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cverl::rollout {

namespace {

constexpr uint32_t kMagic = 0x43565354u;  // 'CVST'
constexpr uint32_t kVersion = 1;

// On-disk shared header. Followed in memory by:
//   request_payload  (payload_capacity bytes)
//   response_payload (payload_capacity bytes)
struct SharedTransportHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t payload_capacity;
  uint64_t request_size;
  uint64_t response_size;
  uint64_t request_id;
  uint32_t response_status;  // 0 = ok, non-zero = error
  uint32_t reserved;
};

constexpr size_t header_bytes() { return sizeof(SharedTransportHeader); }

size_t shm_total_bytes(size_t payload_capacity) {
  return header_bytes() + 2 * payload_capacity;
}

uint8_t* request_buffer(SharedTransportHeader* hdr) {
  return reinterpret_cast<uint8_t*>(hdr) + header_bytes();
}

uint8_t* response_buffer(SharedTransportHeader* hdr) {
  return request_buffer(hdr) + hdr->payload_capacity;
}

std::string normalize_sem_name(const std::string& base, const char* suffix) {
  std::string out = "/";
  out += (base.empty() || base[0] != '/') ? base : base.substr(1);
  out += suffix;
  return out;
}

std::string sys_error_msg(const std::string& what) {
  return what + ": " + std::strerror(errno);
}

void timespec_add_ms(timespec& ts, long ms) {
  ts.tv_sec += ms / 1000;
  ts.tv_nsec += (ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec += ts.tv_nsec / 1000000000L;
    ts.tv_nsec %= 1000000000L;
  }
}

// A trivial little-endian writer. We pin layouts to little-endian since
// every CPU/GPU we target is LE; if that ever changes, swap explicit byte
// writes here.
class Writer {
 public:
  Writer(uint8_t* base, size_t cap) : base_(base), cap_(cap) {}
  size_t pos() const { return pos_; }
  void write_u32(uint32_t v) {
    ensure(sizeof(v));
    std::memcpy(base_ + pos_, &v, sizeof(v));
    pos_ += sizeof(v);
  }
  void write_u64(uint64_t v) {
    ensure(sizeof(v));
    std::memcpy(base_ + pos_, &v, sizeof(v));
    pos_ += sizeof(v);
  }
  void write_f32(float v) {
    ensure(sizeof(v));
    std::memcpy(base_ + pos_, &v, sizeof(v));
    pos_ += sizeof(v);
  }
  void write_f64(double v) {
    ensure(sizeof(v));
    std::memcpy(base_ + pos_, &v, sizeof(v));
    pos_ += sizeof(v);
  }
  void write_bytes(const void* data, size_t len) {
    ensure(len);
    if (len > 0) {
      std::memcpy(base_ + pos_, data, len);
      pos_ += len;
    }
  }
  void write_string(const std::string& s) {
    write_u32(static_cast<uint32_t>(s.size()));
    write_bytes(s.data(), s.size());
  }
  void write_vec_i32(const std::vector<int32_t>& v) {
    write_u32(static_cast<uint32_t>(v.size()));
    write_bytes(v.data(), v.size() * sizeof(int32_t));
  }
  void write_vec_f32(const std::vector<float>& v) {
    write_u32(static_cast<uint32_t>(v.size()));
    write_bytes(v.data(), v.size() * sizeof(float));
  }

 private:
  void ensure(size_t need) {
    if (pos_ + need > cap_) {
      throw std::runtime_error("shared memory transport: payload exceeds capacity");
    }
  }
  uint8_t* base_;
  size_t cap_;
  size_t pos_ = 0;
};

class Reader {
 public:
  Reader(const uint8_t* base, size_t size) : base_(base), size_(size) {}
  bool eof() const { return pos_ >= size_; }
  size_t pos() const { return pos_; }
  uint32_t read_u32() {
    ensure(sizeof(uint32_t));
    uint32_t v;
    std::memcpy(&v, base_ + pos_, sizeof(v));
    pos_ += sizeof(v);
    return v;
  }
  uint64_t read_u64() {
    ensure(sizeof(uint64_t));
    uint64_t v;
    std::memcpy(&v, base_ + pos_, sizeof(v));
    pos_ += sizeof(v);
    return v;
  }
  float read_f32() {
    ensure(sizeof(float));
    float v;
    std::memcpy(&v, base_ + pos_, sizeof(v));
    pos_ += sizeof(v);
    return v;
  }
  double read_f64() {
    ensure(sizeof(double));
    double v;
    std::memcpy(&v, base_ + pos_, sizeof(v));
    pos_ += sizeof(v);
    return v;
  }
  void read_bytes(void* out, size_t n) {
    ensure(n);
    if (n > 0) {
      std::memcpy(out, base_ + pos_, n);
      pos_ += n;
    }
  }
  std::string read_string() {
    uint32_t n = read_u32();
    std::string out(n, '\0');
    read_bytes(out.data(), n);
    return out;
  }
  std::vector<int32_t> read_vec_i32() {
    uint32_t n = read_u32();
    std::vector<int32_t> out(n);
    read_bytes(out.data(), n * sizeof(int32_t));
    return out;
  }
  std::vector<float> read_vec_f32() {
    uint32_t n = read_u32();
    std::vector<float> out(n);
    read_bytes(out.data(), n * sizeof(float));
    return out;
  }

 private:
  void ensure(size_t need) const {
    if (pos_ + need > size_) {
      throw std::runtime_error("shared memory transport: short read");
    }
  }
  const uint8_t* base_;
  size_t size_;
  size_t pos_ = 0;
};

uint64_t serialize_request(const RolloutRequest& req, uint8_t* buffer, size_t cap) {
  Writer w(buffer, cap);
  w.write_u64(req.request_id);
  w.write_u32(req.n);
  w.write_u32(req.max_tokens);
  w.write_f64(req.temperature);
  w.write_f64(req.top_p);
  w.write_u32(static_cast<uint32_t>(req.top_k));
  w.write_u64(req.seed);
  uint32_t flags = 0;
  if (req.return_token_ids) flags |= 1u;
  if (req.return_logprobs) flags |= 2u;
  w.write_u32(flags);
  w.write_string(req.model);
  w.write_u32(static_cast<uint32_t>(req.prompts.size()));
  for (const auto& p : req.prompts) {
    w.write_string(p);
  }
  w.write_u32(static_cast<uint32_t>(req.prompt_token_ids.size()));
  for (const auto& ids : req.prompt_token_ids) {
    w.write_vec_i32(ids);
  }
  w.write_u32(static_cast<uint32_t>(req.stop.size()));
  for (const auto& s : req.stop) {
    w.write_string(s);
  }
  w.write_u32(static_cast<uint32_t>(req.extra_params.size()));
  for (const auto& [k, v] : req.extra_params) {
    w.write_string(k);
    w.write_string(v);
  }
  return static_cast<uint64_t>(w.pos());
}

RolloutRequest deserialize_request(const uint8_t* buffer, size_t size) {
  Reader r(buffer, size);
  RolloutRequest req;
  req.request_id = r.read_u64();
  req.n = r.read_u32();
  req.max_tokens = r.read_u32();
  req.temperature = r.read_f64();
  req.top_p = r.read_f64();
  req.top_k = static_cast<int32_t>(r.read_u32());
  req.seed = r.read_u64();
  uint32_t flags = r.read_u32();
  req.return_token_ids = (flags & 1u) != 0;
  req.return_logprobs = (flags & 2u) != 0;
  req.model = r.read_string();
  uint32_t n_prompts = r.read_u32();
  req.prompts.resize(n_prompts);
  for (uint32_t i = 0; i < n_prompts; ++i) {
    req.prompts[i] = r.read_string();
  }
  uint32_t n_prompt_tokens = r.read_u32();
  req.prompt_token_ids.resize(n_prompt_tokens);
  for (uint32_t i = 0; i < n_prompt_tokens; ++i) {
    req.prompt_token_ids[i] = r.read_vec_i32();
  }
  uint32_t n_stop = r.read_u32();
  req.stop.resize(n_stop);
  for (uint32_t i = 0; i < n_stop; ++i) {
    req.stop[i] = r.read_string();
  }
  uint32_t n_extra = r.read_u32();
  for (uint32_t i = 0; i < n_extra; ++i) {
    std::string k = r.read_string();
    std::string v = r.read_string();
    req.extra_params.emplace(std::move(k), std::move(v));
  }
  return req;
}

uint64_t serialize_response(const RolloutResponse& resp, uint8_t* buffer, size_t cap) {
  Writer w(buffer, cap);
  w.write_u64(resp.request_id);
  w.write_u32(static_cast<uint32_t>(resp.sequences.size()));
  for (const auto& seq : resp.sequences) {
    w.write_u32(seq.prompt_index);
    w.write_u32(seq.sample_index);
    w.write_string(seq.text);
    w.write_vec_i32(seq.token_ids);
    w.write_vec_f32(seq.logprobs);
    w.write_string(seq.finish_reason);
  }
  w.write_u32(static_cast<uint32_t>(resp.metrics.size()));
  for (const auto& [k, v] : resp.metrics) {
    w.write_string(k);
    w.write_string(v);
  }
  return static_cast<uint64_t>(w.pos());
}

RolloutResponse deserialize_response(const uint8_t* buffer, size_t size) {
  Reader r(buffer, size);
  RolloutResponse resp;
  resp.request_id = r.read_u64();
  uint32_t n_seq = r.read_u32();
  resp.sequences.resize(n_seq);
  for (uint32_t i = 0; i < n_seq; ++i) {
    auto& seq = resp.sequences[i];
    seq.prompt_index = r.read_u32();
    seq.sample_index = r.read_u32();
    seq.text = r.read_string();
    seq.token_ids = r.read_vec_i32();
    seq.logprobs = r.read_vec_f32();
    seq.finish_reason = r.read_string();
  }
  uint32_t n_metrics = r.read_u32();
  for (uint32_t i = 0; i < n_metrics; ++i) {
    std::string k = r.read_string();
    std::string v = r.read_string();
    resp.metrics.emplace(std::move(k), std::move(v));
  }
  return resp;
}

bool sem_wait_with_timeout(sem_t* s, long timeout_ms) {
  if (timeout_ms <= 0) {
    while (true) {
      int rc = sem_wait(s);
      if (rc == 0) return true;
      if (errno != EINTR) return false;
    }
  }
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  timespec_add_ms(ts, timeout_ms);
  while (true) {
    int rc = sem_timedwait(s, &ts);
    if (rc == 0) return true;
    if (errno == EINTR) continue;
    return false;
  }
}

}  // namespace

struct SharedMemoryRolloutTransport::Impl {
  SharedMemoryTransportOptions options;
  SharedMemoryRegion region;
  sem_t* req_sem = SEM_FAILED;
  sem_t* resp_sem = SEM_FAILED;
  std::string req_sem_name;
  std::string resp_sem_name;
  bool owner = false;
  uint64_t next_request_id = 1;

  ~Impl() {
    if (req_sem != SEM_FAILED) {
      sem_close(req_sem);
      if (owner) {
        sem_unlink(req_sem_name.c_str());
      }
    }
    if (resp_sem != SEM_FAILED) {
      sem_close(resp_sem);
      if (owner) {
        sem_unlink(resp_sem_name.c_str());
      }
    }
  }
};

SharedMemoryRolloutTransport::SharedMemoryRolloutTransport() = default;

SharedMemoryRolloutTransport::~SharedMemoryRolloutTransport() = default;

std::string SharedMemoryRolloutTransport::name() const {
  return std::string("shm(") + (impl_ ? impl_->options.name : std::string{}) + ")";
}

std::unique_ptr<SharedMemoryRolloutTransport> SharedMemoryRolloutTransport::create_client(
    const SharedMemoryTransportOptions& options) {
  if (options.name.empty()) {
    throw std::invalid_argument("SharedMemoryRolloutTransport requires a name");
  }
  auto self = std::unique_ptr<SharedMemoryRolloutTransport>(new SharedMemoryRolloutTransport());
  self->is_client_ = true;
  self->impl_ = std::make_unique<Impl>();
  self->impl_->options = options;
  self->impl_->owner = true;
  self->impl_->req_sem_name = normalize_sem_name(options.name, ".req");
  self->impl_->resp_sem_name = normalize_sem_name(options.name, ".resp");

  // Best-effort cleanup of stale semaphores from a previous crashed run.
  sem_unlink(self->impl_->req_sem_name.c_str());
  sem_unlink(self->impl_->resp_sem_name.c_str());

  self->impl_->region = SharedMemoryRegion(
      options.name, shm_total_bytes(options.payload_capacity), SharedMemoryOpenMode::Create);
  std::memset(self->impl_->region.data(), 0, self->impl_->region.size());
  auto* hdr = self->impl_->region.as<SharedTransportHeader>();
  hdr->magic = kMagic;
  hdr->version = kVersion;
  hdr->payload_capacity = options.payload_capacity;

  self->impl_->req_sem = sem_open(self->impl_->req_sem_name.c_str(), O_CREAT | O_EXCL, 0600, 0);
  if (self->impl_->req_sem == SEM_FAILED) {
    throw std::runtime_error(sys_error_msg("sem_open " + self->impl_->req_sem_name));
  }
  self->impl_->resp_sem = sem_open(self->impl_->resp_sem_name.c_str(), O_CREAT | O_EXCL, 0600, 0);
  if (self->impl_->resp_sem == SEM_FAILED) {
    throw std::runtime_error(sys_error_msg("sem_open " + self->impl_->resp_sem_name));
  }
  return self;
}

std::unique_ptr<SharedMemoryRolloutTransport> SharedMemoryRolloutTransport::attach_server(
    const SharedMemoryTransportOptions& options) {
  if (options.name.empty()) {
    throw std::invalid_argument("SharedMemoryRolloutTransport requires a name");
  }
  auto self = std::unique_ptr<SharedMemoryRolloutTransport>(new SharedMemoryRolloutTransport());
  self->is_client_ = false;
  self->impl_ = std::make_unique<Impl>();
  self->impl_->options = options;
  self->impl_->owner = false;
  self->impl_->req_sem_name = normalize_sem_name(options.name, ".req");
  self->impl_->resp_sem_name = normalize_sem_name(options.name, ".resp");

  self->impl_->region = SharedMemoryRegion(
      options.name, shm_total_bytes(options.payload_capacity), SharedMemoryOpenMode::OpenExisting);
  auto* hdr = self->impl_->region.as<SharedTransportHeader>();
  if (hdr->magic != kMagic) {
    throw std::runtime_error("SharedMemoryRolloutTransport: bad magic in shared region");
  }
  if (hdr->version != kVersion) {
    throw std::runtime_error("SharedMemoryRolloutTransport: version mismatch");
  }
  if (hdr->payload_capacity != options.payload_capacity) {
    throw std::runtime_error("SharedMemoryRolloutTransport: payload capacity mismatch");
  }

  self->impl_->req_sem = sem_open(self->impl_->req_sem_name.c_str(), 0);
  if (self->impl_->req_sem == SEM_FAILED) {
    throw std::runtime_error(sys_error_msg("sem_open " + self->impl_->req_sem_name));
  }
  self->impl_->resp_sem = sem_open(self->impl_->resp_sem_name.c_str(), 0);
  if (self->impl_->resp_sem == SEM_FAILED) {
    throw std::runtime_error(sys_error_msg("sem_open " + self->impl_->resp_sem_name));
  }
  return self;
}

RolloutResponse SharedMemoryRolloutTransport::generate(const RolloutRequest& request) {
  if (!is_client_) {
    throw std::runtime_error("SharedMemoryRolloutTransport: generate() called on server side");
  }
  auto* hdr = impl_->region.as<SharedTransportHeader>();
  // Use the caller-provided request_id when set, otherwise auto-assign so
  // the server can correlate.
  uint64_t rid = request.request_id != 0 ? request.request_id : impl_->next_request_id++;
  hdr->request_id = rid;
  uint64_t size =
      serialize_request(request, request_buffer(hdr), impl_->options.payload_capacity);
  std::atomic_thread_fence(std::memory_order_release);
  hdr->request_size = size;
  if (sem_post(impl_->req_sem) != 0) {
    throw std::runtime_error(sys_error_msg("sem_post req_sem"));
  }
  if (!sem_wait_with_timeout(impl_->resp_sem, impl_->options.timeout_ms)) {
    throw std::runtime_error("SharedMemoryRolloutTransport: timed out waiting for response");
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  if (hdr->response_status != 0) {
    throw std::runtime_error("SharedMemoryRolloutTransport: server reported error status " +
                             std::to_string(hdr->response_status));
  }
  return deserialize_response(response_buffer(hdr), hdr->response_size);
}

bool SharedMemoryRolloutTransport::serve_one(const Handler& handler) {
  if (is_client_) {
    throw std::runtime_error("SharedMemoryRolloutTransport: serve_one() called on client side");
  }
  if (!sem_wait_with_timeout(impl_->req_sem, impl_->options.timeout_ms)) {
    return false;
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  auto* hdr = impl_->region.as<SharedTransportHeader>();
  RolloutResponse response;
  uint32_t status = 0;
  try {
    RolloutRequest req = deserialize_request(request_buffer(hdr), hdr->request_size);
    if (req.request_id == 0) {
      req.request_id = hdr->request_id;
    }
    response = handler(req);
    if (response.request_id == 0) {
      response.request_id = req.request_id;
    }
  } catch (const std::exception&) {
    status = 1;
  }
  uint64_t resp_size = 0;
  if (status == 0) {
    try {
      resp_size = serialize_response(response, response_buffer(hdr), impl_->options.payload_capacity);
    } catch (const std::exception&) {
      status = 2;
      resp_size = 0;
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
  hdr->response_size = resp_size;
  hdr->response_status = status;
  if (sem_post(impl_->resp_sem) != 0) {
    throw std::runtime_error(sys_error_msg("sem_post resp_sem"));
  }
  return status == 0;
}

void SharedMemoryRolloutTransport::serve_loop(const Handler& handler, std::atomic<bool>& running) {
  while (running.load(std::memory_order_relaxed)) {
    serve_one(handler);
  }
}

}  // namespace cverl::rollout
