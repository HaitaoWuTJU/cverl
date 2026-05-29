#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace cverl::rollout {

enum class SharedMemoryOpenMode {
  Create,
  OpenExisting,
};

class SharedMemoryRegion {
 public:
  SharedMemoryRegion() = default;
  SharedMemoryRegion(const std::string& name, size_t size, SharedMemoryOpenMode mode);
  ~SharedMemoryRegion();

  SharedMemoryRegion(const SharedMemoryRegion&) = delete;
  SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;
  SharedMemoryRegion(SharedMemoryRegion&& other) noexcept;
  SharedMemoryRegion& operator=(SharedMemoryRegion&& other) noexcept;

  static std::string make_name(const std::string& prefix);

  void unlink();
  void close();

  const std::string& name() const { return name_; }
  size_t size() const { return size_; }
  void* data() { return data_; }
  const void* data() const { return data_; }

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(data_);
  }

 private:
  std::string name_;
  size_t size_ = 0;
  int fd_ = -1;
  void* data_ = nullptr;
  bool owner_ = false;
};

struct SharedRolloutHeader {
  uint32_t magic = 0x4356524c;  // CVRL
  uint32_t version = 1;
  uint64_t request_id = 0;
  uint32_t status = 0;
  uint32_t prompt_count = 0;
  uint32_t max_prompt_tokens = 0;
  uint32_t max_response_tokens = 0;
  uint32_t reserved = 0;
};

}  // namespace cverl::rollout
