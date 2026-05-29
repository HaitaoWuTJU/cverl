#include "cverl/rollout/shared_memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cverl::rollout {
namespace {

std::runtime_error sys_error(const std::string& what) {
  return std::runtime_error(what + ": " + std::strerror(errno));
}

std::string normalize_name(const std::string& name) {
  if (name.empty()) {
    throw std::invalid_argument("shared memory name is required");
  }
  return name[0] == '/' ? name : "/" + name;
}

}  // namespace

SharedMemoryRegion::SharedMemoryRegion(const std::string& name, size_t size, SharedMemoryOpenMode mode)
    : name_(normalize_name(name)), size_(size), owner_(mode == SharedMemoryOpenMode::Create) {
  if (size_ == 0) {
    throw std::invalid_argument("shared memory size must be positive");
  }

  int flags = O_RDWR;
  if (mode == SharedMemoryOpenMode::Create) {
    flags |= O_CREAT | O_EXCL;
  }
  fd_ = shm_open(name_.c_str(), flags, 0600);
  if (fd_ < 0) {
    throw sys_error("shm_open " + name_);
  }
  if (mode == SharedMemoryOpenMode::Create && ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
    int saved = errno;
    close();
    errno = saved;
    throw sys_error("ftruncate " + name_);
  }

  data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (data_ == MAP_FAILED) {
    data_ = nullptr;
    int saved = errno;
    close();
    errno = saved;
    throw sys_error("mmap " + name_);
  }
}

SharedMemoryRegion::~SharedMemoryRegion() {
  close();
}

SharedMemoryRegion::SharedMemoryRegion(SharedMemoryRegion&& other) noexcept {
  *this = std::move(other);
}

SharedMemoryRegion& SharedMemoryRegion::operator=(SharedMemoryRegion&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  close();
  name_ = std::move(other.name_);
  size_ = other.size_;
  fd_ = other.fd_;
  data_ = other.data_;
  owner_ = other.owner_;
  other.size_ = 0;
  other.fd_ = -1;
  other.data_ = nullptr;
  other.owner_ = false;
  return *this;
}

std::string SharedMemoryRegion::make_name(const std::string& prefix) {
  return "/" + prefix + "_" + std::to_string(static_cast<unsigned long long>(getpid())) + "_" +
         std::to_string(reinterpret_cast<uintptr_t>(&prefix));
}

void SharedMemoryRegion::unlink() {
  if (!name_.empty()) {
    shm_unlink(name_.c_str());
  }
  owner_ = false;
}

void SharedMemoryRegion::close() {
  if (data_ != nullptr) {
    munmap(data_, size_);
    data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  if (owner_ && !name_.empty()) {
    shm_unlink(name_.c_str());
    owner_ = false;
  }
}

}  // namespace cverl::rollout
