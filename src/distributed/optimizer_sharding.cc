#include "cverl/distributed/optimizer_sharding.h"

#include <stdexcept>

namespace cverl::distributed {

std::vector<int64_t> greedy_parameter_owner_by_size(const std::vector<int64_t>& parameter_bytes,
                                                    int64_t data_parallel) {
  if (data_parallel <= 0) {
    throw std::invalid_argument("data_parallel must be positive");
  }
  std::vector<int64_t> bytes_by_rank(static_cast<size_t>(data_parallel), 0);
  std::vector<int64_t> owner_by_parameter;
  owner_by_parameter.reserve(parameter_bytes.size());
  for (int64_t bytes : parameter_bytes) {
    if (bytes < 0) {
      throw std::invalid_argument("parameter byte size must be non-negative");
    }
    int64_t owner = 0;
    for (int64_t rank = 1; rank < data_parallel; ++rank) {
      if (bytes_by_rank[static_cast<size_t>(rank)] < bytes_by_rank[static_cast<size_t>(owner)]) {
        owner = rank;
      }
    }
    owner_by_parameter.push_back(owner);
    bytes_by_rank[static_cast<size_t>(owner)] += bytes;
  }
  return owner_by_parameter;
}

std::vector<int64_t> owned_parameter_indices(const std::vector<int64_t>& owner_by_parameter,
                                             int64_t data_rank) {
  if (data_rank < 0) {
    throw std::invalid_argument("data_rank must be non-negative");
  }
  std::vector<int64_t> out;
  for (size_t i = 0; i < owner_by_parameter.size(); ++i) {
    if (owner_by_parameter[i] < 0) {
      throw std::invalid_argument("owner rank must be non-negative");
    }
    if (owner_by_parameter[i] == data_rank) {
      out.push_back(static_cast<int64_t>(i));
    }
  }
  return out;
}

}  // namespace cverl::distributed
