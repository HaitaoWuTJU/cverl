#pragma once

#include <cstdint>
#include <vector>

namespace cverl::distributed {

std::vector<int64_t> greedy_parameter_owner_by_size(const std::vector<int64_t>& parameter_bytes,
                                                    int64_t data_parallel);

std::vector<int64_t> owned_parameter_indices(const std::vector<int64_t>& owner_by_parameter,
                                             int64_t data_rank);

}  // namespace cverl::distributed
