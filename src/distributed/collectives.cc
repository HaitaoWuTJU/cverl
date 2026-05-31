#include "cverl/distributed/collectives.h"

#include <stdexcept>

namespace cverl::distributed {
namespace {

void require_single_rank_group(const std::vector<int64_t>& group) {
  if (group.size() != 1 || group[0] != 0) {
    throw std::invalid_argument("SingleProcessCollectives only supports group {0}");
  }
}

}  // namespace

torch::Tensor Collectives::send_recv(const torch::Tensor& input,
                                     int64_t send_peer,
                                     const torch::Tensor& like,
                                     int64_t recv_peer) {
  send(input, send_peer);
  return recv_like(like, recv_peer);
}

torch::Tensor SingleProcessCollectives::broadcast(const torch::Tensor& input,
                                                  int64_t root,
                                                  const std::vector<int64_t>& group) {
  require_single_rank_group(group);
  if (root != 0) {
    throw std::invalid_argument("SingleProcessCollectives can only broadcast from root 0");
  }
  return input.clone();
}

torch::Tensor SingleProcessCollectives::all_reduce(const torch::Tensor& input,
                                                   ReduceOp /*op*/,
                                                   const std::vector<int64_t>& group) {
  require_single_rank_group(group);
  return input;
}

torch::Tensor SingleProcessCollectives::all_gather(const torch::Tensor& input,
                                                   const std::vector<int64_t>& group,
                                                   int64_t /*dim*/) {
  require_single_rank_group(group);
  return input;
}

torch::Tensor SingleProcessCollectives::reduce_scatter(const torch::Tensor& input,
                                                       ReduceOp /*op*/,
                                                       const std::vector<int64_t>& group,
                                                       int64_t /*dim*/) {
  require_single_rank_group(group);
  return input;
}

void SingleProcessCollectives::send(const torch::Tensor& /*input*/, int64_t peer) {
  if (peer != 0) {
    throw std::invalid_argument("SingleProcessCollectives can only send to rank 0");
  }
}

torch::Tensor SingleProcessCollectives::recv_like(const torch::Tensor& like, int64_t peer) {
  if (peer != 0) {
    throw std::invalid_argument("SingleProcessCollectives can only recv from rank 0");
  }
  return torch::empty_like(like);
}

torch::Tensor SingleProcessCollectives::send_recv(const torch::Tensor& input,
                                                  int64_t send_peer,
                                                  const torch::Tensor& like,
                                                  int64_t recv_peer) {
  if (send_peer != 0 || recv_peer != 0) {
    throw std::invalid_argument("SingleProcessCollectives can only send_recv with rank 0");
  }
  (void)input;
  return torch::empty_like(like);
}

}  // namespace cverl::distributed
