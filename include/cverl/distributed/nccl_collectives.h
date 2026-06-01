#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <nccl.h>
#include <torch/torch.h>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

struct NcclUniqueIdBytes {
  std::vector<char> bytes;
};

NcclUniqueIdBytes create_nccl_unique_id();
void write_nccl_unique_id_file(const std::string& path, const NcclUniqueIdBytes& id);
NcclUniqueIdBytes read_nccl_unique_id_file(const std::string& path);

class NcclCollectives final : public Collectives {
 public:
  NcclCollectives(int64_t rank,
                  int64_t world_size,
                  int device_index,
                  const NcclUniqueIdBytes& unique_id,
                  bool synchronize_after_collective = true);
  ~NcclCollectives() override;

  NcclCollectives(const NcclCollectives&) = delete;
  NcclCollectives& operator=(const NcclCollectives&) = delete;
  NcclCollectives(NcclCollectives&&) = delete;
  NcclCollectives& operator=(NcclCollectives&&) = delete;

  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return world_size_; }
  bool synchronize_after_collective() const { return synchronize_after_collective_; }
  void set_synchronize_after_collective(bool value) { synchronize_after_collective_ = value; }
  void synchronize();
  void barrier() override;
  torch::Tensor broadcast(const torch::Tensor& input, int64_t root, const std::vector<int64_t>& group) override;
  torch::Tensor all_reduce(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group) override;
  torch::Tensor all_gather(const torch::Tensor& input, const std::vector<int64_t>& group, int64_t dim) override;
  torch::Tensor reduce_scatter(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group, int64_t dim) override;
  torch::Tensor all_to_all(const torch::Tensor& input, const std::vector<int64_t>& group, int64_t dim) override;
  void send(const torch::Tensor& input, int64_t peer) override;
  torch::Tensor recv_like(const torch::Tensor& like, int64_t peer) override;
  torch::Tensor send_recv(const torch::Tensor& input, int64_t send_peer, const torch::Tensor& recv_like, int64_t recv_peer) override;

 private:
  struct PendingOp {
    cudaEvent_t done = nullptr;
    std::vector<torch::Tensor> keep_alive;
  };

  int64_t rank_;
  int64_t world_size_;
  int device_index_;
  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;
  bool synchronize_after_collective_ = true;
  std::vector<PendingOp> pending_ops_;
  std::unordered_map<std::string, ncclComm_t> subgroup_comms_;

  void finish_nccl_op(std::vector<torch::Tensor> keep_alive);
  void collect_finished_pending_ops();
  ncclComm_t communicator_for_group(const std::vector<int64_t>& group);
};

}  // namespace cverl::distributed
