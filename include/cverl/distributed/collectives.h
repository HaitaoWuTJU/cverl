#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace cverl::distributed {

enum class ReduceOp {
  Sum,
  Mean,
  Max,
};

class Collectives {
 public:
  virtual ~Collectives() = default;

  virtual int64_t rank() const = 0;
  virtual int64_t world_size() const = 0;
  virtual void barrier() = 0;
  virtual torch::Tensor all_reduce(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group) = 0;
  virtual torch::Tensor all_gather(const torch::Tensor& input, const std::vector<int64_t>& group, int64_t dim) = 0;
  virtual torch::Tensor reduce_scatter(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group, int64_t dim) = 0;
  virtual void send(const torch::Tensor& input, int64_t peer) = 0;
  virtual torch::Tensor recv_like(const torch::Tensor& like, int64_t peer) = 0;
};

class SingleProcessCollectives final : public Collectives {
 public:
  int64_t rank() const override { return 0; }
  int64_t world_size() const override { return 1; }
  void barrier() override {}
  torch::Tensor all_reduce(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group) override;
  torch::Tensor all_gather(const torch::Tensor& input, const std::vector<int64_t>& group, int64_t dim) override;
  torch::Tensor reduce_scatter(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group, int64_t dim) override;
  void send(const torch::Tensor& input, int64_t peer) override;
  torch::Tensor recv_like(const torch::Tensor& like, int64_t peer) override;
};

}  // namespace cverl::distributed
