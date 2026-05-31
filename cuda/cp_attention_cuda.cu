#include "cverl/distributed/cp_attention_cuda.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace cverl::distributed {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void require_f32_cuda_contiguous(const torch::Tensor& t, const char* name) {
  if (!t.defined() || !t.is_cuda() || !t.is_contiguous() || t.scalar_type() != torch::kFloat32) {
    throw std::invalid_argument(std::string(name) + " must be contiguous CUDA float32");
  }
}

__global__ void cp_ring_attention_forward_kernel(const float* __restrict__ q,
                                                 const float* __restrict__ k,
                                                 const float* __restrict__ v,
                                                 const int64_t* __restrict__ key_begin_positions,
                                                 float* __restrict__ out,
                                                 int B,
                                                 int H,
                                                 int Tq,
                                                 int Tk,
                                                 int D,
                                                 int V,
                                                 int num_blocks,
                                                 int query_begin,
                                                 int original_sequence_length,
                                                 int shard_size,
                                                 float scale) {
  const int row = blockIdx.x;
  const int t = row % Tq;
  const int bh = row / Tq;
  const int h = bh % H;
  const int b = bh / H;
  const int q_pos = query_begin + t;
  const bool valid_query = q_pos < original_sequence_length;
  const int q_base = (((b * H + h) * Tq + t) * D);
  const int out_base = (((b * H + h) * Tq + t) * V);
  const int kv_bh_base_k = ((b * H + h) * Tk) * D;
  const int kv_bh_base_v = ((b * H + h) * Tk) * V;

  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    if (!valid_query) {
      out[out_base + j] = 0.0f;
      continue;
    }

    float row_max = -CUDART_INF_F;
    for (int block = 0; block < num_blocks; ++block) {
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      for (int local = 0; local < shard_size; ++local) {
        const int k_pos = key_begin + local;
        if (k_pos > q_pos || k_pos >= original_sequence_length) {
          continue;
        }
        const int kt = block * shard_size + local;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += q[q_base + d] * k[kv_bh_base_k + kt * D + d];
        }
        score *= scale;
        row_max = fmaxf(row_max, score);
      }
    }

    if (!isfinite(row_max)) {
      out[out_base + j] = 0.0f;
      continue;
    }

    float row_sum = 0.0f;
    float acc = 0.0f;
    for (int block = 0; block < num_blocks; ++block) {
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      for (int local = 0; local < shard_size; ++local) {
        const int k_pos = key_begin + local;
        if (k_pos > q_pos || k_pos >= original_sequence_length) {
          continue;
        }
        const int kt = block * shard_size + local;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += q[q_base + d] * k[kv_bh_base_k + kt * D + d];
        }
        const float prob_num = expf(score * scale - row_max);
        row_sum += prob_num;
        acc += prob_num * v[kv_bh_base_v + kt * V + j];
      }
    }
    out[out_base + j] = acc / fmaxf(row_sum, 1.0e-20f);
  }
}

}  // namespace

bool cp_attention_cuda_available() {
  return true;
}

torch::Tensor cp_ring_attention_cuda_forward(const torch::Tensor& query_local,
                                             const torch::Tensor& key_ring,
                                             const torch::Tensor& value_ring,
                                             const std::vector<int64_t>& key_begin_positions,
                                             int64_t query_begin,
                                             int64_t original_sequence_length,
                                             int64_t shard_size,
                                             double scale) {
  require_f32_cuda_contiguous(query_local, "query_local");
  require_f32_cuda_contiguous(key_ring, "key_ring");
  require_f32_cuda_contiguous(value_ring, "value_ring");
  if (query_local.dim() != 4 || key_ring.dim() != 4 || value_ring.dim() != 4) {
    throw std::invalid_argument("CP attention CUDA tensors must be [B,H,T,D]");
  }
  if (query_begin < 0 || original_sequence_length < 0 || shard_size <= 0 || key_begin_positions.empty()) {
    throw std::invalid_argument("invalid CP attention CUDA sequence metadata");
  }
  const int64_t B = query_local.size(0);
  const int64_t H = query_local.size(1);
  const int64_t Tq = query_local.size(2);
  const int64_t D = query_local.size(3);
  const int64_t Tk = key_ring.size(2);
  const int64_t V = value_ring.size(3);
  if (key_ring.size(0) != B || value_ring.size(0) != B || key_ring.size(1) != H || value_ring.size(1) != H ||
      value_ring.size(2) != Tk || key_ring.size(3) != D ||
      Tk < shard_size * static_cast<int64_t>(key_begin_positions.size())) {
    throw std::invalid_argument("CP attention CUDA shape mismatch");
  }

  auto positions = torch::tensor(key_begin_positions, torch::TensorOptions().dtype(torch::kLong))
                       .to(query_local.device(), /*non_blocking=*/false)
                       .contiguous();
  auto out = torch::empty({B, H, Tq, V}, query_local.options());

  constexpr int threads = 128;
  const int rows = static_cast<int>(B * H * Tq);
  cp_ring_attention_forward_kernel<<<rows, threads>>>(
      query_local.data_ptr<float>(),
      key_ring.data_ptr<float>(),
      value_ring.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      static_cast<int>(B),
      static_cast<int>(H),
      static_cast<int>(Tq),
      static_cast<int>(Tk),
      static_cast<int>(D),
      static_cast<int>(V),
      static_cast<int>(key_begin_positions.size()),
      static_cast<int>(query_begin),
      static_cast<int>(original_sequence_length),
      static_cast<int>(shard_size),
      static_cast<float>(scale));
  check_cuda(cudaGetLastError(), "cp_ring_attention_forward_kernel");
  return out;
}

}  // namespace cverl::distributed
