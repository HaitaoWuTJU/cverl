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

void validate_shapes(const torch::Tensor& query_local,
                     const torch::Tensor& key_ring,
                     const torch::Tensor& value_ring,
                     const std::vector<int64_t>& key_begin_positions,
                     int64_t query_begin,
                     int64_t original_sequence_length,
                     int64_t shard_size) {
  if (query_local.dim() != 4 || key_ring.dim() != 4 || value_ring.dim() != 4) {
    throw std::invalid_argument("CP attention CUDA tensors must be [B,H,T,D]");
  }
  if (query_begin < 0 || original_sequence_length < 0 || shard_size <= 0 || key_begin_positions.empty()) {
    throw std::invalid_argument("invalid CP attention CUDA sequence metadata");
  }
  const int64_t B = query_local.size(0);
  const int64_t H = query_local.size(1);
  const int64_t D = query_local.size(3);
  const int64_t Tk = key_ring.size(2);
  if (key_ring.size(0) != B || value_ring.size(0) != B || key_ring.size(1) != H || value_ring.size(1) != H ||
      value_ring.size(2) != Tk || key_ring.size(3) != D ||
      Tk < shard_size * static_cast<int64_t>(key_begin_positions.size())) {
    throw std::invalid_argument("CP attention CUDA shape mismatch");
  }
}

torch::Tensor positions_tensor(const std::vector<int64_t>& key_begin_positions, torch::Device device) {
  return torch::tensor(key_begin_positions, torch::TensorOptions().dtype(torch::kLong)).to(device).contiguous();
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

__global__ void cp_ring_attention_backward_q_kernel(const float* __restrict__ grad_out,
                                                    const float* __restrict__ q,
                                                    const float* __restrict__ k,
                                                    const float* __restrict__ v,
                                                    const int64_t* __restrict__ key_begin_positions,
                                                    float* __restrict__ dq,
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
  const int q_base = (((b * H + h) * Tq + t) * D);
  const int go_base = (((b * H + h) * Tq + t) * V);
  const int kv_bh_base_k = ((b * H + h) * Tk) * D;
  const int kv_bh_base_v = ((b * H + h) * Tk) * V;

  if (q_pos >= original_sequence_length) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      dq[q_base + d] = 0.0f;
    }
    return;
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
      row_max = fmaxf(row_max, score * scale);
    }
  }

  float row_sum = 0.0f;
  float dot_go_o = 0.0f;
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
      float dot_go_v = 0.0f;
      for (int j = 0; j < V; ++j) {
        dot_go_v += grad_out[go_base + j] * v[kv_bh_base_v + kt * V + j];
      }
      dot_go_o += prob_num * dot_go_v;
    }
  }
  row_sum = fmaxf(row_sum, 1.0e-20f);
  dot_go_o /= row_sum;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
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
        for (int kd = 0; kd < D; ++kd) {
          score += q[q_base + kd] * k[kv_bh_base_k + kt * D + kd];
        }
        float dot_go_v = 0.0f;
        for (int j = 0; j < V; ++j) {
          dot_go_v += grad_out[go_base + j] * v[kv_bh_base_v + kt * V + j];
        }
        const float prob = expf(score * scale - row_max) / row_sum;
        const float ds = prob * (dot_go_v - dot_go_o);
        acc += ds * k[kv_bh_base_k + kt * D + d];
      }
    }
    dq[q_base + d] = acc * scale;
  }
}

__global__ void cp_ring_attention_backward_kv_kernel(const float* __restrict__ grad_out,
                                                     const float* __restrict__ q,
                                                     const float* __restrict__ k,
                                                     const float* __restrict__ v,
                                                     const int64_t* __restrict__ key_begin_positions,
                                                     float* __restrict__ dk,
                                                     float* __restrict__ dv,
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
  const int kv_row = blockIdx.x;
  const int kt = kv_row % Tk;
  const int bh = kv_row / Tk;
  const int h = bh % H;
  const int b = bh / H;
  const int block = kt / shard_size;
  const int local = kt - block * shard_size;
  if (block >= num_blocks) {
    return;
  }
  const int key_pos = static_cast<int>(key_begin_positions[block]) + local;
  const int k_base = (((b * H + h) * Tk + kt) * D);
  const int v_base = (((b * H + h) * Tk + kt) * V);
  const int q_bh_base = ((b * H + h) * Tq) * D;
  const int go_bh_base = ((b * H + h) * Tq) * V;

  if (key_pos >= original_sequence_length) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      dk[k_base + d] = 0.0f;
    }
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      dv[v_base + j] = 0.0f;
    }
    return;
  }

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float grad_k = 0.0f;
    for (int tq = 0; tq < Tq; ++tq) {
      const int q_pos = query_begin + tq;
      if (q_pos < key_pos || q_pos >= original_sequence_length) {
        continue;
      }
      const int q_base = q_bh_base + tq * D;
      const int go_base = go_bh_base + tq * V;
      float row_max = -CUDART_INF_F;
      for (int b2 = 0; b2 < num_blocks; ++b2) {
        const int key_begin2 = static_cast<int>(key_begin_positions[b2]);
        for (int l2 = 0; l2 < shard_size; ++l2) {
          const int kp2 = key_begin2 + l2;
          if (kp2 > q_pos || kp2 >= original_sequence_length) {
            continue;
          }
          const int kt2 = b2 * shard_size + l2;
          float score = 0.0f;
          for (int kd = 0; kd < D; ++kd) {
            score += q[q_base + kd] * k[((b * H + h) * Tk + kt2) * D + kd];
          }
          row_max = fmaxf(row_max, score * scale);
        }
      }
      float row_sum = 0.0f;
      float dot_go_o = 0.0f;
      for (int b2 = 0; b2 < num_blocks; ++b2) {
        const int key_begin2 = static_cast<int>(key_begin_positions[b2]);
        for (int l2 = 0; l2 < shard_size; ++l2) {
          const int kp2 = key_begin2 + l2;
          if (kp2 > q_pos || kp2 >= original_sequence_length) {
            continue;
          }
          const int kt2 = b2 * shard_size + l2;
          float score = 0.0f;
          for (int kd = 0; kd < D; ++kd) {
            score += q[q_base + kd] * k[((b * H + h) * Tk + kt2) * D + kd];
          }
          const float prob_num = expf(score * scale - row_max);
          row_sum += prob_num;
          float dot_go_v = 0.0f;
          for (int j = 0; j < V; ++j) {
            dot_go_v += grad_out[go_base + j] * v[((b * H + h) * Tk + kt2) * V + j];
          }
          dot_go_o += prob_num * dot_go_v;
        }
      }
      row_sum = fmaxf(row_sum, 1.0e-20f);
      dot_go_o /= row_sum;
      float score = 0.0f;
      for (int kd = 0; kd < D; ++kd) {
        score += q[q_base + kd] * k[k_base + kd];
      }
      float dot_go_v = 0.0f;
      for (int j = 0; j < V; ++j) {
        dot_go_v += grad_out[go_base + j] * v[v_base + j];
      }
      const float prob = expf(score * scale - row_max) / row_sum;
      grad_k += prob * (dot_go_v - dot_go_o) * q[q_base + d] * scale;
    }
    dk[k_base + d] = grad_k;
  }

  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    float grad_v = 0.0f;
    for (int tq = 0; tq < Tq; ++tq) {
      const int q_pos = query_begin + tq;
      if (q_pos < key_pos || q_pos >= original_sequence_length) {
        continue;
      }
      const int q_base = q_bh_base + tq * D;
      const int go_base = go_bh_base + tq * V;
      float row_max = -CUDART_INF_F;
      for (int b2 = 0; b2 < num_blocks; ++b2) {
        const int key_begin2 = static_cast<int>(key_begin_positions[b2]);
        for (int l2 = 0; l2 < shard_size; ++l2) {
          const int kp2 = key_begin2 + l2;
          if (kp2 > q_pos || kp2 >= original_sequence_length) {
            continue;
          }
          const int kt2 = b2 * shard_size + l2;
          float score = 0.0f;
          for (int kd = 0; kd < D; ++kd) {
            score += q[q_base + kd] * k[((b * H + h) * Tk + kt2) * D + kd];
          }
          row_max = fmaxf(row_max, score * scale);
        }
      }
      float row_sum = 0.0f;
      for (int b2 = 0; b2 < num_blocks; ++b2) {
        const int key_begin2 = static_cast<int>(key_begin_positions[b2]);
        for (int l2 = 0; l2 < shard_size; ++l2) {
          const int kp2 = key_begin2 + l2;
          if (kp2 > q_pos || kp2 >= original_sequence_length) {
            continue;
          }
          const int kt2 = b2 * shard_size + l2;
          float score = 0.0f;
          for (int kd = 0; kd < D; ++kd) {
            score += q[q_base + kd] * k[((b * H + h) * Tk + kt2) * D + kd];
          }
          row_sum += expf(score * scale - row_max);
        }
      }
      float score = 0.0f;
      for (int kd = 0; kd < D; ++kd) {
        score += q[q_base + kd] * k[k_base + kd];
      }
      const float prob = expf(score * scale - row_max) / fmaxf(row_sum, 1.0e-20f);
      grad_v += prob * grad_out[go_base + j];
    }
    dv[v_base + j] = grad_v;
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
  validate_shapes(query_local, key_ring, value_ring, key_begin_positions, query_begin, original_sequence_length, shard_size);
  const int64_t B = query_local.size(0);
  const int64_t H = query_local.size(1);
  const int64_t Tq = query_local.size(2);
  const int64_t D = query_local.size(3);
  const int64_t Tk = key_ring.size(2);
  const int64_t V = value_ring.size(3);

  auto positions = positions_tensor(key_begin_positions, query_local.device());
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

std::vector<torch::Tensor> cp_ring_attention_cuda_backward(const torch::Tensor& grad_out,
                                                           const torch::Tensor& query_local,
                                                           const torch::Tensor& key_ring,
                                                           const torch::Tensor& value_ring,
                                                           const std::vector<int64_t>& key_begin_positions,
                                                           int64_t query_begin,
                                                           int64_t original_sequence_length,
                                                           int64_t shard_size,
                                                           double scale) {
  require_f32_cuda_contiguous(grad_out, "grad_out");
  require_f32_cuda_contiguous(query_local, "query_local");
  require_f32_cuda_contiguous(key_ring, "key_ring");
  require_f32_cuda_contiguous(value_ring, "value_ring");
  validate_shapes(query_local, key_ring, value_ring, key_begin_positions, query_begin, original_sequence_length, shard_size);
  if (grad_out.dim() != 4 || grad_out.size(0) != query_local.size(0) || grad_out.size(1) != query_local.size(1) ||
      grad_out.size(2) != query_local.size(2) || grad_out.size(3) != value_ring.size(3)) {
    throw std::invalid_argument("CP attention CUDA grad_out shape mismatch");
  }
  const int64_t B = query_local.size(0);
  const int64_t H = query_local.size(1);
  const int64_t Tq = query_local.size(2);
  const int64_t D = query_local.size(3);
  const int64_t Tk = key_ring.size(2);
  const int64_t V = value_ring.size(3);
  auto positions = positions_tensor(key_begin_positions, query_local.device());
  auto dq = torch::empty_like(query_local);
  auto dk = torch::empty_like(key_ring);
  auto dv = torch::empty_like(value_ring);

  constexpr int threads = 128;
  const int q_rows = static_cast<int>(B * H * Tq);
  cp_ring_attention_backward_q_kernel<<<q_rows, threads>>>(
      grad_out.data_ptr<float>(),
      query_local.data_ptr<float>(),
      key_ring.data_ptr<float>(),
      value_ring.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      dq.data_ptr<float>(),
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
  check_cuda(cudaGetLastError(), "cp_ring_attention_backward_q_kernel");

  const int kv_rows = static_cast<int>(B * H * Tk);
  cp_ring_attention_backward_kv_kernel<<<kv_rows, threads>>>(
      grad_out.data_ptr<float>(),
      query_local.data_ptr<float>(),
      key_ring.data_ptr<float>(),
      value_ring.data_ptr<float>(),
      positions.data_ptr<int64_t>(),
      dk.data_ptr<float>(),
      dv.data_ptr<float>(),
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
  check_cuda(cudaGetLastError(), "cp_ring_attention_backward_kv_kernel");
  return {dq, dk, dv};
}

}  // namespace cverl::distributed
