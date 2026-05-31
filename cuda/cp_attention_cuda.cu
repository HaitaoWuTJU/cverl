#include "cverl/distributed/cp_attention_cuda.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace cverl::distributed {
namespace {

constexpr int kCpAttentionThreads = 128;

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

bool is_supported_attention_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

void require_attention_cuda_contiguous(const torch::Tensor& t, const char* name) {
  if (!t.defined() || !t.is_cuda() || !t.is_contiguous() || !is_supported_attention_dtype(t.scalar_type())) {
    throw std::invalid_argument(std::string(name) + " must be contiguous CUDA float32/float16/bfloat16");
  }
}

void require_same_attention_dtype(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
  if (a.scalar_type() != b.scalar_type()) {
    throw std::invalid_argument(std::string(a_name) + " and " + b_name + " must use the same dtype");
  }
}

void require_f32_cuda_contiguous(const torch::Tensor& t, const char* name) {
  if (!t.defined() || !t.is_cuda() || !t.is_contiguous() || t.scalar_type() != torch::kFloat32) {
    throw std::invalid_argument(std::string(name) + " must be contiguous CUDA float32");
  }
}

void require_f32_cuda_contiguous_lse(const torch::Tensor& t,
                                     const torch::Tensor& query_local,
                                     const char* name) {
  require_f32_cuda_contiguous(t, name);
  if (t.dim() != 3 || t.size(0) != query_local.size(0) || t.size(1) != query_local.size(1) ||
      t.size(2) != query_local.size(2)) {
    throw std::invalid_argument(std::string(name) + " must be contiguous CUDA float32 [B,H,Tq]");
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
  if (key_ring.device() != query_local.device() || value_ring.device() != query_local.device()) {
    throw std::invalid_argument("CP attention CUDA tensors must be on the same CUDA device");
  }
  if (key_ring.size(0) != B || value_ring.size(0) != B || key_ring.size(1) != H || value_ring.size(1) != H ||
      value_ring.size(2) != Tk || key_ring.size(3) != D ||
      Tk != shard_size * static_cast<int64_t>(key_begin_positions.size())) {
    throw std::invalid_argument("CP attention CUDA shape mismatch");
  }
}

torch::Tensor positions_tensor(const std::vector<int64_t>& key_begin_positions, torch::Device device) {
  return torch::tensor(key_begin_positions, torch::TensorOptions().dtype(torch::kLong))
      .to(torch::TensorOptions().dtype(torch::kLong).device(device))
      .contiguous();
}

__device__ float block_reduce_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ float block_reduce_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t>
__global__ void cp_ring_attention_forward_kernel(const scalar_t* __restrict__ q,
                                                 const scalar_t* __restrict__ k,
                                                 const scalar_t* __restrict__ v,
                                                 const int64_t* __restrict__ key_begin_positions,
                                                 scalar_t* __restrict__ out,
                                                 float* __restrict__ lse,
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

  __shared__ float shared_row_max;
  __shared__ float shared_row_sum;
  __shared__ float reduction[kCpAttentionThreads];

  float local_row_max = -INFINITY;
  if (valid_query) {
    const int total_keys = num_blocks * shard_size;
    for (int flat = threadIdx.x; flat < total_keys; flat += blockDim.x) {
      const int block = flat / shard_size;
      const int local = flat - block * shard_size;
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      const int k_pos = key_begin + local;
      if (k_pos <= q_pos && k_pos < original_sequence_length) {
        const int kt = block * shard_size + local;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[kv_bh_base_k + kt * D + d]);
        }
        local_row_max = fmaxf(local_row_max, score * scale);
      }
    }
  }
  const float row_max = block_reduce_max(local_row_max, reduction);
  if (threadIdx.x == 0) {
    shared_row_max = row_max;
  }
  __syncthreads();

  float local_row_sum = 0.0f;
  if (valid_query && isfinite(shared_row_max)) {
    const int total_keys = num_blocks * shard_size;
    for (int flat = threadIdx.x; flat < total_keys; flat += blockDim.x) {
      const int block = flat / shard_size;
      const int local = flat - block * shard_size;
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      const int k_pos = key_begin + local;
      if (k_pos <= q_pos && k_pos < original_sequence_length) {
        const int kt = block * shard_size + local;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[kv_bh_base_k + kt * D + d]);
        }
        local_row_sum += expf(score * scale - shared_row_max);
      }
    }
  }
  const float row_sum = block_reduce_sum(local_row_sum, reduction);
  if (threadIdx.x == 0) {
    shared_row_sum = row_sum;
    lse[row] = isfinite(shared_row_max) ? shared_row_max + logf(fmaxf(row_sum, 1.0e-20f)) : -INFINITY;
  }
  __syncthreads();

  if (D <= blockDim.x && V <= blockDim.x) {
    __shared__ float shared_score;
    __shared__ float acc_shared[kCpAttentionThreads];

    if (threadIdx.x < V) {
      acc_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    const bool valid_row = valid_query && isfinite(shared_row_max);
    for (int block = 0; block < num_blocks; ++block) {
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      for (int local = 0; local < shard_size; ++local) {
        const int k_pos = key_begin + local;
        const bool valid_key = valid_row && k_pos <= q_pos && k_pos < original_sequence_length;
        const int kt = block * shard_size + local;
        float score_part = 0.0f;
        if (valid_key && threadIdx.x < D) {
          score_part =
              static_cast<float>(q[q_base + threadIdx.x]) *
              static_cast<float>(k[kv_bh_base_k + kt * D + threadIdx.x]);
        }
        const float score = block_reduce_sum(score_part, reduction);
        if (threadIdx.x == 0) {
          shared_score = score;
        }
        __syncthreads();

        if (valid_key && threadIdx.x < V) {
          const float prob = expf(shared_score * scale - shared_row_max) /
                             fmaxf(shared_row_sum, 1.0e-20f);
          acc_shared[threadIdx.x] += prob * static_cast<float>(v[kv_bh_base_v + kt * V + threadIdx.x]);
        }
        __syncthreads();
      }
    }

    if (threadIdx.x < V) {
      const float value = valid_row ? acc_shared[threadIdx.x] : 0.0f;
      out[out_base + threadIdx.x] = static_cast<scalar_t>(value);
    }
    return;
  }

  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    if (!valid_query || !isfinite(shared_row_max)) {
      out[out_base + j] = static_cast<scalar_t>(0.0f);
      continue;
    }
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
          score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[kv_bh_base_k + kt * D + d]);
        }
        const float prob_num = expf(score * scale - shared_row_max);
        acc += prob_num * static_cast<float>(v[kv_bh_base_v + kt * V + j]);
      }
    }
    out[out_base + j] = static_cast<scalar_t>(acc / fmaxf(shared_row_sum, 1.0e-20f));
  }
}

template <typename scalar_t>
__global__ void cp_ring_attention_backward_query_dot_kernel(const scalar_t* __restrict__ grad_out,
                                                            const scalar_t* __restrict__ q,
                                                            const scalar_t* __restrict__ k,
                                                            const scalar_t* __restrict__ v,
                                                            const float* __restrict__ lse,
                                                            const int64_t* __restrict__ key_begin_positions,
                                                            float* __restrict__ query_dot,
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
  const float row_lse = lse[row];

  __shared__ float reduction[kCpAttentionThreads];

  float local_dot = 0.0f;
  if (q_pos < original_sequence_length && isfinite(row_lse)) {
    const int total_keys = num_blocks * shard_size;
    for (int flat = threadIdx.x; flat < total_keys; flat += blockDim.x) {
      const int block = flat / shard_size;
      const int local = flat - block * shard_size;
      const int key_begin = static_cast<int>(key_begin_positions[block]);
      const int k_pos = key_begin + local;
      if (k_pos <= q_pos && k_pos < original_sequence_length) {
        const int kt = block * shard_size + local;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
          score += static_cast<float>(q[q_base + d]) * static_cast<float>(k[kv_bh_base_k + kt * D + d]);
        }
        const float prob = expf(score * scale - row_lse);
        float dot_go_v = 0.0f;
        for (int j = 0; j < V; ++j) {
          dot_go_v += static_cast<float>(grad_out[go_base + j]) * static_cast<float>(v[kv_bh_base_v + kt * V + j]);
        }
        local_dot += prob * dot_go_v;
      }
    }
  }

  const float dot = block_reduce_sum(local_dot, reduction);
  if (threadIdx.x == 0) {
    query_dot[row] = dot;
  }
}

template <typename scalar_t>
__global__ void cp_ring_attention_backward_q_kernel(const scalar_t* __restrict__ grad_out,
                                                    const scalar_t* __restrict__ q,
                                                    const scalar_t* __restrict__ k,
                                                    const scalar_t* __restrict__ v,
                                                    const float* __restrict__ lse,
                                                    const float* __restrict__ query_dot,
                                                    const int64_t* __restrict__ key_begin_positions,
                                                    scalar_t* __restrict__ dq,
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
  const float row_lse = lse[row];
  const float dot_go_o = query_dot[row];

  if (q_pos >= original_sequence_length || !isfinite(row_lse)) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      dq[q_base + d] = static_cast<scalar_t>(0.0f);
    }
    return;
  }

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
          score += static_cast<float>(q[q_base + kd]) * static_cast<float>(k[kv_bh_base_k + kt * D + kd]);
        }
        float dot_go_v = 0.0f;
        for (int j = 0; j < V; ++j) {
          dot_go_v += static_cast<float>(grad_out[go_base + j]) * static_cast<float>(v[kv_bh_base_v + kt * V + j]);
        }
        const float prob = expf(score * scale - row_lse);
        const float ds = prob * (dot_go_v - dot_go_o);
        acc += ds * static_cast<float>(k[kv_bh_base_k + kt * D + d]);
      }
    }
    dq[q_base + d] = static_cast<scalar_t>(acc * scale);
  }
}

template <typename scalar_t>
__global__ void cp_ring_attention_backward_kv_kernel(const scalar_t* __restrict__ grad_out,
                                                     const scalar_t* __restrict__ q,
                                                     const scalar_t* __restrict__ k,
                                                     const scalar_t* __restrict__ v,
                                                     const float* __restrict__ lse,
                                                     const float* __restrict__ query_dot,
                                                     const int64_t* __restrict__ key_begin_positions,
                                                     scalar_t* __restrict__ dk,
                                                     scalar_t* __restrict__ dv,
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
      dk[k_base + d] = static_cast<scalar_t>(0.0f);
    }
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      dv[v_base + j] = static_cast<scalar_t>(0.0f);
    }
    return;
  }

  if (D <= blockDim.x && V <= blockDim.x) {
    __shared__ float reduction[kCpAttentionThreads];
    __shared__ float shared_score;
    __shared__ float shared_dot_go_v;
    __shared__ float grad_k_shared[kCpAttentionThreads];
    __shared__ float grad_v_shared[kCpAttentionThreads];

    if (threadIdx.x < D) {
      grad_k_shared[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x < V) {
      grad_v_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int tq = 0; tq < Tq; ++tq) {
      const int q_pos = query_begin + tq;
      const float row_lse = lse[(b * H + h) * Tq + tq];
      const bool valid = q_pos >= key_pos && q_pos < original_sequence_length && isfinite(row_lse);
      const int q_base = q_bh_base + tq * D;
      const int go_base = go_bh_base + tq * V;

      float score_part = 0.0f;
      if (valid && threadIdx.x < D) {
        score_part = static_cast<float>(q[q_base + threadIdx.x]) * static_cast<float>(k[k_base + threadIdx.x]);
      }
      const float score = block_reduce_sum(score_part, reduction);
      if (threadIdx.x == 0) {
        shared_score = score;
      }
      __syncthreads();

      float dot_part = 0.0f;
      if (valid && threadIdx.x < V) {
        dot_part = static_cast<float>(grad_out[go_base + threadIdx.x]) *
                   static_cast<float>(v[v_base + threadIdx.x]);
      }
      const float dot_go_v = block_reduce_sum(dot_part, reduction);
      if (threadIdx.x == 0) {
        shared_dot_go_v = dot_go_v;
      }
      __syncthreads();

      if (valid) {
        const float prob = expf(shared_score * scale - row_lse);
        if (threadIdx.x < D) {
          const float dot_go_o = query_dot[(b * H + h) * Tq + tq];
          grad_k_shared[threadIdx.x] +=
              prob * (shared_dot_go_v - dot_go_o) * static_cast<float>(q[q_base + threadIdx.x]) * scale;
        }
        if (threadIdx.x < V) {
          grad_v_shared[threadIdx.x] += prob * static_cast<float>(grad_out[go_base + threadIdx.x]);
        }
      }
      __syncthreads();
    }

    if (threadIdx.x < D) {
      dk[k_base + threadIdx.x] = static_cast<scalar_t>(grad_k_shared[threadIdx.x]);
    }
    if (threadIdx.x < V) {
      dv[v_base + threadIdx.x] = static_cast<scalar_t>(grad_v_shared[threadIdx.x]);
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
      const float row_lse = lse[(b * H + h) * Tq + tq];
      if (!isfinite(row_lse)) {
        continue;
      }
      const int q_base = q_bh_base + tq * D;
      const int go_base = go_bh_base + tq * V;
      const float dot_go_o = query_dot[(b * H + h) * Tq + tq];
      float score = 0.0f;
      for (int kd = 0; kd < D; ++kd) {
        score += static_cast<float>(q[q_base + kd]) * static_cast<float>(k[k_base + kd]);
      }
      float dot_go_v = 0.0f;
      for (int j = 0; j < V; ++j) {
        dot_go_v += static_cast<float>(grad_out[go_base + j]) * static_cast<float>(v[v_base + j]);
      }
      const float prob = expf(score * scale - row_lse);
      grad_k += prob * (dot_go_v - dot_go_o) * static_cast<float>(q[q_base + d]) * scale;
    }
    dk[k_base + d] = static_cast<scalar_t>(grad_k);
  }

  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    float grad_v = 0.0f;
    for (int tq = 0; tq < Tq; ++tq) {
      const int q_pos = query_begin + tq;
      if (q_pos < key_pos || q_pos >= original_sequence_length) {
        continue;
      }
      const float row_lse = lse[(b * H + h) * Tq + tq];
      if (!isfinite(row_lse)) {
        continue;
      }
      const int q_base = q_bh_base + tq * D;
      const int go_base = go_bh_base + tq * V;
      float score = 0.0f;
      for (int kd = 0; kd < D; ++kd) {
        score += static_cast<float>(q[q_base + kd]) * static_cast<float>(k[k_base + kd]);
      }
      const float prob = expf(score * scale - row_lse);
      grad_v += prob * static_cast<float>(grad_out[go_base + j]);
    }
    dv[v_base + j] = static_cast<scalar_t>(grad_v);
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
  return cp_ring_attention_cuda_forward_with_lse(
      query_local, key_ring, value_ring, key_begin_positions, query_begin, original_sequence_length, shard_size, scale)
      .at(0);
}

std::vector<torch::Tensor> cp_ring_attention_cuda_forward_with_lse(const torch::Tensor& query_local,
                                                                   const torch::Tensor& key_ring,
                                                                   const torch::Tensor& value_ring,
                                                                   const std::vector<int64_t>& key_begin_positions,
                                                                   int64_t query_begin,
                                                                   int64_t original_sequence_length,
                                                                   int64_t shard_size,
                                                                   double scale) {
  require_attention_cuda_contiguous(query_local, "query_local");
  require_attention_cuda_contiguous(key_ring, "key_ring");
  require_attention_cuda_contiguous(value_ring, "value_ring");
  require_same_attention_dtype(query_local, key_ring, "query_local", "key_ring");
  require_same_attention_dtype(query_local, value_ring, "query_local", "value_ring");
  validate_shapes(query_local, key_ring, value_ring, key_begin_positions, query_begin, original_sequence_length, shard_size);
  const int64_t B = query_local.size(0);
  const int64_t H = query_local.size(1);
  const int64_t Tq = query_local.size(2);
  const int64_t D = query_local.size(3);
  const int64_t Tk = key_ring.size(2);
  const int64_t V = value_ring.size(3);

  auto positions = positions_tensor(key_begin_positions, query_local.device());
  auto out = torch::empty({B, H, Tq, V}, query_local.options());
  auto lse = torch::empty({B, H, Tq},
                          torch::TensorOptions().dtype(torch::kFloat32).device(query_local.device()));

  constexpr int threads = kCpAttentionThreads;
  const int rows = static_cast<int>(B * H * Tq);
  auto stream = at::cuda::getCurrentCUDAStream(query_local.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query_local.scalar_type(), "cp_ring_attention_forward", [&] {
    cp_ring_attention_forward_kernel<scalar_t><<<rows, threads, 0, stream>>>(
        query_local.data_ptr<scalar_t>(),
        key_ring.data_ptr<scalar_t>(),
        value_ring.data_ptr<scalar_t>(),
        positions.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        lse.data_ptr<float>(),
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
  });
  check_cuda(cudaGetLastError(), "cp_ring_attention_forward_kernel");
  return {out, lse};
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
  const auto forward = cp_ring_attention_cuda_forward_with_lse(
      query_local, key_ring, value_ring, key_begin_positions, query_begin, original_sequence_length, shard_size, scale);
  return cp_ring_attention_cuda_backward_with_lse(grad_out,
                                                 query_local,
                                                 key_ring,
                                                 value_ring,
                                                 forward.at(1),
                                                 key_begin_positions,
                                                 query_begin,
                                                 original_sequence_length,
                                                 shard_size,
                                                 scale);
}

std::vector<torch::Tensor> cp_ring_attention_cuda_backward_with_lse(const torch::Tensor& grad_out,
                                                                    const torch::Tensor& query_local,
                                                                    const torch::Tensor& key_ring,
                                                                    const torch::Tensor& value_ring,
                                                                    const torch::Tensor& query_lse,
                                                                    const std::vector<int64_t>& key_begin_positions,
                                                                    int64_t query_begin,
                                                                    int64_t original_sequence_length,
                                                                    int64_t shard_size,
                                                                    double scale) {
  require_attention_cuda_contiguous(grad_out, "grad_out");
  require_attention_cuda_contiguous(query_local, "query_local");
  require_attention_cuda_contiguous(key_ring, "key_ring");
  require_attention_cuda_contiguous(value_ring, "value_ring");
  require_same_attention_dtype(query_local, key_ring, "query_local", "key_ring");
  require_same_attention_dtype(query_local, value_ring, "query_local", "value_ring");
  require_same_attention_dtype(query_local, grad_out, "query_local", "grad_out");
  require_f32_cuda_contiguous_lse(query_lse, query_local, "query_lse");
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
  auto query_dot = torch::empty({B, H, Tq},
                                torch::TensorOptions().dtype(torch::kFloat32).device(query_local.device()));

  constexpr int threads = kCpAttentionThreads;
  const int q_rows = static_cast<int>(B * H * Tq);
  auto stream = at::cuda::getCurrentCUDAStream(query_local.get_device()).stream();
  const int kv_rows = static_cast<int>(B * H * Tk);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, query_local.scalar_type(), "cp_ring_attention_backward", [&] {
    cp_ring_attention_backward_query_dot_kernel<scalar_t><<<q_rows, threads, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        query_local.data_ptr<scalar_t>(),
        key_ring.data_ptr<scalar_t>(),
        value_ring.data_ptr<scalar_t>(),
        query_lse.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        query_dot.data_ptr<float>(),
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
    check_cuda(cudaGetLastError(), "cp_ring_attention_backward_query_dot_kernel");

    cp_ring_attention_backward_q_kernel<scalar_t><<<q_rows, threads, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        query_local.data_ptr<scalar_t>(),
        key_ring.data_ptr<scalar_t>(),
        value_ring.data_ptr<scalar_t>(),
        query_lse.data_ptr<float>(),
        query_dot.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        dq.data_ptr<scalar_t>(),
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

    cp_ring_attention_backward_kv_kernel<scalar_t><<<kv_rows, threads, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        query_local.data_ptr<scalar_t>(),
        key_ring.data_ptr<scalar_t>(),
        value_ring.data_ptr<scalar_t>(),
        query_lse.data_ptr<float>(),
        query_dot.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        dk.data_ptr<scalar_t>(),
        dv.data_ptr<scalar_t>(),
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
  });
  check_cuda(cudaGetLastError(), "cp_ring_attention_backward_kv_kernel");
  return {dq, dk, dv};
}

}  // namespace cverl::distributed
