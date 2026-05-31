#include "cverl/model/qwen_linear_attention_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cverl {
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

int value_tile_size(int value_dim) {
  const char* env = std::getenv("CVERL_LINEAR_ATTN_VALUE_TILE");
  int tile = (env == nullptr || *env == '\0') ? 32 : std::stoi(env);
  if (tile <= 0) {
    throw std::invalid_argument("CVERL_LINEAR_ATTN_VALUE_TILE must be positive");
  }
  return std::min(tile, value_dim);
}

__global__ void qwen_linear_attn_forward_kernel(const float* __restrict__ q,
                                                const float* __restrict__ k,
                                                const float* __restrict__ v,
                                                const float* __restrict__ beta,
                                                const float* __restrict__ g,
                                                float* __restrict__ out,
                                                float* __restrict__ states,
                                                bool save_states,
                                                int B,
                                                int H,
                                                int T,
                                                int K,
                                                int V) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  extern __shared__ float state[];
  int state_elems = K * V;
  for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
    state[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int state_base = ((((b * H + h) * T) * K) * V);
  for (int t = 0; t < T; ++t) {
    float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] *= et;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * V + j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * V + j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float o = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        o += state[kk * V + j] * q[q_base + t * K + kk];
      }
      out[v_base + t * V + j] = o;
    }
    if (save_states) {
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        states[state_base + t * state_elems + idx] = state[idx];
      }
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_forward_tiled_kernel(const float* __restrict__ q,
                                                      const float* __restrict__ k,
                                                      const float* __restrict__ v,
                                                      const float* __restrict__ beta,
                                                      const float* __restrict__ g,
                                                      float* __restrict__ out,
                                                      float* __restrict__ states,
                                                      bool save_states,
                                                      int B,
                                                      int H,
                                                      int T,
                                                      int K,
                                                      int V,
                                                      int value_tile) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  const int tile_begin = blockIdx.y * value_tile;
  const int tile_cols = value_tile < (V - tile_begin) ? value_tile : (V - tile_begin);
  if (tile_cols <= 0) {
    return;
  }

  extern __shared__ float state[];
  const int state_elems = K * tile_cols;
  for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
    state[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int state_base = ((((b * H + h) * T) * K) * V);
  for (int t = 0; t < T; ++t) {
    const float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] *= et;
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * tile_cols + local_j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * tile_cols + local_j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float o = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        o += state[kk * tile_cols + local_j] * q[q_base + t * K + kk];
      }
      out[v_base + t * V + j] = o;
    }
    if (save_states) {
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        const int kk = idx / tile_cols;
        const int local_j = idx - kk * tile_cols;
        states[state_base + t * K * V + kk * V + tile_begin + local_j] = state[idx];
      }
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_forward_checkpointed_kernel(const float* __restrict__ q,
                                                             const float* __restrict__ k,
                                                             const float* __restrict__ v,
                                                             const float* __restrict__ beta,
                                                             const float* __restrict__ g,
                                                             float* __restrict__ out,
                                                             float* __restrict__ checkpoints,
                                                             int checkpoint_interval,
                                                             int B,
                                                             int H,
                                                             int T,
                                                             int K,
                                                             int V,
                                                             int C) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  extern __shared__ float state[];
  int state_elems = K * V;
  for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
    state[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int checkpoint_base = (((b * H + h) * C) * K) * V;
  for (int t = 0; t < T; ++t) {
    if (t % checkpoint_interval == 0) {
      const int c = t / checkpoint_interval;
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        checkpoints[checkpoint_base + c * state_elems + idx] = state[idx];
      }
      __syncthreads();
    }

    float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] *= et;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * V + j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * V + j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float o = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        o += state[kk * V + j] * q[q_base + t * K + kk];
      }
      out[v_base + t * V + j] = o;
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_forward_checkpointed_tiled_kernel(const float* __restrict__ q,
                                                                   const float* __restrict__ k,
                                                                   const float* __restrict__ v,
                                                                   const float* __restrict__ beta,
                                                                   const float* __restrict__ g,
                                                                   float* __restrict__ out,
                                                                   float* __restrict__ checkpoints,
                                                                   int checkpoint_interval,
                                                                   int B,
                                                                   int H,
                                                                   int T,
                                                                   int K,
                                                                   int V,
                                                                   int C,
                                                                   int value_tile) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  const int tile_begin = blockIdx.y * value_tile;
  const int tile_cols = value_tile < (V - tile_begin) ? value_tile : (V - tile_begin);
  if (tile_cols <= 0) {
    return;
  }

  extern __shared__ float state[];
  const int state_elems = K * tile_cols;
  for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
    state[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int checkpoint_base = (((b * H + h) * C) * K) * V;
  for (int t = 0; t < T; ++t) {
    if (t % checkpoint_interval == 0) {
      const int c = t / checkpoint_interval;
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        const int kk = idx / tile_cols;
        const int local_j = idx - kk * tile_cols;
        checkpoints[checkpoint_base + c * K * V + kk * V + tile_begin + local_j] = state[idx];
      }
      __syncthreads();
    }

    const float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] *= et;
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * tile_cols + local_j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * tile_cols + local_j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float o = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        o += state[kk * tile_cols + local_j] * q[q_base + t * K + kk];
      }
      out[v_base + t * V + j] = o;
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_backward_kernel(const float* __restrict__ grad_out,
                                                 const float* __restrict__ q,
                                                 const float* __restrict__ k,
                                                 const float* __restrict__ v,
                                                 const float* __restrict__ beta,
                                                 const float* __restrict__ g,
                                                 const float* __restrict__ states,
                                                 float* __restrict__ dq,
                                                 float* __restrict__ dk,
                                                 float* __restrict__ dv,
                                                 float* __restrict__ dbeta,
                                                 float* __restrict__ dg,
                                                 int B,
                                                 int H,
                                                 int T,
                                                 int K,
                                                 int V) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  extern __shared__ float shared[];
  float* dstate = shared;
  float* dspre = shared + K * V;
  int state_elems = K * V;
  for (int idx = threadIdx.x; idx < state_elems * 2; idx += blockDim.x) {
    shared[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int state_base = ((((b * H + h) * T) * K) * V);
  for (int t = T - 1; t >= 0; --t) {
    const float et = expf(g[scalar_base + t]);

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        acc += states[state_base + t * state_elems + kk * V + j] * grad_out[v_base + t * V + j];
      }
      dq[q_base + t * K + kk] = acc;
    }
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      dstate[idx] += q[q_base + t * K + kk] * grad_out[v_base + t * V + j];
      dspre[idx] = dstate[idx];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float ddelta = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
      }
      dv[v_base + t * V + j] = ddelta * beta[scalar_base + t];
    }
    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * state_elems + kk * V + j];
        float s_pre = s_prev * et;
        float kv = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          float sp = (t == 0) ? 0.0f : states[state_base + (t - 1) * state_elems + kk2 * V + j] * et;
          kv += sp * k[q_base + t * K + kk2];
        }
        float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
        float ddelta = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
        }
        acc += dstate[kk * V + j] * delta - s_pre * ddelta * beta[scalar_base + t];
      }
      dk[q_base + t * K + kk] = acc;
    }
    if (threadIdx.x == 0) {
      float db = 0.0f;
      for (int j = 0; j < V; ++j) {
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * state_elems + kk * V + j];
          kv += s_prev * et * k[q_base + t * K + kk];
        }
        float ddelta = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
        }
        db += ddelta * (v[v_base + t * V + j] - kv);
      }
      dbeta[scalar_base + t] = db;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * state_elems + idx];
      float ddelta = 0.0f;
      for (int kk2 = 0; kk2 < K; ++kk2) {
        ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
      }
      dspre[idx] += -k[q_base + t * K + kk] * ddelta * beta[scalar_base + t];
      dstate[idx] = dspre[idx] * et;
      atomicAdd(&dg[scalar_base + t], dspre[idx] * s_prev * et);
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_backward_tiled_kernel(const float* __restrict__ grad_out,
                                                       const float* __restrict__ q,
                                                       const float* __restrict__ k,
                                                       const float* __restrict__ v,
                                                       const float* __restrict__ beta,
                                                       const float* __restrict__ g,
                                                       const float* __restrict__ states,
                                                       float* __restrict__ dq,
                                                       float* __restrict__ dk,
                                                       float* __restrict__ dv,
                                                       float* __restrict__ dbeta,
                                                       float* __restrict__ dg,
                                                       int B,
                                                       int H,
                                                       int T,
                                                       int K,
                                                       int V,
                                                       int value_tile) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  const int tile_begin = blockIdx.y * value_tile;
  const int tile_cols = value_tile < (V - tile_begin) ? value_tile : (V - tile_begin);
  if (tile_cols <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* dstate = shared;
  float* dspre = shared + K * tile_cols;
  const int state_elems = K * tile_cols;
  for (int idx = threadIdx.x; idx < state_elems * 2; idx += blockDim.x) {
    shared[idx] = 0.0f;
  }
  __syncthreads();

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int state_base = ((((b * H + h) * T) * K) * V);
  for (int t = T - 1; t >= 0; --t) {
    const float et = expf(g[scalar_base + t]);

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        acc += states[state_base + t * K * V + kk * V + j] * grad_out[v_base + t * V + j];
      }
      atomicAdd(&dq[q_base + t * K + kk], acc);
    }
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      const int kk = idx / tile_cols;
      const int local_j = idx - kk * tile_cols;
      const int j = tile_begin + local_j;
      dstate[idx] += q[q_base + t * K + kk] * grad_out[v_base + t * V + j];
      dspre[idx] = dstate[idx];
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float ddelta = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        ddelta += dstate[kk * tile_cols + local_j] * k[q_base + t * K + kk];
      }
      dv[v_base + t * V + j] = ddelta * beta[scalar_base + t];
    }
    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        const float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * K * V + kk * V + j];
        const float s_pre = s_prev * et;
        float kv = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          const float sp = (t == 0) ? 0.0f : states[state_base + (t - 1) * K * V + kk2 * V + j] * et;
          kv += sp * k[q_base + t * K + kk2];
        }
        const float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
        float ddelta = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          ddelta += dstate[kk2 * tile_cols + local_j] * k[q_base + t * K + kk2];
        }
        acc += dstate[kk * tile_cols + local_j] * delta - s_pre * ddelta * beta[scalar_base + t];
      }
      atomicAdd(&dk[q_base + t * K + kk], acc);
    }
    if (threadIdx.x == 0) {
      float db = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          const float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * K * V + kk * V + j];
          kv += s_prev * et * k[q_base + t * K + kk];
        }
        float ddelta = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          ddelta += dstate[kk * tile_cols + local_j] * k[q_base + t * K + kk];
        }
        db += ddelta * (v[v_base + t * V + j] - kv);
      }
      atomicAdd(&dbeta[scalar_base + t], db);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      const int kk = idx / tile_cols;
      const int local_j = idx - kk * tile_cols;
      const int j = tile_begin + local_j;
      const float s_prev = (t == 0) ? 0.0f : states[state_base + (t - 1) * K * V + kk * V + j];
      float ddelta = 0.0f;
      for (int kk2 = 0; kk2 < K; ++kk2) {
        ddelta += dstate[kk2 * tile_cols + local_j] * k[q_base + t * K + kk2];
      }
      dspre[idx] += -k[q_base + t * K + kk] * ddelta * beta[scalar_base + t];
      dstate[idx] = dspre[idx] * et;
      atomicAdd(&dg[scalar_base + t], dspre[idx] * s_prev * et);
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_backward_recompute_kernel(const float* __restrict__ grad_out,
                                                           const float* __restrict__ q,
                                                           const float* __restrict__ k,
                                                           const float* __restrict__ v,
                                                           const float* __restrict__ beta,
                                                           const float* __restrict__ g,
                                                           float* __restrict__ dq,
                                                           float* __restrict__ dk,
                                                           float* __restrict__ dv,
                                                           float* __restrict__ dbeta,
                                                           float* __restrict__ dg,
                                                           int B,
                                                           int H,
                                                           int T,
                                                           int K,
                                                           int V) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  extern __shared__ float shared[];
  const int state_elems = K * V;
  float* dstate = shared;
  float* prev = shared + state_elems;
  float* state = shared + state_elems * 2;

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);

  for (int idx = threadIdx.x; idx < state_elems * 3; idx += blockDim.x) {
    shared[idx] = 0.0f;
  }
  __syncthreads();

  for (int t = T - 1; t >= 0; --t) {
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      prev[idx] = 0.0f;
    }
    __syncthreads();

    for (int s = 0; s < t; ++s) {
      const float es = expf(g[scalar_base + s]);
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        prev[idx] *= es;
      }
      __syncthreads();

      for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * V + j] * k[q_base + s * K + kk];
        }
        float delta = (v[v_base + s * V + j] - kv) * beta[scalar_base + s];
        for (int kk = 0; kk < K; ++kk) {
          prev[kk * V + j] += k[q_base + s * K + kk] * delta;
        }
      }
      __syncthreads();
    }

    const float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] = prev[idx] * et;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * V + j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * V + j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        acc += state[kk * V + j] * grad_out[v_base + t * V + j];
      }
      dq[q_base + t * K + kk] = acc;
    }
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      dstate[idx] += q[q_base + t * K + kk] * grad_out[v_base + t * V + j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float ddelta = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
      }
      dv[v_base + t * V + j] = ddelta * beta[scalar_base + t];
    }

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        float kv = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          kv += prev[kk2 * V + j] * et * k[q_base + t * K + kk2];
        }
        float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
        float ddelta = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
        }
        acc += dstate[kk * V + j] * delta - prev[kk * V + j] * et * ddelta * beta[scalar_base + t];
      }
      dk[q_base + t * K + kk] = acc;
    }

    if (threadIdx.x == 0) {
      float db = 0.0f;
      for (int j = 0; j < V; ++j) {
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * V + j] * et * k[q_base + t * K + kk];
        }
        float ddelta = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
        }
        db += ddelta * (v[v_base + t * V + j] - kv);
      }
      dbeta[scalar_base + t] = db;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      float ddelta = 0.0f;
      for (int kk2 = 0; kk2 < K; ++kk2) {
        ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
      }
      const float dpre = dstate[idx] - k[q_base + t * K + kk] * ddelta * beta[scalar_base + t];
      dstate[idx] = dpre * et;
      atomicAdd(&dg[scalar_base + t], dpre * prev[idx] * et);
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_backward_checkpointed_kernel(const float* __restrict__ grad_out,
                                                              const float* __restrict__ q,
                                                              const float* __restrict__ k,
                                                              const float* __restrict__ v,
                                                              const float* __restrict__ beta,
                                                              const float* __restrict__ g,
                                                              const float* __restrict__ checkpoints,
                                                              float* __restrict__ dq,
                                                              float* __restrict__ dk,
                                                              float* __restrict__ dv,
                                                              float* __restrict__ dbeta,
                                                              float* __restrict__ dg,
                                                              int checkpoint_interval,
                                                              int B,
                                                              int H,
                                                              int T,
                                                              int K,
                                                              int V,
                                                              int C) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  extern __shared__ float shared[];
  const int state_elems = K * V;
  float* dstate = shared;
  float* prev = shared + state_elems;
  float* state = shared + state_elems * 2;

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int checkpoint_base = (((b * H + h) * C) * K) * V;

  for (int idx = threadIdx.x; idx < state_elems * 3; idx += blockDim.x) {
    shared[idx] = 0.0f;
  }
  __syncthreads();

  for (int t = T - 1; t >= 0; --t) {
    const int c = t / checkpoint_interval;
    const int begin = c * checkpoint_interval;
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      prev[idx] = checkpoints[checkpoint_base + c * state_elems + idx];
    }
    __syncthreads();

    for (int s = begin; s < t; ++s) {
      const float es = expf(g[scalar_base + s]);
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        prev[idx] *= es;
      }
      __syncthreads();

      for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * V + j] * k[q_base + s * K + kk];
        }
        float delta = (v[v_base + s * V + j] - kv) * beta[scalar_base + s];
        for (int kk = 0; kk < K; ++kk) {
          prev[kk * V + j] += k[q_base + s * K + kk] * delta;
        }
      }
      __syncthreads();
    }

    const float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] = prev[idx] * et;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * V + j] * k[q_base + t * K + kk];
      }
      float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * V + j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        acc += state[kk * V + j] * grad_out[v_base + t * V + j];
      }
      dq[q_base + t * K + kk] = acc;
    }
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      dstate[idx] += q[q_base + t * K + kk] * grad_out[v_base + t * V + j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
      float ddelta = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
      }
      dv[v_base + t * V + j] = ddelta * beta[scalar_base + t];
    }

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int j = 0; j < V; ++j) {
        float kv = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          kv += prev[kk2 * V + j] * et * k[q_base + t * K + kk2];
        }
        float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
        float ddelta = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
        }
        acc += dstate[kk * V + j] * delta - prev[kk * V + j] * et * ddelta * beta[scalar_base + t];
      }
      dk[q_base + t * K + kk] = acc;
    }

    if (threadIdx.x == 0) {
      float db = 0.0f;
      for (int j = 0; j < V; ++j) {
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * V + j] * et * k[q_base + t * K + kk];
        }
        float ddelta = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          ddelta += dstate[kk * V + j] * k[q_base + t * K + kk];
        }
        db += ddelta * (v[v_base + t * V + j] - kv);
      }
      dbeta[scalar_base + t] = db;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      int kk = idx / V;
      int j = idx - kk * V;
      float ddelta = 0.0f;
      for (int kk2 = 0; kk2 < K; ++kk2) {
        ddelta += dstate[kk2 * V + j] * k[q_base + t * K + kk2];
      }
      const float dpre = dstate[idx] - k[q_base + t * K + kk] * ddelta * beta[scalar_base + t];
      dstate[idx] = dpre * et;
      atomicAdd(&dg[scalar_base + t], dpre * prev[idx] * et);
    }
    __syncthreads();
  }
}

__global__ void qwen_linear_attn_backward_checkpointed_tiled_kernel(const float* __restrict__ grad_out,
                                                                    const float* __restrict__ q,
                                                                    const float* __restrict__ k,
                                                                    const float* __restrict__ v,
                                                                    const float* __restrict__ beta,
                                                                    const float* __restrict__ g,
                                                                    const float* __restrict__ checkpoints,
                                                                    float* __restrict__ dq,
                                                                    float* __restrict__ dk,
                                                                    float* __restrict__ dv,
                                                                    float* __restrict__ dbeta,
                                                                    float* __restrict__ dg,
                                                                    int checkpoint_interval,
                                                                    int B,
                                                                    int H,
                                                                    int T,
                                                                    int K,
                                                                    int V,
                                                                    int C,
                                                                    int value_tile) {
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh - b * H;
  const int tile_begin = blockIdx.y * value_tile;
  const int tile_cols = value_tile < (V - tile_begin) ? value_tile : (V - tile_begin);
  if (tile_cols <= 0) {
    return;
  }

  extern __shared__ float shared[];
  const int state_elems = K * tile_cols;
  float* dstate = shared;
  float* prev = shared + state_elems;
  float* state = shared + state_elems * 2;

  const int q_base = (((b * H + h) * T) * K);
  const int v_base = (((b * H + h) * T) * V);
  const int scalar_base = ((b * H + h) * T);
  const int checkpoint_base = (((b * H + h) * C) * K) * V;

  for (int idx = threadIdx.x; idx < state_elems * 3; idx += blockDim.x) {
    shared[idx] = 0.0f;
  }
  __syncthreads();

  for (int t = T - 1; t >= 0; --t) {
    const int c = t / checkpoint_interval;
    const int begin = c * checkpoint_interval;
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      const int kk = idx / tile_cols;
      const int local_j = idx - kk * tile_cols;
      const int j = tile_begin + local_j;
      prev[idx] = checkpoints[checkpoint_base + c * K * V + kk * V + j];
    }
    __syncthreads();

    for (int s = begin; s < t; ++s) {
      const float es = expf(g[scalar_base + s]);
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        prev[idx] *= es;
      }
      __syncthreads();

      for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
        const int j = tile_begin + local_j;
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * tile_cols + local_j] * k[q_base + s * K + kk];
        }
        const float delta = (v[v_base + s * V + j] - kv) * beta[scalar_base + s];
        for (int kk = 0; kk < K; ++kk) {
          prev[kk * tile_cols + local_j] += k[q_base + s * K + kk] * delta;
        }
      }
      __syncthreads();
    }

    const float et = expf(g[scalar_base + t]);
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      state[idx] = prev[idx] * et;
    }
    __syncthreads();
    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float kv = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        kv += state[kk * tile_cols + local_j] * k[q_base + t * K + kk];
      }
      const float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
      for (int kk = 0; kk < K; ++kk) {
        state[kk * tile_cols + local_j] += k[q_base + t * K + kk] * delta;
      }
    }
    __syncthreads();

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        acc += state[kk * tile_cols + local_j] * grad_out[v_base + t * V + j];
      }
      atomicAdd(&dq[q_base + t * K + kk], acc);
    }
    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      const int kk = idx / tile_cols;
      const int local_j = idx - kk * tile_cols;
      const int j = tile_begin + local_j;
      dstate[idx] += q[q_base + t * K + kk] * grad_out[v_base + t * V + j];
    }
    __syncthreads();

    for (int local_j = threadIdx.x; local_j < tile_cols; local_j += blockDim.x) {
      const int j = tile_begin + local_j;
      float ddelta = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        ddelta += dstate[kk * tile_cols + local_j] * k[q_base + t * K + kk];
      }
      dv[v_base + t * V + j] = ddelta * beta[scalar_base + t];
    }

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
      float acc = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        float kv = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          kv += prev[kk2 * tile_cols + local_j] * et * k[q_base + t * K + kk2];
        }
        const float delta = (v[v_base + t * V + j] - kv) * beta[scalar_base + t];
        float ddelta = 0.0f;
        for (int kk2 = 0; kk2 < K; ++kk2) {
          ddelta += dstate[kk2 * tile_cols + local_j] * k[q_base + t * K + kk2];
        }
        acc += dstate[kk * tile_cols + local_j] * delta -
               prev[kk * tile_cols + local_j] * et * ddelta * beta[scalar_base + t];
      }
      atomicAdd(&dk[q_base + t * K + kk], acc);
    }

    if (threadIdx.x == 0) {
      float db = 0.0f;
      for (int local_j = 0; local_j < tile_cols; ++local_j) {
        const int j = tile_begin + local_j;
        float kv = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          kv += prev[kk * tile_cols + local_j] * et * k[q_base + t * K + kk];
        }
        float ddelta = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          ddelta += dstate[kk * tile_cols + local_j] * k[q_base + t * K + kk];
        }
        db += ddelta * (v[v_base + t * V + j] - kv);
      }
      atomicAdd(&dbeta[scalar_base + t], db);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      const int kk = idx / tile_cols;
      const int local_j = idx - kk * tile_cols;
      float ddelta = 0.0f;
      for (int kk2 = 0; kk2 < K; ++kk2) {
        ddelta += dstate[kk2 * tile_cols + local_j] * k[q_base + t * K + kk2];
      }
      const float dpre = dstate[idx] - k[q_base + t * K + kk] * ddelta * beta[scalar_base + t];
      dstate[idx] = dpre * et;
      atomicAdd(&dg[scalar_base + t], dpre * prev[idx] * et);
    }
    __syncthreads();
  }
}

}  // namespace

bool qwen_linear_attention_cuda_available() {
  return true;
}

std::tuple<torch::Tensor, torch::Tensor> qwen_linear_attention_cuda_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    bool save_states) {
  require_f32_cuda_contiguous(query, "query");
  require_f32_cuda_contiguous(key, "key");
  require_f32_cuda_contiguous(value, "value");
  require_f32_cuda_contiguous(beta, "beta");
  require_f32_cuda_contiguous(g, "g");
  int B = static_cast<int>(query.size(0));
  int H = static_cast<int>(query.size(1));
  int T = static_cast<int>(query.size(2));
  int K = static_cast<int>(query.size(3));
  int V = static_cast<int>(value.size(3));
  auto out = torch::empty({B, H, T, V}, query.options());
  auto states = save_states ? torch::empty({B, H, T, K, V}, query.options())
                            : torch::empty({0}, query.options());
  const int value_tile = value_tile_size(V);
  const int value_tiles = (V + value_tile - 1) / value_tile;
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(value_tile) * sizeof(float);
  if (shared > 96 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA tiled forward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_forward_tiled_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute tiled forward");
  qwen_linear_attn_forward_tiled_kernel<<<dim3(B * H, value_tiles), 256, shared>>>(
      query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), out.data_ptr<float>(),
      save_states ? states.data_ptr<float>() : nullptr, save_states,
      B, H, T, K, V, value_tile);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_forward_tiled_kernel");
  return {out, states};
}

std::tuple<torch::Tensor, torch::Tensor> qwen_linear_attention_cuda_forward_checkpointed(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    int64_t checkpoint_interval) {
  require_f32_cuda_contiguous(query, "query");
  require_f32_cuda_contiguous(key, "key");
  require_f32_cuda_contiguous(value, "value");
  require_f32_cuda_contiguous(beta, "beta");
  require_f32_cuda_contiguous(g, "g");
  if (checkpoint_interval <= 0) {
    throw std::invalid_argument("checkpoint_interval must be positive");
  }
  int B = static_cast<int>(query.size(0));
  int H = static_cast<int>(query.size(1));
  int T = static_cast<int>(query.size(2));
  int K = static_cast<int>(query.size(3));
  int V = static_cast<int>(value.size(3));
  int interval = static_cast<int>(checkpoint_interval);
  int C = (T + interval - 1) / interval;
  auto out = torch::empty({B, H, T, V}, query.options());
  auto checkpoints = torch::empty({B, H, C, K, V}, query.options());
  const int value_tile = value_tile_size(V);
  const int value_tiles = (V + value_tile - 1) / value_tile;
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(value_tile) * sizeof(float);
  if (shared > 96 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA tiled checkpointed forward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_forward_checkpointed_tiled_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute tiled checkpointed forward");
  qwen_linear_attn_forward_checkpointed_tiled_kernel<<<dim3(B * H, value_tiles), 256, shared>>>(
      query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), out.data_ptr<float>(),
      checkpoints.data_ptr<float>(), interval, B, H, T, K, V, C, value_tile);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_forward_checkpointed_tiled_kernel");
  return {out, checkpoints};
}

std::vector<torch::Tensor> qwen_linear_attention_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& states) {
  require_f32_cuda_contiguous(grad_out, "grad_out");
  require_f32_cuda_contiguous(query, "query");
  require_f32_cuda_contiguous(key, "key");
  require_f32_cuda_contiguous(value, "value");
  require_f32_cuda_contiguous(beta, "beta");
  require_f32_cuda_contiguous(g, "g");
  require_f32_cuda_contiguous(states, "states");
  int B = static_cast<int>(query.size(0));
  int H = static_cast<int>(query.size(1));
  int T = static_cast<int>(query.size(2));
  int K = static_cast<int>(query.size(3));
  int V = static_cast<int>(value.size(3));
  auto dq = torch::zeros_like(query);
  auto dk = torch::zeros_like(key);
  auto dv = torch::empty_like(value);
  auto dbeta = torch::zeros_like(beta);
  auto dg = torch::zeros_like(g);
  const int value_tile = value_tile_size(V);
  const int value_tiles = (V + value_tile - 1) / value_tile;
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(value_tile) * 2 * sizeof(float);
  if (shared > 192 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA tiled backward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_backward_tiled_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute tiled backward");
  qwen_linear_attn_backward_tiled_kernel<<<dim3(B * H, value_tiles), 256, shared>>>(
      grad_out.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), states.data_ptr<float>(), dq.data_ptr<float>(),
      dk.data_ptr<float>(), dv.data_ptr<float>(), dbeta.data_ptr<float>(), dg.data_ptr<float>(),
      B, H, T, K, V, value_tile);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_backward_tiled_kernel");
  return {dq, dk, dv, dbeta, dg};
}

std::vector<torch::Tensor> qwen_linear_attention_cuda_backward_recompute(
    const torch::Tensor& grad_out,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g) {
  require_f32_cuda_contiguous(grad_out, "grad_out");
  require_f32_cuda_contiguous(query, "query");
  require_f32_cuda_contiguous(key, "key");
  require_f32_cuda_contiguous(value, "value");
  require_f32_cuda_contiguous(beta, "beta");
  require_f32_cuda_contiguous(g, "g");
  int B = static_cast<int>(query.size(0));
  int H = static_cast<int>(query.size(1));
  int T = static_cast<int>(query.size(2));
  int K = static_cast<int>(query.size(3));
  int V = static_cast<int>(value.size(3));
  if (T <= 0) {
    return {torch::empty_like(query), torch::empty_like(key), torch::empty_like(value),
            torch::empty_like(beta), torch::empty_like(g)};
  }
  auto zero_checkpoint = torch::zeros({B, H, 1, K, V}, query.options());
  return qwen_linear_attention_cuda_backward_checkpointed(
      grad_out, query, key, value, beta, g, zero_checkpoint, T);
}

std::vector<torch::Tensor> qwen_linear_attention_cuda_backward_checkpointed(
    const torch::Tensor& grad_out,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& checkpoints,
    int64_t checkpoint_interval) {
  require_f32_cuda_contiguous(grad_out, "grad_out");
  require_f32_cuda_contiguous(query, "query");
  require_f32_cuda_contiguous(key, "key");
  require_f32_cuda_contiguous(value, "value");
  require_f32_cuda_contiguous(beta, "beta");
  require_f32_cuda_contiguous(g, "g");
  require_f32_cuda_contiguous(checkpoints, "checkpoints");
  if (checkpoint_interval <= 0) {
    throw std::invalid_argument("checkpoint_interval must be positive");
  }
  int B = static_cast<int>(query.size(0));
  int H = static_cast<int>(query.size(1));
  int T = static_cast<int>(query.size(2));
  int K = static_cast<int>(query.size(3));
  int V = static_cast<int>(value.size(3));
  int interval = static_cast<int>(checkpoint_interval);
  int C = (T + interval - 1) / interval;
  if (checkpoints.dim() != 5 || checkpoints.size(0) != B || checkpoints.size(1) != H ||
      checkpoints.size(2) != C || checkpoints.size(3) != K || checkpoints.size(4) != V) {
    throw std::invalid_argument("checkpoint tensor shape mismatch");
  }
  auto dq = torch::zeros_like(query);
  auto dk = torch::zeros_like(key);
  auto dv = torch::empty_like(value);
  auto dbeta = torch::zeros_like(beta);
  auto dg = torch::zeros_like(g);
  const int value_tile = value_tile_size(V);
  const int value_tiles = (V + value_tile - 1) / value_tile;
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(value_tile) * 3 * sizeof(float);
  if (shared > 192 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA tiled checkpointed backward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_backward_checkpointed_tiled_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute tiled checkpointed backward");
  qwen_linear_attn_backward_checkpointed_tiled_kernel<<<dim3(B * H, value_tiles), 256, shared>>>(
      grad_out.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), checkpoints.data_ptr<float>(), dq.data_ptr<float>(),
      dk.data_ptr<float>(), dv.data_ptr<float>(), dbeta.data_ptr<float>(), dg.data_ptr<float>(),
      interval, B, H, T, K, V, C, value_tile);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_backward_checkpointed_tiled_kernel");
  return {dq, dk, dv, dbeta, dg};
}

}  // namespace cverl
