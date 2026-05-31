#include "cverl/model/qwen_linear_attention_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>

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
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(V) * sizeof(float);
  if (shared > 96 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA kernel shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_forward_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute forward");
  qwen_linear_attn_forward_kernel<<<B * H, 256, shared>>>(
      query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), out.data_ptr<float>(),
      save_states ? states.data_ptr<float>() : nullptr, save_states,
      B, H, T, K, V);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_forward_kernel");
  return {out, states};
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
  auto dq = torch::empty_like(query);
  auto dk = torch::empty_like(key);
  auto dv = torch::empty_like(value);
  auto dbeta = torch::zeros_like(beta);
  auto dg = torch::zeros_like(g);
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(V) * 2 * sizeof(float);
  if (shared > 192 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA backward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_backward_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute backward");
  qwen_linear_attn_backward_kernel<<<B * H, 256, shared>>>(
      grad_out.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), states.data_ptr<float>(), dq.data_ptr<float>(),
      dk.data_ptr<float>(), dv.data_ptr<float>(), dbeta.data_ptr<float>(), dg.data_ptr<float>(),
      B, H, T, K, V);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_backward_kernel");
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
  auto dq = torch::empty_like(query);
  auto dk = torch::empty_like(key);
  auto dv = torch::empty_like(value);
  auto dbeta = torch::zeros_like(beta);
  auto dg = torch::zeros_like(g);
  size_t shared = static_cast<size_t>(K) * static_cast<size_t>(V) * 3 * sizeof(float);
  if (shared > 192 * 1024) {
    throw std::runtime_error("Qwen linear attention CUDA recompute backward shared memory limit exceeded");
  }
  check_cuda(cudaFuncSetAttribute(qwen_linear_attn_backward_recompute_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared)),
             "cudaFuncSetAttribute recompute backward");
  qwen_linear_attn_backward_recompute_kernel<<<B * H, 256, shared>>>(
      grad_out.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
      beta.data_ptr<float>(), g.data_ptr<float>(), dq.data_ptr<float>(), dk.data_ptr<float>(),
      dv.data_ptr<float>(), dbeta.data_ptr<float>(), dg.data_ptr<float>(), B, H, T, K, V);
  check_cuda(cudaGetLastError(), "qwen_linear_attn_backward_recompute_kernel");
  return {dq, dk, dv, dbeta, dg};
}

}  // namespace cverl
