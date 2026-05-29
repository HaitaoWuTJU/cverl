# cverl

`cverl` is a C++/CUDA rewrite target for the RL training and inference stack
inspired by `verl`. The first milestone keeps a stable C ABI while using
LibTorch/ATen for tensor math, autograd, and optimizer-friendly C++ integration.

The project does not depend on Python at runtime, Ray, TensorDict, or the Python
PyTorch API. It does depend on LibTorch, either from a standalone LibTorch
install or from the `torch` Python wheel's bundled C++ package at build time.

## Current Scope

Implemented kernels:

- Masked reductions: `masked_sum`, `masked_mean`, `masked_whiten`
- KL penalty: `k1`, `abs`, `k2`, `k3`
- KL penalty backward gradient for `logprob`
- GAE advantage and returns
- GRPO outcome advantage
- PPO clipped policy loss
- PPO clipped policy loss backward gradient for `log_prob`

Current tensor support:

- contiguous `float32`
- dense 2D tensors with shape `[batch, seq]`
- CPU device

The C ABI wraps raw pointers into `torch::Tensor` views and delegates math to
LibTorch. Backward APIs use LibTorch autograd rather than manually derived
gradient loops.

CUDA support is scaffolded in the build system. GPU execution should come
through LibTorch/CUDA first; custom CUDA kernels should only be added for proven
hot spots.

## Layout

```text
cverl/
  include/cverl/        Public C ABI headers
  src/                  C ABI wrappers and LibTorch implementation
  cuda/                 CUDA implementation placeholder
  tests/                Native C++ tests
  tools/                Future golden-data and comparison tools
  docs/                 Design notes
  CMakeLists.txt        CMake build
  Makefile              Minimal build for CPU-only environments
```

## Build

### CMake

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build
./build/test_core_algos_cpu
./build/test_torch_backend
./build/test_simple_grpo_trainer
```

### Make

For environments with Python `torch` installed:

```sh
make test
```

## CUDA Build Path

The CUDA option is reserved for GPU environments:

```sh
cmake -S . -B build-cuda -DCVERL_ENABLE_CUDA=ON
cmake --build build-cuda
```

At the moment this only validates the CUDA build path. Kernel implementations
will be added after the LibTorch behavior is covered by golden tests.

The `minimal_ppo_step` executable shows a native C++ PPO-style training step
using a LibTorch model, optimizer, and `cverl` RL losses:

```sh
./build/minimal_ppo_step
```

The `toy_grpo_trainer` executable runs a complete synthetic GRPO/PPO training
loop in C++:

```sh
./build/toy_grpo_trainer 32
```

It creates toy prompts, samples responses from a LibTorch policy, computes an
accuracy reward, builds GRPO advantages, applies PPO updates, and prints CSV
metrics.

Checkpointing is available through the trainer API and CLI:

```sh
./build/toy_grpo_trainer 8 4 4 --save-prefix /tmp/cverl-toy
./build/toy_grpo_trainer 8 4 4 --load-prefix /tmp/cverl-toy
```

## Hugging Face Downloads

`cverl` provides a small C++ wrapper around the Python `huggingface_hub`
download API. This keeps C++ training code in control while reusing Hugging
Face's supported cache, auth, revision, and pattern filtering behavior.

```sh
./build/hf_download Qwen/Qwen3.5-0.8B --dry-run
./build/hf_download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B
```

The wrapper requires `huggingface_hub` to be installed in the selected Python
environment. Use `--python /path/to/python` if the default `python3` is not the
right interpreter.

## API Example

```c
#include "cverl/cverl.h"

float rewards[8] = {0};
float values[8] = {0};
float mask[8] = {1};
float advantages[8];
float returns[8];

cverl_const_tensor2d_t rewards_t = {
    rewards, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_const_tensor2d_t values_t = {
    values, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_const_tensor2d_t mask_t = {
    mask, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_tensor2d_t adv_t = {
    advantages, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_tensor2d_t ret_t = {
    returns, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};

cverl_status_t status = cverl_gae_advantage_return_f32_cpu(
    rewards_t, values_t, mask_t, 1.0f, 1.0f, adv_t, ret_t);
```

## Correctness Strategy

The LibTorch implementation is the source of truth for current kernels.
Golden-data tests can be generated from the Python `verl` implementation. The
current golden format covers forward outputs for KL, GAE, GRPO, PPO loss, plus
autograd reference gradients for KL with respect to `logprob` and PPO loss with
respect to `log_prob`.

1. Generate random inputs and reference outputs from `verl`.
2. Serialize them to a simple binary/JSON test format.
3. Load the same cases in C++.
4. Compare outputs with fp32 tolerances.

Example:

```sh
cmake -S . -B build
cmake --build build

PYTHONPATH=../verl python3 tools/golden_dump.py build/golden_core_algos.bin
./build/compare_golden build/golden_core_algos.bin
```

The `golden_dump.py` script requires an environment with `torch`, `numpy`, and
the Python `verl` checkout importable via `PYTHONPATH`.

Planned tolerance for fp32 kernels:

- `rtol = 1e-5`
- `atol = 1e-6`

## Roadmap

1. Add Python-generated golden tests for the current CPU kernels.
2. Use LibTorch CUDA for GPU execution and profile bottlenecks.
3. Add dtype support for `float16` and `bfloat16` where numerically appropriate.
4. Add a native tensor/batch abstraction beyond simple 2D tensors.
5. Build a minimal C++ trainer runtime.
6. Add native model backend pieces using existing C++ APIs where possible:
   optimizer/checkpointing via LibTorch, distributed collectives via NCCL, and
   attention/math via FlashAttention/cuBLAS/CUTLASS.

## Non-goals for the Current Milestone

- Full `verl` feature parity
- Ray-compatible scheduling
- PyTorch/FSDP/Megatron integration
- vLLM/SGLang replacement
- Production LLM training runtime

Those pieces can be revisited after the native RL math core is stable and
covered by tests.
