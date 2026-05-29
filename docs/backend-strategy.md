# Backend Strategy

The long-term goal is a C++/CUDA RL training and inference stack. That does not
mean every low-level component should be rewritten from scratch.

## Principles

- Keep the stable public surface as a small C ABI where practical.
- Use mature C++/CUDA libraries for large, well-solved subsystems.
- Write custom kernels only where the RL workload has a clear gap or measurable
  bottleneck.
- Keep a CPU reference path for correctness and a GPU path for performance.

## Preferred Existing Libraries

- Tensor, autograd, and model prototyping: LibTorch / ATen C++ API
- GPU math: cuBLAS, cuDNN, CUTLASS
- Collectives: NCCL
- Attention kernels: FlashAttention or equivalent C++/CUDA integration
- Tensor checkpoint format: safetensors or a compatible native format

## cverl Layers

1. `cverl`: minimal C ABI and CPU reference kernels.
2. `cverl_torch`: optional LibTorch backend for C++ trainer prototyping.
3. CUDA kernels: custom implementations added only after CPU/LibTorch behavior is
   covered by golden tests.
4. Runtime: native trainer, optimizer, checkpointing, rollout, and distributed
   execution.

The `minimal_ppo_step` example shows the intended near-term integration style:
LibTorch owns model parameters, autograd, and optimizer state while `cverl_torch`
provides RL-specific losses and advantage computation. Once behavior is stable,
hot kernels can be replaced by CUDA implementations behind the same tests.

This lets us avoid depending on Python while still reusing C++ APIs from mature
ML systems.
