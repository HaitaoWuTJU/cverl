# Backend Strategy

The long-term goal is a C++/CUDA RL training and inference stack. That does not
mean every low-level component should be rewritten from scratch.

## Principles

- Keep the stable public surface as a small C ABI where practical.
- Use mature C++/CUDA libraries for large, well-solved subsystems.
- Write custom kernels only where the RL workload has a clear gap or measurable
  bottleneck.
- Keep a LibTorch reference path for correctness and use CUDA-enabled LibTorch
  before writing custom kernels.

## Preferred Existing Libraries

- Tensor, autograd, and model prototyping: LibTorch / ATen C++ API
- GPU math: cuBLAS, cuDNN, CUTLASS
- Collectives: NCCL
- Attention kernels: FlashAttention or equivalent C++/CUDA integration
- Tensor checkpoint format: safetensors or a compatible native format

## cverl Layers

1. `cverl`: minimal C ABI backed by LibTorch tensor/autograd operations.
2. CUDA execution through LibTorch first.
3. Custom CUDA kernels only after LibTorch behavior is covered by golden tests
   and profiling proves a clear bottleneck.
4. Runtime: native trainer, optimizer, checkpointing, rollout, and distributed
   execution.
5. Distributed topology and collectives: plan DP/TP/PP with
   `cverl::distributed::Topology`, then use NCCL-backed collectives in GPU
   builds.

The `minimal_ppo_step` example shows the intended near-term integration style:
LibTorch owns model parameters, tensor operations, autograd, and optimizer state
while `cverl` provides RL-specific losses and advantage computation. Once
behavior is stable, hot kernels can be replaced by CUDA implementations behind
the same tests.

The `toy_grpo_trainer` executable is the current runnable training nucleus. It
uses synthetic prompts and rewards, but exercises the full C++ loop:

- rollout sampling from a LibTorch policy
- reward computation
- GRPO advantage construction
- PPO loss and backward
- optimizer step
- scalar metrics

This lets us avoid depending on Python while still reusing C++ APIs from mature
ML systems.

The distributed runtime starts with explicit topology planning rather than
embedding rank arithmetic inside model code. See `docs/distributed-runtime.md`
for the DP/TP/PP mapping, memory policy, and network/NCCL configuration plan.
