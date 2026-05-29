# Rollout Transport Plan

The first complete GRPO/GSM8K chain should reuse vLLM or SGLang for text
generation through their OpenAI-compatible HTTP endpoints. That path is stable
and keeps cverl independent from rollout-server internals.

For high-throughput colocated training, the transport should be upgraded in
stages:

1. HTTP JSON transport for compatibility with stock vLLM/SGLang.
2. POSIX shared memory for local sidecars that exchange token IDs, logprobs,
   rewards, and sequence metadata without JSON serialization.
3. CUDA IPC for same-node GPU tensors when the rollout sidecar can export CUDA
   memory handles.
4. NCCL send/recv or collectives for GPU-to-GPU transfer between cverl ranks and
   rollout ranks.
5. GPUDirect RDMA/NCCL network tuning for cross-node transfer.

Important boundary: stock vLLM/SGLang public serving APIs do not expose their
internal GPU tensors or NCCL communicators. Direct NCCL/RDMA transfer therefore
requires a custom worker/plugin/sidecar in the rollout process. Without that
cooperation, cverl can only optimize the client side around HTTP.

The current code provides the POSIX shared-memory foundation:

- `include/cverl/rollout/shared_memory.h`
- `src/rollout/shared_memory.cc`
- `tests/rollout/test_shared_memory.cc`

This is intended for structured CPU-side payloads. CUDA IPC and NCCL transport
should be implemented behind the same rollout transport boundary rather than
inside trainer logic.
