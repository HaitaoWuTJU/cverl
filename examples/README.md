# cverl Examples

This directory contains shell launchers only. C++ sources and tests belong under
`tools/`, `src/`, or `tests/`.

Current examples:

- `run_gsm8k_grpo_trainer_cpu.sh`: Qwen3.5-0.8B CPU entry point for debugging.
- `run_gsm8k_grpo_trainer_cuda.sh`: Qwen3.5-0.8B CUDA trainer entry point with
  in-process rollout.
- `run_h20_nccl_weight_sync.sh`: H20 NCCL collectives and parameter sync check.

HTTP rollout launchers are intentionally absent from the core path. Efficient
rollout should use in-process workers or vLLM/Megatron/FlashAttention-backed
plugin workers that keep tokens and weights on GPU and synchronize actor
weights through NCCL/CUDA IPC.
