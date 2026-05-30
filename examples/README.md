# cverl Examples

This directory contains shell launchers only. C++ sources and tests belong under
`tools/`, `src/`, or `tests/`.

Current examples:

- `run_gsm8k_grpo_trainer_cpu.sh`: Qwen3.5-0.8B CPU entry point for debugging.
- `run_gsm8k_grpo_trainer_cuda.sh`: Qwen3.5-0.8B CUDA trainer entry point with
  in-process rollout.
- `run_h20_nccl_weight_sync.sh`: H20 NCCL collectives and parameter sync check.
- `run_vllm_online_grpo.sh`: online GSM8K GRPO driver where cverl updates the
  live Qwen actor tensors and synchronizes them to vLLM through Native RL NCCL.

HTTP is still treated as a debug/control surface. The online vLLM script uses
HTTP for OpenAI-compatible generation while the weight payload uses Native RL
NCCL; the next optimization boundary is replacing generation transport with a
vLLM/Megatron plugin worker or CUDA IPC path.
