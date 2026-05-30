# cverl Examples

This directory is for runnable shell examples.

- `run_vllm_gsm8k_train.sh`: starts a local vLLM OpenAI-compatible server,
  probes `/v1/completions`, then runs GSM8K GRPO/PPO through
  `gsm8k_grpo_trainer`.
- `run_sglang_gsm8k_train.sh`: same GSM8K GRPO/PPO path with SGLang.
- `run_vllm_gsm8k_smoke.sh` / `run_sglang_gsm8k_smoke.sh`: legacy quick
  launchers retained for small regression runs.

C++ executable sources live under `tools/` next to the related subsystem.
