# cverl Examples

This directory is for runnable shell examples.

- `run_sglang_gsm8k_smoke.sh`: starts a local SGLang OpenAI-compatible server,
  probes `/v1/completions`, then runs the GSM8K GRPO/PPO smoke through cverl.
- `run_vllm_gsm8k_smoke.sh`: same GSM8K GRPO/PPO path with vLLM.

C++ executable sources live under `tools/` next to the related subsystem.
