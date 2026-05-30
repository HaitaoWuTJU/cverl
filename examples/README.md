# cverl Examples

Useful runnable examples and smoke scripts live here.

## Scripts

- `scripts/run_sglang_gsm8k_smoke.sh`: starts a local SGLang OpenAI-compatible
  server, probes `/v1/completions`, then runs the GSM8K GRPO/PPO smoke through
  cverl.

## C++ Examples

- `rollout/gsm8k_grpo_smoke.cc`: GSM8K rollout-to-update smoke with loopback or
  HTTP rollout transport.
- `rollout/gsm8k_rollout_pipeline.cc`: rollout + rule reward pipeline.
- `bench/cverl_gsm8k_grpo_bench.cc`: cverl GSM8K GRPO benchmark path.
- `torch/minimal_ppo_step.cc`: minimal PPO clipped loss update.
- `torch/toy_grpo_trainer.cc`: tiny GRPO trainer example.
- `model/hf_embedding_grpo_trainer.cc`: HF model loader plus GRPO-style trainer
  example.
