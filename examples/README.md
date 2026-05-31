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
- `run_vllm_tp2_online_grpo_h20.sh`: 4-card H20 node launcher using vLLM TP=2
  on GPU0-1 and cverl trainer/sender on GPU2, with GPU3 left free for the next
  trainer DP/TP step.
- `run_vllm_static_split_4gpu_h20.sh`: 4-card static role split for the 8-card
  production design. GPU0-1 run vLLM TP=2; GPU2-3 run the PP/TP trainer. This
  avoids vLLM sleep/wake phase switching. Use `ROLLOUT_BACKEND=oracle` for a
  deterministic reward-variance trainer smoke; default `vllm` uses real
  generation and may produce zero reward on tiny GSM8K samples.
- `run_gsm8k_ppo_8gpu_h20.sh`: full 8-card GSM8K PPO trainer entry point.
  It first writes rollout batches from GSM8K, then runs the Qwen PP/TP PPO
  trainer on GPU0-7 with default `PP=4,TP=2`. The default
  `ROLLOUT_BACKEND=oracle` guarantees reward variance for regression; set
  `ROLLOUT_BACKEND=vllm` to consume real vLLM generations.

Multi-GPU regression and benchmark entry points:

```bash
# Pass/fail baseline for Megatron-style Qwen PPO training.
NCCL_SOCKET_IFNAME=eth1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  tools/distributed/run_qwen_multigpu_regression.sh ../models/Qwen3.5-0.8B

# Timing baseline; writes build/bench/qwen_multigpu_ppo_bench.csv.
NCCL_SOCKET_IFNAME=eth1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  tools/bench/qwen_multigpu_ppo_bench.sh ../models/Qwen3.5-0.8B

# Real rollout batch -> PP/TP PPO trainer. Use ROLLOUT_BACKEND=vllm with a
# running vLLM server, or oracle for a deterministic reward-variance check.
ROLLOUT_BACKEND=oracle NCCL_SOCKET_IFNAME=eth1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  tools/distributed/run_rollout_to_pp_tp_ppo.sh ../models/Qwen3.5-0.8B

# Full 8-card GSM8K PPO path, defaulting to deterministic oracle rollout.
NCCL_SOCKET_IFNAME=bond1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  examples/run_gsm8k_ppo_8gpu_h20.sh
```

Useful distributed planning command:

```bash
./build/cverl_parallel_plan --dp 2 --pp 2 --cp 2 --tp 2 \
  --world-size 16 --local-world-size 4 --gpus-per-node 4 \
  --micro-batches 4 --num-layers 24 --all-ranks
```

HTTP is still treated as a debug/control surface. The online vLLM script uses
HTTP for OpenAI-compatible generation while the weight payload uses Native RL
NCCL; the next optimization boundary is replacing generation transport with a
vLLM/Megatron plugin worker or CUDA IPC path.
