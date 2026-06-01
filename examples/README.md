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
- `run_qwen3_30b_a3b_sft_pp8_h20.sh`: 8-card H20 GSM8K SFT entry point for
  `Qwen/Qwen3-30B-A3B`, defaulting to `PP=8,TP=1,DP=1,EP=1,SP=1`,
  bf16, and configurable `GRAD_ACCUM_STEPS`.
- `run_qwen3_30b_a3b_moe_parallel_h20.sh`: 8-card H20 Qwen3 MoE regression
  launcher. It defaults to `DP=1,PP=2,CP=1,TP=2,EP=2,SP=1` and `TRAIN_MODE=sft`;
  set `TRAIN_MODE=ppo` for the PPO loss path, or set `CP_SIZE=2,TP_SIZE=1` to
  exercise context parallel attention with expert parallel dispatch.
- `run_resnet152_cifar10_bench.sh`: single-GPU ResNet152/CIFAR10 benchmark
  comparing C++/LibTorch against Python/PyTorch with the same model, data,
  optimizer, batch size, epoch count, seed, and TF32 setting.
- `prepare_cifar10_hf.sh`: local/shared-filesystem CIFAR10 preparation helper.
  It downloads `uoft-cs/cifar10` from Hugging Face as Parquet and exports
  raw `train_*` and `test_*` tensor caches for both C++ and Python benchmarks.

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

# Full 48-layer Qwen3-30B-A3B SFT smoke on 8 H20 GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  examples/run_qwen3_30b_a3b_sft_pp8_h20.sh

# Qwen3 MoE 8-card parallel regression: PP + TP + EP.
NCCL_SOCKET_IFNAME=bond1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  examples/run_qwen3_30b_a3b_moe_parallel_h20.sh

# Qwen3 MoE CP + EP regression.
NCCL_SOCKET_IFNAME=bond1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  CP_SIZE=2 TP_SIZE=1 examples/run_qwen3_30b_a3b_moe_parallel_h20.sh

# ResNet152/CIFAR10 10-epoch C++ vs Python timing benchmark.
DATA_ROOT=/data/workspace/ceph/zw/rl/datasets/cifar10 examples/prepare_cifar10_hf.sh
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH_SIZE=256 \
  examples/run_resnet152_cifar10_bench.sh
```

Useful distributed planning command:

```bash
./build/cverl_parallel_plan --dp 2 --pp 2 --cp 2 --tp 2 --ep 2 --sp 2 \
  --world-size 32 --local-world-size 4 --gpus-per-node 4 \
  --micro-batches 4 --num-layers 24 --all-ranks
```

Topology planning now understands `DP x PP x CP x TP x EP` as rank-mapping
dimensions. `SP` is Megatron-style sequence parallelism inside TP groups, so it
does not multiply world size. The Qwen3 MoE trainer now supports sparse EP
dispatch with `TP=1`, `TP+EP` expert tensor sharding, and `CP+EP` when `TP=1`.
Experts are sharded by expert id across EP ranks; expert gate/up/down matrices
are sharded across TP ranks following Megatron's column/row split; non-expert
parameters are replicated across EP and synchronize gradients. MoE routes use
equal-capacity token dispatch plus return all-to-all with autograd, and local
expert compute is packed into padded batched GEMMs. The trainer assigns
different microbatches to different EP ranks, so EP lanes are not duplicating
the same local tokens. cuBLAS grouped GEMM/TransformerEngine kernels, production
capacity/drop policy, `CP+TP` in the same Qwen3 MoE run, and real SP
all-gather/reduce-scatter are still the next optimization boundary.

HTTP is still treated as a debug/control surface. The online vLLM script uses
HTTP for OpenAI-compatible generation while the weight payload uses Native RL
NCCL; the next optimization boundary is replacing generation transport with a
vLLM/Megatron plugin worker or CUDA IPC path.
