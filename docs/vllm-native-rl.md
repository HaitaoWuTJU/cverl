# vLLM Native RL Path

The current efficient rollout target is:

```text
cverl trainer -> vLLM Native RL/NCCL weight sync -> vLLM rollout -> cverl PPO/GRPO update
```

vLLM should be reused first because its production runtime already provides
paged attention/KV cache, continuous batching, scheduling, FlashAttention
backends, and TP/PP serving behavior. cverl should not rewrite those pieces
until vLLM becomes the bottleneck.

## Transport Rule

- HTTP is allowed only for small control-plane calls: pause, resume, init weight
  transfer, and update metadata.
- Tensor payloads must use vLLM Native RL NCCL or CUDA IPC weight transfer.
- Checkpoint export/reload is a debug path, not the online RL path.

## Implemented In This Repo

- `tools/rollout/vllm_native_rl_control.py`
  - `pause`
  - `resume`
  - `sleep`
  - `wake`
  - `is-sleeping`
  - `world-size`
  - `init-nccl`
  - `update-nccl`
- `tools/rollout/vllm_weight_manifest.cc`
  - emits vLLM `update_info` metadata from the cverl Qwen3.5 policy parameter
    list without serializing tensor payloads.
- `_cverl_vllm_bridge`
  - pybind bridge exposing cverl C++/LibTorch Qwen3.5 live parameters as
    Python `torch.Tensor` objects.
- `tools/rollout/vllm_nccl_weight_sender.py`
  - trainer-side sender that initializes vLLM's Native RL NCCL communicator and
    calls `NCCLWeightTransferEngine.trainer_send_weights` on the live cverl
    tensors.

## Weight Sync

Build the bridge:

```bash
cmake -S . -B build-h20-nccl -DCVERL_ENABLE_NCCL=ON -DCVERL_BUILD_PYTHON_BRIDGE=ON
cmake --build build-h20-nccl -j2 --target _cverl_vllm_bridge
```

Run a vLLM server with Native RL endpoints enabled and NCCL weight transfer
configured, then send live cverl parameters:

```bash
PYTHONPATH=build-h20-nccl:$PYTHONPATH \
tools/rollout/vllm_nccl_weight_sender.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --base-url http://127.0.0.1:8000 \
  --master-port 29577 \
  --world-size 2 \
  --param-dtype bfloat16
```

`world-size` is `1 + vLLM_worker_world_size`: trainer rank 0 plus every vLLM
worker rank. The script posts only init/update metadata to vLLM; tensors are
broadcast by vLLM's NCCL packed weight-transfer path.
`--param-dtype` accepts `float32`, `bfloat16`/`bf16`, and `float16`/`fp16`; the
bridge exposes live tensors in that dtype and emits matching vLLM
`update_info.dtype_names`.

## Online GRPO Driver

`tools/rollout/vllm_online_grpo.py` is the current end-to-end integration point.
It keeps one live `_cverl_vllm_bridge.QwenPolicy` object for both PPO updates
and NCCL weight sync, so the tensors sent to vLLM are the tensors AdamW just
updated.

```bash
PYTHONPATH=build-h20-nccl:$PYTHONPATH \
tools/rollout/vllm_online_grpo.py \
  --dataset data/gsm8k-train-smoke.jsonl \
  --model-dir ../models/Qwen3.5-0.8B \
  --tokenizer-path ../models/Qwen3.5-0.8B/tokenizer.json \
  --served-model-name /path/or/name/served/by/vllm \
  --base-url http://127.0.0.1:8000 \
  --device cuda:0 \
  --param-dtype bfloat16 \
  --steps 8 \
  --prompts 4 \
  --n 4 \
  --world-size 2 \
  --master-port 29577
```

For trainer-only validation without a vLLM server, use
`--rollout-backend oracle --no-weight-sync`. This still exercises the live Qwen
GRPO/PPO update path and is useful before enabling vLLM Native RL sync.

## Static Rollout/Trainer Split

The production direction is static role assignment, not phase switching:

```text
rollout pool:
  vLLM stays alive, uses TP, owns paged KV cache and scheduler state

trainer pool:
  cverl/Megatron-style trainer owns gradients, optimizer states, PP/TP/DP/CP
```

On an 8-card node this maps naturally to:

```text
GPU 0-3: rollout, vLLM TP=4
GPU 4-7: trainer, FSDP/TP/PP/CP
```

The current test node has 4 GPUs, so the regression is the scaled-down version:

```text
GPU 0-1: rollout, vLLM TP=2
GPU 2-3: trainer, PP/TP world=2
```

Run:

```bash
examples/run_vllm_static_split_4gpu_h20.sh
```

This keeps vLLM running while the trainer uses a separate GPU set. The current
script still dumps rollout batches to files for compatibility with the existing
PP/TP trainer entry. The next production step is to replace that handoff with
the NCCL GPU batch data plane, then run shard-wise trainer-to-vLLM weight sync
without stopping either side.

For a deterministic trainer update smoke on 4 GPUs, set
`ROLLOUT_BACKEND=oracle`; this still uses the same trainer GPU assignment but
generates reward variance intentionally. With `ROLLOUT_BACKEND=vllm`, Qwen3.5
0.8B may produce zero correct GSM8K answers in a tiny sample, so the
infrastructure can run while PPO correctly has zero advantage and no parameter
update.

`sleep`/`wake` remain implemented in `tools/rollout/vllm_native_rl_control.py`
for debug and emergency memory recovery, but they are not the default efficient
online RL lifecycle.
