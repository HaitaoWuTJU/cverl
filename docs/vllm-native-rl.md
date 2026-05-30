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
