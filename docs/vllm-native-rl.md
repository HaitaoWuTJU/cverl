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

## Current Missing Piece

The live cverl C++ trainer parameters still need a trainer-side sender
compatible with vLLM's `NCCLWeightTransferEngine.trainer_send_weights`.
vLLM's Python implementation builds a `StatelessProcessGroup`, creates a
`PyNcclCommunicator`, and broadcasts packed uint8 tensor buffers. The cverl
sender should implement the same wire protocol in C++ or expose cverl tensors
to the Python sender without copying through checkpoint files.

Until that sender is implemented, `update-nccl` should not be called against a
running vLLM server unless a matching trainer-side NCCL producer is running.
