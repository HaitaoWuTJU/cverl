# Efficient Online RL Path

HTTP is not part of the core rollout/training path. It is the wrong mechanism
for maximum-efficiency online RL because:

- token batches are serialized through JSON and copied through CPU memory;
- parameter updates require checkpoint materialization or server reload;
- stock OpenAI-compatible endpoints do not expose GPU tensor pointers,
  CUDA IPC handles, or NCCL communicators.

The efficient path is:

1. Trainer actor, reference, and optimizer own GPU parameters through LibTorch.
2. Rollout workers are colocated C++/CUDA sidecars, vLLM worker plugins, or
   Megatron-backed workers, not external HTTP clients.
3. Prompt token IDs and generated token IDs/logprobs move through shared memory
   for CPU metadata and CUDA IPC for same-node GPU tensors.
4. Updated actor weights synchronize from trainer ranks to rollout ranks with
   NCCL broadcast/send-recv. Same-node uses P2P/NVLink/PCIe; cross-node uses
   NCCL over RDMA.
5. DP gradient sync, TP collectives, PP activation transfer, and rollout weight
   sync share the same NCCL topology and stream policy.

Implemented foundation:

- `cverl::distributed::Collectives::broadcast`
- `cverl::distributed::NcclCollectives::broadcast`
- `cverl::distributed::broadcast_parameters_from_root`
- `cverl::rollout::RolloutWorker`
- `cverl::rollout::PolicyRolloutWorker`
- `cverl::rollout::DynamicRolloutWorker`
- `cverl::rollout::synchronize_rollout_actor_weights`
- CPU unit test: `test_weight_sync`
- CPU unit test: `test_rollout_worker`
- plugin loader unit test: `test_plugin_worker`
- NCCL test coverage inside `test_nccl_collectives`

`gsm8k_grpo_trainer` may still export HF checkpoints for debugging or offline
inspection, but checkpoint reload is not the intended online parameter-update
path. A production rollout integration should register rollout-side parameter
tensors and call `broadcast_parameters_from_root` after each optimizer step or
after each configured synchronization interval.

The integration boundary is a rollout worker interface:

```cpp
struct RolloutWorker {
  virtual GenerationOutput generate(const TokenBatch& prompts,
                                    const GenerationConfig& config) = 0;
  virtual std::vector<cverl::distributed::ParameterView> actor_parameters() = 0;
};
```

vLLM, FlashAttention, and Megatron are still important dependencies to reuse.
The reuse point is their CUDA kernels, paged-attention cache, scheduler,
tensor-parallel sharding, pipeline scheduling, and optimizer/communication
patterns. The boundary must be C++/CUDA worker/plugin APIs exposing GPU tensors,
not HTTP endpoints.

See `docs/rollout-worker-plugin.md` for the plugin ABI.
