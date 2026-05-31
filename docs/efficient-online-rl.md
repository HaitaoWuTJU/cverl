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
- `cverl::rollout::NCCLGpuBatchSender`
- `cverl::rollout::NCCLGpuBatchReceiver`
- CPU unit test: `test_weight_sync`
- CPU unit test: `test_rollout_worker`
- plugin loader unit test: `test_plugin_worker`
- NCCL test coverage inside `test_nccl_collectives`
- NCCL rollout batch test coverage inside `test_nccl_gpu_batch_transport`

`NCCLGpuBatchSender` sends one descriptor header followed by dtype-coalesced
GPU slabs. A typical PPO rollout batch has int64 token/group IDs and float
mask/logprob/reward/advantage tensors, so the data plane uses header + int64
slab + float slab instead of one NCCL message per tensor.

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

vLLM is the first rollout backend to reuse. The reuse point is its paged
attention/KV cache, continuous batching scheduler, FlashAttention backends,
tensor-parallel serving, and Native RL weight-transfer path. cverl should only
replace those pieces after measurement shows vLLM is the bottleneck.

See `docs/rollout-worker-plugin.md` for the plugin ABI.
See `docs/vllm-native-rl.md` for the vLLM Native RL control/sync path.
