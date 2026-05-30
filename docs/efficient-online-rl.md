# Efficient Online RL Path

HTTP is a debug transport only. It is useful for bootstrapping rollout against
stock vLLM/SGLang, but it is the wrong mechanism for maximum-efficiency online
RL because:

- token batches are serialized through JSON and copied through CPU memory;
- parameter updates require checkpoint materialization or server reload;
- stock OpenAI-compatible endpoints do not expose GPU tensor pointers,
  CUDA IPC handles, or NCCL communicators.

The efficient path is:

1. Trainer actor, reference, and optimizer own GPU parameters through LibTorch.
2. Rollout workers are colocated C++/CUDA sidecars or vLLM/SGLang worker
   plugins, not external HTTP clients.
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
- CPU unit test: `test_weight_sync`
- NCCL test coverage inside `test_nccl_collectives`

`gsm8k_grpo_trainer` may still export HF checkpoints for debugging or offline
inspection, but checkpoint reload is not the intended online parameter-update
path. A production rollout integration should register rollout-side parameter
tensors and call `broadcast_parameters_from_root` after each optimizer step or
after each configured synchronization interval.

The next integration boundary is a rollout worker interface with three calls:

```cpp
struct RolloutWorker {
  virtual void generate(const TokenBatch& prompts, TokenBatch* responses) = 0;
  virtual std::vector<cverl::distributed::ParameterView> actor_parameters() = 0;
  virtual void synchronize_actor_weights(int64_t trainer_root) = 0;
};
```

For stock vLLM/SGLang this requires a worker/plugin/sidecar, because their HTTP
server process does not expose internal weight tensors to a C++ client.
