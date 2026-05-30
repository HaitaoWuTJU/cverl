# Rollout Worker Plugin ABI

cverl's efficient rollout path uses `cverl::rollout::RolloutWorker`, not HTTP.
vLLM/Megatron integrations should expose a shared library that returns a worker
object backed by the engine's internal CUDA tensors.

Required symbols:

```cpp
extern "C" cverl::rollout::RolloutWorker* cverl_create_rollout_worker(
    const char* config_json);

extern "C" void cverl_destroy_rollout_worker(
    cverl::rollout::RolloutWorker* worker);
```

The plugin worker must implement:

- `generate(TokenBatch, GenerationConfig)`: accept GPU prompt token tensors and
  return GPU token/logprob tensors.
- `actor_parameters()`: return engine-side actor parameter tensors in the same
  order/names as the trainer policy, so cverl can synchronize them through NCCL
  or CUDA IPC without checkpoint serialization.

Expected backend mapping:

- vLLM plugin: wrap the worker/model runner where weights, KV cache, scheduler,
  and FlashAttention/paged-attention kernels already live.
- Megatron plugin: wrap TP/PP shards and expose shard-local parameters through
  `ParameterView`; cverl's DP/TP/PP groups then decide which ranks broadcast or
  reduce.

The plugin ABI intentionally does not define JSON/HTTP generation calls. The
only JSON string is startup configuration.
