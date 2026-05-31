# Distributed Runtime Plan

This document defines the distributed shape that `cverl` code should target.
The implementation should use mature communication libraries first: NCCL for GPU
collectives, CUDA streams/events for overlap, and LibTorch/ATen tensors as the
initial integration surface.

## Parallel Axes

`cverl::distributed::Topology` uses four orthogonal axes:

- Data parallelism (`DP`): independent replicas over different prompt/response
  batches. Gradients are reduced across the DP group.
- Tensor parallelism (`TP`): shard large matrix multiplications and attention
  heads inside a layer. TP ranks should remain on one node whenever possible so
  collectives use NVLink/NVSwitch rather than inter-node fabric.
- Pipeline parallelism (`PP`): split transformer layers into stages. Activation
  tensors move point-to-point between adjacent stages.
- Context parallelism (`CP`): shard the sequence/context dimension for long
  contexts. CP ranks keep the same model-layer ownership and exchange sequence
  shards through all-gather/ring-attention style collectives.

The default rank mapping is:

```text
global_rank = (((data_rank * pipeline_parallel) + pipeline_rank)
               * context_parallel + context_rank)
              * tensor_parallel + tensor_rank
```

This keeps each TP group contiguous in rank space. Launchers should map
contiguous TP ranks onto GPUs connected by NVLink/NVSwitch. CP groups should
also stay within a node for short/medium contexts; for very long contexts they
can span nodes only when the attention implementation overlaps communication.

## Communication Patterns

Expected collectives by parallel axis:

- DP:
  - reduce-scatter gradients after backward
  - all-gather parameters before forward when optimizer state is sharded
  - all-reduce only for small scalar metrics
- TP:
  - all-reduce or reduce-scatter/all-gather around row/column parallel linear
    layers
  - all-gather KV/head shards only where the attention implementation requires
    full heads
- PP:
  - send/recv activations in forward
  - send/recv activation gradients in backward
  - schedule with enough micro-batches to keep stages occupied
- CP:
  - shard sequence tensors before attention
  - all-gather or ring-exchange context shards where attention needs remote
    tokens
  - reduce-scatter sequence gradients when full-context activations were
    materialized

`include/cverl/distributed/collectives.h` is intentionally an interface. CPU
tests use `SingleProcessCollectives`; CUDA/NCCL builds can enable
`NcclCollectives` with `-DCVERL_ENABLE_NCCL=ON`.

`NcclCollectives` supports two completion modes. The default constructor mode
keeps legacy smoke tests simple by synchronizing the host after every
collective. The production PP/TP PPO trainer passes
`--nccl-sync-after-collective false` by default: NCCL work is ordered on a
dedicated non-blocking stream, returned tensors are made visible to the current
CUDA stream with events, and temporary inputs are kept alive until completion.
Explicit barriers, checkpoint boundaries, and communicator destruction still
call `synchronize()`.

The PP/TP trainer also keeps per-micro-batch loss metrics on device and only
gathers one compact metric tensor per step. Do not call `Tensor::item()` inside
the 1F1B micro-batch loop; it serializes the CUDA stream and destroys overlap.
Rank-0 step metric aggregation uses direct CPU tensor accessors after the
single all-gather transfer; avoid reintroducing per-field `item()` dispatch.
Gradient buckets avoid `torch::cat` for single-tensor flushes; preserve this
fast path for small replicated TP parameters and bucket-boundary tail tensors.
Single-rank DP gradient sync and reduce-scatter return local gradients without
issuing collectives; this matters for PP/TP-only bring-up runs.
Qwen TP replicated parameters are classified once at trainer startup; the
per-step gradient sync should only walk the precomputed tensor list.
Pipeline activation slot vectors are allocated once before the step loop and
cleared in place each step to avoid allocator churn in long PPO/GRPO runs.
Static topology helpers such as model-parallel rank lists and DP bucket numel
are computed before the loop; do not rebuild them every step.

## Current TP/DP Implementation

The first sharding layer is implemented in
`include/cverl/distributed/parallel_ops.h`:

- `column_parallel_linear`: shard output rows of a projection.
- `row_parallel_linear`: shard input columns and all-reduce partial outputs.
- `tensor_parallel_mlp_swiglu`: shard Qwen/SwiGLU gate and up projections by
  intermediate dimension, then shard down projection by input dimension.
- `data_parallel_sync_gradients`: all-reduce or average parameter gradients
  across a DP group. It accepts an optional communication dtype, so BF16/FP16
  training can choose FP32 gradient communication for numerical stability or
  BF16/FP16 communication to save bandwidth.
- `qwen3_5_pp_tp_ppo_trainer` exposes this policy as
  `--dp-grad-comm-dtype model|float32|bfloat16|float16` and
  `--tp-grad-comm-dtype model|float32|bfloat16|float16`. `model` keeps the
  existing per-gradient dtype behavior; explicit dtypes force bucket
  all-reduce/reduce-scatter payload dtype.

Qwen3.5 has TP entry points for:

- MLP: gate/up column-parallel, down row-parallel.
- Full attention: q projection and o projection are sharded by attention heads.
  The current safe path requires `TP <= num_key_value_heads`; for
  Qwen3.5-0.8B, use `TP=2` and combine the remaining GPUs with DP until the
  replicated-KV `TP > KV heads` path has a dedicated golden test.
- Linear attention: q/k/v projection rows, depthwise conv channels, z/a/b
  projections, recurrent head state, and out projection are sharded by linear
  heads.

These entry points are separate from the dense forward path so correctness can
be preserved while the trainer is migrated stage by stage.

The current 4-GPU validation target is `DP=2,TP=2` on H20:

- TP groups run Qwen3.5 dense-vs-TP hidden-state comparison through the first
  full-attention layer.
- DP groups synchronize gradients for matching TP ranks.
- `NCCL_SOCKET_IFNAME=eth1` is required on the tested container.

## Memory Policy

The runtime config separates memory decisions from model code:

- BF16 parameters and activations by default.
- FP32 reductions where needed for numerical stability.
- Activation checkpointing enabled by default.
- Qwen linear-attention CUDA training defaults to chunk checkpointing rather
  than full-state saving or pure recompute. Full-state mode is only for golden
  tests/debugging; pure recompute is available by env override and runs a
  tiled recompute backward kernel without allocating full states, checkpoints,
  or per-chunk replay cache. Chunk replay backward can also run in compact
  per-value-tile mode, reducing temporary replay cache from
  `B*H*chunk*K*V` to `B*H*chunk*K*value_tile` at the cost of serializing value
  tiles inside each checkpoint chunk.
- Sharded optimizer and sharded gradients enabled by default.
- DP flat sharded AdamW supports both Megatron-style no-master BF16/FP16
  parameter shards and FP32 master shards. Gradients and Adam moments stay
  FP32 in both modes.
- Flat sharded AdamW keeps grad-norm math on device and transfers one compact
  norm tensor to host per step. Avoid splitting this back into separate
  `Tensor::item()` calls on local norm, global norm, and clipped norm.
- Non-flat AdamW exposes tensor-valued grad norm helpers so the PP/TP trainer
  can pack local sq norm, global sq norm, and local norm into one host transfer.
- Flat DP optimizer skips reduce-scatter and parameter all-gather when
  `DP=1`; PP/TP-only runs should not pay for degenerate DP collectives.
- DP flat optimizer checkpoints default to shard-only format: each DP rank
  writes its flat parameter shard and Adam states, not a duplicate full
  parameter copy. Resume reconstructs local parameters with DP all-gather.
  `--flat-checkpoint-save-model-params true` keeps the older debug-friendly
  full-parameter rank checkpoint format.
- PP/TP trainer checkpoint manifests record the training policy that changes
  numerical behavior or restore compatibility: model dtype, DP/TP gradient
  communication dtype, bucket sizes, micro-batch/sequence shape, PPO loss
  settings, and optimizer sharding/master-weight mode.
- CPU offload is available as a policy switch, not a default. It saves memory
  but usually costs throughput.

For Qwen-style models, memory pressure should be handled in this order:

1. BF16 weights/activations.
2. Activation checkpointing on transformer blocks.
3. Gradient accumulation through micro-batches.
4. ZeRO-style optimizer and gradient sharding across DP ranks.
5. Tensor parallelism for large projections and attention heads.
6. Pipeline parallelism when the model no longer fits in one TP group.
7. CPU/NVMe offload only when fitting the model is more important than speed.

## GPU and Network Utilization

Launcher responsibilities:

- Set `CUDA_VISIBLE_DEVICES` so local ranks map to physically close GPUs.
- Keep TP groups within a node when `tensor_parallel <= gpus_per_node`.
- Spread DP groups across nodes to use inter-node bandwidth for large gradient
  reductions.
- Use multiple NICs through `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`.
- Provide a topology file through `NCCL_TOPO_FILE` when the cluster needs manual
  routing hints.

`Topology::nccl_env()` generates baseline NCCL environment values from
`NetworkPolicy`; production launchers can add cluster-specific tuning.

On the H20 test node, NCCL needed `NCCL_SOCKET_IFNAME=eth1`; without it,
bootstrap failed with `no socket interface found`. The CMake path also needs to
prefer the NCCL package bundled with the CUDA-enabled PyTorch wheel over older
system NCCL libraries.

## Integration Order

1. Keep CPU topology tests strict and deterministic.
2. Add NCCL collectives implementation for GPU builds.
3. Add multi-rank Qwen module correctness tests against dense outputs.
4. Add PP stage ownership and activation send/recv around decoder layer ranges.
5. Add DP gradient reduce-scatter and optimizer state sharding.
6. Add memory accounting per rank and fail early when a plan cannot fit.
7. Connect the trainer to the TP/DP Qwen path.

The current CPU environment can validate topology, rank groups, config, and
single-process collectives. Actual NCCL throughput, stream overlap, and NIC
binding must be validated on the target GPU cluster.

## Implemented Runtime Primitives

Current code exposes the distributed shape directly:

- `Topology`: DP/PP/CP/TP rank mapping, data/tensor/pipeline/context/model
  groups, and per-stage layer ranges.
- `parallel_ops`: TP row/column parallel projections, Qwen MLP/attention TP
  paths, and DP gradient sync.
- `pipeline`: PP peer selection plus forward/backward activation send/recv
  wrappers on the `Collectives` interface.
- `context_parallel`: sequence-dimension shard/gather helpers used as the
  base for CP attention. It also provides a differentiable all-gather-KV
  causal-attention reference for local query shards. The attention math now
  uses FlashAttention-style streaming over KV blocks, maintaining row max,
  row sum, and value accumulator instead of materializing the full `[Q,K]`
  score/probability matrix. Qwen CP full attention consumes blocks in the
  same order as the CP ring schedule. The current ring-exchange path obtains
  those blocks through grouped `send_recv` and has an autograd wrapper that
  sends owner-gradient blocks around the reverse ring so K/V owners receive
  accumulated gradients without all-gather. Ring-exchanged KV attention now
  uses a recompute-autograd path: forward saves Q/K/V inputs and block
  metadata, not per-block score/probability tensors; backward recomputes the
  streaming attention and returns gradients to the ring-exchange owner-gradient
  path. Full industrial CP training still needs a fused CUDA ring-attention
  kernel and a lower-latency reduce-scatter ring schedule to replace the
  current per-owner ring accumulation.
- `qwen3_5_pp_tp_ppo_trainer`: accepts DP/PP/CP/TP topology dimensions and
  records all four axes in metrics and checkpoint manifests. For `CP>1,TP=1`,
  PP stages now pass local sequence shards, Qwen range forward keeps
  RMSNorm/MLP/projections local, full attention runs local-Q plus differentiable
  CP ring-exchanged K/V blocks, linear attention exchanges projected QKV/Z/B/A
  activations in rank order instead of gathering hidden states, and the final
  PPO logprob slice gathers hidden with autograd so gradients reduce-scatter
  back to owner shards. `TP>1` with `CP>1` is intentionally rejected until
  TP-sharded CP attention and MLP are fused into one Megatron-style execution
  path.

The next efficiency step is replacing the eager CP exchange paths with fused
ring-attention and recurrent-state/ring CUDA kernels, using a lower-latency
backward reduce-scatter schedule, and grouping PP send/recv plus TP/DP/CP
collectives into larger scheduled communication batches.

Run the CP/NCCL smoke on a multi-GPU node with:

```sh
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 examples/run_cp_attention_nccl_smoke.sh
```

CPU regression coverage includes:

- `test_distributed_topology`: CP topology, padded sequence sharding,
  streaming and recompute ring-attention math, CP=3 ring exchange
  owner-gradient accumulation, and gather/ring-exchange KV gradient checks.
- `test_qwen3_5_cp_forward`: synthetic Qwen3.5 full-attention layer with
  deterministic weights, validating that CP rank-local forward shards match
  the corresponding dense forward slices.
