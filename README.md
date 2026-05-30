# cverl

`cverl` is a C++/CUDA rewrite target for the RL training and inference stack
inspired by `verl`. The first milestone keeps a stable C ABI while using
LibTorch/ATen for tensor math, autograd, and optimizer-friendly C++ integration.

The project does not depend on Python at runtime, Ray, TensorDict, or the Python
PyTorch API. It does depend on LibTorch, either from a standalone LibTorch
install or from the `torch` Python wheel's bundled C++ package at build time.

## Current Scope

Implemented kernels:

- Masked reductions: `masked_sum`, `masked_mean`, `masked_whiten`
- KL penalty: `k1`, `abs`, `k2`, `k3`
- KL penalty backward gradient for `logprob`
- GAE advantage and returns
- GRPO outcome advantage
- PPO clipped policy loss
- PPO clipped policy loss backward gradient for `log_prob`

Current tensor support:

- contiguous `float32`
- dense 2D tensors with shape `[batch, seq]`
- CPU device

Current model support:

- Hugging Face safetensors metadata and tensor loading
- Qwen3.5 text-only transformer prefill forward
- Qwen3.5 linear attention, full attention, MLP, RMSNorm, RoPE, and tied lm head
- Native C++ CLI smoke for hidden states and logits

Current data support:

- Hugging Face datasets download bridge through Python `datasets`
- Prompt/answer JSONL materialization for C++ training loops
- Pure C++ JSONL reader for prompt/answer examples, using `simdjson` when
  available and falling back to `nlohmann/json`

Current distributed support:

- DP/TP/PP topology config and rank-group planning
- GPU/NIC/NCCL environment policy generation
- Memory policy for BF16/FP32 reduction, activation checkpointing, and sharding
- Optional NCCL collectives backend for CUDA builds
- Tensor-parallel linear helpers: column-parallel, row-parallel, and SwiGLU MLP
- Qwen3.5 TP entry points for MLP, full attention, and linear attention
- Data-parallel gradient all-reduce/average helper
- Single-process collectives for CPU tests

Current efficient rollout foundation:

- `RolloutWorker` interface for vLLM/Megatron/native CUDA worker plugins
- direct GPU parameter registration via `ParameterView`
- NCCL broadcast for trainer-to-rollout actor weight synchronization
- POSIX shared-memory region only for small CPU metadata/control messages

The C ABI wraps raw pointers into `torch::Tensor` views and delegates math to
LibTorch. Backward APIs use LibTorch autograd rather than manually derived
gradient loops.

CUDA support is scaffolded in the build system. GPU execution should come
through LibTorch/CUDA first; custom CUDA kernels should only be added for proven
hot spots.

## Layout

```text
cverl/
  include/cverl/        Public C ABI headers
  src/                  C ABI wrappers and LibTorch implementation
  cuda/                 CUDA implementation placeholder
  tests/                Native C++ tests
  tools/                Future golden-data and comparison tools
  docs/                 Design notes
  CMakeLists.txt        CMake build
  Makefile              Minimal build for CPU-only environments
```

## Build

### CMake

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build
./build/test_core_algos_cpu
./build/test_torch_backend
./build/test_simple_grpo_trainer
./build/test_distributed_topology
```

### Make

For environments with Python `torch` installed:

```sh
make test
```

## CUDA Build Path

The CUDA option is reserved for GPU environments:

```sh
cmake -S . -B build-cuda -DCVERL_ENABLE_CUDA=ON
cmake --build build-cuda
```

At the moment this only validates the CUDA build path. Kernel implementations
will be added after the LibTorch behavior is covered by golden tests.

## Distributed Runtime

The distributed runtime is being built around explicit DP/TP/PP topology:

```text
global_rank = ((data_rank * pipeline_parallel) + pipeline_rank) * tensor_parallel + tensor_rank
```

This keeps tensor-parallel groups contiguous so launchers can place them on
NVLink/NVSwitch-connected GPUs. Data-parallel groups are intended for
reduce-scatter/all-gather across replicas, and pipeline-parallel groups use
send/recv between adjacent stages with micro-batching.

The current CPU-buildable layer lives in:

- `include/cverl/distributed/topology.h`
- `include/cverl/distributed/collectives.h`
- `docs/distributed-runtime.md`

GPU builds should add an NCCL implementation behind the `Collectives` interface
instead of hand-writing communication primitives. A 4-GPU local smoke can be run
on a CUDA/NCCL node with:

```sh
cmake -S . -B build-h20-nccl \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DCVERL_ENABLE_CUDA=ON \
  -DCVERL_ENABLE_NCCL=ON
cmake --build build-h20-nccl

NCCL_SOCKET_IFNAME=eth1 \
NCCL_LIB_DIR=/path/to/python/site-packages/nvidia/nccl/lib \
WORLD_SIZE=4 \
tools/distributed/run_nccl_smoke.sh build-h20-nccl
```

For Qwen3.5-0.8B TP correctness with full-attention layers, use TP size 2
because the model has 2 KV heads:

```sh
NCCL_SOCKET_IFNAME=eth1 \
NCCL_LIB_DIR=/path/to/python/site-packages/nvidia/nccl/lib \
WORLD_SIZE=2 \
LAYERS=4 \
tools/distributed/run_qwen_tp_smoke.sh ./models/Qwen3.5-0.8B
```

The 4-GPU H20 smoke uses `DP=2,TP=2`: each data-parallel replica runs a
2-rank tensor-parallel Qwen forward check, and matching tensor ranks synchronize
gradients through a DP NCCL communicator:

```sh
NCCL_SOCKET_IFNAME=eth1 \
NCCL_LIB_DIR=/path/to/python/site-packages/nvidia/nccl/lib \
DP_SIZE=2 \
TP_SIZE=2 \
LAYERS=4 \
tools/distributed/run_qwen_dp_tp_smoke.sh ./models/Qwen3.5-0.8B
```

The `minimal_ppo_step` executable shows a native C++ PPO-style training step
using a LibTorch model, optimizer, and `cverl` RL losses:

```sh
./build/minimal_ppo_step
```

The `toy_grpo_trainer` executable runs a complete synthetic GRPO/PPO training
loop in C++:

```sh
./build/toy_grpo_trainer 32
```

It creates toy prompts, samples responses from a LibTorch policy, computes an
accuracy reward, builds GRPO advantages, applies PPO updates, and prints CSV
metrics.

Checkpointing is available through the trainer API and CLI:

```sh
./build/toy_grpo_trainer 8 4 4 --save-prefix /tmp/cverl-toy
./build/toy_grpo_trainer 8 4 4 --load-prefix /tmp/cverl-toy
```

## Hugging Face Downloads

`cverl` provides a small C++ wrapper around the Python `huggingface_hub`
download API. This keeps C++ training code in control while reusing Hugging
Face's supported cache, auth, revision, and pattern filtering behavior.

```sh
./build/hf_download Qwen/Qwen3.5-0.8B --dry-run
./build/hf_download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B
```

The wrapper requires `huggingface_hub` to be installed in the selected Python
environment. Use `--python /path/to/python` if the default `python3` is not the
right interpreter.

## Hugging Face Datasets

Datasets are loaded through a small C++ wrapper around Python `datasets`, then
materialized as prompt/answer JSONL. Native training code reads that JSONL from
C++ via `cverl::data::load_prompt_answer_jsonl`.

GSM8K smoke:

```sh
python -m pip install datasets
./build/hf_dataset_download gsm8k \
  --name main \
  --split train \
  --output-file ./data/gsm8k-train.jsonl \
  --max-examples 8 \
  --print-samples 1
```

The generated JSONL uses stable fields:

```json
{"prompt":"...","answer":"..."}
```

Use `--prompt-field` and `--answer-field` for datasets with different source
column names. For compatibility with newer Hugging Face dataset repository
names, the CLI maps `gsm8k` to `openai/gsm8k`.

## GSM8K Speed Compare

`tools/bench/compare_gsm8k_speed.sh` runs the same GSM8K GRPO/PPO micro-training
workload through Python `verl` core algorithms and native `cverl` LibTorch code.
Both sides use the same JSONL data, batch shape, seed, model dimensions, PPO
epochs, and optimizer settings:

```sh
source /data/home/marvinhtwu/miniconda3/etc/profile.d/conda.sh
conda activate verl-cpu

PYTHON=$(which python) \
VERL_DIR=../verl \
STEPS=32 \
PROMPTS=8 \
RESPONSES=4 \
SEQ_LEN=16 \
ACTION_DIM=64 \
HIDDEN_DIM=128 \
PPO_EPOCHS=2 \
tools/bench/compare_gsm8k_speed.sh
```

The script writes raw per-backend CSV metrics to `build/bench/` and prints
`cverl_vs_verl_speedup`. Set `DEVICE=cuda:0` when both builds/environments
support CUDA.

## Efficient Rollout

cverl's core rollout path must keep tokens and weights on GPU. Reuse vLLM
kernels/schedulers, FlashAttention, and Megatron TP/PP/optimizer components
through worker plugins or colocated sidecars, not through OpenAI HTTP APIs.
Rollout-side actor parameters are registered as GPU tensors and synchronized
from trainer ranks with NCCL/CUDA IPC.

See `docs/efficient-online-rl.md`.

## HF Model Loading

`cverl` can inspect a downloaded Hugging Face model directory from C++ and load
individual safetensors weights into LibTorch tensors:

```sh
./build/inspect_hf_model ./models/Qwen3.5-0.8B
./build/inspect_hf_model ./models/Qwen3.5-0.8B --find embed
./build/inspect_hf_model ./models/Qwen3.5-0.8B --load-tensor model.language_model.embed_tokens.weight
./build/inspect_hf_model ./models/Qwen3.5-0.8B --load-all
```

This covers config parsing, safetensors metadata, tensor indexing, and
single-tensor or full-weight loading into LibTorch tensors.

## Qwen3.5 Forward

`cverl` includes a first native C++/LibTorch Qwen3.5 text forward path:

```sh
./build/qwen3_5_forward ./models/Qwen3.5-0.8B --tokens 1,2 --layers 24
./build/qwen3_5_forward ./models/Qwen3.5-0.8B --tokens 1,2 --layers 24 --logits
```

CPU correctness can be checked against Hugging Face Transformers:

```sh
python tools/model/compare_qwen3_5_forward.py ./models/Qwen3.5-0.8B --tokens 1,2 --layers 24
python tools/model/compare_qwen3_5_forward.py ./models/Qwen3.5-0.8B --tokens 1,2 --layers 24 --logits
```

The comparison script runs the C++ executable, dumps the C++ output to a small
binary tensor format, runs the same input through Hugging Face on CPU, and
reports max/mean absolute error plus top-k token agreement for logits.

Implemented pieces:

- `embed_tokens`
- decoder layer residual structure
- Qwen3.5 RMSNorm weight convention
- linear attention via LibTorch depthwise `conv1d` and recurrent gated delta rule
- full attention with grouped KV heads, causal masking, RoPE, q/k norm, and output gate
- SwiGLU MLP
- final norm
- tied lm head through `embed_tokens.weight`

Current limitations:

- text-only path; vision/multimodal modules are not wired yet
- prefill/no-cache forward; incremental KV/recurrent cache is not implemented yet
- correctness is checked against Hugging Face CPU forward for hidden states and logits
- CPU path is intended for validation, not performance

The lightweight HF-weight RL regression path uses Qwen embeddings as the policy
input and action head over a small vocabulary subset:

```sh
./build/hf_embedding_grpo_trainer ./models/Qwen3.5-0.8B --steps 8 --vocab-subset 64
```

This verifies the path `HF safetensors -> LibTorch tensors -> GRPO/PPO training`
for fast CPU tests.

For end-to-end RL on the actual Qwen3.5 weights, use `gsm8k_grpo_trainer
--policy qwen --model-dir ./models/Qwen3.5-0.8B`. It wraps `Qwen35TextModel` in a
`Qwen3_5CausalLmPolicy` nn::Module so AdamW + PPO autograd actually
update the loaded fp32 parameters; `tests/torch/test_qwen3_5_grpo_step`
verifies the param_delta is non-zero on H20 with mixed-correctness GRPO
groups.

## API Example

```c
#include "cverl/cverl.h"

float rewards[8] = {0};
float values[8] = {0};
float mask[8] = {1};
float advantages[8];
float returns[8];

cverl_const_tensor2d_t rewards_t = {
    rewards, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_const_tensor2d_t values_t = {
    values, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_const_tensor2d_t mask_t = {
    mask, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_tensor2d_t adv_t = {
    advantages, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};
cverl_tensor2d_t ret_t = {
    returns, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, 2, 4};

cverl_status_t status = cverl_gae_advantage_return_f32_cpu(
    rewards_t, values_t, mask_t, 1.0f, 1.0f, adv_t, ret_t);
```

## Correctness Strategy

The LibTorch implementation is the source of truth for current kernels.
Golden-data tests can be generated from the Python `verl` implementation. The
current golden format covers forward outputs for KL, GAE, GRPO, PPO loss, plus
autograd reference gradients for KL with respect to `logprob` and PPO loss with
respect to `log_prob`.

1. Generate random inputs and reference outputs from `verl`.
2. Serialize them to a simple binary/JSON test format.
3. Load the same cases in C++.
4. Compare outputs with fp32 tolerances.

Example:

```sh
cmake -S . -B build
cmake --build build

PYTHONPATH=../verl python3 tools/golden_dump.py build/golden_core_algos.bin
./build/compare_golden build/golden_core_algos.bin
```

The `golden_dump.py` script requires an environment with `torch`, `numpy`, and
the Python `verl` checkout importable via `PYTHONPATH`.

Planned tolerance for fp32 kernels:

- `rtol = 1e-5`
- `atol = 1e-6`

## Roadmap

1. Add Python-generated golden tests for the current CPU kernels.
2. Use LibTorch CUDA for GPU execution and profile bottlenecks.
3. Add dtype support for `float16` and `bfloat16` where numerically appropriate.
4. Add a native tensor/batch abstraction beyond simple 2D tensors.
5. Connect the minimal trainer runtime to `Qwen35TextModel` log-prob generation.
6. Add NCCL-backed collectives and shard Qwen3.5 with DP/TP/PP.
7. Add native model backend pieces using existing C++ APIs where possible:
   optimizer/checkpointing via LibTorch, distributed collectives via NCCL, and
   attention/math via FlashAttention/cuBLAS/CUTLASS.

## Non-goals for the Current Milestone

- Full `verl` feature parity
- Ray-compatible scheduling
- Production LLM training runtime

Those pieces can be revisited after the native RL math core is stable and
covered by tests.
