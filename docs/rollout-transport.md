# Rollout Transport Plan

The first complete GRPO/GSM8K chain should reuse vLLM or SGLang for text
generation through their OpenAI-compatible HTTP endpoints. That path is stable
and keeps cverl independent from rollout-server internals.

For high-throughput colocated training, the transport should be upgraded in
stages:

1. HTTP JSON transport for compatibility with stock vLLM/SGLang.
2. POSIX shared memory for local sidecars that exchange token IDs, logprobs,
   rewards, and sequence metadata without JSON serialization.
3. CUDA IPC for same-node GPU tensors when the rollout sidecar can export CUDA
   memory handles.
4. NCCL send/recv or collectives for GPU-to-GPU transfer between cverl ranks and
   rollout ranks.
5. GPUDirect RDMA/NCCL network tuning for cross-node transfer.

Important boundary: stock vLLM/SGLang public serving APIs do not expose their
internal GPU tensors or NCCL communicators. Direct NCCL/RDMA transfer therefore
requires a custom worker/plugin/sidecar in the rollout process. Without that
cooperation, cverl can only optimize the client side around HTTP.

## Abstraction

All transports implement `cverl::rollout::RolloutTransport`:

- `include/cverl/rollout/transport.h`  — `RolloutRequest`, `RolloutResponse`,
  abstract base class, plus `LoopbackRolloutTransport` for tests.
- `include/cverl/rollout/http_transport.h` — `HttpRolloutTransport`, libcurl
  client for OpenAI-compatible `/v1/completions` and `/v1/chat/completions`.
- `include/cverl/rollout/shared_memory.h` — POSIX shared-memory primitive for
  local sidecars (CPU-side payloads).
- `include/cverl/rollout/shared_memory_transport.h` —
  `SharedMemoryRolloutTransport`, packs `RolloutRequest` / `RolloutResponse`
  into a shm region and signals round-trips via two named POSIX semaphores.
  `create_client()` is the trainer side; `attach_server()` + `serve_one()` /
  `serve_loop()` is the rollout sidecar side.

Trainer code talks to the abstract base class. Replacing HTTP with
shared-memory or CUDA IPC later does not require touching trainer logic.

## Pipeline binaries

`tools/rollout/gsm8k_rollout_pipeline.cc` exercises the upper half of the
chain (rollout + reward, no trainer):

```
gsm8k_rollout_pipeline \
  --dataset path/to/gsm8k.jsonl \
  --transport loopback \
  --prompts 4 --n 4 \
  --reward-method strict
```

`tools/rollout/gsm8k_grpo_smoke.cc` runs the full closed loop on CPU:
GSM8K -> RolloutTransport -> rule reward -> Tokenizer -> TinyCausalPolicy
logprobs -> GRPO advantages -> PPO clipped step. KL penalty against a frozen
reference policy is enabled with `--kl-coef > 0` and `--kl-penalty
{k1,abs,k2,k3}`. The tokenizer is selected via `--tokenizer {byte,hf}`;
`hf` requires `--tokenizer-path /path/to/tokenizer.json` and uses
HfBpeTokenizer (pure C++ HF byte-level BPE).

```
gsm8k_grpo_smoke \
  --dataset path/to/gsm8k.jsonl \
  --transport loopback \
  --tokenizer hf --tokenizer-path /path/to/Qwen3.5/tokenizer.json \
  --prompts 4 --n 4 --steps 4 \
  --max-prompt-tokens 96 --max-response-tokens 64 \
  --hidden-dim 16 \
  --kl-coef 0.05 --kl-penalty k2
```

## Tokenizer backends

All tokenizer-aware code (rollout batch builder, smoke loop) takes a
`const cverl::text::Tokenizer&`. Two backends ship today:

- `cverl::text::ByteTokenizer` — vocab=260, byte-per-id, dependency-free.
  CPU smokes / CI only.
- `cverl::text::HfBpeTokenizer` — pure C++, loads HF `tokenizer.json`,
  byte-level BPE matching Qwen / Llama-3 / GPT-2 family bit-for-bit.
  Cross-validated against HF Python via
  `tools/text/dump_hf_tokenizer_fixtures.py` + `tests/text/fixtures/...`.

A future Rust-backed `HfTokenizersCpp` wrapper (mlc-ai/tokenizers-cpp) can
plug into the same interface for parity verification on GPU machines that
have the Rust toolchain.

Both binaries accept `--transport http --base-url http://host:8000 --model
<model-id>` to drive a real vLLM/SGLang server. The trainer code (advantage
computation, PPO loss, optimizer step) does not change between transports.

## Tests

- `tests/rollout/test_shared_memory.cc` — cross-process POSIX shm round-trip.
- `tests/rollout/test_shared_memory_transport.cc` — fork()-based round trip
  through `SharedMemoryRolloutTransport`: parent is the client, child is an
  echo server, both ends serialize and round-trip a full `RolloutRequest` /
  `RolloutResponse`.
- `tests/rollout/test_transport.cc` — abstract interface + loopback transport.
- `tests/rollout/test_http_transport.cc` — drives `HttpRolloutTransport`
  against a tiny in-process loopback HTTP server that emulates the
  `/v1/completions` response shape, verifying request body fields, response
  parsing and per-prompt routing.
- `tests/reward/test_gsm8k_reward.cc` — strict + flexible extraction, ground
  truth normalization, and end-to-end reward scoring.
- `tests/text/test_byte_tokenizer.cc` — encode/decode round trip, BOS/EOS,
  truncation, non-ASCII bytes.
- `tests/rollout/test_rollout_batch.cc` — tokenized batch shape, left-padded
  prompts, right-padded responses, mask alignment, reward routing.
- `tests/rollout/test_gsm8k_grpo_smoke.cc` — fake rollout response ->
  TinyCausalPolicy logprobs -> one PPO step; asserts finite loss and that
  parameters actually moved.
- `tests/torch/test_reference_policy_kl.cc` — `clone_as_reference` produces
  a frozen detached copy, KL is ~0 when policy==ref, ref stays put after
  policy updates, KL gradient flows back to the policy parameters.
- `tests/text/test_hf_bpe_tokenizer.cc` — round-trips a synthetic HF
  tokenizer.json + cross-validates encode/decode against fixtures generated
  by Python `tokenizers`.

All tests are wired into `make test` and run on CPU only.
