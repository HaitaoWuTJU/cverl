# SGLang Rollout Smoke

This repo can use SGLang through the same OpenAI-compatible HTTP transport used
for vLLM.

On the H20 CUDA 12.9 node, the tested setup was:

```bash
python -m pip install --no-cache-dir --no-deps sglang==0.5.12.post1
python -m pip install --no-cache-dir IPython
python -m pip install --no-cache-dir --no-deps \
  sglang-kernel==0.4.2.post2 \
  --index-url https://docs.sglang.ai/whl/cu129/
python -m pip install --no-cache-dir --no-deps soundfile==0.13.1
```

The full smoke can be run with:

```bash
PATCH_SGLANG_ACTIVATION_JIT=1 \
BUILD_DIR=build-h20-nccl \
MODEL_PATH=../models/Qwen3.5-0.8B \
examples/run_sglang_gsm8k_smoke.sh
```

The script starts SGLang, checks `/v1/models`, sends one `/v1/completions`
probe, then runs:

```text
GSM8K -> SGLang HTTP rollout -> C++ GSM8K reward -> GRPO advantage -> PPO clipped update
```

The trainer side now uses **the real Qwen3.5 policy** (`--policy qwen
--model-dir $MODEL_PATH`) and the **HF tokenizer** (`--tokenizer hf
--tokenizer-path $MODEL_PATH/tokenizer.json`), so the PPO step actually
updates Qwen3.5 weights that share the same vocabulary as SGLang's
serving model. Earlier versions of this script trained an unrelated
ByteTokenizer/TinyCausalPolicy mini-MLP — its `param_delta` did not
reflect anything SGLang generated.

Defaults (overridable via env):

- `POLICY=qwen` — switch to `tiny` for the old vocab=260 ByteTokenizer
  trainer if you want to debug the rollout side without paying the Qwen
  forward cost.
- `QWEN_MAX_LAYERS=2` — only the first N transformer blocks participate
  in the trainer forward/backward. Set `-1` to train every layer.
- `TRAINER_DEVICE=cuda`, `TRAINER_DEVICES=1` — SGLang gets GPU 0,
  trainer gets GPU 1 by default. Set `TRAINER_DEVICE=cpu` for a
  fall-back debug path.
- `WEIGHT_DECAY=0.0` — AdamW weight decay. Stays at 0 by default so a
  zero-reward batch yields `param_delta=0` rather than silently
  decaying every parameter.
- `LR=3e-5`, `KL_COEF=0.0`, `STEPS=1`, `PPO_EPOCHS=1`,
  `MAX_PROMPT_TOKENS=256`, `MAX_RESPONSE_TOKENS=256`,
  `REWARD_METHOD=flexible`, `TEMPERATURE=0.9`, `TOP_P=0.95`.

Current H20 result with Qwen3.5-0.8B (legacy `--policy tiny`,
`--tokenizer byte`):

```text
total_seq=16
mean_reward=0.187500
success_rate=0.187500
param_delta=27.592876
rollout+update seconds=16.590672
```

The new `--policy qwen --tokenizer hf` path produces a `param_delta` on
the actual Qwen3.5 fp32 parameters and is the value to compare across
runs.

Notes:

- Use the cu129 `sglang-kernel` wheel. The default PyPI wheel tried to load
  CUDA 13 libraries on this node.
- `--linear-attn-backend flashinfer --mamba-backend flashinfer` avoided the
  Triton linear-attention dtype mismatch seen with `--dtype float16`.
- `PATCH_SGLANG_ACTIVATION_JIT=1` applies a local compatibility patch to
  SGLang's installed `activation.cuh`; without it, CUDA 12.9 NVCC failed to
  compile the bf16 activation JIT for this SGLang wheel.
