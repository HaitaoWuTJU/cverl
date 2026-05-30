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
examples/scripts/run_sglang_gsm8k_smoke.sh
```

The script starts SGLang, checks `/v1/models`, sends one `/v1/completions`
probe, then runs:

```text
GSM8K -> SGLang HTTP rollout -> C++ GSM8K reward -> GRPO advantage -> PPO clipped update
```

Current H20 result with Qwen3.5-0.8B:

```text
total_seq=16
mean_reward=0.187500
success_rate=0.187500
param_delta=27.592876
rollout+update seconds=16.590672
```

Notes:

- Use the cu129 `sglang-kernel` wheel. The default PyPI wheel tried to load
  CUDA 13 libraries on this node.
- `--linear-attn-backend flashinfer --mamba-backend flashinfer` avoided the
  Triton linear-attention dtype mismatch seen with `--dtype float16`.
- `PATCH_SGLANG_ACTIVATION_JIT=1` applies a local compatibility patch to
  SGLang's installed `activation.cuh`; without it, CUDA 12.9 NVCC failed to
  compile the bf16 activation JIT for this SGLang wheel.
