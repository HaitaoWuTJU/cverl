# SGLang Rollout Smoke

End-to-end smoke for the cverl GRPO loop running against a real SGLang
serving Qwen3.5-0.8B on H20:

```text
GSM8K JSONL -> SGLang HTTP rollout -> cverl C++ rule reward
                                    -> GRPO advantage
                                    -> Qwen3_5CausalLmPolicy fp32 PPO step
```

The trainer side wraps `Qwen35TextModel` in a `Qwen3_5CausalLmPolicy`
nn::Module so AdamW + PPO autograd actually update the loaded fp32
parameters. SGLang and the trainer share the same Qwen3.5 weights, so
the optimizer step is meaningful (no Tiny / toy MLP in the loop).

## SGLang environment (H20 / CUDA 12.9 driver, torch cu128)

SGLang lives in its own Python env so we don't break the cverl build's
torch 2.11.0+cu128. Tested combo:

```bash
CONDA=/path/to/your/miniconda3
$CONDA/bin/conda create -p /home/$(id -un)/sglang-env python=3.12 -y --offline

SGENV=/home/$(id -un)/sglang-env
PIP="$SGENV/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple/ --no-cache-dir"

# Trainer core (SGLang's exact pin is torch==2.9.1; sgl-kernel 0.3.21 ABI-locks
# to that version).
$PIP --no-deps torch==2.9.1
$PIP transformers==4.57.1 numpy

# Server-essential deps (subset of sglang 0.5.9 requirements - we skip
# diffusion / extra grpc bits).
$PIP fastapi uvicorn uvloop psutil pyzmq orjson pydantic pillow scipy einops \
     sentencepiece tiktoken aiohttp requests setproctitle msgspec \
     partial_json_parser pybase64 packaging interegular nvidia-ml-py \
     prometheus-client python-multipart \
     outlines==0.1.11 xgrammar==0.1.27 llguidance==0.7.30 \
     compressed-tensors gguf hf_transfer modelscope ninja datasets \
     blobfile==3.0.0 build apache-tvm-ffi==0.1.5 nvidia-cutlass-dsl==4.3.4 \
     cuda-python==12.9 \
     openai==2.6.1 anthropic openai-harmony==0.0.4 \
     timm==1.0.16 torchao==0.9.0 torchaudio==2.9.1 torchvision \
     IPython decord2

# SGLang core + kernels.
$PIP --no-deps sglang==0.5.9 sgl-kernel==0.3.21
$PIP --no-deps torch_memory_saver==0.0.9 soundfile==0.13.1
$PIP --no-deps flashinfer_python==0.6.3 flashinfer_cubin==0.6.3

# Re-pin torch since outlines/torchao pulled it forward; sgl-kernel 0.3.21 ABI
# matches torch 2.9.1 only.
$PIP --no-deps torch==2.9.1

# cuDNN 9.10 has a known nn.Conv3d perf bug that sglang refuses to start with.
# We export SGLANG_DISABLE_CUDNN_CHECK=1 in the launcher anyway, but bumping
# the wheel is cleaner.
$PIP nvidia-cudnn-cu12==9.16.0.29
```

Verify:

```bash
$SGENV/bin/python -c "import sglang, sgl_kernel, flashinfer; print('ok')"
```

## Running the smoke

```bash
SGLANG_PYTHON=/home/$(id -un)/sglang-env/bin/python \
BUILD_DIR=/path/to/cverl/build-h20-nccl \
MODEL_PATH=/path/to/Qwen3.5-0.8B \
PORT=8011 \
PROMPTS=4 N=4 STEPS=2 PPO_EPOCHS=1 KL_COEF=0.05 \
examples/run_sglang_gsm8k_smoke.sh
```

What the script does:

1. Launches `sglang.launch_server` on `${SGLANG_DEVICES:-0}` with
   `--mamba-backend flashinfer --sampling-backend pytorch
   --disable-cuda-graph --dtype bfloat16 --mem-fraction-static 0.45
   --trust-remote-code` (these are the flags Qwen3.5-0.8B's hybrid
   linear/full-attention stack actually accepts in sglang 0.5.9; the
   older `--linear-attn-backend flashinfer` flag was removed).
2. Polls `/v1/models` until the server is ready.
3. Sends one `/v1/completions` probe.
4. Runs `gsm8k_grpo_smoke --policy qwen --tokenizer hf --device cuda
   --endpoint chat` on `${TRAINER_DEVICES:-1}` with a system prompt
   that asks for a `#### N` final-line answer (without it the base
   Qwen3.5 model rarely emits a parseable answer and reward is always
   0 -> advantage 0 -> no update).

## What the numbers should look like

H20, Qwen3.5-0.8B, 4 prompts × 4 samples × 2 steps × 1 PPO epoch:

```text
step,transport,total_seq,mean_reward,success_rate,no_answer,loss,pg_loss,kl_loss,ppo_kl,clipfrac,param_delta,seconds
1,http(chat@http://127.0.0.1:8011),16,0.375000,0.375000,0,-0.000509,-0.000509,0.000000,0.000000,0.000000,7967.436881,14.762884
2,http(chat@http://127.0.0.1:8011),16,0.125000,0.125000,0,0.244531,0.023786,4.414908,0.000000,0.000000,5074.043493,14.698378
```

`param_delta` here is on the actual fp32 Qwen3.5 weights (`--qwen-max-layers
2` truncates the layer stack to keep each iteration ~15 s on H20). `mean_reward
= 0.375` means SGLang generated correct GSM8K answers on 6 / 16 samples,
which is consistent with greedy Qwen3.5-0.8B math performance.

## Knobs

| env var | default | effect |
|--|--|--|
| `SGLANG_PYTHON` | `/home/$(id -un)/sglang-env/bin/python` | python with sglang+sgl_kernel installed |
| `BUILD_DIR` | `${ROOT}/build-h20-nccl` | where `gsm8k_grpo_smoke` lives |
| `MODEL_PATH` | `${ROOT}/../models/Qwen3.5-0.8B` | safetensors + tokenizer.json |
| `SGLANG_DEVICES` / `TRAINER_DEVICES` | `0` / `1` | per-side CUDA device split |
| `MEM_FRACTION_STATIC` | `0.45` | SGLang memory pool |
| `CONTEXT_LENGTH` | `1024` | SGLang max ctx |
| `POLICY` | `qwen` | trainer backbone (`qwen` or `tiny` for debug) |
| `QWEN_MAX_LAYERS` | `2` | trainer layer-stack truncation; `-1` for all |
| `KL_COEF` / `KL_PENALTY` | `0.0` / `k2` | KL penalty against frozen policy |
| `WEIGHT_DECAY` | `0.0` | AdamW weight decay (`0` so reward=0 batches don't decay) |
| `LR` | `3e-5` | AdamW LR |
| `TEMPERATURE` / `TOP_P` | `1.1` / `0.95` | sampling for non-trivial within-group variance |
| `STEPS` / `PPO_EPOCHS` / `PROMPTS` / `N` | `1 / 1 / 4 / 4` | trainer batch shape |
| `MAX_TOKENS` / `MAX_PROMPT_TOKENS` / `MAX_RESPONSE_TOKENS` | `256` | length budgets |
| `SYSTEM_PROMPT` | (math tutor `#### N` recipe) | chat template prompt |

## Notes

- The base `Qwen3.5-0.8B` model is registered as
  `Qwen3_5ForConditionalGeneration` (multimodal). SGLang's Qwen VL processor
  needs `decord` to import; without it sglang raises `No processor
  registered for architecture: ['Qwen3_5ForConditionalGeneration']` even
  for text-only requests. The install instructions above include `decord2`.
- `SGLANG_DISABLE_CUDNN_CHECK=1` is exported by the launcher.
- The trainer side runs the cverl build's torch (2.11.0+cu128, system NCCL
  + libcverl.a) so we set `LD_LIBRARY_PATH` separately from SGLang's env.
