# vLLM Rollout Smoke

End-to-end smoke for the cverl GRPO loop running against a real vLLM
serving Qwen3.5-0.8B on H20:

```text
GSM8K JSONL -> vLLM HTTP rollout -> cverl C++ rule reward
                                  -> GRPO advantage
                                  -> Qwen3_5CausalLmPolicy fp32 PPO step
```

Full 24-layer Qwen3.5 (`QWEN_MAX_LAYERS=-1`) on the trainer side; vLLM and
the trainer share the same fp32 weights so the optimizer update is
meaningful.

## vLLM environment (H20 / CUDA 12.9 driver, torch cu128)

vLLM 0.22.0's pypi wheel links against `libcudart.so.13` (CUDA 13). On a
CUDA 12 box you need the **`+cu129`** wheel from `wheels.vllm.ai` — it
links `libcudart.so.12` and runs on cu128 driver/runtime.

```bash
CONDA=/path/to/miniconda3
$CONDA/bin/conda create -p /home/$(id -un)/vllm-env python=3.12 -y --offline

VENV=/home/$(id -un)/vllm-env
PIP="$VENV/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple/ --no-cache-dir"

# 1. torch + supporting wheels (cu128).
#    pypi default torch wheel is cu130; we go through pytorch.org's cu128
#    index via the corp HTTP proxy.
$VENV/bin/pip install --proxy http://star-proxy.oa.com:3128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --no-cache-dir torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128

# 2. vLLM core: install --no-deps from the wheels.vllm.ai cu129 wheel.
$PIP --no-deps "vllm==0.22.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cu129/

# 3. Server-essential deps (subset of vllm 0.22.0 requirements; we skip
#    the cu13-locked ones: humming-kernels[cu13], tilelang, tokenspeed-mla,
#    nvidia-cutlass-dsl[cu13], quack-kernels - they are not needed for
#    standard text generation on Qwen3.5).
$PIP regex cachetools psutil sentencepiece numpy requests tqdm blake3 \
     py-cpuinfo "tokenizers>=0.21.1" "safetensors>=0.6.2" \
     "protobuf>=5.29.6" "fastapi[standard]>=0.115.0" "aiohttp>=3.13.3" \
     "openai>=2.0.0" "pydantic>=2.12.0" "prometheus_client>=0.18.0" pillow \
     "prometheus-fastapi-instrumentator>=7.0.0" "tiktoken>=0.6.0" \
     lm-format-enforcer==0.11.3 "outlines_core==0.2.14" diskcache==5.6.3 \
     lark==1.2.2 "xgrammar>=0.2.0" "typing_extensions>=4.10" "filelock>=3.16.1" \
     partial-json-parser "pyzmq>=25.0.0" msgspec "gguf>=0.17.0" \
     "mistral_common[image]>=1.11.2" "opencv-python-headless>=4.13.0" pyyaml einops \
     "compressed-tensors==0.15.0.1" depyf==0.20.0 cloudpickle watchfiles \
     python-json-logger ninja pybase64 cbor2 ijson setproctitle "openai-harmony>=0.0.3" \
     anthropic numba==0.65.0 "apache-tvm-ffi==0.1.9" \
     "fastsafetensors>=0.2.2" "llguidance>=1.7.0" \
     "model-hosting-container-standards>=0.1.14" mcp \
     "opentelemetry-sdk>=1.27.0" "opentelemetry-api>=1.27.0" \
     "opentelemetry-exporter-otlp>=1.27.0" "opentelemetry-semantic-conventions-ai>=0.4.1" \
     pynvml tabulate transformers==4.57.1

$PIP --no-deps "setuptools<81.0.0" "nvidia-cudnn-frontend<1.19.0,>=1.13.0" "six>=1.16.0"
$PIP --no-deps "flashinfer-python==0.6.11.post2" "flashinfer-cubin==0.6.11.post2"
$PIP --no-deps "nvidia-cutlass-dsl>=4.5.0"      # plain cu12 cutlass-dsl, not [cu13]
$PIP --no-deps "cuda-bindings>=12.9.4,<13"      # torch cu128 needs <13
```

Verify:

```bash
$VENV/bin/python -c "import torch, vllm, flashinfer; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
# 2.11.0+cu128 True 0.22.0
```

## Running the smoke

```bash
VLLM_PYTHON=/home/$(id -un)/vllm-env/bin/python \
BUILD_DIR=/path/to/cverl-build/gpu-nccl \
MODEL_PATH=/path/to/Qwen3.5-0.8B \
PORT=8021 \
QWEN_MAX_LAYERS=-1 PROMPTS=1 N=4 STEPS=8 KL_COEF=0.05 \
examples/run_vllm_gsm8k_smoke.sh
```

What the script does:

1. Launches `vllm.entrypoints.openai.api_server` on `${VLLM_DEVICES:-0}`
   with `--max-model-len 1024 --dtype bfloat16
   --gpu-memory-utilization 0.45 --trust-remote-code` (defaults that
   leave ~50 GiB on H20 for the trainer).
2. Polls `/v1/models`, sends one `/v1/completions` probe.
3. Runs `gsm8k_grpo_smoke --policy qwen --tokenizer hf --device cuda
   --endpoint chat --qwen-max-layers -1` on `${TRAINER_DEVICES:-1}` with
   a system prompt that asks for a `#### N` final-line answer.
4. Trainer side has `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   set to handle the activation fragmentation pattern that the linear-
   attention 256-step loop generates.

## Memory ceiling on a single H20 (24-layer fp32 trainer)

vLLM holds ~43 GiB on its GPU; the trainer holds Qwen3.5 fp32 weights
(~3.2 GiB), grad (~3.2 GiB), AdamW state (~6.4 GiB), and activations.
The dominant cost is the linear-attention forward: it loops over the
sequence in Python and keeps every intermediate state alive for
autograd. On H20 the working configurations are:

| B | P+R | result |
|--|--|--|
| 2 | 384 | OK, ~50 GiB activations |
| 4 | 256 | OK |
| 4 | 384 | OK |
| 4 | 512 | OOM |
| 8 | 256 | OK |
| 8 | 384 | OOM |

The script defaults are `PROMPTS=1 N=4 MAX_PROMPT_TOKENS=128
MAX_RESPONSE_TOKENS=256` (B=4, ctx=384) which fits.

## What the numbers should look like

H20, vLLM serving Qwen3.5-0.8B bf16, trainer 24-layer fp32, 1 prompt × 4
samples × 8 steps × 1 PPO epoch, KL_COEF=0.05:

```text
step,transport,total_seq,mean_reward,success_rate,no_answer,loss,pg_loss,kl_loss,ppo_kl,clipfrac,param_delta,seconds
1,http(chat@127.0.0.1:8021),4,1.000000,1.000000,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.572016
2,http(chat@127.0.0.1:8021),4,0.000000,0.000000,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.598597
...
7,http(chat@127.0.0.1:8021),4,0.250000,0.250000,0,0.034465,0.034465,0.000000,0.000000,0.000000,7813.044868,0.598548
8,http(chat@127.0.0.1:8021),4,0.000000,0.000000,0,0.018944,0.000000,0.378875,0.000000,0.000000,6552.989591,0.600023
```

`param_delta` is the L1 change across **all** 24-layer Qwen3.5 fp32
parameters. It's 0 on steps where every group has identical reward
(within-group std=0 → GRPO advantage=0 → pg_loss=0). It is non-zero on
step 7 (1/4 correct → real PPO gradient) and on step 8 (KL gradient
flows even when reward variance is 0, plus AdamW momentum from step 7).

`seconds` per rollout is dominated by vLLM batched generation (~0.6 s for
4 samples × 256 tokens; roughly 4× faster than SGLang at the same shape
on this box).

## Knobs

| env var | default | effect |
|--|--|--|
| `VLLM_PYTHON` | `/home/$(id -un)/vllm-env/bin/python` | python with vllm installed |
| `BUILD_DIR` | `${ROOT}/build-h20-nccl` | where `gsm8k_grpo_smoke` lives |
| `MODEL_PATH` | `${ROOT}/../models/Qwen3.5-0.8B` | safetensors + tokenizer.json |
| `VLLM_DEVICES` / `TRAINER_DEVICES` | `0` / `1` | per-side CUDA device |
| `GPU_MEMORY_UTILIZATION` | `0.45` | vLLM memory pool |
| `MAX_MODEL_LEN` | `1024` | vLLM max ctx |
| `POLICY` | `qwen` | trainer backbone |
| `QWEN_MAX_LAYERS` | `-1` | full 24 layers |
| `KL_COEF` / `KL_PENALTY` | `0.05` / `k2` | KL penalty against frozen policy |
| `WEIGHT_DECAY` | `0.0` | AdamW weight decay |
| `LR` | `3e-5` | AdamW LR |
| `TEMPERATURE` / `TOP_P` | `1.1` / `0.95` | sampling for within-group variance |
| `STEPS` / `PPO_EPOCHS` / `PROMPTS` / `N` | `8 / 1 / 1 / 4` | trainer batch shape |
| `MAX_TOKENS` / `MAX_PROMPT_TOKENS` / `MAX_RESPONSE_TOKENS` | `256 / 128 / 256` | length budgets |
| `SYSTEM_PROMPT` | (math tutor `#### N` recipe) | chat template prompt |

## Notes vs. SGLang

- vLLM doesn't need the `decord` workaround for `Qwen3_5ForConditionalGeneration`;
  it loads the language model directly.
- vLLM doesn't need the `--mamba-backend flashinfer` flag — its Qwen3.5
  model handles linear attention internally.
- vLLM rollout latency is ~0.6 s/batch vs SGLang ~14 s for the same
  shape; the difference is that SGLang 0.5.9's `--disable-cuda-graph
  --sampling-backend pytorch` falls back to the eager path while vLLM
  uses a compiled CUDA graph by default.

## Future work

The 24-layer ceiling on H20 is dominated by the cverl
`linear_attention` per-token loop (one allocation per token × 18 layers
× 256 tokens = ~4600 live tensors in the autograd tape). To raise the
ceiling without touching the algorithm:

- Activation checkpointing per decoder layer.
- Stop-grad through `linear_attention` and only train MLPs +
  `full_attention` (still 6 layers × 4 weights = 24 trained tensor
  groups, dominant compute path).
- Rewrite the per-token loop as a single `chunkwise_recurrent`
  kernel — probably the biggest single perf win.
