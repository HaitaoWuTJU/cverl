#!/usr/bin/env python3
"""Online GSM8K GRPO loop with cverl training tensors and vLLM rollout.

This is the current agile integration path:
  * vLLM handles generation with its production scheduler/KV/attention stack.
  * cverl owns the trainable Qwen actor tensors and runs GRPO/PPO update.
  * The same live tensors are synchronized back to vLLM through Native RL NCCL.

HTTP is used only for rollout requests and small vLLM control endpoints here.
The weight payload path is NCCL, not checkpoint export/reload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests


def normalize_dtype_name(dtype_name: str) -> str:
    return dtype_name.split(".")[-1]


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_jsonl(path: str, prompt_field: str, answer_field: str, max_examples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"prompt": str(obj[prompt_field]), "answer": str(obj[answer_field])})
            if max_examples >= 0 and len(rows) >= max_examples:
                break
    if not rows:
        raise RuntimeError(f"empty dataset: {path}")
    return rows


def answer_number(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.rsplit(marker, 1)[1].strip()
    return answer.strip().split()[-1]


def oracle_rollout(prompts: list[str], answers: list[str], n: int) -> tuple[list[str], list[int], list[int]]:
    responses: list[str] = []
    prompt_indices: list[int] = []
    sample_indices: list[int] = []
    for pidx, ans in enumerate(answers):
        gold = answer_number(ans)
        for sidx in range(n):
            responses.append(f"Reasoning...\n#### {gold if sidx % 2 == 0 else '9999'}")
            prompt_indices.append(pidx)
            sample_indices.append(sidx)
    return responses, prompt_indices, sample_indices


def vllm_rollout(
    base_url: str,
    model: str,
    prompts: list[str],
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> tuple[list[str], list[int], list[int]]:
    responses: list[str] = []
    prompt_indices: list[int] = []
    sample_indices: list[int] = []
    for pidx, prompt in enumerate(prompts):
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed + pidx,
        }
        doc = post_json(base_url, "/v1/completions", payload, timeout)
        choices = doc.get("choices", [])
        if len(choices) != n:
            raise RuntimeError(f"vLLM returned {len(choices)} choices for prompt {pidx}, expected {n}")
        for sidx, choice in enumerate(choices):
            responses.append(str(choice.get("text", "")))
            prompt_indices.append(pidx)
            sample_indices.append(sidx)
    return responses, prompt_indices, sample_indices


class VllmNcclSync:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.group = None
        self.initialized = False

    def sync(self, policy: Any) -> None:
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        params = policy.named_parameters()
        update_info = policy.update_info(
            packed=self.args.packed,
            packed_buffer_size_bytes=self.args.packed_buffer_size_bytes,
            packed_num_buffers=self.args.packed_num_buffers,
        )
        update_info["dtype_names"] = [normalize_dtype_name(x) for x in update_info["dtype_names"]]
        update_info["is_checkpoint_format"] = self.args.checkpoint_format

        if not self.initialized:
            init_payload = {
                "init_info": {
                    "master_address": self.args.master_address,
                    "master_port": self.args.master_port,
                    "rank_offset": self.args.rank_offset,
                    "world_size": self.args.world_size,
                }
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                server_init = executor.submit(
                    post_json,
                    self.args.base_url,
                    "/init_weight_transfer_engine",
                    init_payload,
                    self.args.timeout,
                )
                self.group = NCCLWeightTransferEngine.trainer_init(
                    {
                        "master_address": self.args.master_address,
                        "master_port": self.args.master_port,
                        "world_size": self.args.world_size,
                    }
                )
                server_init.result()
            self.initialized = True

        post_json(self.args.base_url, "/pause?mode=wait", {}, self.args.timeout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            server_update = executor.submit(
                post_json,
                self.args.base_url,
                "/update_weights",
                {"update_info": dict(update_info)},
                self.args.timeout,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iter(params),
                NCCLTrainerSendWeightsArgs(
                    group=self.group,
                    src=0,
                    packed=self.args.packed,
                    packed_buffer_size_bytes=self.args.packed_buffer_size_bytes,
                    packed_num_buffers=self.args.packed_num_buffers,
                ),
            )
            server_update.result()
        post_json(self.args.base_url, "/resume", {}, self.args.timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--param-dtype", default="bfloat16",
                        choices=("float32", "fp32", "bfloat16", "bf16", "float16", "fp16"))
    parser.add_argument("--qwen-max-layers", type=int, default=-1,
                        help="debug truncation only; -1 trains/syncs the full model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rollout-backend", choices=("vllm", "oracle"), default="vllm")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--reward-method", choices=("strict", "flexible"), default="flexible")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--measure-param-delta", action="store_true")
    parser.add_argument("--sync-initial", action="store_true")
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--no-weight-sync", action="store_true")
    parser.add_argument("--master-address", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29577)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rank-offset", type=int, default=1)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packed-buffer-size-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--packed-num-buffers", type=int, default=2)
    parser.add_argument("--checkpoint-format", action=argparse.BooleanOptionalAction, default=True,
                        help="send HF checkpoint names and let vLLM map them into kernel-format parameters")
    parser.add_argument("--dump-rollout-dir", default="",
                        help="write each real rollout batch as JSON for the C++ PP/TP trainer")
    parser.add_argument("--skip-train", action="store_true",
                        help="only generate and dump rollout batches; do not update local cverl policy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    data = load_jsonl(args.dataset, args.prompt_field, args.answer_field, args.max_examples)
    policy = None
    syncer = None
    if not args.skip_train:
        import torch
        import _cverl_vllm_bridge

        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            raise RuntimeError("CUDA device requested but torch.cuda is not available")
        if args.device.startswith("cuda"):
            torch.cuda.set_device(int(args.device.split(":", 1)[1]))

        policy = _cverl_vllm_bridge.QwenPolicy(
            args.model_dir, args.tokenizer_path, args.device, args.param_dtype, args.qwen_max_layers)
        syncer = None if args.no_weight_sync else VllmNcclSync(args)

    if args.sync_initial and syncer is not None:
        syncer.sync(policy)

    served_model = args.served_model_name or args.model_dir
    print("step,rollout,total_seq,mean_reward,success_rate,adv_abs_sum,loss,pg_loss,kl_loss,ppo_kl,clipfrac,param_delta,rollout_seconds,train_seconds,sync_seconds")
    for step in range(1, args.steps + 1):
        batch = [data[((step - 1) * args.prompts + i) % len(data)] for i in range(args.prompts)]
        prompts = [x["prompt"] for x in batch]
        answers = [x["answer"] for x in batch]

        t0 = time.perf_counter()
        if args.rollout_backend == "oracle":
            responses, prompt_indices, sample_indices = oracle_rollout(prompts, answers, args.n)
        else:
            responses, prompt_indices, sample_indices = vllm_rollout(
                args.base_url, served_model, prompts, args.n, args.max_tokens,
                args.temperature, args.top_p, args.seed + step * 100003, args.timeout)
        t1 = time.perf_counter()

        rollout_doc = None
        if args.dump_rollout_dir:
            rollout_doc = {
                "step": step,
                "prompts": prompts,
                "answers": answers,
                "responses": responses,
                "prompt_indices": prompt_indices,
                "sample_indices": sample_indices,
            }

        if args.dump_rollout_dir:
            dump_dir = Path(args.dump_rollout_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = dump_dir / f"rollout_step_{step:06d}.json.tmp"
            out_path = dump_dir / f"rollout_step_{step:06d}.json"
            tmp_path.write_text(json.dumps(rollout_doc, ensure_ascii=False))
            tmp_path.replace(out_path)

        if args.skip_train:
            print(
                f"{step},{args.rollout_backend},{len(responses)},"
                f"0.000000,0.000000,0.000000,0.000000,"
                f"0.000000,0.000000,0.000000,0.000000,"
                f"0.000000,{t1 - t0:.6f},0.000000,0.000000",
                flush=True,
            )
            continue

        metrics = policy.gsm8k_grpo_update(
            prompts, responses, answers, prompt_indices, sample_indices,
            args.max_prompt_tokens, args.max_response_tokens, args.ppo_epochs,
            args.lr, args.weight_decay, args.clip_ratio, args.kl_coef,
            args.reward_method, args.measure_param_delta)
        t2 = time.perf_counter()

        sync_seconds = 0.0
        if syncer is not None and args.sync_every > 0 and step % args.sync_every == 0:
            s0 = time.perf_counter()
            syncer.sync(policy)
            sync_seconds = time.perf_counter() - s0

        print(
            f"{step},{args.rollout_backend},{metrics['total_seq']},"
            f"{metrics['mean_reward']:.6f},{metrics['success_rate']:.6f},"
            f"{metrics['adv_abs_sum']:.6f},{metrics['loss']:.6f},"
            f"{metrics['pg_loss']:.6f},{metrics['kl_loss']:.6f},"
            f"{metrics['ppo_kl']:.6f},{metrics['clipfrac']:.6f},"
            f"{metrics['param_delta']:.6f},{t1 - t0:.6f},{t2 - t1:.6f},{sync_seconds:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
