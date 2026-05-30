#!/usr/bin/env python3
"""Trainer-side vLLM Native RL NCCL sender for live cverl tensors.

The cverl Qwen policy is constructed by the C++/LibTorch pybind bridge. The
sender passes those live torch.Tensor handles to vLLM's
NCCLWeightTransferEngine.trainer_send_weights, so the payload is NCCL broadcast
from GPU memory. By default, metadata uses HF checkpoint names so vLLM can reuse
its mature model-specific weight loader to map into packed kernel parameters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys

import requests


def post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def normalize_dtype_name(dtype_name: str) -> str:
    # pybind renders torch dtype as "torch.float32"; vLLM update_info expects
    # the attribute suffix used by getattr(torch, dtype_name).
    return dtype_name.split(".")[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--param-dtype", default="float32",
                        choices=("float32", "fp32", "bfloat16", "bf16", "float16", "fp16"),
                        help="dtype used to load live cverl trainable parameters")
    parser.add_argument("--qwen-max-layers", type=int, default=-1,
                        help="debug truncation only; -1 sends the full model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--master-address", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True,
                        help="trainer rank + all vLLM worker ranks")
    parser.add_argument("--rank-offset", type=int, default=1)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packed-buffer-size-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--packed-num-buffers", type=int, default=2)
    parser.add_argument("--checkpoint-format", action=argparse.BooleanOptionalAction, default=True,
                        help="send HF checkpoint names and let vLLM map them into kernel-format parameters")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-server-init", action="store_true")
    parser.add_argument("--skip-update-metadata", action="store_true")
    parser.add_argument("--skip-pause", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="construct live cverl tensors and print metadata without opening NCCL")
    args = parser.parse_args()

    import torch
    import _cverl_vllm_bridge

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM NCCL weight sender requires CUDA")
    device_index = int(args.device.split(":", 1)[1]) if args.device.startswith("cuda:") else 0
    torch.cuda.set_device(device_index)

    policy = _cverl_vllm_bridge.QwenPolicy(
        args.model_dir, args.tokenizer_path, args.device, args.param_dtype, args.qwen_max_layers)
    params = policy.named_parameters()
    update_info = policy.update_info(
        packed=args.packed,
        packed_buffer_size_bytes=args.packed_buffer_size_bytes,
        packed_num_buffers=args.packed_num_buffers,
    )
    update_info["dtype_names"] = [normalize_dtype_name(x) for x in update_info["dtype_names"]]
    update_info["is_checkpoint_format"] = args.checkpoint_format

    if args.dry_run:
        print(json.dumps({
            "params": len(params),
            "first": {
                "name": params[0][0],
                "shape": list(params[0][1].shape),
                "device": str(params[0][1].device),
                "dtype": str(params[0][1].dtype).split(".")[-1],
            },
            "packed": args.packed,
        }, indent=2))
        return 0

    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    server_init = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        if not args.skip_server_init:
            init_payload = {
                "init_info": {
                    "master_address": args.master_address,
                    "master_port": args.master_port,
                    "rank_offset": args.rank_offset,
                    "world_size": args.world_size,
                }
            }
            server_init = executor.submit(
                post_json, args.base_url, "/init_weight_transfer_engine", init_payload, args.timeout)

        group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=args.master_address,
                master_port=args.master_port,
                world_size=args.world_size,
            )
        )
        if server_init is not None:
            print(json.dumps(server_init.result(), indent=2))

        if not args.skip_pause:
            print(json.dumps(post_json(args.base_url, "/pause?mode=wait", {}, args.timeout), indent=2))

        server_update = None
        if not args.skip_update_metadata:
            server_update = executor.submit(
                post_json, args.base_url, "/update_weights", {"update_info": dict(update_info)}, args.timeout)

        NCCLWeightTransferEngine.trainer_send_weights(
            iter(params),
            NCCLTrainerSendWeightsArgs(
                group=group,
                src=0,
                packed=args.packed,
                packed_buffer_size_bytes=args.packed_buffer_size_bytes,
                packed_num_buffers=args.packed_num_buffers,
            ),
        )
        if server_update is not None:
            print(json.dumps(server_update.result(), indent=2))

        if not args.skip_pause:
            print(json.dumps(post_json(args.base_url, "/resume", {}, args.timeout), indent=2))
    finally:
        executor.shutdown(wait=False)

    print(json.dumps({"sent": len(params), "packed": args.packed}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
