#!/usr/bin/env python3
"""Control-plane helper for vLLM Native RL weight transfer.

This script intentionally does not send weights through HTTP. HTTP is used only
to pause/resume vLLM and to pass small init/update metadata. The actual tensor
payload is transferred by vLLM's Native RL NCCL weight-transfer engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_json(base_url: str, path: str, timeout: float) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_update_info(path: str) -> dict:
    doc = json.loads(Path(path).read_text())
    if "update_info" in doc:
        return doc["update_info"]
    return doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("world-size")

    init = sub.add_parser("init-nccl")
    init.add_argument("--master-address", default="127.0.0.1")
    init.add_argument("--master-port", type=int, required=True)
    init.add_argument("--world-size", type=int, required=True)
    init.add_argument("--rank-offset", type=int, default=1)

    update = sub.add_parser("update-nccl")
    update.add_argument("--manifest", required=True)

    args = parser.parse_args()

    if args.cmd == "pause":
        print(json.dumps(post_json(args.base_url, "/pause?mode=wait", {}, args.timeout), indent=2))
    elif args.cmd == "resume":
        print(json.dumps(post_json(args.base_url, "/resume", {}, args.timeout), indent=2))
    elif args.cmd == "world-size":
        print(json.dumps(get_json(args.base_url, "/get_world_size?include_dp=true", args.timeout), indent=2))
    elif args.cmd == "init-nccl":
        payload = {
            "init_info": {
                "master_address": args.master_address,
                "master_port": args.master_port,
                "rank_offset": args.rank_offset,
                "world_size": args.world_size,
            }
        }
        print(json.dumps(post_json(args.base_url, "/init_weight_transfer_engine", payload, args.timeout), indent=2))
    elif args.cmd == "update-nccl":
        payload = {"update_info": load_update_info(args.manifest)}
        print(json.dumps(post_json(args.base_url, "/update_weights", payload, args.timeout), indent=2))
    else:
        raise AssertionError(args.cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
