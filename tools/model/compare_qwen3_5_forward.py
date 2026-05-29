#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText


def parse_tokens(text: str) -> list[int]:
    tokens = [int(item) for item in text.split(",") if item]
    if not tokens:
        raise ValueError("--tokens must contain at least one id")
    return tokens


def load_hf_model(model_dir: str, layers: int | None):
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.eval()
    if layers is not None:
        model.model.language_model.config.num_hidden_layers = layers
    return model


def run_cpp(exe: Path, model_dir: str, tokens: list[int], layers: int | None, logits: bool, output_path: Path):
    cmd = [
        str(exe),
        model_dir,
        "--tokens",
        ",".join(str(t) for t in tokens),
        "--save-output",
        str(output_path),
    ]
    if layers is not None:
        cmd += ["--layers", str(layers)]
    if logits:
        cmd += ["--logits"]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def load_cpp_tensor(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    if data[:8] != b"CVTENSR1":
        raise ValueError(f"invalid cverl tensor dump: {path}")
    offset = 8
    rank = int(np.frombuffer(data, dtype="<i8", count=1, offset=offset)[0])
    offset += 8
    shape = np.frombuffer(data, dtype="<i8", count=rank, offset=offset).astype(np.int64)
    offset += 8 * rank
    array = np.frombuffer(data, dtype="<f4", offset=offset).copy().reshape(tuple(shape.tolist()))
    return torch.from_numpy(array)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--exe", default="build/qwen3_5_forward")
    parser.add_argument("--tokens", default="1,2")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)
    exe = Path(args.exe)
    with tempfile.TemporaryDirectory() as tmp:
        cpp_path = Path(tmp) / "cpp_output.pt"
        cpp_stdout = run_cpp(exe, args.model_dir, tokens, args.layers, args.logits, cpp_path)
        cpp = load_cpp_tensor(cpp_path).float()

    model = load_hf_model(args.model_dir, args.layers)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        if args.logits:
            ref = model(input_ids=input_ids).logits.float()
        else:
            ref = model.model.language_model(input_ids=input_ids).last_hidden_state.float()

    diff = (cpp - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_norm = ref.abs().max().item()
    rel_max = max_abs / max(ref_norm, 1e-12)
    ok = torch.allclose(cpp, ref, rtol=args.rtol, atol=args.atol)

    report = {
        "tokens": tokens,
        "layers": args.layers,
        "logits": args.logits,
        "shape": list(cpp.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_abs_max": ref_norm,
        "rel_max": rel_max,
        "allclose": bool(ok),
    }
    if args.logits:
        report["cpp_top5"] = cpp[0, -1].topk(5).indices.tolist()
        report["hf_top5"] = ref[0, -1].topk(5).indices.tolist()

    print(cpp_stdout.rstrip())
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
