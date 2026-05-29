#!/usr/bin/env python3
"""Dump tokenizer ground-truth fixtures.

Two purposes:

1. Synthetic mode (--synthetic): build a tiny, deterministic byte-level BPE
   tokenizer.json from scratch and dump probe-string fixtures next to it.
   The C++ test_hf_bpe_tokenizer.cc uses this to verify the BPE algorithm
   without depending on a downloaded model.

2. Real mode (--tokenizer-json): load a real HF tokenizer.json (Qwen / GPT-2
   / Llama-3) and dump probe-string fixtures. The C++ tests then verify the
   pure C++ implementation matches HF Python bit-for-bit.

The fixture file is a JSON array of objects:
    {"text": "...", "ids": [..], "decoded": "..."}
The C++ side reloads with nlohmann/json and asserts equality.

Synthetic tokenizer.json layout (matches the HF "BPE" format the C++ code
loads):
    {
      "model": {"type": "BPE", "vocab": {...}, "merges": [...]},
      "added_tokens": [...],
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "..."}, ...}
    }
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence


# HF GPT-2 byte_to_unicode mapping. Mirrors the C++ implementation.
def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_PRE_TOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)


def build_synthetic_tokenizer(out_dir: Path, probes: Sequence[str]) -> None:
    """Tiny byte-level BPE: vocab = every byte's unicode form, plus a
    handful of hand-picked merges, plus 3 special tokens. Big enough to
    exercise the BPE merge step without resembling a real model.
    """
    encoder = bytes_to_unicode()
    # Base vocab: 256 byte-tokens + a few merges below.
    vocab: Dict[str, int] = {}
    for b in range(256):
        vocab[encoder[b]] = b
    next_id = 256

    merges: List[str] = []

    def add_merge(a: str, b: str) -> None:
        nonlocal next_id
        merges.append(f"{a} {b}")
        ab = a + b
        if ab not in vocab:
            vocab[ab] = next_id
            next_id += 1

    # A few merges for an English ASCII tokenizer. The unicode form of
    # ASCII bytes is just the ASCII character itself in this mapping.
    # Space (0x20) maps to 'Ġ', the GPT-2 sentinel for word-initial space.
    g = encoder[0x20]  # 'Ġ'
    add_merge(g, "h")        # " h"
    add_merge(g + "h", "e")  # " he"
    add_merge(g + "he", "l") # " hel"
    add_merge(g + "hel", "l") # " hell"
    add_merge(g + "hell", "o") # " hello"
    add_merge(g, "w")        # " w"
    add_merge(g + "w", "o")  # " wo"
    add_merge(g + "wo", "r") # " wor"
    add_merge(g + "wor", "l")# " worl"
    add_merge(g + "worl", "d")# " world"
    add_merge("a", "n")       # "an"
    add_merge("a", "b")       # "ab"
    add_merge("ab", "c")      # "abc"

    # Specials.
    added = [
        {
            "id": next_id,
            "content": "<|pad|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": next_id + 1,
            "content": "<|endoftext|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": next_id + 2,
            "content": "<|im_start|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ]
    for tok in added:
        vocab[tok["content"]] = tok["id"]
    next_id += 3

    tok_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
        "added_tokens": added,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRE_TOKENIZER_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    with tok_path.open("w") as f:
        json.dump(tok_json, f, ensure_ascii=False)

    # Now compute probe fixtures using HF tokenizers.
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        raise SystemExit(
            "synthetic fixture generation requires the 'tokenizers' Python package "
            "(conda activate verl-cpu)."
        )

    tok = Tokenizer.from_file(str(tok_path))
    fixtures = []
    for probe in probes:
        enc = tok.encode(probe, add_special_tokens=False)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        fixtures.append({"text": probe, "ids": list(map(int, enc.ids)), "decoded": decoded})
    fix_path = out_dir / "fixtures.json"
    with fix_path.open("w") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    print(f"synthetic tokenizer at {tok_path}")
    print(f"fixtures at {fix_path}: {len(fixtures)} probes")


def dump_real_fixtures(tokenizer_json: Path, out_path: Path, probes: Sequence[str]) -> None:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        raise SystemExit("real fixture mode requires the 'tokenizers' Python package")
    tok = Tokenizer.from_file(str(tokenizer_json))
    fixtures = []
    for probe in probes:
        enc = tok.encode(probe, add_special_tokens=False)
        decoded = tok.decode(enc.ids, skip_special_tokens=False)
        fixtures.append({"text": probe, "ids": list(map(int, enc.ids)), "decoded": decoded})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    print(f"wrote {len(fixtures)} fixtures to {out_path}")


_DEFAULT_PROBES = [
    "Hello world",
    "hello world",
    " hello",
    "abc",
    "an apple",
    "Q: What is 2+2?\nA: 4",
    "1234567890",
    "<|endoftext|>",
    "<|im_start|>user\nWhat?<|im_end|>",
    "tabs\tand\nnewlines",
    "naïve résumé",  # non-ASCII
    "  ",  # double space
    "",  # empty
]


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    sp_synth = sub.add_parser("synthetic")
    sp_synth.add_argument("--out-dir", type=Path, required=True)

    sp_real = sub.add_parser("real")
    sp_real.add_argument("--tokenizer-json", type=Path, required=True)
    sp_real.add_argument("--out", type=Path, required=True)

    p.add_argument("--probes-file", type=Path, default=None,
                   help="Optional file with one probe string per line (UTF-8). "
                        "Falls back to a hardcoded default set.")
    args = p.parse_args()

    if args.probes_file:
        probes = args.probes_file.read_text().splitlines()
    else:
        probes = list(_DEFAULT_PROBES)

    if args.mode == "synthetic":
        build_synthetic_tokenizer(args.out_dir, probes)
    elif args.mode == "real":
        dump_real_fixtures(args.tokenizer_json, args.out, probes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
