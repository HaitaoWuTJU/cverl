#!/usr/bin/env python3
"""Download a Hugging Face dataset split into prompt/answer JSONL for C++."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


DATASET_ALIASES = {
    "gsm8k": "openai/gsm8k",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--token")
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    dataset_id = DATASET_ALIASES.get(args.dataset, args.dataset)
    if args.local_files_only:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    dataset = load_dataset(
        path=dataset_id,
        name=args.name,
        split=args.split,
        cache_dir=args.cache_dir or None,
        token=args.token or None,
        download_mode="reuse_dataset_if_exists",
        verification_mode="basic_checks",
        trust_remote_code=args.trust_remote_code,
    )
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", encoding="utf-8") as f:
        for row in dataset:
            if args.prompt_field not in row or args.answer_field not in row:
                raise KeyError(
                    f"dataset row missing {args.prompt_field!r} or {args.answer_field!r}; "
                    f"available keys: {sorted(row.keys())}"
                )
            item = {
                "prompt": "" if row[args.prompt_field] is None else str(row[args.prompt_field]),
                "answer": "" if row[args.answer_field] is None else str(row[args.answer_field]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows += 1
            if args.max_examples >= 0 and rows >= args.max_examples:
                break

    print(f"CVERL_HF_DATASET_FILE={output}")
    print(f"CVERL_HF_DATASET_ROWS={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
