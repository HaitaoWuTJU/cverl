#!/usr/bin/env python3
"""Small stable wrapper around huggingface_hub for C++ callers."""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import model_info, snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--token")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        info = model_info(args.repo_id, revision=args.revision, token=args.token)
        print(f"CVERL_HF_REPO_ID={args.repo_id}")
        print(f"CVERL_HF_SHA={info.sha}")
        print(f"CVERL_HF_FILE_COUNT={len(info.siblings or [])}")
        print(f"CVERL_HF_LOCAL_DIR={args.local_dir or ''}")
        return 0

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir or None,
        cache_dir=args.cache_dir or None,
        token=args.token or None,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
        local_files_only=args.local_files_only,
    )
    print(f"CVERL_HF_LOCAL_DIR={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
