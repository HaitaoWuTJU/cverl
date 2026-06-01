#!/usr/bin/env python3
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--keep-pt", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def image_to_chw_float(image_obj) -> np.ndarray:
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        elif image_obj.get("path"):
            image = Image.open(image_obj["path"]).convert("RGB")
        else:
            raise ValueError("unsupported HF image record")
    elif isinstance(image_obj, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_obj)).convert("RGB")
    else:
        image = image_obj.convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def main() -> None:
    args = parse_args()
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    table = pq.read_table(args.parquet)
    names = set(table.column_names)
    image_col = "img" if "img" in names else "image"
    label_col = "label"
    if image_col not in names or label_col not in names:
        raise ValueError(f"expected image/label columns in {args.parquet}, got {table.column_names}")

    images_py = table[image_col].to_pylist()
    labels_py = table[label_col].to_pylist()
    images_np = np.stack([image_to_chw_float(img) for img in images_py], axis=0)
    labels_np = np.asarray(labels_py, dtype=np.int64)

    images = torch.from_numpy(images_np).contiguous()
    labels = torch.from_numpy(labels_np).contiguous()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)
    images = ((images - mean) / std).contiguous()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_pt:
        torch.save(images, out_dir / f"{args.split}_images.pt")
        torch.save(labels, out_dir / f"{args.split}_labels.pt")
    images.numpy().astype(np.float32, copy=False).tofile(out_dir / f"{args.split}_images.f32")
    labels.numpy().astype(np.int64, copy=False).tofile(out_dir / f"{args.split}_labels.i64")
    (out_dir / f"{args.split}_meta.json").write_text(
        json.dumps(
            {
                "source": "huggingface:uoft-cs/cifar10",
                "parquet": str(args.parquet),
                "split": args.split,
                "examples": int(images.shape[0]),
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "label_dtype": str(labels.dtype),
                "normalized": True,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"exported {images.shape[0]} examples to {out_dir}")


if __name__ == "__main__":
    main()
