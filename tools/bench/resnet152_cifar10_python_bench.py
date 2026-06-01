#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out)


class ResNet152Cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(256, 36, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 10)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cifar-10-batches-bin")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--allow-tf32", type=str, default="true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.warmup_batches < 0:
        raise ValueError("epochs and batch-size must be positive; warmup-batches must be non-negative")
    args.allow_tf32 = args.allow_tf32.lower() in {"1", "true", "yes", "on"}
    return args


def load_cifar10_split(data_dir: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(data_dir)
    raw_images = root / f"{split}_images.f32"
    raw_labels = root / f"{split}_labels.i64"
    if raw_images.exists() and raw_labels.exists():
        image_arr = np.fromfile(raw_images, dtype=np.float32)
        if image_arr.size % (3 * 32 * 32) != 0:
            raise ValueError(f"invalid CIFAR10 raw image cache: {raw_images}")
        n = image_arr.size // (3 * 32 * 32)
        label_arr = np.fromfile(raw_labels, dtype=np.int64)
        if label_arr.size != n:
            raise ValueError(f"CIFAR10 raw image/label count mismatch in {data_dir}")
        images = torch.from_numpy(image_arr.reshape(n, 3, 32, 32)).contiguous()
        labels = torch.from_numpy(label_arr).contiguous()
        return images, labels

    tensor_images = root / f"{split}_images.pt"
    tensor_labels = root / f"{split}_labels.pt"
    if tensor_images.exists() and tensor_labels.exists():
        images = torch.load(tensor_images, map_location="cpu").to(torch.float32).contiguous()
        labels = torch.load(tensor_labels, map_location="cpu").to(torch.int64).contiguous()
        if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0] or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(f"invalid CIFAR10 tensor cache in {data_dir}")
        return images, labels

    rows = []
    if split == "train":
        paths = [root / f"data_batch_{i}.bin" for i in range(1, 6)]
    elif split == "test":
        paths = [root / "test_batch.bin"]
    else:
        raise ValueError(f"unsupported CIFAR10 split: {split}")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing CIFAR10 file: {path}")
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size % 3073 != 0:
            raise ValueError(f"invalid CIFAR10 binary file size: {path}")
        rows.append(raw.reshape(-1, 3073))
    arr = np.concatenate(rows, axis=0)
    labels = torch.from_numpy(arr[:, 0].astype(np.int64, copy=True))
    images = torch.from_numpy(arr[:, 1:].astype(np.float32, copy=True)).view(-1, 3, 32, 32).div_(255.0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)
    images = (images - mean) / std
    return images.contiguous(), labels.contiguous()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batches(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    max_batches: int,
) -> float:
    model.train()
    n = images.shape[0]
    batches = math.ceil(n / batch_size)
    run_count = batches if max_batches <= 0 else min(batches, max_batches)
    perm = torch.randperm(n, device=images.device)
    loss_sum = torch.zeros((), device=images.device, dtype=torch.float32)
    seen = 0
    for batch in range(run_count):
        start = batch * batch_size
        size = min(batch_size, n - start)
        idx = perm.narrow(0, start, size)
        x = images.index_select(0, idx)
        y = labels.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach().to(torch.float32) * float(size)
        seen += size
    return float(loss_sum.cpu().item()) / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int) -> tuple[float, float]:
    model.eval()
    n = images.shape[0]
    loss_sum = torch.zeros((), device=images.device, dtype=torch.float32)
    correct_sum = torch.zeros((), device=images.device, dtype=torch.int64)
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        x = images.narrow(0, start, size)
        y = labels.narrow(0, start, size)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum = loss_sum + loss.detach().to(torch.float32) * float(size)
        correct_sum = correct_sum + logits.argmax(dim=1).eq(y).sum().to(torch.int64)
    examples = max(n, 1)
    return float(loss_sum.cpu().item()) / examples, float(correct_sum.cpu().item()) / examples


def main() -> None:
    process_begin = time.perf_counter()
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    data_begin = time.perf_counter()
    train_images, train_labels = load_cifar10_split(args.data_dir, "train")
    test_images, test_labels = load_cifar10_split(args.data_dir, "test")
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    sync_if_cuda(device)
    data_seconds = time.perf_counter() - data_begin

    model_begin = time.perf_counter()
    model = ResNet152Cifar().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    sync_if_cuda(device)
    model_init_seconds = time.perf_counter() - model_begin

    warmup_seconds = 0.0
    if args.warmup_batches > 0:
        warmup_begin = time.perf_counter()
        run_batches(model, optimizer, train_images, train_labels, args.batch_size, args.warmup_batches)
        sync_if_cuda(device)
        warmup_seconds = time.perf_counter() - warmup_begin

    metrics_file = None
    writer = None
    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["backend", "epoch", "seconds", "avg_loss", "examples", "batches", "batch_size"],
        )
        writer.writeheader()

    examples = train_images.shape[0]
    test_examples = test_images.shape[0]
    batches = math.ceil(examples / args.batch_size)
    epoch_train_seconds = 0.0
    final_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        sync_if_cuda(device)
        begin = time.perf_counter()
        final_loss = run_batches(model, optimizer, train_images, train_labels, args.batch_size, 0)
        sync_if_cuda(device)
        seconds = time.perf_counter() - begin
        epoch_train_seconds += seconds
        if writer is not None:
            writer.writerow(
                {
                    "backend": "python_pytorch",
                    "epoch": epoch,
                    "seconds": f"{seconds:.9f}",
                    "avg_loss": f"{final_loss:.9f}",
                    "examples": examples,
                    "batches": batches,
                    "batch_size": args.batch_size,
                }
            )
            metrics_file.flush()
        print(f"backend=python_pytorch epoch={epoch} seconds={seconds:.6f} avg_loss={final_loss:.6f}", flush=True)

    eval_begin = time.perf_counter()
    sync_if_cuda(device)
    train_loss, train_accuracy = evaluate(model, train_images, train_labels, args.batch_size)
    test_loss, test_accuracy = evaluate(model, test_images, test_labels, args.batch_size)
    sync_if_cuda(device)
    eval_seconds = time.perf_counter() - eval_begin
    end_to_end_seconds = time.perf_counter() - process_begin

    if metrics_file is not None:
        metrics_file.close()
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "backend": "python_pytorch",
                    "model": "resnet152_cifar",
                    "dataset": "cifar10_train",
                    "device": args.device,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "warmup_batches": args.warmup_batches,
                    "lr": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "allow_tf32": args.allow_tf32,
                    "examples": examples,
                    "test_examples": test_examples,
                    "batches_per_epoch": batches,
                    "total_seconds": end_to_end_seconds,
                    "end_to_end_seconds": end_to_end_seconds,
                    "epoch_train_seconds": epoch_train_seconds,
                    "seconds_per_epoch": epoch_train_seconds / args.epochs,
                    "data_seconds": data_seconds,
                    "model_init_seconds": model_init_seconds,
                    "warmup_seconds": warmup_seconds,
                    "eval_seconds": eval_seconds,
                    "final_loss": final_loss,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                indent=2,
            )
            + "\n"
        )
    print(
        "metrics backend=python_pytorch "
        f"train_loss={train_loss:.6f} train_acc={train_accuracy:.6f} "
        f"test_loss={test_loss:.6f} test_acc={test_accuracy:.6f}",
        flush=True,
    )
    print(
        "stages backend=python_pytorch "
        f"data_seconds={data_seconds:.6f} model_init_seconds={model_init_seconds:.6f} "
        f"warmup_seconds={warmup_seconds:.6f} epoch_train_seconds={epoch_train_seconds:.6f} "
        f"eval_seconds={eval_seconds:.6f} end_to_end_seconds={end_to_end_seconds:.6f}",
        flush=True,
    )
    print(
        "summary backend=python_pytorch "
        f"epochs={args.epochs} total_seconds={end_to_end_seconds:.6f} "
        f"seconds_per_epoch={epoch_train_seconds / args.epochs:.6f} final_loss={final_loss:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
