#!/usr/bin/env python3
"""Generate binary golden test data from the Python verl implementation.

Run this from the repository that contains the Python `verl` checkout, for
example:

    PYTHONPATH=../verl python3 tools/golden_dump.py build/golden_core_algos.bin

The output is consumed by `compare_golden`.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import struct
import sys
from typing import BinaryIO

import numpy as np
import torch


MAGIC = b"CVERLGD1"
VERSION = 3

KIND_KL = 1
KIND_GAE = 2
KIND_GRPO = 3
KIND_PPO = 4

KL_K1 = 0
KL_ABS = 1
KL_K2 = 2
KL_K3 = 3

AGG_TOKEN_MEAN = 0
AGG_SEQ_MEAN_TOKEN_SUM = 1
AGG_SEQ_MEAN_TOKEN_MEAN = 2


def _import_verl():
    try:
        from verl.trainer.ppo.core_algos import (
            compute_gae_advantage_return,
            compute_grpo_outcome_advantage,
            compute_policy_loss_vanilla,
            kl_penalty,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import verl. Set PYTHONPATH to the Python verl checkout, "
            "for example: PYTHONPATH=../verl python3 tools/golden_dump.py build/golden.bin"
        ) from exc

    return compute_gae_advantage_return, compute_grpo_outcome_advantage, compute_policy_loss_vanilla, kl_penalty


def _write_u32(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("<I", value))


def _write_i64_array(f: BinaryIO, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    _write_u32(f, arr.size)
    f.write(arr.tobytes(order="C"))


def _write_f32_tensor(f: BinaryIO, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {arr.shape}")
    _write_u32(f, arr.shape[0])
    _write_u32(f, arr.shape[1])
    f.write(arr.tobytes(order="C"))


def _rand_mask(rows: int, cols: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (rows, cols), dtype=torch.float32)
    for row in range(rows):
        if mask[row].sum().item() < 2:
            mask[row, -2:] = 1.0
    return mask


def _groups(rows: int, group_size: int) -> np.ndarray:
    groups = np.arange(rows, dtype=np.int64) // group_size
    random.shuffle(groups)
    return groups


class ActorConfigLike(dict):
    def __init__(self, clip_ratio: float, clip_low: float | None, clip_high: float | None, clip_c: float):
        super().__init__(clip_ratio_c=clip_c, global_batch_info={})
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_low
        self.clip_ratio_high = clip_high
        self.global_batch_info = {}


def make_records(seed: int):
    compute_gae, compute_grpo, compute_policy_loss_vanilla, kl_penalty = _import_verl()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    records = []

    rows, cols = 8, 16
    logp_base = torch.randn(rows, cols, dtype=torch.float32) * 2.0
    ref = torch.randn(rows, cols, dtype=torch.float32) * 2.0
    for penalty_id, name in [(KL_K1, "kl"), (KL_ABS, "abs"), (KL_K2, "k2"), (KL_K3, "k3")]:
        logp = logp_base.detach().clone().requires_grad_(True)
        out = kl_penalty(logp, ref, name)
        out.sum().backward()
        records.append((KIND_KL, penalty_id, logp.detach(), ref, out.detach(), logp.grad.detach().clone()))

    rewards = torch.randn(rows, cols, dtype=torch.float32)
    values = torch.randn(rows, cols, dtype=torch.float32)
    mask = _rand_mask(rows, cols)
    adv, ret = compute_gae(rewards, values, mask, torch.tensor(0.97), torch.tensor(0.91))
    records.append((KIND_GAE, 0.97, 0.91, rewards, values, mask, adv, ret))

    rewards = torch.randn(rows, cols, dtype=torch.float32) * _rand_mask(rows, cols)
    mask = _rand_mask(rows, cols)
    group_ids = _groups(rows, 2)
    for norm in [False, True]:
        adv, ret = compute_grpo(rewards, mask, group_ids, norm_adv_by_std_in_grpo=norm)
        records.append((KIND_GRPO, group_ids, 1.0e-6, int(norm), rewards, mask, adv, ret))

    old_log_prob = torch.randn(rows, cols, dtype=torch.float32) * 0.5
    log_prob = (old_log_prob + torch.randn(rows, cols, dtype=torch.float32) * 0.25).detach().requires_grad_(True)
    advantages = torch.randn(rows, cols, dtype=torch.float32)
    mask = _rand_mask(rows, cols)
    cfg = ActorConfigLike(clip_ratio=0.2, clip_low=None, clip_high=None, clip_c=3.0)
    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        loss_agg_mode="token-mean",
        config=cfg,
        rollout_is_weights=None,
    )
    loss.backward()
    grad_log_prob = log_prob.grad.detach().clone()
    records.append(
        (
            KIND_PPO,
            0.2,
            -1.0,
            -1.0,
            3.0,
            AGG_TOKEN_MEAN,
            old_log_prob,
            log_prob,
            advantages,
            mask,
            float(loss.item()),
            float(metrics["actor/pg_clipfrac"]),
            float(metrics["actor/ppo_kl"]),
            float(metrics["actor/pg_clipfrac_lower"]),
            grad_log_prob,
        )
    )

    return records


def write_records(path: pathlib.Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        _write_u32(f, VERSION)
        _write_u32(f, len(records))
        for rec in records:
            kind = rec[0]
            _write_u32(f, kind)
            if kind == KIND_KL:
                _, penalty, logp, ref, out, grad_logp = rec
                _write_u32(f, penalty)
                _write_f32_tensor(f, logp)
                _write_f32_tensor(f, ref)
                _write_f32_tensor(f, out)
                _write_f32_tensor(f, grad_logp)
            elif kind == KIND_GAE:
                _, gamma, lam, rewards, values, mask, adv, ret = rec
                f.write(struct.pack("<ff", gamma, lam))
                _write_f32_tensor(f, rewards)
                _write_f32_tensor(f, values)
                _write_f32_tensor(f, mask)
                _write_f32_tensor(f, adv)
                _write_f32_tensor(f, ret)
            elif kind == KIND_GRPO:
                _, group_ids, eps, norm, rewards, mask, adv, ret = rec
                f.write(struct.pack("<fI", eps, norm))
                _write_i64_array(f, group_ids)
                _write_f32_tensor(f, rewards)
                _write_f32_tensor(f, mask)
                _write_f32_tensor(f, adv)
                _write_f32_tensor(f, ret)
            elif kind == KIND_PPO:
                (
                    _,
                    clip,
                    clip_low,
                    clip_high,
                    clip_c,
                    agg,
                    old_log_prob,
                    log_prob,
                    advantages,
                    mask,
                    loss,
                    clipfrac,
                    ppo_kl,
                    clipfrac_lower,
                    grad_log_prob,
                ) = rec
                f.write(struct.pack("<ffffI", clip, clip_low, clip_high, clip_c, agg))
                _write_f32_tensor(f, old_log_prob)
                _write_f32_tensor(f, log_prob)
                _write_f32_tensor(f, advantages)
                _write_f32_tensor(f, mask)
                f.write(struct.pack("<ffff", loss, clipfrac, ppo_kl, clipfrac_lower))
                _write_f32_tensor(f, grad_log_prob)
            else:
                raise ValueError(f"unknown record kind {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    records = make_records(args.seed)
    write_records(args.output, records)
    print(f"wrote {len(records)} golden records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
