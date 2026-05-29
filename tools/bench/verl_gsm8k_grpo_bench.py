#!/usr/bin/env python3
"""GSM8K GRPO/PPO benchmark using Python verl core algorithms."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch


def import_verl():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla

    return compute_grpo_outcome_advantage, compute_policy_loss_vanilla


class TinyPolicy(torch.nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(obs)))


class ActorConfigLike(dict):
    def __init__(self, clip_ratio: float, clip_low: float | None, clip_high: float | None, clip_c: float):
        super().__init__(clip_ratio_c=clip_c, global_batch_info={})
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_low
        self.clip_ratio_high = clip_high
        self.global_batch_info = {}


@dataclass
class Example:
    prompt: str
    answer: str


def fnv1a64(text: str, seed: int) -> int:
    mask = (1 << 64) - 1
    h = (1469598103934665603 ^ seed) & mask
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & mask
    return h


def load_jsonl(path: str) -> list[Example]:
    out: list[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(Example(prompt=str(row["prompt"]), answer=str(row["answer"])))
    return out


def make_targets(data: list[Example], step: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    mask = (1 << 64) - 1
    rows: list[list[int]] = []
    for row in range(args.prompts):
        ex = data[(step * args.prompts + row) % len(data)]
        state = fnv1a64(ex.prompt + "\n" + ex.answer, args.seed + step)
        values: list[int] = []
        for _ in range(args.seq_len):
            state ^= state >> 12
            state &= mask
            state ^= (state << 25) & mask
            state &= mask
            state ^= state >> 27
            state &= mask
            values.append(((state * 2685821657736338717) & mask) % args.action_dim)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.long, device=device)


def action_log_probs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--responses", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    compute_grpo_outcome_advantage, compute_policy_loss_vanilla = import_verl()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = load_jsonl(args.dataset)
    if not data:
        raise RuntimeError("dataset is empty")

    policy = TinyPolicy(args.action_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    actor_cfg = ActorConfigLike(clip_ratio=0.2, clip_low=None, clip_high=None, clip_c=3.0)

    batch = args.prompts * args.responses
    total_sequences = args.steps * batch
    total_tokens = total_sequences * args.seq_len * args.ppo_epochs
    loss_value = 0.0
    reward_value = 0.0
    kl_value = 0.0
    clipfrac_value = 0.0

    start = time.perf_counter()
    for step in range(args.steps):
        prompt_targets = make_targets(data, step, args, device)
        targets = prompt_targets.repeat_interleave(args.responses, 0)
        obs = torch.nn.functional.one_hot(targets, args.action_dim).to(torch.float32)
        response_mask = torch.ones((batch, args.seq_len), dtype=torch.float32, device=device)
        group_ids = torch.arange(args.prompts, dtype=torch.long).repeat_interleave(args.responses).cpu().numpy()

        with torch.no_grad():
            old_logits = policy(obs)
            probs = torch.softmax(old_logits, -1)
            actions = probs.reshape(batch * args.seq_len, args.action_dim).multinomial(1).reshape(batch, args.seq_len)
            old_log_probs = action_log_probs(old_logits, actions).detach()

        token_correct = (actions == targets).to(torch.float32) * response_mask
        scalar_rewards = token_correct.sum(-1) / response_mask.sum(-1)
        token_rewards = torch.zeros((batch, args.seq_len), dtype=torch.float32, device=device)
        token_rewards[:, args.seq_len - 1] = scalar_rewards

        advantages, _ = compute_grpo_outcome_advantage(
            token_rewards, response_mask, group_ids, epsilon=1.0e-6, norm_adv_by_std_in_grpo=True
        )

        for _ in range(args.ppo_epochs):
            logits = policy(obs)
            log_probs = action_log_probs(logits, actions)
            loss, metrics = compute_policy_loss_vanilla(
                old_log_prob=old_log_probs,
                log_prob=log_probs,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode="token-mean",
                config=actor_cfg,
                rollout_is_weights=None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            kl_value = float(metrics["actor/ppo_kl"])
            clipfrac_value = float(metrics["actor/pg_clipfrac"])
        reward_value = float(scalar_rewards.mean().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start

    print("backend,verl")
    print(f"dataset_rows,{len(data)}")
    print(f"steps,{args.steps}")
    print(f"prompts,{args.prompts}")
    print(f"responses,{args.responses}")
    print(f"seq_len,{args.seq_len}")
    print(f"action_dim,{args.action_dim}")
    print(f"hidden_dim,{args.hidden_dim}")
    print(f"ppo_epochs,{args.ppo_epochs}")
    print(f"device,{args.device}")
    print(f"seconds,{seconds:.6f}")
    print(f"sequences_per_second,{total_sequences / seconds:.6f}")
    print(f"train_tokens_per_second,{total_tokens / seconds:.6f}")
    print(f"last_loss,{loss_value:.6f}")
    print(f"last_avg_reward,{reward_value:.6f}")
    print(f"last_ppo_kl,{kl_value:.6f}")
    print(f"last_clipfrac,{clipfrac_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
