from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import (
    AffineTransform,
    Distribution,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from hw2 import MLP, Trajectory, sample_trajectory


class MLPPolicy(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        self.mlp = MLP(
            input_size=3,
            output_size=2,
            width=width,
            depth=depth,
        )

    def action_dist(self, obs: torch.Tensor) -> Distribution:
        batch, _ = obs.shape

        mean_log_std = self.mlp(obs)
        mean = torch.clamp(mean_log_std[:, 0], -3, 3)
        log_std = torch.clamp(mean_log_std[:, 1], -5, 2)
        std = torch.exp(log_std)
        return TransformedDistribution(
            Normal(mean, std),
            [TanhTransform(), AffineTransform(0.5, 0.5)],
        )


def policy_gradients_loss(traj: Trajectory) -> Tensor:
    reverse_cumsum_rewards = traj.rewards.flip(0).cumsum(0).flip(0)

    rewards_mean = reverse_cumsum_rewards.mean()
    rewards_std = reverse_cumsum_rewards.std() + 1e-8
    normalized_rewards = (reverse_cumsum_rewards - rewards_mean) / rewards_std

    log_probs = traj.log_probs

    return -(log_probs * normalized_rewards).sum()


def train_reinforce(
    env,
    policy: nn.Module,
    *,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_iterations: int = 200,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> nn.Module:
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(max_iterations):
        trajs = [sample_trajectory(env, policy) for _ in range(batch_size)]  # type: ignore

        all_rewards = torch.stack([t.rewards for t in trajs])
        if verbose:
            print(f"Iteration {epoch}, mean reward: {all_rewards.mean().item():.4f}")

        losses = [policy_gradients_loss(traj) for traj in trajs]
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"  policy loss: {loss.item():.4f}")

    return policy
