from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hw2 import Trajectory, sample_trajectory


@dataclass
class REINFORCETrainingResult:
    policy: nn.Module
    mean_rewards: list[float]
    policy_losses: list[float]


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
    plot_callback=None,
    plot_interval: int = 50,
) -> REINFORCETrainingResult:
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    mean_rewards_history = []
    policy_losses_history = []

    for epoch in range(max_iterations):
        trajs = [sample_trajectory(env, policy) for _ in range(batch_size)]  # type: ignore

        all_rewards = torch.stack([t.rewards for t in trajs])
        mean_reward = all_rewards.mean().item()
        mean_rewards_history.append(mean_reward)

        if verbose:
            print(f"Iteration {epoch}, mean reward: {mean_reward:.4f}", flush=True)

        losses = [policy_gradients_loss(traj) for traj in trajs]
        loss = torch.stack(losses).mean()
        policy_losses_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"  policy loss: {loss.item():.4f}", flush=True)

        # Call plot callback if provided
        if plot_callback is not None and epoch % plot_interval == 0:
            plot_callback(policy, epoch, env)

    return REINFORCETrainingResult(
        policy=policy,
        mean_rewards=mean_rewards_history,
        policy_losses=policy_losses_history,
    )
