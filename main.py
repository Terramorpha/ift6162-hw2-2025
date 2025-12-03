import pdb
import sys
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import (
    AffineTransform,
    Distribution,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

import hw2
from calciner import CalcinerEnv, ConstantTemperatureController, evaluate_baseline
from hw2 import (
    MLP,
    Controller,
    Trajectory,
    importance_sampling_policy_gradients_loss,
    policy_gradients_loss,
    sample_trajectory,
)


def handler(_a, _b, tb):
    pdb.post_mortem()


# sys.excepthook = handler


@dataclass
class MLPPolicy:
    mlp: MLP

    def __init__(self, depth: int, width: int):
        self.mlp = MLP(
            input_size=3,
            output_size=2,
            width=32,
            depth=2,
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


class MLPActorCritic(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        self.trunk = MLP(
            input_size=3,
            output_size=width,
            width=width,
            depth=depth,
        )

        # Single layer heads
        self.actor_head = nn.Linear(width, 2)
        self.critic_head = nn.Linear(width, 1)

    def action_dist(self, obs: torch.Tensor) -> Distribution:
        batch, _ = obs.shape

        features = self.trunk(obs)
        mean_log_std = self.actor_head(features)
        mean = torch.clamp(mean_log_std[:, 0], -3, 3)
        log_std = torch.clamp(mean_log_std[:, 1], -5, 2)
        std = torch.exp(log_std)
        return TransformedDistribution(
            Normal(mean, std),
            [TanhTransform(), AffineTransform(0.5, 0.5)],
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        return self.critic_head(features)


env = CalcinerEnv()


model = MLPActorCritic(3, 8)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 64
n_epochs = 3  # PPO epochs per batch

γ = 0.99
λ = 0.95

for epoch in range(200):
    # Collect fresh trajectories (no replay buffer - PPO is on-policy)
    trajs = [sample_trajectory(env, model).detach() for _ in range(batch_size)]

    all_rewards = torch.stack([t.rewards for t in trajs])
    print(f"Iteration {epoch}, mean reward:", all_rewards.mean().item())

    # Multiple epochs over the same batch (standard PPO)
    for k in range(n_epochs):
        # Critic update
        value_losses = [hw2.td_loss(model, traj, γ) for traj in trajs]
        critic_loss = torch.stack(value_losses).mean()

        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        # Policy update
        policy_losses = [hw2.ppo_loss(model, traj, γ, λ=λ) for traj in trajs]
        policy_loss = torch.stack(policy_losses).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    print(f"  critic: {critic_loss.item():.2f}, policy: {policy_loss.item():.4f}")

    if epoch % 10 == 0:
        res = evaluate_baseline(
            env, Controller(model, u_min=env.u_min, u_max=env.u_max), n_episodes=32
        )
        print(res)


baseline_results = evaluate_baseline(env, ConstantTemperatureController())
policy_results = evaluate_baseline(
    env, Controller(model, u_min=env.u_min, u_max=env.u_max)
)
