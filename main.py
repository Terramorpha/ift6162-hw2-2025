import copy
import pdb
import random
import sys
from dataclasses import dataclass
from typing import Protocol

import numpy as np
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


def train_ppo():
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


class DeterministicActorMLP(nn.Module):
    mlp: MLP

    def __init__(self, depth, width):
        super().__init__()

        self.mlp = MLP(
            3,
            1,
            width,
            depth,
        )

    def action(self, obs: torch.Tensor, /) -> torch.Tensor:
        _, _ = obs.shape

        return torch.sigmoid(self.mlp(obs))


class QCriticMLP(nn.Module):
    mlp: MLP

    def __init__(self, depth, width):
        super().__init__()

        self.mlp = MLP(
            input_size=4,
            output_size=1,
            width=width,
            depth=depth,
        )

    def q_value(self, obs: Tensor, action: Tensor) -> Tensor:
        _, _ = obs.shape
        _, _ = action.shape

        concat = torch.concatenate([obs, action], dim=-1)

        return self.mlp(concat)


@dataclass
class MinQ:
    q1: hw2.QCritic
    q2: hw2.QCritic

    def q_value(self, obs: Tensor, action: Tensor) -> Tensor:
        v1 = self.q1.q_value(obs, action)
        v2 = self.q2.q_value(obs, action)

        return torch.min(v1, v2)


def polyak_update(model, ema_model, decay=0.999):
    """Update EMA model parameters with polyak averaging."""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

        for buffer, ema_buffer in zip(model.buffers(), ema_model.buffers()):
            ema_buffer.data = decay * ema_buffer.data + (1 - decay) * buffer.data


def train_td3(
    seed=0,
    batch_size=16,
    buffer_size=256,
    γ=0.99,
    exploration_noise=0.1,
    lr=1e-3,
    warmup_steps=50,
    max_epochs=5000,
    network_depth=3,
    network_width=16,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    env = CalcinerEnv()

    online_actor = DeterministicActorMLP(network_depth, network_width)
    online_q_1 = QCriticMLP(network_depth, network_width)
    online_q_2 = QCriticMLP(network_depth, network_width)

    target_actor = copy.deepcopy(online_actor)
    target_q_1 = copy.deepcopy(online_q_1)
    target_q_2 = copy.deepcopy(online_q_2)

    actor_opt = torch.optim.Adam(online_actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(online_q_1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(online_q_2.parameters(), lr=lr)

    trajectory_buffer = []
    best_reward = -float('inf')

    for epoch in range(max_epochs):
        # Decay exploration noise over time
        current_noise = exploration_noise * max(0.05, 1.0 - epoch / 1000)
        
        # Collect new trajectories with exploration noise
        new_trajs = [
            hw2.sample_trajectory_noisy(env, online_actor, current_noise).detach()
            for _ in range(batch_size)
        ]
        trajectory_buffer.extend(new_trajs)

        # Evict oldest when buffer is full
        if len(trajectory_buffer) > buffer_size:
            trajectory_buffer = trajectory_buffer[-buffer_size:]

        # Sample from buffer
        trajs = random.sample(
            trajectory_buffer, min(batch_size, len(trajectory_buffer))
        )

        mean_reward = torch.stack([t.rewards.mean() for t in trajs]).mean().item()
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            
        if epoch % 100 == 0:
            print(f"[Seed {seed}] Epoch {epoch}, mean: {mean_reward:.4f}, best: {best_reward:.4f}")

        # Skip training during warmup
        if epoch < warmup_steps:
            continue

        # Q updates
        target_q = MinQ(target_q_1, target_q_2)

        for q_net, opt in [(online_q_1, q1_opt), (online_q_2, q2_opt)]:
            opt.zero_grad()
            loss = torch.stack(
                [
                    hw2.q_td_loss(q_net, target_q, target_actor, traj, γ)
                    for traj in trajs
                ]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=0.5)
            opt.step()

        # Delayed policy + target updates
        if epoch % 2 == 0:
            actor_opt.zero_grad()
            loss = torch.stack(
                [
                    hw2.actor_gradient_ascent_loss(
                        online_actor, online_q_1, traj.observations
                    )
                    for traj in trajs
                ]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_actor.parameters(), max_norm=0.5)
            actor_opt.step()

            polyak_update(online_actor, target_actor)
            polyak_update(online_q_1, target_q_1)
            polyak_update(online_q_2, target_q_2)
    
    return best_reward


# Quick hyperparameter comparison
if __name__ == "__main__":
    configs = [
        {"lr": 1e-3, "network_width": 16, "exploration_noise": 0.1, "γ": 0.99},
        {"lr": 5e-4, "network_width": 24, "exploration_noise": 0.15, "γ": 0.98},
    ]

    for config_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {config_idx}: {config}")
        print(f"{'='*60}")
        results = []
        for seed in range(2):
            best = train_td3(seed=seed, max_epochs=2000, **config)
            results.append(best)
            print(f"  Seed {seed} final best: {best:.4f}")
        print(f"Config {config_idx} mean: {np.mean(results):.4f} ± {np.std(results):.4f}\n")
