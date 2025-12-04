from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hw2 import Critic, Trajectory, sample_trajectory, MLP


class MLPActorCritic(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        self.trunk = MLP(
            input_size=3,
            output_size=width,
            width=width,
            depth=depth,
        )

        self.actor_head = nn.Linear(width, 2)
        self.critic_head = nn.Linear(width, 1)

    def action_dist(self, obs: torch.Tensor):
        from torch.distributions import (
            AffineTransform,
            Normal,
            TanhTransform,
            TransformedDistribution,
        )

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


def compute_residuals(critic, traj: Trajectory, γ: float) -> Tensor:
    observations = torch.cat(
        [traj.observations, traj.final_observation.unsqueeze(0)],
    )

    δ = (
        traj.rewards
        + γ * critic.value(observations[1:, :]).squeeze()
        - critic.value(observations[:-1, :]).squeeze()
    )

    return δ


def discounted_returns(rewards: Tensor, γ: float) -> Tensor:
    out = torch.zeros_like(rewards)

    out[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        out[i] = rewards[i] + γ * out[i + 1]

    return out


def generalized_advantage_estimation(critic, traj: Trajectory, γ: float, λ: float):
    δₜ = compute_residuals(critic, traj, γ)

    return discounted_returns(δₜ, γ * λ)


def ppo_loss(
    model,
    traj: Trajectory,
    γ: float,
    λ: float,
    ε: float = 0.2,
) -> Tensor:
    advantages = generalized_advantage_estimation(model, traj, γ, λ).detach()

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    sampling_log_probs = traj.log_probs
    model_log_probs = model.action_dist(traj.observations).log_prob(
        traj.actions.squeeze(-1)
    )

    log_ratio = model_log_probs - sampling_log_probs
    ratios = torch.exp(log_ratio)

    clipped_ratios = torch.clamp(ratios, 1 - ε, 1 + ε)

    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    return policy_loss


def td_loss(model, traj: Trajectory, γ: float) -> Tensor:
    residuals = compute_residuals(model, traj, γ)
    return (residuals**2).sum()


def train_ppo(
    env,
    model: nn.Module,
    *,
    batch_size: int = 64,
    n_epochs: int = 3,
    lr: float = 1e-3,
    γ: float = 0.99,
    λ: float = 0.95,
    max_iterations: int = 200,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> nn.Module:
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    critic_loss = None
    policy_loss = None

    for epoch in range(max_iterations):
        trajs = [sample_trajectory(env, model).detach() for _ in range(batch_size)]  # type: ignore

        all_rewards = torch.stack([t.rewards for t in trajs])
        if verbose:
            print(f"Iteration {epoch}, mean reward: {all_rewards.mean().item():.4f}")

        for k in range(n_epochs):
            value_losses = [td_loss(model, traj, γ) for traj in trajs]
            critic_loss = torch.stack(value_losses).mean()

            optimizer.zero_grad()
            critic_loss.backward()
            optimizer.step()

            policy_losses = [ppo_loss(model, traj, γ, λ=λ) for traj in trajs]
            policy_loss = torch.stack(policy_losses).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if (
            verbose
            and epoch % 10 == 0
            and critic_loss is not None
            and policy_loss is not None
        ):
            print(
                f"  critic: {critic_loss.item():.2f}, policy: {policy_loss.item():.4f}"
            )

    return model
