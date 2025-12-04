from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hw2 import Critic, Trajectory, sample_trajectory


@dataclass
class PPOTrainingResult:
    model: nn.Module
    mean_rewards: list[float]
    critic_losses: list[float]
    policy_losses: list[float]


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
    log_ratio = torch.clamp(log_ratio, -20, 20)
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
) -> PPOTrainingResult:
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mean_rewards_history = []
    critic_losses_history = []
    policy_losses_history = []

    for epoch in range(max_iterations):
        trajs = [sample_trajectory(env, model).detach() for _ in range(batch_size)]  # type: ignore

        all_rewards = torch.stack([t.rewards for t in trajs])
        mean_reward = all_rewards.mean().item()
        mean_rewards_history.append(mean_reward)

        if verbose:
            print(f"Iteration {epoch}, mean reward: {mean_reward:.4f}")

        critic_loss_val = 0.0
        policy_loss_val = 0.0

        for k in range(n_epochs):
            value_losses = [td_loss(model, traj, γ) for traj in trajs]
            critic_loss = torch.stack(value_losses).mean()
            critic_loss_val = critic_loss.item()

            optimizer.zero_grad()
            critic_loss.backward()
            optimizer.step()

            policy_losses = [ppo_loss(model, traj, γ, λ=λ) for traj in trajs]
            policy_loss = torch.stack(policy_losses).mean()
            policy_loss_val = policy_loss.item()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        critic_losses_history.append(critic_loss_val)
        policy_losses_history.append(policy_loss_val)

        if verbose and epoch % 10 == 0:
            print(f"  critic: {critic_loss_val:.2f}, policy: {policy_loss_val:.4f}")

    return PPOTrainingResult(
        model=model,
        mean_rewards=mean_rewards_history,
        critic_losses=critic_losses_history,
        policy_losses=policy_losses_history,
    )
