from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import torch
import torch.nn as nn
from calciner import CalcinerEnv
from torch import Tensor
from torch.distributions import Distribution


@dataclass
class Trajectory:
    observations: Tensor
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    final_observation: Tensor

    def detach(self) -> Self:
        return Trajectory(
            self.observations.detach(),
            self.actions.detach(),
            self.log_probs.detach(),
            self.rewards.detach(),
            self.final_observation.detach(),
        )


class Actor(Protocol):
    def action_dist(self, obs: Tensor, /) -> Distribution:
        pass


class Critic(Protocol):
    def value(self, obs: Tensor, /) -> Tensor:
        pass


class ActorCritic(Actor, Critic, Protocol):
    pass


@dataclass
class Controller:
    policy: Actor
    u_min: float
    u_max: float

    def get_action(self, obs: np.ndarray) -> float:
        u_01 = (
            self.policy.action_dist(torch.from_numpy(obs).unsqueeze(0))
            .sample()
            .squeeze(-1)
            .item()
        )

        return (u_01 * (self.u_max - self.u_min)) + self.u_min


def sample_trajectory(env: CalcinerEnv, policy: Actor) -> Trajectory:
    observations = []
    actions = []
    log_probs = []
    rewards = []
    obs = env.reset()

    done = False
    while not done:
        torch_obs = torch.from_numpy(obs)

        observations.append(torch_obs)

        action_dist = policy.action_dist(torch_obs.unsqueeze(0))

        action = action_dist.sample()

        log_prob = action_dist.log_prob(action).squeeze(0)

        actions.append(action)
        log_probs.append(log_prob)

        scaled_action = action.item() * (env.u_max - env.u_min) + env.u_min

        obs, reward, done, info = env.step(scaled_action)
        rewards.append(reward)
    else:
        return Trajectory(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            rewards=Tensor(rewards),
            final_observation=Tensor(obs),
        )


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int, depth: int):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(width, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def policy_gradients_loss(traj: Trajectory) -> Tensor:
    reverse_cumsum_rewards = traj.rewards.flip(0).cumsum(0).flip(0)

    rewards_mean = reverse_cumsum_rewards.mean()
    rewards_std = reverse_cumsum_rewards.std() + 1e-8
    normalized_rewards = (reverse_cumsum_rewards - rewards_mean) / rewards_std

    log_probs = traj.log_probs

    return -(log_probs * normalized_rewards).sum()


def importance_sampling_policy_gradients_loss(
    policy: Actor, traj: Trajectory
) -> Tensor:
    reverse_cumsum_rewards = traj.rewards.flip(0).cumsum(0).flip(0)

    sampling_log_probs = traj.log_probs
    model_log_probs = policy.action_dist(traj.observations).log_prob(
        traj.actions.squeeze(-1)
    )

    ratios = torch.exp(model_log_probs - sampling_log_probs).detach()

    return -(model_log_probs * ratios * reverse_cumsum_rewards).sum()


def discounted_returns(rewards: Tensor, γ: float) -> Tensor:
    out = torch.zeros_like(rewards)

    out[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        out[i] = rewards[i] + γ * out[i + 1]

    return out


def compute_residuals(critic: Critic, traj: Trajectory, γ: float) -> Tensor:
    observations = torch.cat(
        [traj.observations, traj.final_observation.unsqueeze(0)],
    )

    δ = (
        traj.rewards
        + γ * critic.value(observations[1:, :]).squeeze()
        - critic.value(observations[:-1, :]).squeeze()
    )

    return δ


def generalized_advantage_estimation(
    critic: Critic, traj: Trajectory, γ: float, λ: float
):
    δₜ = compute_residuals(critic, traj, γ)

    return discounted_returns(δₜ, γ * λ)


def ppo_loss(
    model: ActorCritic,
    traj: Trajectory,
    γ: float,
    λ: float,
    ε: float = 0.2,
) -> Tensor:
    advantages = generalized_advantage_estimation(model, traj, γ, λ).detach()

    # Normalize advantages
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


def td_loss(model: Critic, traj: Trajectory, γ: float) -> Tensor:
    residuals = compute_residuals(model, traj, γ)
    return (residuals**2).sum()
