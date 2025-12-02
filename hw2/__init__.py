from dataclasses import dataclass
from typing import Protocol

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


class Policy(Protocol):
    def action_dist(self, obs: Tensor) -> Distribution:
        pass


def sample_trajectory(env: CalcinerEnv, policy: Policy) -> Trajectory:
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

        log_prob = action_dist.log_prob(action)

        actions.append(action)
        log_probs.append(log_prob)

        scaled_action = action.item() * (env.u_max - env.u_min) + env.u_min

        obs, reward, done, info = env.step(scaled_action)
        rewards.append(reward)
    return Trajectory(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        log_probs=torch.stack(log_probs),
        rewards=torch.Tensor(rewards),
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


def policy_gradients_loss(traj: Trajectory):
    reverse_cumsum_rewards = traj.rewards.flip(0).cumsum(0).flip(0)
    neglogliks = -traj.log_probs

    return (neglogliks * reverse_cumsum_rewards).sum()
