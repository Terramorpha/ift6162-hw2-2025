from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


@dataclass
class Trajectory:
    observations: Tensor
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    final_observation: Tensor

    def detach(self):
        return Trajectory(
            self.observations.detach(),
            self.actions.detach(),
            self.log_probs.detach(),
            self.rewards.detach(),
            self.final_observation.detach(),
        )


class StochasticActor(Protocol):
    def action_dist(self, obs: Tensor, /) -> Distribution: ...


class Critic(Protocol):
    def value(self, obs: Tensor, /) -> Tensor: ...


class ActorCritic(StochasticActor, Critic, Protocol):
    pass


class DeterministicActor(Protocol):
    def action(self, obs: Tensor, /) -> Tensor: ...


class QCritic(Protocol):
    def q_value(self, obs: Tensor, act: Tensor, /) -> Tensor: ...


@dataclass
class Controller:
    policy: StochasticActor
    u_min: float
    u_max: float

    def get_action(self, obs: np.ndarray) -> float:
        u_01 = (
            self.policy.action_dist(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            .sample()
            .squeeze(-1)
            .item()
        )

        return (u_01 * (self.u_max - self.u_min)) + self.u_min


def sample_trajectory(env, policy) -> Trajectory:
    observations = []
    actions = []
    log_probs = []
    rewards = []
    obs = env.reset()

    done = False
    while not done:
        torch_obs = torch.tensor(obs, dtype=torch.float32)

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
            rewards=torch.tensor(rewards, dtype=torch.float32),
            final_observation=torch.tensor(obs, dtype=torch.float32),
        )


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int, depth: int):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
