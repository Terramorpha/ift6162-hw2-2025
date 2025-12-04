import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hw2 import Trajectory, MLP


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
    q1: nn.Module
    q2: nn.Module

    def q_value(self, obs: Tensor, action: Tensor) -> Tensor:
        v1 = self.q1.q_value(obs, action)  # type: ignore
        v2 = self.q2.q_value(obs, action)  # type: ignore

        return torch.min(v1, v2)


def polyak_update(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

        for buffer, ema_buffer in zip(model.buffers(), ema_model.buffers()):
            ema_buffer.data = decay * ema_buffer.data + (1 - decay) * buffer.data


def q_td_loss(
    online,
    target_critic,
    target_actor,
    traj: Trajectory,
    γ: float,
    σ: float = 0.1,
) -> Tensor:
    current_state = traj.observations
    current_actions = traj.actions
    next_state = torch.concatenate(
        [traj.observations[1:, :], traj.final_observation.unsqueeze(0)]
    )

    current_q = online.q_value(current_state, current_actions)

    next_actions = target_actor.action(next_state)

    noise = torch.clip(σ * torch.randn(next_actions.shape), -0.5, 0.5)
    perturbed_actions = (next_actions + noise).clip(0.0, 1.0)

    delta = current_q - (
        traj.rewards
        + γ
        * target_critic.q_value(
            next_state,
            perturbed_actions,
        )
    )

    return (delta**2).sum()


def actor_gradient_ascent_loss(actor, target_critic, observations: Tensor) -> Tensor:
    q = target_critic.q_value(observations, actor.action(observations))

    return (-q).sum()


def sample_trajectory_noisy(env, actor, noise_scale: float) -> Trajectory:
    observations = []
    actions = []
    rewards = []
    obs = env.reset()

    done = False
    while not done:
        torch_obs = torch.from_numpy(obs)
        observations.append(torch_obs)

        with torch.no_grad():
            action = actor.action(torch_obs.unsqueeze(0)).squeeze(0)
            action = action + noise_scale * torch.randn_like(action)
            action = torch.clamp(action, 0, 1)

        actions.append(action)

        scaled_action = action.item() * (env.u_max - env.u_min) + env.u_min
        obs, reward, done, _ = env.step(scaled_action)
        rewards.append(reward)

    return Trajectory(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        log_probs=torch.ones(len(rewards)),
        rewards=torch.tensor(rewards),
        final_observation=torch.from_numpy(obs),
    )


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size: int) -> list[Trajectory]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def train_td3(
    env,
    actor: nn.Module,
    critic_1: nn.Module,
    critic_2: nn.Module,
    *,
    batch_size: int = 16,
    buffer_size: int = 256,
    lr: float = 1e-3,
    γ: float = 0.99,
    exploration_noise: float = 0.1,
    max_epochs: int = 5000,
    warmup_steps: int = 50,
    policy_delay: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    target_actor = copy.deepcopy(actor)
    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_opt = torch.optim.Adam(critic_1.parameters(), lr=lr)
    q2_opt = torch.optim.Adam(critic_2.parameters(), lr=lr)

    replay_buffer = ReplayBuffer(buffer_size)
    best_reward = -float("inf")

    for epoch in range(max_epochs):
        current_noise = exploration_noise * max(0.05, 1.0 - epoch / 1000)

        new_trajs = [
            sample_trajectory_noisy(env, actor, current_noise).detach()
            for _ in range(batch_size)
        ]
        for traj in new_trajs:
            replay_buffer.add(traj)

        trajs = replay_buffer.sample(batch_size)

        mean_reward = torch.stack([t.rewards.mean() for t in trajs]).mean().item()

        if mean_reward > best_reward:
            best_reward = mean_reward

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, mean: {mean_reward:.4f}, best: {best_reward:.4f}")

        if epoch < warmup_steps:
            continue

        target_q = MinQ(target_critic_1, target_critic_2)

        for q_net, opt in [(critic_1, q1_opt), (critic_2, q2_opt)]:
            opt.zero_grad()
            loss = torch.stack(
                [q_td_loss(q_net, target_q, target_actor, traj, γ) for traj in trajs]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=0.5)
            opt.step()

        if epoch % policy_delay == 0:
            actor_opt.zero_grad()
            loss = torch.stack(
                [
                    actor_gradient_ascent_loss(actor, critic_1, traj.observations)
                    for traj in trajs
                ]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_opt.step()

            polyak_update(actor, target_actor)
            polyak_update(critic_1, target_critic_1)
            polyak_update(critic_2, target_critic_2)

    return actor, critic_1, critic_2
