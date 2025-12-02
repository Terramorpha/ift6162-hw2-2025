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

from calciner import CalcinerEnv
from hw2 import MLP, Trajectory, policy_gradients_loss, sample_trajectory


def handler(_a, _b, tb):
    pdb.post_mortem()


sys.excepthook = handler


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
        mean = mean_log_std[:, 0].clamp(-5, 2)
        log_std = mean_log_std[:, 1]
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)),
            [TanhTransform(), AffineTransform(0.5, 0.5)],
        )


env = CalcinerEnv()


policy = MLPPolicy(3, 8)


optimizer = torch.optim.SGD(policy.mlp.parameters(), lr=1e-7)

batch_size = 1 << 7


for i in range(1000):
    trajs = [sample_trajectory(env, policy) for _ in range(batch_size)]

    all_rewards = torch.stack([t.rewards for t in trajs])

    print("mean:", all_rewards.mean())

    loss = sum(policy_gradients_loss(traj) for traj in trajs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
