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
from hw2 import MLP
from hw2.ppo import train_ppo
from hw2.reinforce import train_reinforce
from hw2.td3 import train_td3


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, depth: int, width: int):
        super().__init__()
        self.mlp = MLP(
            input_size=obs_dim,
            output_size=2,
            width=width,
            depth=depth,
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


def train_reinforce_simple(seed=0):
    env = CalcinerEnv()
    obs_dim = env.reset().shape[0]
    policy = MLPPolicy(obs_dim=obs_dim, depth=2, width=32)

    result = train_reinforce(
        env,
        policy,
        batch_size=64,
        lr=1e-3,
        max_iterations=200,
        seed=seed,
        verbose=True,
    )

    return result


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, depth: int, width: int):
        super().__init__()
        self.trunk = MLP(
            input_size=obs_dim,
            output_size=width,
            width=width,
            depth=depth,
        )

        self.actor_head = nn.Linear(width, 2)
        self.critic_head = nn.Linear(width, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def action_dist(self, obs: torch.Tensor):
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


def train_ppo_simple(seed=0):
    env = CalcinerEnv()
    obs_dim = env.reset().shape[0]
    model = MLPActorCritic(obs_dim=obs_dim, depth=3, width=8)

    result = train_ppo(
        env,
        model,
        batch_size=64,
        n_epochs=3,
        lr=1e-3,
        γ=0.99,
        λ=0.95,
        max_iterations=200,
        seed=seed,
        verbose=True,
    )

    return result


class DeterministicActorMLP(nn.Module):
    def __init__(self, obs_dim: int, depth: int, width: int):
        super().__init__()

        self.mlp = MLP(
            obs_dim,
            1,
            width,
            depth,
        )

    def action(self, obs: torch.Tensor, /) -> torch.Tensor:
        return torch.sigmoid(self.mlp(obs))


class QCriticMLP(nn.Module):
    def __init__(self, obs_dim: int, depth: int, width: int):
        super().__init__()

        self.mlp = MLP(
            input_size=obs_dim + 1,
            output_size=1,
            width=width,
            depth=depth,
        )

    def q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        concat = torch.concatenate([obs, action], dim=-1)
        return self.mlp(concat)


def train_td3_simple(seed=0):
    env = CalcinerEnv()
    obs_dim = env.reset().shape[0]

    actor = DeterministicActorMLP(obs_dim=obs_dim, depth=3, width=16)
    critic_1 = QCriticMLP(obs_dim=obs_dim, depth=3, width=16)
    critic_2 = QCriticMLP(obs_dim=obs_dim, depth=3, width=16)

    result = train_td3(
        env,
        actor,
        critic_1,
        critic_2,
        batch_size=16,
        buffer_size=256,
        lr=1e-3,
        γ=0.99,
        exploration_noise=0.1,
        max_epochs=5000,
        warmup_steps=50,
        policy_delay=2,
        seed=seed,
        verbose=True,
    )

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train_simple.py [reinforce|ppo|td3]")
        sys.exit(1)

    algorithm = sys.argv[1].lower()

    if algorithm == "reinforce":
        print("Training REINFORCE...")
        result = train_reinforce_simple()
        print("Training complete!")
        print(f"Final mean reward: {result.mean_rewards[-1]:.4f}")
    elif algorithm == "ppo":
        print("Training PPO...")
        result = train_ppo_simple()
        print("Training complete!")
        print(f"Final mean reward: {result.mean_rewards[-1]:.4f}")
    elif algorithm == "td3":
        print("Training TD3...")
        result = train_td3_simple()
        print("Training complete!")
        print(f"Final mean reward: {result.mean_rewards[-1]:.4f}")
        print(f"Best reward: {result.best_rewards[-1]:.4f}")
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available: reinforce, ppo, td3")
        sys.exit(1)
