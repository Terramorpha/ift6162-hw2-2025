"""
Main training script for IFT6162 HW2.

Usage:
    python main.py reinforce    - Train REINFORCE on simple environment
    python main.py ppo_simple   - Train PPO on simple environment
    python main.py td3          - Train TD3 on simple environment
    python main.py ppo_complex  - Train PPO on complex surrogate environment
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.nn.functional import softplus

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calciner import (
    CalcinerEnv,
    SpatiallyAwareDynamics,
    SurrogateCalcinerEnv,
    SurrogateModel,
)
from hw2.reinforce import train_reinforce
from hw2.ppo import train_ppo
from hw2.td3 import train_td3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


# =============================================================================
# Simple Environment Policy Networks (3D state)
# =============================================================================


class SimplePolicy(nn.Module):
    """Simple MLP policy for 3D state (CalcinerEnv)."""

    def __init__(self, state_dim=3, hidden_dim=64, n_layers=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, 2)  # Beta distribution parameters

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        self.actor_head.bias.data[0] = 1.0
        self.actor_head.bias.data[1] = 0.0

    def action_dist(self, obs: torch.Tensor):
        features = self.trunk(obs)
        params = self.actor_head(features)
        alpha = softplus(params[:, 0]) + 1.0
        beta = softplus(params[:, 1]) + 1.0
        return Beta(alpha, beta)


class SimpleActorCritic(nn.Module):
    """Actor-Critic for simple 3D environment."""

    def __init__(self, state_dim=3, hidden_dim=64, n_layers=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, 2)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        self.actor_head.bias.data[0] = 1.0
        self.actor_head.bias.data[1] = 0.0
        self.critic_head.bias.data[0] = -50.0  # Reasonable for simple env

    def action_dist(self, obs: torch.Tensor):
        features = self.trunk(obs)
        params = self.actor_head(features)
        alpha = softplus(params[:, 0]) + 1.0
        beta = softplus(params[:, 1]) + 1.0
        return Beta(alpha, beta)

    def value(self, obs: torch.Tensor):
        features = self.trunk(obs)
        return self.critic_head(features)


class SimpleTD3Actor(nn.Module):
    """Deterministic actor for TD3."""

    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def action(self, obs: torch.Tensor):
        return self.network(obs)


class SimpleTD3Critic(nn.Module):
    """Q-function critic for TD3."""

    def __init__(self, state_dim=3, action_dim=1, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def q_value(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


# =============================================================================
# Complex Environment Policy (140D state with obs normalization)
# =============================================================================


class MLPActorCritic(nn.Module):
    """MLP Actor-Critic for complex 140D environment with observation normalization."""

    def __init__(self, state_dim=140, hidden_dim=256, n_layers=3):
        super().__init__()
        self.state_dim = state_dim

        # Observation normalization
        obs_mean = torch.zeros(state_dim)
        obs_mean[:100] = 0.05
        obs_mean[100:] = 900.0
        self.register_buffer("obs_mean", obs_mean)

        obs_std = torch.ones(state_dim)
        obs_std[:100] = 0.1
        obs_std[100:] = 200.0
        self.register_buffer("obs_std", obs_std)

        # MLP trunk
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, 2)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        self.actor_head.bias.data[0] = 2.0
        self.actor_head.bias.data[1] = -1.0
        self.critic_head.bias.data[0] = -2000.0

    def _normalize_obs(self, obs: torch.Tensor):
        obs_mean: torch.Tensor = self.obs_mean  # type: ignore
        obs_std: torch.Tensor = self.obs_std  # type: ignore
        return (obs - obs_mean) / (obs_std + 1e-8)

    def action_dist(self, obs: torch.Tensor):
        obs_norm = self._normalize_obs(obs)
        features = self.trunk(obs_norm)
        params = self.actor_head(features)
        alpha = softplus(params[:, 0]) + 1.0
        beta = softplus(params[:, 1]) + 1.0
        return Beta(alpha, beta)

    def value(self, obs: torch.Tensor):
        obs_norm = self._normalize_obs(obs)
        features = self.trunk(obs_norm)
        return self.critic_head(features)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_simple_trajectory(policy, iteration, env, save_dir):
    """Plot trajectory for simple environment."""
    obs = env.reset(seed=42)

    temperatures = []
    alphas = []
    rewards = []
    alpha_targets = []

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if hasattr(policy, "action_dist"):
                action = policy.action_dist(obs_t).mean
            else:  # TD3
                action = policy.action(obs_t)

        T_g = action.item() * (env.u_max - env.u_min) + env.u_min
        temperatures.append(T_g)
        alphas.append(obs[0])  # Current alpha
        alpha_targets.append(obs[1])  # Target alpha_min

        obs, reward, done, info = env.step(T_g)
        rewards.append(reward)

    # Final alpha
    alphas.append(obs[0])
    alpha_targets.append(obs[1])

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Temperature
    axes[0].plot(temperatures, linewidth=2, color="red")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(f"Iteration {iteration}: Temperature Profile")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([900, 1300])

    # Conversion
    axes[1].plot(alphas, linewidth=2, color="blue", label="α")
    axes[1].plot(
        alpha_targets, linewidth=1, color="red", linestyle="--", label="α_target"
    )
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Conversion (α)")
    axes[1].set_title(f"Conversion (final: {alphas[-1]:.3f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cumulative reward
    cumulative = np.cumsum(rewards)
    axes[2].plot(cumulative, linewidth=2, color="green")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Cumulative Reward")
    axes[2].set_title(f"Reward (total: {cumulative[-1]:.1f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_dir}/iter_{iteration:04d}.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_complex_trajectory(policy, iteration, env, save_dir):
    """Plot trajectory for complex environment."""
    obs = env.reset(seed=42)

    temperatures = []
    alphas = []
    rewards = []

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_dist = policy.action_dist(obs_t)
            action = action_dist.mean

        T_g = action.item() * (env.u_max - env.u_min) + env.u_min
        temperatures.append(T_g)

        obs, reward, done, info = env.step(T_g)
        alphas.append(info["alpha"])
        rewards.append(reward)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Temperature
    axes[0].plot(temperatures, linewidth=2, color="red")
    axes[0].axhline(1300, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(f"Iteration {iteration}: Temperature Profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([900, 1350])

    # Conversion
    axes[1].plot(alphas, linewidth=2, color="blue")
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.5, label="Target")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Conversion (α)")
    axes[1].set_title(f"Conversion (final: {alphas[-1]:.3f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cumulative reward
    cumulative = np.cumsum(rewards)
    axes[2].plot(cumulative, linewidth=2, color="green")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Cumulative Reward")
    axes[2].set_title(f"Reward (total: {cumulative[-1]:.1f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_dir}/iter_{iteration:04d}.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curves(result, method_name, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean rewards
    axes[0].plot(result.mean_rewards, linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title(f"{method_name}: Learning Curve")
    axes[0].grid(True, alpha=0.3)

    # Critic/value loss
    if hasattr(result, "critic_losses"):
        axes[1].plot(result.critic_losses, linewidth=1.5, color="orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Critic Loss")
        axes[1].set_title("Value Function Loss")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No critic loss\n(REINFORCE)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Value Function Loss")

    # Policy/actor loss
    loss_key = "policy_losses" if hasattr(result, "policy_losses") else "actor_losses"
    if hasattr(result, loss_key):
        losses = getattr(result, loss_key)
        axes[2].plot(losses, linewidth=1.5, color="green")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Policy Loss")
        axes[2].set_title("Policy Loss")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}", flush=True)
    plt.close()


# =============================================================================
# Training Experiments
# =============================================================================


def run_reinforce_simple(seed=0):
    """Train REINFORCE on simple environment."""
    print("\n" + "=" * 60)
    print("Experiment 1: REINFORCE on Simple Environment (3D state)")
    print("=" * 60)

    env = CalcinerEnv(episode_length=40)
    policy = SimplePolicy(state_dim=3, hidden_dim=64, n_layers=2)

    print(
        f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}", flush=True
    )

    # Callback for plotting
    plot_dir = "figures/reinforce_simple_trajectories"

    def callback(model, iteration, env):
        plot_simple_trajectory(model, iteration, env, plot_dir)

    result = train_reinforce(
        env,
        policy,
        batch_size=64,
        lr=1e-3,
        max_iterations=200,
        seed=seed,
        verbose=True,
        plot_callback=callback,
        plot_interval=50,
    )

    # Save results
    torch.save(result.policy, "models/reinforce_simple_model.pth")
    with open("results/reinforce_simple_results.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot training curves
    plot_training_curves(
        result, "REINFORCE (Simple)", "figures/reinforce_simple_training.png"
    )

    print(f"\nResults:", flush=True)
    print(f"  Initial reward: {result.mean_rewards[0]:.2f}", flush=True)
    print(f"  Final reward: {result.mean_rewards[-1]:.2f}", flush=True)
    print(f"  Best reward: {max(result.mean_rewards):.2f}", flush=True)

    return result


def run_ppo_simple(seed=1):
    """Train PPO on simple environment."""
    print("\n" + "=" * 60)
    print("Experiment 2: PPO on Simple Environment (3D state)")
    print("=" * 60)

    env = CalcinerEnv(episode_length=40)
    policy = SimpleActorCritic(state_dim=3, hidden_dim=64, n_layers=2)

    print(
        f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}", flush=True
    )

    # Callback for plotting
    plot_dir = "figures/ppo_simple_trajectories"

    def callback(model, iteration, env):
        plot_simple_trajectory(model, iteration, env, plot_dir)

    result = train_ppo(
        env,
        policy,
        batch_size=64,
        n_epochs=10,
        lr=1e-3,
        γ=0.99,
        λ=0.95,
        max_iterations=200,
        seed=seed,
        verbose=True,
        ent_coef=0.01,
        plot_callback=callback,
        plot_interval=50,
    )

    # Save results
    torch.save(result.model, "models/ppo_simple_model.pth")
    with open("results/ppo_simple_results.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot training curves
    plot_training_curves(result, "PPO (Simple)", "figures/ppo_simple_training.png")

    print(f"\nResults:", flush=True)
    print(f"  Initial reward: {result.mean_rewards[0]:.2f}", flush=True)
    print(f"  Final reward: {result.mean_rewards[-1]:.2f}", flush=True)
    print(f"  Best reward: {max(result.mean_rewards):.2f}", flush=True)

    return result


def run_td3_simple(seed=2):
    """Train TD3 on simple environment."""
    print("\n" + "=" * 60)
    print("Experiment 3: TD3 on Simple Environment (3D state)")
    print("=" * 60)

    env = CalcinerEnv(episode_length=40)
    actor = SimpleTD3Actor(state_dim=3, hidden_dim=64)
    critic_1 = SimpleTD3Critic(state_dim=3, action_dim=1, hidden_dim=64)
    critic_2 = SimpleTD3Critic(state_dim=3, action_dim=1, hidden_dim=64)

    total_params = (
        sum(p.numel() for p in actor.parameters())
        + sum(p.numel() for p in critic_1.parameters())
        + sum(p.numel() for p in critic_2.parameters())
    )
    print(f"Total parameters: {total_params:,}", flush=True)

    # Callback for plotting
    plot_dir = "figures/td3_simple_trajectories"

    def callback(model, iteration, env):
        plot_simple_trajectory(model, iteration, env, plot_dir)

    result = train_td3(
        env,
        actor,
        critic_1,
        critic_2,
        batch_size=64,
        buffer_size=2000,
        lr=1e-4,  # Very conservative LR
        γ=0.99,
        exploration_noise=0.15,
        max_epochs=300,
        warmup_steps=50,
        policy_delay=4,  # Update actor less frequently
        seed=seed,
        verbose=True,
        plot_callback=callback,
        plot_interval=50,
    )

    # Save results
    torch.save(
        {
            "actor": result.actor,
            "critic_1": result.critic_1,
            "critic_2": result.critic_2,
        },
        "models/td3_simple_model.pth",
    )
    with open("results/td3_simple_results.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot training curves
    plot_training_curves(result, "TD3 (Simple)", "figures/td3_simple_training.png")

    print(f"\nResults:", flush=True)
    print(f"  Initial reward: {result.mean_rewards[0]:.2f}", flush=True)
    print(f"  Final reward: {result.mean_rewards[-1]:.2f}", flush=True)
    print(f"  Best reward: {max(result.best_rewards):.2f}", flush=True)

    return result


def run_ppo_complex(seed=3):
    """Train PPO on complex surrogate environment."""
    print("\n" + "=" * 60)
    print("Experiment 4: PPO on Complex Environment (140D state)")
    print("=" * 60)

    # Load surrogate
    model_path = Path("models/surrogate_model.pt")
    checkpoint = torch.load(model_path, weights_only=False)
    N_z = checkpoint["N_z"]

    surrogate_model = SpatiallyAwareDynamics(N_z=N_z)
    surrogate_model.load_state_dict(checkpoint["model_state_dict"])
    surrogate_model.eval()

    norm_params = {k: np.array(v) for k, v in checkpoint["norm_params"].items()}
    surrogate = SurrogateModel(surrogate_model, norm_params)

    env = SurrogateCalcinerEnv(
        surrogate, episode_length=50, alpha_min=0.95, control_T_s=False
    )

    policy = MLPActorCritic(state_dim=140, hidden_dim=256, n_layers=3)

    print(
        f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}", flush=True
    )

    # Callback for plotting
    plot_dir = "figures/ppo_complex_trajectories"

    def callback(model, iteration, env):
        plot_complex_trajectory(model, iteration, env, plot_dir)

    result = train_ppo(
        env,
        policy,
        batch_size=64,
        n_epochs=10,
        lr=3e-4,
        γ=0.99,
        λ=1.0,
        max_iterations=200,
        seed=seed,
        verbose=True,
        ent_coef=0.01,
        plot_callback=callback,
        plot_interval=50,
    )

    # Save results
    torch.save(result.model, "models/ppo_complex_model.pth")
    with open("results/ppo_complex_results.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot training curves
    plot_training_curves(result, "PPO (Complex)", "figures/ppo_complex_training.png")

    print(f"\nResults:", flush=True)
    print(f"  Initial reward: {result.mean_rewards[0]:.2f}", flush=True)
    print(f"  Final reward: {result.mean_rewards[-1]:.2f}", flush=True)
    print(f"  Best reward: {max(result.mean_rewards):.2f}", flush=True)

    return result


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_simple_model(policy, env, n_episodes=20):
    """Evaluate a policy on the simple environment."""
    episode_rewards = []
    episode_energies = []
    episode_violations = []
    episode_final_alphas = []

    for ep in range(n_episodes):
        obs = env.reset(seed=100 + ep)
        total_reward = 0.0
        total_energy = 0.0
        violations = 0
        final_alpha = 0.0

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                if hasattr(policy, "action_dist"):
                    action = policy.action_dist(obs_t).mean
                else:  # TD3 deterministic
                    action = policy.action(obs_t)

            T_g = action.item() * (env.u_max - env.u_min) + env.u_min
            obs, reward, done, info = env.step(T_g)

            total_reward += reward
            total_energy += info["power"]
            final_alpha = info["alpha"]

            # Check constraint violation
            if info["alpha"] < env.alpha_min[env.t - 1]:
                violations += 1

        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_violations.append(violations)
        episode_final_alphas.append(final_alpha)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_energy": np.mean(episode_energies),
        "mean_violations": np.mean(episode_violations),
        "mean_final_alpha": np.mean(episode_final_alphas),
    }


def evaluate_complex_model(policy, surrogate, n_episodes=20):
    """Evaluate a policy on the complex surrogate environment."""
    env = SurrogateCalcinerEnv(
        surrogate, episode_length=50, alpha_min=0.95, control_T_s=False
    )

    episode_rewards = []
    episode_energies = []
    episode_violations = []
    episode_final_alphas = []

    for ep in range(n_episodes):
        obs = env.reset(seed=100 + ep)
        total_reward = 0.0
        total_energy = 0.0
        violations = 0
        final_alpha = 0.0

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_dist = policy.action_dist(obs_t)
                action = action_dist.mean

            T_g = action.item() * (env.u_max - env.u_min) + env.u_min
            obs, reward, done, info = env.step(T_g)

            total_reward += reward
            total_energy += info["energy"]
            if info["violation"] > 0:
                violations += 1
            final_alpha = info["alpha"]

        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_violations.append(violations)
        episode_final_alphas.append(final_alpha)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_energy": np.mean(episode_energies),
        "mean_violations": np.mean(episode_violations),
        "mean_final_alpha": np.mean(episode_final_alphas),
    }


def evaluate_baseline_simple(env, T_g=1100.0, n_episodes=20):
    """Evaluate constant temperature baseline on simple env."""
    episode_rewards = []
    episode_energies = []
    episode_violations = []
    episode_final_alphas = []

    for ep in range(n_episodes):
        obs = env.reset(seed=100 + ep)
        total_reward = 0.0
        total_energy = 0.0
        violations = 0
        final_alpha = 0.0

        done = False
        while not done:
            obs, reward, done, info = env.step(T_g)
            total_reward += reward
            total_energy += info["power"]
            final_alpha = info["alpha"]
            if info["alpha"] < env.alpha_min[env.t - 1]:
                violations += 1

        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_violations.append(violations)
        episode_final_alphas.append(final_alpha)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_energy": np.mean(episode_energies),
        "mean_violations": np.mean(episode_violations),
        "mean_final_alpha": np.mean(episode_final_alphas),
    }


def evaluate_baseline_complex(surrogate, T_g=1300.0, n_episodes=20):
    """Evaluate constant temperature baseline on complex env."""
    env = SurrogateCalcinerEnv(
        surrogate, episode_length=50, alpha_min=0.95, control_T_s=False
    )

    episode_rewards = []
    episode_energies = []
    episode_violations = []
    episode_final_alphas = []

    for ep in range(n_episodes):
        obs = env.reset(seed=100 + ep)
        total_reward = 0.0
        total_energy = 0.0
        violations = 0
        final_alpha = 0.0

        done = False
        while not done:
            obs, reward, done, info = env.step(T_g)
            total_reward += reward
            total_energy += info["energy"]
            final_alpha = info["alpha"]
            if info["violation"] > 0:
                violations += 1

        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_violations.append(violations)
        episode_final_alphas.append(final_alpha)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_energy": np.mean(episode_energies),
        "mean_violations": np.mean(episode_violations),
        "mean_final_alpha": np.mean(episode_final_alphas),
    }


def run_evaluation():
    """Evaluate all trained models against baselines."""
    print("\n" + "=" * 60)
    print("Evaluating All Trained Models")
    print("=" * 60)

    # Simple environment
    env_simple = CalcinerEnv(episode_length=40)

    # Complex environment
    model_path = Path("models/surrogate_model.pt")
    checkpoint = torch.load(model_path, weights_only=False)
    N_z = checkpoint["N_z"]
    surrogate_model = SpatiallyAwareDynamics(N_z=N_z)
    surrogate_model.load_state_dict(checkpoint["model_state_dict"])
    surrogate_model.eval()
    norm_params = {k: np.array(v) for k, v in checkpoint["norm_params"].items()}
    surrogate = SurrogateModel(surrogate_model, norm_params)

    results = {}

    # Evaluate baselines
    print("\n" + "-" * 60)
    print("Baselines - Simple Environment")
    print("-" * 60)
    baseline_simple = evaluate_baseline_simple(env_simple, T_g=1100.0, n_episodes=20)
    print(f"Constant T_g=1100K:")
    print(
        f"  Reward: {baseline_simple['mean_reward']:.2f} ± {baseline_simple['std_reward']:.2f}"
    )
    print(f"  Energy: {baseline_simple['mean_energy']:.2f}")
    print(f"  Violations: {baseline_simple['mean_violations']:.1f} / 40 steps")
    print(f"  Final α: {baseline_simple['mean_final_alpha']:.4f}")
    results["baseline_simple"] = baseline_simple

    print("\n" + "-" * 60)
    print("Baselines - Complex Environment")
    print("-" * 60)
    baseline_complex = evaluate_baseline_complex(surrogate, T_g=1300.0, n_episodes=20)
    print(f"Constant T_g=1300K:")
    print(
        f"  Reward: {baseline_complex['mean_reward']:.2f} ± {baseline_complex['std_reward']:.2f}"
    )
    print(f"  Energy: {baseline_complex['mean_energy']:.2f}")
    print(f"  Violations: {baseline_complex['mean_violations']:.1f} / 50 steps")
    print(f"  Final α: {baseline_complex['mean_final_alpha']:.4f}")
    results["baseline_complex"] = baseline_complex

    # Evaluate REINFORCE
    if Path("models/reinforce_simple_model.pth").exists():
        print("\n" + "-" * 60)
        print("REINFORCE (Simple Environment)")
        print("-" * 60)
        policy = torch.load("models/reinforce_simple_model.pth", weights_only=False)
        policy.eval()
        reinforce_results = evaluate_simple_model(policy, env_simple, n_episodes=20)
        print(
            f"  Reward: {reinforce_results['mean_reward']:.2f} ± {reinforce_results['std_reward']:.2f}"
        )
        print(f"  Energy: {reinforce_results['mean_energy']:.2f}")
        print(f"  Violations: {reinforce_results['mean_violations']:.1f} / 40 steps")
        print(f"  Final α: {reinforce_results['mean_final_alpha']:.4f}")
        print(
            f"  Improvement over baseline: {reinforce_results['mean_reward'] - baseline_simple['mean_reward']:.2f}"
        )
        results["reinforce"] = reinforce_results

    # Evaluate PPO Simple
    if Path("models/ppo_simple_model.pth").exists():
        print("\n" + "-" * 60)
        print("PPO (Simple Environment)")
        print("-" * 60)
        policy = torch.load("models/ppo_simple_model.pth", weights_only=False)
        policy.eval()
        ppo_simple_results = evaluate_simple_model(policy, env_simple, n_episodes=20)
        print(
            f"  Reward: {ppo_simple_results['mean_reward']:.2f} ± {ppo_simple_results['std_reward']:.2f}"
        )
        print(f"  Energy: {ppo_simple_results['mean_energy']:.2f}")
        print(f"  Violations: {ppo_simple_results['mean_violations']:.1f} / 40 steps")
        print(f"  Final α: {ppo_simple_results['mean_final_alpha']:.4f}")
        print(
            f"  Improvement over baseline: {ppo_simple_results['mean_reward'] - baseline_simple['mean_reward']:.2f}"
        )
        results["ppo_simple"] = ppo_simple_results

    # Evaluate TD3
    if Path("models/td3_simple_model.pth").exists():
        print("\n" + "-" * 60)
        print("TD3 (Simple Environment)")
        print("-" * 60)
        checkpoint = torch.load("models/td3_simple_model.pth", weights_only=False)
        policy = checkpoint["actor"]
        policy.eval()
        td3_results = evaluate_simple_model(policy, env_simple, n_episodes=20)
        print(
            f"  Reward: {td3_results['mean_reward']:.2f} ± {td3_results['std_reward']:.2f}"
        )
        print(f"  Energy: {td3_results['mean_energy']:.2f}")
        print(f"  Violations: {td3_results['mean_violations']:.1f} / 40 steps")
        print(f"  Final α: {td3_results['mean_final_alpha']:.4f}")
        print(
            f"  Improvement over baseline: {td3_results['mean_reward'] - baseline_simple['mean_reward']:.2f}"
        )
        results["td3"] = td3_results

    # Evaluate PPO Complex
    if Path("models/ppo_complex_model.pth").exists():
        print("\n" + "-" * 60)
        print("PPO (Complex Environment)")
        print("-" * 60)
        policy = torch.load("models/ppo_complex_model.pth", weights_only=False)
        policy.eval()
        ppo_complex_results = evaluate_complex_model(policy, surrogate, n_episodes=20)
        print(
            f"  Reward: {ppo_complex_results['mean_reward']:.2f} ± {ppo_complex_results['std_reward']:.2f}"
        )
        print(f"  Energy: {ppo_complex_results['mean_energy']:.2f}")
        print(f"  Violations: {ppo_complex_results['mean_violations']:.1f} / 50 steps")
        print(f"  Final α: {ppo_complex_results['mean_final_alpha']:.4f}")
        print(
            f"  Improvement over baseline: {ppo_complex_results['mean_reward'] - baseline_complex['mean_reward']:.2f}"
        )
        results["ppo_complex"] = ppo_complex_results

    # Save all evaluation results
    eval_results_path = Path("results/evaluation_results.pkl")
    with open(eval_results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n\nAll evaluation results saved to {eval_results_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(
        f"{'Method':<20} {'Reward':<15} {'Energy':<15} {'Violations':<15} {'Final α':<10}"
    )
    print("-" * 75)

    if "baseline_simple" in results:
        r = results["baseline_simple"]
        print(
            f"{'Baseline (Simple)':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )

    if "reinforce" in results:
        r = results["reinforce"]
        print(
            f"{'REINFORCE':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )

    if "ppo_simple" in results:
        r = results["ppo_simple"]
        print(
            f"{'PPO (Simple)':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )

    if "td3" in results:
        r = results["td3"]
        print(
            f"{'TD3':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )

    print()

    if "baseline_complex" in results:
        r = results["baseline_complex"]
        print(
            f"{'Baseline (Complex)':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )

    if "ppo_complex" in results:
        r = results["ppo_complex"]
        print(
            f"{'PPO (Complex)':<20} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} {r['mean_energy']:>6.2f}       {r['mean_violations']:>6.1f}       {r['mean_final_alpha']:>6.4f}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [reinforce|ppo_simple|td3|ppo_complex|evaluate]")
        sys.exit(1)

    experiment = sys.argv[1].lower()

    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    if experiment == "reinforce":
        run_reinforce_simple(seed=0)
    elif experiment == "ppo_simple":
        run_ppo_simple(seed=1)
    elif experiment == "td3":
        run_td3_simple(seed=2)
    elif experiment == "ppo_complex":
        run_ppo_complex(seed=3)
    elif experiment == "evaluate":
        run_evaluation()
    else:
        print(f"Unknown experiment: {experiment}")
        print("Valid options: reinforce, ppo_simple, td3, ppo_complex, evaluate")
        sys.exit(1)

    if experiment != "evaluate":
        print("\n" + "=" * 60)
        print("Experiment Complete!")
        print("=" * 60)
