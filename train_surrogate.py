import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.distributions import (
    AffineTransform,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from calciner import SpatiallyAwareDynamics, SurrogateModel, SurrogateCalcinerEnv
from hw2.ppo import train_ppo


class ConvActorCritic(nn.Module):
    """
    1D CNN-based Actor-Critic for spatially-structured 140D state.

    State structure: 140D = 5 species × 20 cells + 2 temps × 20 cells
    Reshaped to: (7 channels, 20 spatial positions)

    Ultra-lightweight: single aggressive downsample + tiny MLP (~2K params).
    """

    def __init__(self, n_channels: int = 7, n_spatial: int = 20, hidden_dim: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_spatial = n_spatial

        # Single aggressive downsample: 20 -> 5 (stride=4)
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
        )

        # After stride-4 conv: 20 -> 5
        self.flattened_size = 16 * 5

        # Tiny shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (outputs mean and log_std for action distribution)
        self.actor_head = nn.Linear(hidden_dim, 2)

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _reshape_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat observation to (batch, channels, spatial).

        Input: (batch, 140) where 140 = 5*20 + 2*20
        Output: (batch, 7, 20) where 7 = 5 species + 2 temps
        """
        batch_size = obs.shape[0]

        # Split into species (100) and temps (40)
        species_flat = obs[:, :100]  # 5 species × 20 cells
        temps_flat = obs[:, 100:]  # 2 temps × 20 cells

        # Reshape to (batch, channels, spatial)
        species = species_flat.reshape(batch_size, 5, self.n_spatial)
        temps = temps_flat.reshape(batch_size, 2, self.n_spatial)

        # Concatenate along channel dimension
        return torch.cat([species, temps], dim=1)  # (batch, 7, 20)

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observation using conv layers."""
        x = self._reshape_obs(obs)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.trunk(x)
        return x

    def action_dist(self, obs: torch.Tensor):
        """Return action distribution for given observation."""
        features = self._extract_features(obs)
        mean_log_std = self.actor_head(features)
        mean = torch.clamp(mean_log_std[:, 0], -3, 3)
        log_std = torch.clamp(mean_log_std[:, 1], -5, 2)
        std = torch.exp(log_std)
        return TransformedDistribution(
            Normal(mean, std),
            [TanhTransform(), AffineTransform(0.5, 0.5)],
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state value for given observation."""
        features = self._extract_features(obs)
        return self.critic_head(features)


def train_ppo_surrogate(seed=0):
    """Train PPO with CNN on the full 140D surrogate environment."""
    # Load surrogate model
    model_path = Path("models/surrogate_model.pt")
    checkpoint = torch.load(model_path, weights_only=False)
    N_z = checkpoint["N_z"]

    model = SpatiallyAwareDynamics(N_z=N_z)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_params = {k: np.array(v) for k, v in checkpoint["norm_params"].items()}
    surrogate = SurrogateModel(model, norm_params)

    # Create environment
    env = SurrogateCalcinerEnv(
        surrogate, episode_length=50, alpha_min=0.95, control_T_s=False
    )

    # Create CNN-based actor-critic
    ppo_model = ConvActorCritic(
        n_channels=7,  # 5 species + 2 temps
        n_spatial=20,  # 20 spatial cells
        hidden_dim=32,
    )

    print(f"Model parameters: {sum(p.numel() for p in ppo_model.parameters()):,}")

    # Train
    result = train_ppo(
        env,
        ppo_model,
        batch_size=16,
        n_epochs=3,
        lr=3e-4,
        γ=0.99,
        λ=0.95,
        max_iterations=500,
        seed=seed,
        verbose=True,
    )

    return result


if __name__ == "__main__":
    print("Training PPO with 1D CNN on 140D surrogate environment...")
    result = train_ppo_surrogate()
    print("\nTraining complete!")
    print(f"Final mean reward: {result.mean_rewards[-1]:.4f}")

    # Optionally plot learning curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(result.mean_rewards)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean Reward")
        axes[0].set_title("Learning Curve")
        axes[0].grid(True)

        axes[1].plot(result.critic_losses)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Critic Loss")
        axes[1].set_title("Critic Loss")
        axes[1].grid(True)

        axes[2].plot(result.policy_losses)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Policy Loss")
        axes[2].set_title("Policy Loss")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("figures/ppo_surrogate_training.png", dpi=150)
        print("Learning curves saved to figures/ppo_surrogate_training.png")
    except ImportError:
        print("matplotlib not available for plotting")
