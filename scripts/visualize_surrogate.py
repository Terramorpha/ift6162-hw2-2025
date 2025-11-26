#!/usr/bin/env python3
"""
Visualize Real vs Surrogate Dynamics

Creates Figure 3-style heatmaps comparing physics simulator and neural surrogate
under challenging conditions (cold start with time-varying controls).

Usage:
    python scripts/visualize_surrogate.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import CalcinerSimulator, SpatiallyAwareDynamics, SurrogateModel
from calciner.physics import N_SPECIES, L

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})


def load_surrogate(model_path: Path) -> SurrogateModel:
    """Load trained surrogate model."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    N_z = checkpoint['N_z']
    model = SpatiallyAwareDynamics(N_z=N_z)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    norm_params = {k: np.array(v) for k, v in checkpoint['norm_params'].items()}
    return SurrogateModel(model, norm_params, device=torch.device('cpu'))


def create_cold_start_initial_condition(simulator: CalcinerSimulator) -> np.ndarray:
    """
    Create a 'cold start' initial condition matching paper's Figure 3.
    - Uniform low concentrations
    - Uniform low temperature (600K)
    This shows the full transient from cold reactor to steady state.
    """
    N_z = simulator.N_z
    c = np.zeros((N_SPECIES, N_z))
    
    # Initial concentrations from paper: mostly inert gas
    c[0, :] = 0.1   # Kaolinite (AB2) - low initial
    c[1, :] = 0.1   # Quartz (Q) - low initial  
    c[2, :] = 0.1   # Metakaolin (A) - low initial
    c[3, :] = 19.65 # N2 (air) - high initial (like paper)
    c[4, :] = 0.1   # H2O (B) - low initial
    
    # Uniform cold temperature
    T_s = np.ones(N_z) * 600.0
    T_g = np.ones(N_z) * 600.0
    
    return simulator.state_to_vector(c, T_s, T_g)


def create_control_sequence(n_steps: int, scenario: str = 'varying') -> np.ndarray:
    """
    Create control sequence for evaluation.
    
    Scenarios:
    - 'constant': Fixed controls (easy)
    - 'varying': Time-varying controls with jumps (harder)
    - 'ramp': Gradual ramp-up (medium)
    """
    u_seq = np.zeros((n_steps, 2))
    
    if scenario == 'constant':
        u_seq[:, 0] = 1261.15  # T_g_in from paper
        u_seq[:, 1] = 657.15   # T_s_in from paper
        
    elif scenario == 'varying':
        # Time-varying with jumps - challenging for surrogate
        for t in range(n_steps):
            # Base values from paper
            T_g_base = 1261.15
            T_s_base = 657.15
            
            # Add sinusoidal variation
            T_g_var = 100 * np.sin(2 * np.pi * t / 40)
            T_s_var = 50 * np.sin(2 * np.pi * t / 60 + np.pi/4)
            
            # Add step changes
            if 20 <= t < 40:
                T_g_var += 80
            if 50 <= t < 70:
                T_s_var -= 40
            
            u_seq[t, 0] = np.clip(T_g_base + T_g_var, 1000, 1400)
            u_seq[t, 1] = np.clip(T_s_base + T_s_var, 580, 750)
            
    elif scenario == 'ramp':
        # Gradual ramp from cold to hot
        for t in range(n_steps):
            frac = min(t / 30, 1.0)
            u_seq[t, 0] = 900 + frac * 350  # 900 -> 1250
            u_seq[t, 1] = 550 + frac * 150  # 550 -> 700
    
    return u_seq


def run_comparison(simulator: CalcinerSimulator, surrogate: SurrogateModel,
                   n_steps: int = 80, scenario: str = 'varying') -> dict:
    """Run both physics and surrogate from cold start with varying controls."""
    
    # Cold start initial condition (like paper's Figure 3)
    x0 = create_cold_start_initial_condition(simulator)
    
    # Time-varying control sequence
    u_seq = create_control_sequence(n_steps, scenario)
    
    # Run physics simulator
    physics_traj = np.zeros((n_steps + 1, simulator.state_dim))
    physics_traj[0] = x0
    x = x0.copy()
    for t in range(n_steps):
        x = simulator.step(x, u_seq[t])
        physics_traj[t + 1] = x
    
    # Run surrogate
    with torch.no_grad():
        x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
        
        surrogate_traj = np.zeros((n_steps + 1, simulator.state_dim))
        surrogate_traj[0] = x0
        x_surr = x0_torch
        for t in range(n_steps):
            u_torch = torch.tensor(u_seq[t], dtype=torch.float32).unsqueeze(0)
            x_surr = surrogate.step(x_surr, u_torch)
            surrogate_traj[t + 1] = x_surr.numpy().squeeze()
    
    return {
        'physics': physics_traj,
        'surrogate': surrogate_traj,
        'controls': u_seq,
        'N_z': simulator.N_z,
        'n_steps': n_steps,
        'dt': simulator.dt,
        'scenario': scenario,
    }


def plot_figure3_style(results: dict, save_path: Path, source: str = 'physics'):
    """
    Create Figure 3-style plot with 8 vertically stacked panels.
    Shows: c_AB2, c_A, c_B, c_air, c_Q, T_s, T_g, and controls
    """
    traj = results[source]
    N_z = results['N_z']
    n_steps = results['n_steps']
    dt = results['dt']
    controls = results['controls']
    
    # Parse state
    conc_dim = N_SPECIES * N_z
    c = traj[:, :conc_dim].reshape(-1, N_SPECIES, N_z)  # (T, 5, N_z)
    T_s = traj[:, conc_dim:conc_dim + N_z]  # (T, N_z)
    T_g = traj[:, conc_dim + N_z:]  # (T, N_z)
    
    # Axes
    z = np.linspace(0, L, N_z)
    t = np.arange(n_steps + 1) * dt
    T_mesh, Z_mesh = np.meshgrid(t, z)
    
    # Species order matching paper: AB2, A, B, air, Q
    species_order = [0, 2, 4, 3, 1]  # Kaolinite, Metakaolin, H2O, N2, Quartz
    species_labels = [
        r'$c_{AB_2}$ [mol/m$^3$]',  # Kaolinite
        r'$c_A$ [mol/m$^3$]',        # Metakaolin
        r'$c_B$ [mol/m$^3$]',        # H2O
        r'$c_{air}$ [mol/m$^3$]',    # N2
        r'$c_Q$ [mol/m$^3$]',        # Quartz
    ]
    
    fig, axes = plt.subplots(8, 1, figsize=(8, 16), sharex=True)
    
    # Plot concentrations
    for idx, (sp_idx, label) in enumerate(zip(species_order, species_labels)):
        ax = axes[idx]
        data = c[:, sp_idx, :].T  # (N_z, T)
        pcm = ax.pcolormesh(T_mesh, Z_mesh, data, cmap='viridis', shading='auto')
        ax.set_ylabel('Length [m]')
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(label)
    
    # Plot T_s
    ax = axes[5]
    pcm = ax.pcolormesh(T_mesh, Z_mesh, T_s.T, cmap='viridis', shading='auto')
    ax.set_ylabel('Length [m]')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'$T_s$ [K]')
    
    # Plot T_g
    ax = axes[6]
    pcm = ax.pcolormesh(T_mesh, Z_mesh, T_g.T, cmap='viridis', shading='auto')
    ax.set_ylabel('Length [m]')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r'$T_g$ [K]')
    
    # Plot controls
    ax = axes[7]
    ax.plot(t[:-1], controls[:, 0], 'r-', label=r'$T_{g,in}$', linewidth=1.5)
    ax.plot(t[:-1], controls[:, 1], 'b-', label=r'$T_{s,in}$', linewidth=1.5)
    ax.set_ylabel('Control [K]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='right')
    ax.set_ylim([500, 1400])
    ax.grid(True, alpha=0.3)
    
    title = 'Physics Simulator' if source == 'physics' else 'Neural Surrogate'
    fig.suptitle(f'{title}: States in Time and Space', fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved {source} plot to {save_path}")


def plot_comparison_grid(results: dict, save_path: Path):
    """
    Create side-by-side comparison: Physics vs Surrogate for key variables.
    4 rows (c_AB2, c_A, T_s, T_g) × 3 columns (Physics, Surrogate, Error)
    """
    physics = results['physics']
    surrogate = results['surrogate']
    N_z = results['N_z']
    n_steps = results['n_steps']
    dt = results['dt']
    
    # Parse states
    conc_dim = N_SPECIES * N_z
    c_phys = physics[:, :conc_dim].reshape(-1, N_SPECIES, N_z)
    c_surr = surrogate[:, :conc_dim].reshape(-1, N_SPECIES, N_z)
    T_s_phys = physics[:, conc_dim:conc_dim + N_z]
    T_s_surr = surrogate[:, conc_dim:conc_dim + N_z]
    T_g_phys = physics[:, conc_dim + N_z:]
    T_g_surr = surrogate[:, conc_dim + N_z:]
    
    # Axes
    z = np.linspace(0, L, N_z)
    t = np.arange(n_steps + 1) * dt
    T_mesh, Z_mesh = np.meshgrid(t, z)
    
    # Variables to plot
    variables = [
        (c_phys[:, 0, :], c_surr[:, 0, :], r'$c_{AB_2}$ (Kaolinite)', 'viridis'),
        (c_phys[:, 2, :], c_surr[:, 2, :], r'$c_A$ (Metakaolin)', 'viridis'),
        (T_s_phys, T_s_surr, r'$T_s$ (Solid Temp)', 'plasma'),
        (T_g_phys, T_g_surr, r'$T_g$ (Gas Temp)', 'plasma'),
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    
    for row, (phys_data, surr_data, label, cmap) in enumerate(variables):
        vmin = min(phys_data.min(), surr_data.min())
        vmax = max(phys_data.max(), surr_data.max())
        
        # Physics
        ax = axes[row, 0]
        pcm = ax.pcolormesh(T_mesh, Z_mesh, phys_data.T, cmap=cmap, 
                            shading='auto', vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title('Physics', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{label}\nLength [m]')
        fig.colorbar(pcm, ax=ax, pad=0.02)
        
        # Surrogate
        ax = axes[row, 1]
        pcm = ax.pcolormesh(T_mesh, Z_mesh, surr_data.T, cmap=cmap,
                            shading='auto', vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title('Surrogate', fontsize=12, fontweight='bold')
        fig.colorbar(pcm, ax=ax, pad=0.02)
        
        # Absolute Error
        ax = axes[row, 2]
        error = np.abs(phys_data - surr_data)
        pcm = ax.pcolormesh(T_mesh, Z_mesh, error.T, cmap='Reds', shading='auto')
        if row == 0:
            ax.set_title('|Error|', fontsize=12, fontweight='bold')
        fig.colorbar(pcm, ax=ax, pad=0.02)
        
        if row == 3:
            for col in range(3):
                axes[row, col].set_xlabel('Time [s]')
    
    # Compute overall metrics
    total_phys = np.concatenate([physics[:, :conc_dim], physics[:, conc_dim:]], axis=1)
    total_surr = np.concatenate([surrogate[:, :conc_dim], surrogate[:, conc_dim:]], axis=1)
    mse = np.mean((total_phys - total_surr) ** 2)
    mre = np.mean(np.abs(total_phys - total_surr) / (np.abs(total_phys) + 1e-6))
    
    fig.suptitle(
        f'Physics vs Surrogate Comparison ({results["scenario"]} controls)\n'
        f'MSE: {mse:.2e} | MRE: {mre:.1%}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved comparison to {save_path}")
    
    return {'mse': mse, 'mre': mre}


def main():
    print("=" * 70)
    print("Surrogate vs Physics Visualization (Challenging Evaluation)")
    print("=" * 70)
    
    model_path = Path(__file__).parent.parent / "models" / "surrogate_model.pt"
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        print("  Run 'python scripts/train.py' first")
        return
    
    # Load model and simulator
    print("\nLoading surrogate model...")
    surrogate = load_surrogate(model_path)
    simulator = CalcinerSimulator(N_z=20, dt=0.1)
    
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Run comparison with time-varying controls (challenging)
    print("\nRunning comparison (cold start + varying controls, 80 steps = 8s)...")
    results = run_comparison(simulator, surrogate, n_steps=80, scenario='varying')
    
    # Generate Figure 3-style plots for both
    print("\nGenerating Figure 3-style plots...")
    plot_figure3_style(results, figures_dir / "physics_figure3.png", source='physics')
    plot_figure3_style(results, figures_dir / "surrogate_figure3.png", source='surrogate')
    
    # Generate comparison grid
    print("\nGenerating comparison grid...")
    metrics = plot_comparison_grid(results, figures_dir / "surrogate_comparison.png")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Metrics (varying controls, cold start):")
    print(f"  MSE: {metrics['mse']:.2e}")
    print(f"  MRE: {metrics['mre']:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

