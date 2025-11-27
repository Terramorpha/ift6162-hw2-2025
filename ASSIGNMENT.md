# Assignment: Deep RL for Flash Calciner Control

Control an industrial reactor that converts clay to metakaolin by heating. You control gas inlet temperature $T_{g,in} \in [900, 1300]$ K. Higher temperatures speed up the reaction but waste energy. The goal: achieve target conversion $\alpha \geq \alpha_{min}$ while minimizing heater power.

| Part | State | Task |
|------|-------|------|
| **1** | 3D (scalar dynamics) | Implement REINFORCE, PPO, TD3 (50 pts) |
| **2** | 140D (PDE surrogate) | Scale up your best algorithm (50 pts) |

Run `python scripts/demo.py` to verify setup. Read `docs/model.md` for full physics.

## Part 1: Simplified Problem (50 points)

The outlet conversion $\alpha \in [0,1]$ follows first-order lag dynamics toward a temperature-dependent equilibrium:

$$\alpha_{k+1} = e^{-\Delta t/\tau} \alpha_k + (1 - e^{-\Delta t/\tau}) \, \alpha_{ss}(u_k)$$

where $\tau = 2$ s, $\Delta t = 0.5$ s, and the steady-state conversion $\alpha_{ss}(T) = 0.999/(1 + \exp(-0.025(T - 1000)))$ captures Arrhenius kinetics. Low temperatures give ~50% conversion; high temperatures approach 100%.

**Environment**: State is $[\alpha, \alpha_{min}, t/T] \in \mathbb{R}^3$. Action is $T_{g,in} \in [900, 1300]$ K. Reward is $-\text{energy} - 10 \cdot \max(0, \alpha_{min} - \alpha)^2$.

Implement three algorithms:

**REINFORCE** (15 pts): Policy gradient with baseline. Use Monte Carlo returns. Train 200+ episodes.

**PPO** (20 pts): Clipped surrogate objective with value network. Compare sample efficiency to REINFORCE.

**TD3** (15 pts): Twin critics, delayed updates, target smoothing. Use replay buffer.

Compare all three against `ConstantTemperatureController(T_g_in=1261)` baseline. Report learning curves, final energy consumption, and constraint violations. Does the policy learn to modulate temperature based on current $\alpha$ and $\alpha_{min}$?

---

## Part 2: Full 140D Problem (50 points)

The full state tracks 5 species + 2 temperatures across 20 spatial cells (140D). Physics simulation is too slow (~25 ms/step) for RL. We provide a neural surrogate (60× faster, 19% error) wrapped in `SurrogateCalcinerEnv`.

**Algorithm** (25 pts): Adapt your best Part 1 algorithm to 140D. You'll need a neural network policy—linear won't work. Should you use fully-connected layers or 1D convolutions to capture spatial structure? Start simple: just look at outlet conversion, then incorporate spatial profiles.

**Evaluation** (15 pts): Validate on the true physics simulator (`CalcinerSimulator`), not just the surrogate. Report energy and violations. Visualize spatial profiles—does the policy create smooth gradients?

The surrogate is differentiable if you want to exploit that.

---

## Deliverables

Code, 6-page report (algorithms, setup, results, discussion), trained models.

## Grading

REINFORCE (15), PPO (20), TD3 (15), Part 2 algorithm (25), Part 2 evaluation (15). Partial credit given.