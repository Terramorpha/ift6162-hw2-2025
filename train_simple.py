from calciner import CalcinerEnv
from hw2.ppo import MLPActorCritic, train_ppo
from hw2.reinforce import MLPPolicy, train_reinforce
from hw2.td3 import DeterministicActorMLP, QCriticMLP, train_td3


def train_reinforce_simple(seed=0):
    env = CalcinerEnv()
    policy = MLPPolicy(depth=2, width=32)

    trained_policy = train_reinforce(
        env,
        policy,
        batch_size=64,
        lr=1e-3,
        max_iterations=200,
        seed=seed,
        verbose=True,
    )

    return trained_policy


def train_ppo_simple(seed=0):
    env = CalcinerEnv()
    model = MLPActorCritic(depth=3, width=8)

    trained_model = train_ppo(
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

    return trained_model


def train_td3_simple(seed=0):
    env = CalcinerEnv()

    actor = DeterministicActorMLP(depth=3, width=16)
    critic_1 = QCriticMLP(depth=3, width=16)
    critic_2 = QCriticMLP(depth=3, width=16)

    trained_actor, trained_critic_1, trained_critic_2 = train_td3(
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

    return trained_actor, trained_critic_1, trained_critic_2


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train_simple.py [reinforce|ppo|td3]")
        sys.exit(1)

    algorithm = sys.argv[1].lower()

    if algorithm == "reinforce":
        print("Training REINFORCE...")
        policy = train_reinforce_simple()
        print("Training complete!")
    elif algorithm == "ppo":
        print("Training PPO...")
        model = train_ppo_simple()
        print("Training complete!")
    elif algorithm == "td3":
        print("Training TD3...")
        actor, critic_1, critic_2 = train_td3_simple()
        print("Training complete!")
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available: reinforce, ppo, td3")
        sys.exit(1)
