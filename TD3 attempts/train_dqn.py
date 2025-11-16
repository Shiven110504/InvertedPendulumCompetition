import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from rl.models import FCQ, ReplayBuffer
from rl.dqn_agent import DQN
from rl.pendulum_env import PendulumBalanceEnv


def train(episodes=1000,
          max_steps=1000,
          seed=0,
          batch_size=256,
          warmup_batches=5,
          update_freq=50,
          learning_rate=5e-4,
          gamma=0.99,
          eval_every=10,
          soft_tau=0.01,
          hidden=(256, 128),
          # Env-related knobs
          per_joint_scale=0.2,
          ctrl_kp=150.0,
          ctrl_kd=5.2,
          control_dt=0.005,
          save_dir="rl/checkpoints"):

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = PendulumBalanceEnv(
        per_joint_scale=per_joint_scale,
        ctrl_kp=ctrl_kp,
        ctrl_kd=ctrl_kd,
        control_dt=control_dt,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # With 3**6 = 729 actions by default, a slightly larger head can help.
    online = FCQ(obs_dim, act_dim, hidden_dim=hidden)
    target = FCQ(obs_dim, act_dim, hidden_dim=hidden)
    target.load_state_dict(online.state_dict())

    opt = optim.RMSprop(online.parameters(), lr=learning_rate)
    rb = ReplayBuffer(batch_size=batch_size)
    agent = DQN(env, rb, online, target, opt, warmup_batches, update_freq, epochs=1, gamma=gamma)

    min_steps_to_learn = warmup_batches * batch_size
    scores, evals = [], []
    decay_steps = max(1, int(episodes * 0.8))

    for e in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + e)
        eps = max(1.0 - e / decay_steps, 0.05)
        ep_ret = 0.0

        for t in range(max_steps):
            state, reward, terminated, truncated, _ = agent.interaction_step(state, eps)
            ep_ret += reward

            if len(rb) >= min_steps_to_learn:
                agent.learn()
                if soft_tau is not None:
                    agent.soft_update_weights(tau=soft_tau)

            if terminated or truncated:
                break

        scores.append(ep_ret)

        if soft_tau is None and e % update_freq == 0:
            agent.soft_update_weights(tau=1.0)

        if e % eval_every == 0:
            evals.append(agent.demo(render=False))

        if e % 50 == 0:
            print(f"Episode {e}/{episodes} | 50-mean: {np.mean(scores[-50:]):.2f} | mean: {np.mean(scores):.2f} | eps: {eps:.2f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(online.state_dict(), os.path.join(save_dir, "dqn_online.pt"))
    torch.save(target.state_dict(), os.path.join(save_dir, "dqn_target.pt"))

    # Plot evals if any
    if len(evals) > 0:
        plt.figure()
        plt.plot(evals, label="Evaluation score")
        window_size = min(50, len(evals))
        if window_size >= 3:
            moving_averages = np.convolve(evals, np.ones(window_size), 'valid') / window_size
            xs = np.arange(window_size, len(evals) + 1)
            plt.plot(xs, moving_averages, label=f"{window_size}-Episode Moving Average")
        plt.xlabel("Eval #")
        plt.ylabel("Return")
        plt.title("DQN Pendulum Evaluation Scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_eval.png"))

    return scores, evals


if __name__ == "__main__":
    train()
