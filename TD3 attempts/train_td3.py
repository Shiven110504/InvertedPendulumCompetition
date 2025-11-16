import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl.models import ReplayBuffer
from rl.pendulum_env_cont import PendulumBalanceEnvCont
from rl.algos.td3 import TD3, TD3Config


class RunningNorm:
    def __init__(self, shape: int):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)


def plot_eval_metrics(save_dir: str,
                      eval_returns: list,
                      eval_scores: list,
                      eval_balance_counts: list):
    if len(eval_scores) == 0:
        return
    indices = np.arange(len(eval_scores))
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(indices, eval_returns, label="Training Return @ Eval", color="tab:orange")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(indices, eval_scores, label="Eval Reward", color="tab:blue")
    axes[1].set_ylabel("Eval Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(indices, eval_balance_counts, label="Eval Balance Count", color="tab:green")
    axes[2].set_ylabel("Balance Count")
    axes[2].set_xlabel("Evaluation Index")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "eval_metrics.png")
    fig.savefig(fig_path)
    plt.close(fig)


def _steps_for_balance(env: PendulumBalanceEnvCont, balance_target: int) -> int:
    if balance_target <= 0:
        return 1
    start = getattr(env, "_next_push_time", 0.5)
    duration = getattr(env, "_push_duration", 0.1)
    settle = 0.5
    gap = getattr(env, "_push_gap", 4.0)
    t_final = start + duration + settle + max(0, balance_target - 1) * gap
    steps = int(np.ceil(t_final / max(env.control_dt, 1e-6)))
    return max(1, steps)


def _linear_noise(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + (end - start) * progress)


def _evaluate(agent: TD3,
              env: PendulumBalanceEnvCont,
              obs_norm: RunningNorm,
              max_steps: int,
              seed: int,
              noise_mask: np.ndarray,
              force_zero_action: bool = False) -> Tuple[float, int]:
    state, _ = env.reset(seed=seed)
    total = 0.0
    balance_count = 0
    for _ in range(max_steps):
        if force_zero_action:
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        else:
            action = agent.act(obs_norm.normalize(state), noise_std=0.0, noise_mask=noise_mask)
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        balance_count = int(info.get("balance_count", balance_count))
        if terminated or truncated:
            break
    return total, balance_count


def train_td3(max_steps: Optional[int] = None,
              seed: int = 0,
              batch_size: int = 256,
              warmup_steps: int = 10000,
              start_noise_std: float = 0.005,
              end_noise_std: float = 0.001,
              eval_every: int = 20,
              save_dir: str = "rl/checkpoints",
              control_dt: Optional[float] = None,
              reward_mode: str = "custom_reward",
              action_scale: float = 1.0,
              residual_scale: float = 0.2,
              training_episodes: int = 2000,
              refine_episodes: int = 1000,
              noise_actuator_mask: Optional[np.ndarray] = None,
              load_actor_path: Optional[str] = None,
              load_critic_path: Optional[str] = None,
              load_norm_path: Optional[str] = None,
              actor_lr: float = 1e-4,
              critic_lr: float = 1e-4,
              policy_noise: float = 0.1,
              noise_clip: float = 0.25,
              replay_max_size: int = 200_000,
              norm_freeze_steps: Optional[int] = 500_000,
              balance_target: int = 10,
              residual_enable_threshold: int = 5,
              residual_disable_threshold: int = 3) -> Tuple[list, list]:

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_kwargs = dict(control_dt=control_dt,
                      reward_mode=reward_mode,
                      action_scale=action_scale,
                      residual_scale=residual_scale)
    env = PendulumBalanceEnvCont(**env_kwargs)
    eval_env = PendulumBalanceEnvCont(**env_kwargs)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if max_steps is None:
        max_steps = _steps_for_balance(env, balance_target)
    if noise_actuator_mask is None:
        noise_mask = np.zeros(action_dim, dtype=np.float32)
        noise_mask[0] = 1.0
    else:
        noise_mask = np.asarray(noise_actuator_mask, dtype=np.float32).reshape(-1)
        if noise_mask.shape[0] != action_dim:
            raise ValueError("noise_actuator_mask must match action dimension.")

    cfg = TD3Config(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
    )
    agent = TD3(cfg)

    if load_actor_path is not None:
        if not os.path.exists(load_actor_path):
            raise FileNotFoundError(f"Actor checkpoint not found: {load_actor_path}")
        state_dict = torch.load(load_actor_path, map_location=agent.actor.device)
        agent.actor.load_state_dict(state_dict)
        agent.actor_target.load_state_dict(state_dict)
        print(f"Loaded actor weights from {load_actor_path}")
    if load_critic_path is not None:
        if not os.path.exists(load_critic_path):
            raise FileNotFoundError(f"Critic checkpoint not found: {load_critic_path}")
        state_dict = torch.load(load_critic_path, map_location=agent.critic.device)
        agent.critic.load_state_dict(state_dict)
        agent.critic_target.load_state_dict(state_dict)
        print(f"Loaded critic weights from {load_critic_path}")

    rb = ReplayBuffer(max_size=replay_max_size, batch_size=batch_size)
    obs_norm = RunningNorm(state_dim)
    norm_frozen = False
    if load_norm_path is not None:
        if not os.path.exists(load_norm_path):
            raise FileNotFoundError(f"Normalizer file not found: {load_norm_path}")
        with np.load(load_norm_path) as norm_stats:
            obs_norm.load_state_dict(dict(mean=norm_stats["mean"],
                                          var=norm_stats["var"],
                                          count=norm_stats["count"]))
        print(f"Loaded observation normalizer from {load_norm_path}")
        norm_frozen = True

    scores, eval_scores = [], []
    eval_balance_counts: list = []
    eval_train_returns: list = []
    total_env_steps = 0
    best_eval = -np.inf
    best_actor_state = None
    best_critic_state = None
    best_obs_norm_state = None
    total_episode_count = training_episodes + refine_episodes
    residual_enabled = False
    residual_success = 0
    residual_fail = 0
    if total_episode_count <= 0:
        raise ValueError("training_episodes + refine_episodes must be positive.")

    for ep_idx in range(1, total_episode_count + 1):
        training_phase = ep_idx <= training_episodes
        state, _ = env.reset(seed=seed + ep_idx)
        if not norm_frozen:
            obs_norm.update(state[None, :])
        ep_ret = 0.0
        episode_balance = 0

        current_noise = 0.0
        for _ in range(max_steps):
            if total_env_steps < warmup_steps or not residual_enabled:
                action = np.zeros(action_dim, dtype=np.float32)
                current_noise = 0.0
            else:
                if training_phase and training_episodes > 1:
                    progress = (ep_idx - 1) / max(1, training_episodes - 1)
                    current_noise = _linear_noise(start_noise_std, end_noise_std, progress)
                elif training_phase:
                    current_noise = end_noise_std
                else:
                    current_noise = 0.0
                nstate = obs_norm.normalize(state)
                action = agent.act(nstate, noise_std=current_noise, noise_mask=noise_mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rb.store((state, action, reward, next_state, done))
            if not norm_frozen:
                obs_norm.update(next_state[None, :])
                if norm_freeze_steps is not None and total_env_steps >= norm_freeze_steps:
                    norm_frozen = True
                    print(f"[{reward_mode}] Observation normalizer frozen at step {total_env_steps}.")
            state = next_state
            ep_ret += reward
            total_env_steps += 1
            episode_balance = int(info.get("balance_count", episode_balance))

            if total_env_steps >= warmup_steps and len(rb) >= batch_size:
                batch = rb.draw_samples(batch_size)
                states, actions, rewards, next_states, dones = batch
                norm_states = obs_norm.normalize(states)
                norm_next_states = obs_norm.normalize(next_states)
                agent.train_step((norm_states, actions, rewards, norm_next_states, dones))

            if done:
                break

        scores.append(ep_ret)
        if episode_balance >= 1:
            residual_success += 1
            residual_fail = 0
        else:
            residual_fail += 1
            residual_success = 0

        if not residual_enabled and residual_success >= residual_enable_threshold:
            residual_enabled = True
            print(f"[{reward_mode}] Residual control enabled at episode {ep_idx}.")
        if residual_enabled and residual_fail >= residual_disable_threshold:
            residual_enabled = False
            residual_fail = 0
            print(f"[{reward_mode}] Residual control disabled after {residual_disable_threshold} zero-balance episodes.")

        if eval_every > 0 and ep_idx % eval_every == 0:
            eval_seed = seed + 1000 + ep_idx
            eval_return, balance_eval = _evaluate(agent, eval_env, obs_norm, max_steps, eval_seed, noise_mask)
            baseline_return, baseline_balance = _evaluate(
                agent, eval_env, obs_norm, max_steps, eval_seed, noise_mask, force_zero_action=True)
            eval_scores.append(eval_return)
            eval_balance_counts.append(balance_eval)
            eval_train_returns.append(ep_ret)
            phase = "train" if training_phase else "refine"
            noise_msg = f" | Noise {current_noise:.3f}" if training_phase and residual_enabled else ""
            mode_msg = "ON" if residual_enabled else "BASE"
            print(
                f"[{reward_mode}] Ep {ep_idx} ({phase}) [{mode_msg}] | "
                f"TrainRet {ep_ret:.2f} | EvalRet {eval_return:.2f} | Balance {balance_eval}"
                f"{noise_msg} | BaseEval {baseline_return:.2f} ({baseline_balance})"
            )

            if eval_return > best_eval:
                best_eval = eval_return
                best_actor_state = agent.actor.state_dict()
                best_critic_state = agent.critic.state_dict()
                best_obs_norm_state = obs_norm.state_dict()
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_actor_state, os.path.join(save_dir, "td3_actor_best.pt"))
                torch.save(best_critic_state, os.path.join(save_dir, "td3_critic_best.pt"))
                np.savez(os.path.join(save_dir, "td3_obs_norm_best.npz"), **best_obs_norm_state)

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "td3_obs_norm.npz"), **obs_norm.state_dict())
    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "td3_actor.pt"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "td3_critic.pt"))
    if best_actor_state is not None:
        torch.save(best_actor_state, os.path.join(save_dir, "td3_actor_best.pt"))
    if best_critic_state is not None:
        torch.save(best_critic_state, os.path.join(save_dir, "td3_critic_best.pt"))
    if best_obs_norm_state is not None:
        np.savez(os.path.join(save_dir, "td3_obs_norm_best.npz"), **best_obs_norm_state)

    plot_eval_metrics(save_dir, eval_train_returns, eval_scores, eval_balance_counts)
    return scores, eval_scores


if __name__ == "__main__":
    print("\n=== Training residual TD3 on baseline controller ===")
    train_td3(start_noise_std=0,end_noise_std=0)
