import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl.algos.td3 import TD3, TD3Config
from rl.models import ReplayBuffer
from rl.baseyaw_env import BaseYawResidualEnv


class RunningNorm:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def update(self, batch: np.ndarray):
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = M2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)


def _steps_for_balance(env: BaseYawResidualEnv, balance_target: int) -> int:
    base_env = env._env  # type: ignore[attr-defined]
    start = getattr(base_env, "_next_push_time", 0.5)
    duration = getattr(base_env, "_push_duration", 0.1)
    settle = 0.5
    gap = getattr(base_env, "_push_gap", 4.0)
    t_final = start + duration + settle + max(0, balance_target - 1) * gap
    steps = int(np.ceil(t_final / max(base_env.control_dt, 1e-6)))
    return max(1, steps)


def _evaluate(agent: TD3,
              env: BaseYawResidualEnv,
              obs_norm: RunningNorm,
              max_steps: int,
              seed: int) -> Tuple[float, int]:
    state, _ = env.reset(seed=seed)
    total = 0.0
    balance = 0
    for _ in range(max_steps):
        action = agent.act(obs_norm.normalize(state), noise_std=0.0)
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        balance = int(info.get("balance_count", balance))
        if terminated or truncated:
            break
    return total, balance


def train_td3_baseyaw(episodes: int = 2000,
                      warmup_steps: int = 10000,
                      start_noise_std: float = 0.003,
                      end_noise_std: float = 0.0005,
                      eval_every: int = 20,
                      save_dir: str = "rl/checkpoints_baseyaw",
                      control_dt: Optional[float] = None,
                      reward_mode: str = "custom_reward",
                      action_scale: float = 1.0,
                      residual_scale: float = 0.2,
                      balance_target: int = 10,
                      batch_size: int = 256,
                      replay_max_size: int = 200_000,
                      norm_freeze_steps: Optional[int] = 300_000,
                      seed: int = 0) -> Tuple[list, list]:

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_kwargs = dict(control_dt=control_dt,
                      reward_mode=reward_mode,
                      action_scale=action_scale,
                      residual_scale=residual_scale)
    env = BaseYawResidualEnv(**env_kwargs)
    eval_env = BaseYawResidualEnv(**env_kwargs)
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    max_steps = _steps_for_balance(env, balance_target)

    cfg = TD3Config(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=1e-4,
        critic_lr=1e-4,
        policy_noise=0.05,
        noise_clip=0.1,
    )
    agent = TD3(cfg)

    rb = ReplayBuffer(max_size=replay_max_size, batch_size=batch_size)
    obs_norm = RunningNorm(state_dim)
    norm_frozen = False

    scores, eval_scores = [], []
    eval_balance = []
    total_env_steps = 0
    best_eval = -np.inf
    best_actor_state = None
    best_critic_state = None
    best_obs_state = None

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + ep)
        if not norm_frozen:
            obs_norm.update(state[None, :])
        ep_ret = 0.0

        for _ in range(max_steps):
            if total_env_steps < warmup_steps:
                action = np.zeros(action_dim, dtype=np.float32)
                noise_std = 0.0
            else:
                frac = (ep - 1) / max(1, episodes - 1)
                noise_std = float(start_noise_std + (end_noise_std - start_noise_std) * frac)
                noise_std = max(noise_std, end_noise_std)
                action = agent.act(obs_norm.normalize(state), noise_std=noise_std)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            rb.store((state, action, reward, next_state, done))
            if not norm_frozen:
                obs_norm.update(next_state[None, :])
                if norm_freeze_steps is not None and total_env_steps >= norm_freeze_steps:
                    norm_frozen = True
                    print(f"[baseyaw] Observation normalizer frozen at step {total_env_steps}.")

            state = next_state
            ep_ret += reward
            total_env_steps += 1

            if total_env_steps >= warmup_steps and len(rb) >= batch_size:
                batch = rb.draw_samples(batch_size)
                s, a, r, s2, d = batch
                agent.train_step((obs_norm.normalize(s), a, r, obs_norm.normalize(s2), d))

            if done:
                break

        scores.append(ep_ret)

        if eval_every > 0 and ep % eval_every == 0:
            eval_ret, bal = _evaluate(agent, eval_env, obs_norm, max_steps, seed + 1000 + ep)
            eval_scores.append(eval_ret)
            eval_balance.append(bal)
            print(f"[baseyaw] Ep {ep} | TrainRet {ep_ret:.2f} | EvalRet {eval_ret:.2f} | Balance {bal} | Noise {noise_std:.4f}")
            if eval_ret > best_eval:
                best_eval = eval_ret
                best_actor_state = agent.actor.state_dict()
                best_critic_state = agent.critic.state_dict()
                best_obs_state = obs_norm.state_dict()
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_actor_state, os.path.join(save_dir, "td3_actor_best.pt"))
                torch.save(best_critic_state, os.path.join(save_dir, "td3_critic_best.pt"))
                np.savez(os.path.join(save_dir, "td3_obs_norm_best.npz"), **best_obs_state)

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "td3_obs_norm.npz"), **obs_norm.state_dict())
    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "td3_actor.pt"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "td3_critic.pt"))
    if best_actor_state is not None:
        torch.save(best_actor_state, os.path.join(save_dir, "td3_actor_best.pt"))
    if best_critic_state is not None:
        torch.save(best_critic_state, os.path.join(save_dir, "td3_critic_best.pt"))
    if best_obs_state is not None:
        np.savez(os.path.join(save_dir, "td3_obs_norm_best.npz"), **best_obs_state)

    return scores, eval_scores


if __name__ == "__main__":
    print("\n=== Training TD3 residual (base yaw only) ===")
    train_td3_baseyaw()
