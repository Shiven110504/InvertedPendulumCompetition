"""Train SAC purely in viewer_env (Run_PendulumEnv-like pushes)."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.viewer_env import ViewerEnvConfig, ViewerResidualEnv
from rl.sac_agent import ReplayBuffer, SACAgent, SACConfig

env_val = os.getenv("RESIDUAL_JOINTS", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC in viewer_env to mirror Run_PendulumEnv pushes.")
    parser.add_argument("--total_steps", type=int, default=6_000_000)
    parser.add_argument("--random_steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_size", type=int, default=2_000_000)
    parser.add_argument("--eval_interval", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default= "new_action_scale_"+env_val)
    parser.add_argument("--save_path", type=str, default= "new_action_scale_"+env_val + "\sac_viewer.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _parse_residual_joints(default_joints: tuple[str, ...]) -> tuple[str, ...]:
    env_val = os.getenv("RESIDUAL_JOINTS", "")
    if env_val.strip():
        joints = tuple(j.strip() for j in env_val.split(",") if j.strip())
        if joints:
            print(f"[train_viewer_sac] Using residual joints from RESIDUAL_JOINTS: {joints}")
            return joints
    return default_joints


def build_envs(seed: int, residual_joints: tuple[str, ...]) -> Dict[str, ViewerResidualEnv]:
    root = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(root, "Robot", "miniArm_with_pendulum.xml")
    cfg = ViewerEnvConfig(
        model_path=model_path,
        residual_joints=residual_joints,
        frame_skip=1,
        frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
        max_episode_steps=200_000,
        push_gap=4.0,
        push_gap_std=0.0,
        push_gap_min=4.0,
        push_duration=0.1,
        push_pause=0.5,
        push_force_start=0.0005,
        push_force_min=0.0005,
        push_force_max=1,
        push_force_start_random=False,
        push_force_increment=0.001,
        push_time_jitter_std=0.0,
        base_yaw_pos_noise=0.0,
    )
    train_env = ViewerResidualEnv(cfg)
    eval_env = ViewerResidualEnv(cfg)
    train_env.rng = np.random.default_rng(seed)
    eval_env.rng = np.random.default_rng(seed + 123)
    return {"train": train_env, "eval": eval_env}


def evaluate_policy(env: ViewerResidualEnv, agent: SACAgent, episodes: int) -> Dict[str, float]:
    rewards = []
    lengths = []
    balances = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0
        info = {}
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        lengths.append(ep_len)
        balances.append(float(info.get("balance_count", 0.0)))
    return {
        "reward": float(np.mean(rewards)),
        "length": float(np.mean(lengths)),
        "balance": float(np.mean(balances)),
    }


def save_reward_plot(history, out_path: str) -> None:
    if not history:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use dataclass default without instantiating.
    default_joints = ViewerEnvConfig.__dataclass_fields__["residual_joints"].default  # type: ignore[index]
    residual_joints = _parse_residual_joints(tuple(default_joints))

    # Derive save/checkpoint dirs from joints to avoid collisions.
    tag = "-".join(residual_joints)
    base_dir = os.path.join("artifacts", f"viewer_only_{tag}")
    checkpoint_dir = base_dir
    save_path = os.path.join(base_dir, "sac_viewer.pt")
    if args.checkpoint_dir != "artifacts/viewer_only":
        checkpoint_dir = args.checkpoint_dir
        save_path = args.save_path

    device = torch.device(args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = build_envs(args.seed, residual_joints)
    env = envs["train"]
    eval_env = envs["eval"]

    sac_cfg = SACConfig(obs_dim=env.observation_size(), act_dim=env.action_size())
    agent = SACAgent(sac_cfg, device=device)
    buffer = ReplayBuffer(env.observation_size(), env.action_size(), capacity=args.replay_size)

    os.makedirs(checkpoint_dir, exist_ok=True)
    episode_reward = 0.0
    episode_length = 0
    episode_rewards_history = []
    start_time = time.time()

    obs = env.reset()
    last_update_stats: Dict[str, float] = {}
    episode_idx = 0

    for step in range(1, args.total_steps + 1):
        if step <= args.random_steps:
            action = np.random.uniform(-1.0, 1.0, size=env.action_size())
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeout = truncated and not terminated
        buffer.add(obs, action, reward, next_obs, 0.0 if timeout else float(terminated))

        episode_reward += reward
        episode_length += 1
        obs = next_obs

        if done:
            episode_rewards_history.append(episode_reward)
            episode_idx += 1
            print(
                f"ep={episode_idx} step={step} reward={episode_reward:.2f} "
                f"len={episode_length} balance={info.get('balance_count', 0)}"
            )
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if step > args.random_steps and buffer.size >= args.batch_size:
            for _ in range(args.updates_per_step):
                last_update_stats = agent.update(buffer, args.batch_size)

        if step % args.eval_interval == 0:
            eval_stats = evaluate_policy(eval_env, agent, args.eval_episodes)
            elapsed = time.time() - start_time
            alpha_val = last_update_stats.get("alpha", float(agent.alpha.item()))
            print(
                f"[eval] step={step} reward={eval_stats['reward']:.2f} "
                f"length={eval_stats['length']:.1f} balance={eval_stats['balance']:.2f} "
                f"elapsed={elapsed/60:.1f}m alpha={alpha_val:.3f}"
            )
            ckpt_path = os.path.join(checkpoint_dir, f"sac_viewer_step{step}.pt")
            agent.save(ckpt_path)
            save_reward_plot(episode_rewards_history, os.path.join(checkpoint_dir, "reward_plot.png"))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    save_reward_plot(episode_rewards_history, os.path.join(checkpoint_dir, "reward_plot.png"))
    print(f"Saved final agent to {save_path}")


if __name__ == "__main__":
    main()
