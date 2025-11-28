"""Entry point for residual SAC training on the MiniArm inverted pendulum."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections import deque
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from rl.residual_env import MiniArmResidualEnv, ResidualEnvConfig
from rl.viewer_env import ViewerEnvConfig, ViewerResidualEnv
from rl.sac_agent import ReplayBuffer, SACAgent, SACConfig


TRAINING_PLAN = {
    "curriculum": [
        "Phase A: no pushes; small init noise on pendulum/base_yaw; survive to 1500 steps.",
        "Phase B: light randomized pushes with timing jitter; horizon 2000.",
        "Phase C: escalating pushes (gap shrinks, force grows on success); horizon 5000.",
    ],
    "steps": 4_000_000,
    "random_steps": 25_000,
    "batch_size": 256,
    "replay_size": 1_000_000,
    "eval_interval": 20_000,
    "eval_episodes": 5,
}

REWARD_DESCRIPTION = {
    "upright_bonus": "2.5 * exp(-(theta/upright_tol)^2)",
    "pend_vel_penalty": "0.15 * |theta_dot|",
    "yaw_penalty": "0.5 * |base_yaw| + 0.1 * |base_yaw_rate|",
    "tip_velocity_penalty": "0.1 * ||tip_vel_xy||",
    "action_penalty": "0.02 * ||a||^2",
    "survival_bonus": "+0.1 per step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual SAC agent for the pendulum task.")
    parser.add_argument("--total_steps", type=int, default=TRAINING_PLAN["steps"])
    parser.add_argument("--random_steps", type=int, default=TRAINING_PLAN["random_steps"])
    parser.add_argument("--batch_size", type=int, default=TRAINING_PLAN["batch_size"])
    parser.add_argument("--replay_size", type=int, default=TRAINING_PLAN["replay_size"])
    parser.add_argument("--eval_interval", type=int, default=TRAINING_PLAN["eval_interval"])
    parser.add_argument("--eval_episodes", type=int, default=TRAINING_PLAN["eval_episodes"])
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
    parser.add_argument("--balance_eval_interval", type=int, default=100_000)
    parser.add_argument("--balance_eval_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_path", type=str, default="artifacts/sac_residual.pt")
    return parser.parse_args()


def build_envs(args: argparse.Namespace) -> Dict[str, MiniArmResidualEnv]:
    root = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(root, "Robot", "miniArm_with_pendulum.xml")
    train_cfg = ResidualEnvConfig(model_path=model_path, frame_skip=args.frame_skip, max_episode_steps=args.max_episode_steps)
    eval_cfg = dataclasses.replace(train_cfg, observation_noise=0.0)
    viewer_cfg = ViewerEnvConfig(
        model_path=model_path,
        frame_skip=1,
        frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
        max_episode_steps=100_000,
        push_gap=4.0,
        push_gap_std=0.0,
        push_gap_min=0.0,
        push_force_start=0.0005,
        push_force_min=0.0005,
        push_force_max=1e6,
        push_force_start_random=False,
        push_force_increment=0.001,
        push_time_jitter_std=0.0,
    )
    env = MiniArmResidualEnv(train_cfg)
    eval_env = MiniArmResidualEnv(eval_cfg)
    viewer_env = ViewerResidualEnv(viewer_cfg)
    env.rng = np.random.default_rng(args.seed)
    eval_env.rng = np.random.default_rng(args.seed + 123)
    viewer_env.rng = np.random.default_rng(args.seed + 999)
    return {"train": env, "eval": eval_env, "viewer": viewer_env}


def evaluate_policy(env: MiniArmResidualEnv, agent: SACAgent, episodes: int) -> Dict[str, float]:
    rewards = []
    lengths = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        step = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += reward
            step += 1
            done = terminated or truncated
        rewards.append(ep_rew)
        lengths.append(step)
    return {"reward": float(np.mean(rewards)), "length": float(np.mean(lengths))}


def evaluate_balance(env: ViewerResidualEnv, agent: SACAgent, episodes: int) -> Dict[str, float]:
    """Average balance_count across viewer-like runs."""
    counts = []
    lengths = []
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        ep_rew = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_len += 1
            ep_rew += reward
            done = terminated or truncated
        counts.append(float(info.get("balance_count", 0.0)))
        lengths.append(ep_len)
        rewards.append(ep_rew)
    return {"avg_balance": float(np.mean(counts)), "avg_length": float(np.mean(lengths)), "avg_reward": float(np.mean(rewards))}


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

    device = torch.device(args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = build_envs(args)
    env = envs["train"]
    eval_env = envs["eval"]
    viewer_env = envs["viewer"]

    sac_cfg = SACConfig(obs_dim=env.observation_size(), act_dim=env.action_size())
    agent = SACAgent(sac_cfg, device=device)
    buffer = ReplayBuffer(env.observation_size(), env.action_size(), capacity=args.replay_size)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Training plan:")
    print(json.dumps(TRAINING_PLAN, indent=2))
    print("Reward policy:")
    print(json.dumps(REWARD_DESCRIPTION, indent=2))

    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()
    last_update_stats: Dict[str, float] = {}
    episode_rewards_history = []
    balance_history = []
    phase_episode_rewards = []
    success_window = deque(maxlen=20)

    phases = [
        {
            "name": "phase_A",
            "cfg": dict(
                max_episode_steps=2000,
                frame_skip=args.frame_skip,
                frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
                push_enabled=False,
                pend_vel_noise=0.1,
                base_yaw_pos_noise=0.05,
            ),
            "noise_targets": {"pend_vel": 0.2, "base_yaw": 0.25},
            "noise_increment": {"pend_vel": 0.01, "base_yaw": 0.02},
            "transition_successes": 12,
        },
        {
            "name": "phase_B",
            "cfg": dict(
                max_episode_steps=5000,
                frame_skip=args.frame_skip,
                frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
                push_enabled=True,
                pend_vel_noise=0.0,
                base_yaw_pos_noise=0.0,
                push_force_min=0.0005,
                push_force_max=0.0105,  # goal upper bound
                push_force_max_init=0.0015,
                push_force_start=0.0005,
                push_force_start_random=True,
                push_force_resample_each=True,
                push_force_increment=0.001,
                push_gap=4.0,
                push_gap_std=0.2,
                push_gap_min=3.0,
                push_duration=0.1,
                push_pause=0.5,
            ),
            "noise_targets": {"pend_vel": 0.0, "base_yaw": 0.0},
            "noise_increment": {"pend_vel": 0.0, "base_yaw": 0.0},
            "force_target": 0.0105,
            "force_increment": 0.001,
            "transition_successes": 12,
        },
        {
            "name": "phase_C",
            "cfg": dict(
                max_episode_steps=50000,
                frame_skip=1,
                frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
                push_enabled=True,
                pend_vel_noise=0.0,
                base_yaw_pos_noise=0.0,
                push_force_min=0.0005,
                push_force_max=0.1,
                push_force_max_init=0.0005,
                push_force_start=0.0005,
                push_force_start_random=False,
                push_force_increment=0.001,
                push_force_resample_each=False,
                push_gap=4.0,
                push_gap_std=0.0,
                push_gap_min=0.0,
                push_duration=0.1,
                push_pause=0.5,
                push_time_jitter_std=0.0,
            ),
            "noise_targets": {"pend_vel": 1.0, "base_yaw": 0.3},
            "noise_increment": {"pend_vel": 0.0, "base_yaw": 0.0},
            "transition_successes": None,
        },
    ]

    noise_state = {"pend_vel": phases[0]["cfg"]["pend_vel_noise"], "base_yaw": phases[0]["cfg"]["base_yaw_pos_noise"]}
    force_max_state = phases[1]["cfg"]["push_force_max"]

    def apply_phase(phase_idx: int):
        phase = phases[phase_idx]
        if phase["name"] == "phase_C":
            # For the test-mirroring phase, drop back to no start noise.
            noise_state["pend_vel"] = phase["cfg"].get("pend_vel_noise", 0.0)
            noise_state["base_yaw"] = phase["cfg"].get("base_yaw_pos_noise", 0.0)
        else:
            noise_state["pend_vel"] = max(noise_state.get("pend_vel", 0.0), phase["cfg"].get("pend_vel_noise", 0.0))
            noise_state["base_yaw"] = max(noise_state.get("base_yaw", 0.0), phase["cfg"].get("base_yaw_pos_noise", 0.0))
        for k, v in phase["cfg"].items():
            setattr(env.cfg, k, v)
            setattr(eval_env.cfg, k, v)
        env.cfg.max_episode_steps = phase["cfg"]["max_episode_steps"]
        eval_env.cfg.max_episode_steps = phase["cfg"]["max_episode_steps"]
        env.cfg.pend_vel_noise = noise_state["pend_vel"]
        env.cfg.base_yaw_pos_noise = noise_state["base_yaw"]
        eval_env.cfg.pend_vel_noise = noise_state["pend_vel"]
        eval_env.cfg.base_yaw_pos_noise = noise_state["base_yaw"]
        if phase["name"] == "phase_B":
            env.cfg.push_force_max = force_max_state
            eval_env.cfg.push_force_max = force_max_state
        elif phase["name"] == "phase_C":
            env.cfg.push_force_max = phase["cfg"]["push_force_max"]
            eval_env.cfg.push_force_max = phase["cfg"]["push_force_max"]
        obs_local = env.reset()
        eval_env.reset()
        os.makedirs(os.path.join(args.checkpoint_dir, phase["name"]), exist_ok=True)
        return phase, obs_local

    phase_idx = 0
    phase, obs = apply_phase(phase_idx)
    phase_dir = os.path.join(args.checkpoint_dir, phase["name"])
    length_window = deque(maxlen=20)
    success_streak = 0

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
            print(f"step={step} episode_reward={episode_reward:.2f} length={episode_length} last_up={info['upright']:.3f}")
            episode_rewards_history.append(episode_reward)
            phase_episode_rewards.append(episode_reward)
            horizon_hit = truncated and not terminated
            success_window.append(horizon_hit)

            # Ramp noise/force on successes.
            if phase["name"] in ("phase_A", "phase_B") and horizon_hit:
                targets = phase["noise_targets"]
                incs = phase["noise_increment"]
                noise_state["pend_vel"] = min(targets["pend_vel"], noise_state["pend_vel"] + incs["pend_vel"])
                noise_state["base_yaw"] = min(targets["base_yaw"], noise_state["base_yaw"] + incs["base_yaw"])
                env.cfg.pend_vel_noise = noise_state["pend_vel"]
                env.cfg.base_yaw_pos_noise = noise_state["base_yaw"]
                eval_env.cfg.pend_vel_noise = noise_state["pend_vel"]
                eval_env.cfg.base_yaw_pos_noise = noise_state["base_yaw"]

            if phase["name"] == "phase_B" and horizon_hit:
                force_max_state = min(phase["force_target"], force_max_state + phase["force_increment"])
                env.cfg.push_force_max = force_max_state
                eval_env.cfg.push_force_max = force_max_state

            # Phase transitions based on success rate in window.
            if phase["name"] == "phase_A":
                if len(success_window) == success_window.maxlen and sum(success_window) >= phase["transition_successes"]:
                    save_reward_plot(phase_episode_rewards, os.path.join(phase_dir, "reward_plot.png"))
                    phase_episode_rewards = []
                    phase_idx += 1
                    phase, obs = apply_phase(phase_idx)
                    phase_dir = os.path.join(args.checkpoint_dir, phase["name"])
                    success_window.clear()
                    length_window = deque(maxlen=20)
                    success_streak = 0
                    print(f"[phase] Transitioned to {phase['name']}")
                    episode_reward = 0.0
                    episode_length = 0
                    continue
            elif phase["name"] == "phase_B":
                if (
                    force_max_state >= phase["force_target"]
                    and len(success_window) == success_window.maxlen
                    and sum(success_window) >= phase["transition_successes"]
                ):
                    save_reward_plot(phase_episode_rewards, os.path.join(phase_dir, "reward_plot.png"))
                    phase_episode_rewards = []
                    phase_idx += 1
                    phase, obs = apply_phase(phase_idx)
                    phase_dir = os.path.join(args.checkpoint_dir, phase["name"])
                    success_window.clear()
                    length_window = deque(maxlen=20)
                    success_streak = 0
                    print(f"[phase] Transitioned to {phase['name']}")
                    episode_reward = 0.0
                    episode_length = 0
                    continue

            # Phase C: optional push tweak on streak of horizon hits.
            if phase["name"] == "phase_C":
                if horizon_hit:
                    success_streak += 1
                else:
                    success_streak = 0
                if success_streak >= 2:
                    env.cfg.push_gap = max(env.cfg.push_gap_min, env.cfg.push_gap * 0.95)
                    eval_env.cfg.push_gap = env.cfg.push_gap
                    success_streak = 0

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
                f"[eval] step={step} reward={eval_stats['reward']:.2f} length={eval_stats['length']:.1f} "
                f"elapsed={elapsed/60:.1f}m alpha={alpha_val:.3f}"
            )

        if step % args.balance_eval_interval == 0:
            bal_stats = evaluate_balance(viewer_env, agent, args.balance_eval_episodes)
            balance_history.append((step, bal_stats))
            ckpt_path = os.path.join(phase_dir, f"sac_residual_step{step}.pt")
            agent.save(ckpt_path)
            print(
                f"[balance] step={step} avg_balance={bal_stats['avg_balance']:.2f} "
                f"avg_len={bal_stats['avg_length']:.1f} avg_rew={bal_stats['avg_reward']:.2f} saved={ckpt_path}"
            )

            save_reward_plot(phase_episode_rewards, os.path.join(phase_dir, "reward_plot.png"))
            with open(os.path.join(args.checkpoint_dir, "balance_history.json"), "w", encoding="utf-8") as f:
                json.dump(balance_history, f, indent=2)

    # Final save + plot for last phase.
    save_reward_plot(phase_episode_rewards, os.path.join(phase_dir, "reward_plot.png"))
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    agent.save(args.save_path)
    print(f"Saved agent to {args.save_path}")


if __name__ == "__main__":
    main()
