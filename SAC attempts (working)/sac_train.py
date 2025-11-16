"""Entry point for training the base-yaw SAC policy."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rl.base_yaw_env import BaseYawEnv
from rl.sac import ReplayBuffer, SACAgent, SACConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC residual policy for base yaw")
    parser.add_argument("--xml", default="Robot/miniArm_with_pendulum.xml", help="MJCF model path")
    parser.add_argument("--output", default="models/sac_base_yaw.pt", help="Where to save the trained policy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--random-steps", type=int, default=5_000, help="Steps taken with random actions")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10_000, help="How often to run evaluation (steps)")
    return parser.parse_args()


def evaluate(agent: SACAgent, env: BaseYawEnv, episodes: int) -> float:
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, eval_mode=True)
            result = env.step(action)
            obs = result.observation
            total += result.reward
            done = result.done
        rewards.append(total)
    return float(np.mean(rewards))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BaseYawEnv(xml_path=args.xml, seed=args.seed)
    eval_env = BaseYawEnv(xml_path=args.xml, seed=args.seed + 1)

    config = SACConfig(device=args.device)
    agent = SACAgent(env.obs_dim, env.action_dim, config)
    replay = ReplayBuffer(env.obs_dim, env.action_dim, args.buffer_size, torch.device(args.device))

    obs = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode = 1
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        if step < args.random_steps:
            action = np.random.uniform(-1.0, 1.0, size=(env.action_dim,))
        else:
            action = agent.act(obs, eval_mode=False)

        result = env.step(action)
        replay.add(obs, action, result.reward, result.observation, result.done)
        obs = result.observation
        episode_reward += result.reward
        episode_steps += 1

        if result.done:
            duration = time.time() - start_time
            print(f"Episode {episode:05d}: reward={episode_reward:.1f} steps={episode_steps} wall_time={duration:.1f}s")
            obs = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode += 1

        if replay.size >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = replay.sample(args.batch_size)
                metrics = agent.update(batch)

        if step % args.eval_interval == 0:
            avg_reward = evaluate(agent, eval_env, args.eval_episodes)
            print(f"Evaluation at step {step}: avg_reward={avg_reward:.1f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    agent.save_policy(args.output, env.ctrl.policy_obs_scale, env.ctrl.residual_limit)
    print(f"Policy saved to {args.output}")


if __name__ == "__main__":
    main()
