"""Compare balance_count between viewer-like env and Run_PendulumEnv."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Tuple

import numpy as np
import torch

from rl.viewer_env import ViewerEnvConfig, ViewerResidualEnv
from rl.sac_agent import SACAgent, SACConfig


def run_viewer_env(agent, episodes: int, model_path: str) -> float:
    cfg = ViewerEnvConfig(
        model_path=model_path,
        frame_skip=1,
        frame_skip_choices=[3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 10, 11, 12],
        max_episode_steps=32_000,
        push_gap=4.0,
        push_gap_std=0.0,
        push_gap_min=0.0,
        push_force_start=0.0005,
        push_force_min=0.0005,
        push_force_max=0.05,
        push_force_start_random=False,
        push_force_increment=0.001,
        push_time_jitter_std=0.0,
        clock_jitter_steps=0,
    )
    env = ViewerResidualEnv(cfg)
    counts = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        counts.append(float(info.get("balance_count", 0.0)))
    return float(np.mean(counts))


def run_run_pendulum(agent_path: str, episodes: int) -> float:
    """Run Run_PendulumEnv.py as a subprocess, parse final balance count."""
    counts = []
    cmd = [sys.executable, "Run_PendulumEnv.py"]
    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["SAC_AGENT_PATH"] = agent_path
    for _ in range(episodes):
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.realpath(__file__)),
            env=env_base,
        )
        output = proc.stdout or ""
        match = re.findall(r"Final Balance Count:\s+(\d+)", output)
        if match:
            counts.append(float(match[-1]))
        else:
            counts.append(0.0)
    return float(np.mean(counts))


def main():
    parser = argparse.ArgumentParser(description="Compare balance_count in viewer_env vs Run_PendulumEnv.")
    parser.add_argument("--agent_path", type=str, default=os.getenv("SAC_AGENT_PATH", ""), help="Path to SAC checkpoint.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.agent_path or not os.path.isfile(args.agent_path):
        raise FileNotFoundError(f"SAC agent path missing/invalid: {args.agent_path}")

    root = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(root, "Robot", "miniArm_with_pendulum.xml")

    # Load agent for viewer_env comparison.
    dummy_env = ViewerResidualEnv(ViewerEnvConfig(model_path=model_path))
    obs_dim = dummy_env.observation_size()
    act_dim = dummy_env.action_size()
    device = torch.device(args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(SACConfig(obs_dim=obs_dim, act_dim=act_dim), device=device)
    agent.load(args.agent_path)

    
    print("Running Run_PendulumEnv.py episodes (GUI required)...")
    run_avg = run_run_pendulum(args.agent_path, args.episodes)
    print(f"Run_PendulumEnv avg final balance_count over {args.episodes} eps: {run_avg:.2f}")

    print("Running viewer_env episodes...")
    viewer_avg = run_viewer_env(agent, args.episodes, model_path)
    print(f"viewer_env avg balance_count over {args.episodes} eps: {viewer_avg:.2f}")


if __name__ == "__main__":
    main()
