"""
Register PendulumEnv with Gymnasium for CleanRL compatibility.

TRAINING:
=========

# SAC (recommended - off-policy, sample efficient)
python cleanrl/cleanrl/sac_continuous_action.py \
    --env-id PendulumBalance-v0 \
    --total-timesteps 2000000 \
    --save-model

# PPO (on-policy, parallelizable)
python cleanrl/cleanrl/ppo_continuous_action.py \
    --env-id PendulumBalance-v0 \
    --total-timesteps 5000000 \
    --num-envs 256 \
    --save-model

# TD3 (deterministic off-policy)
python cleanrl/cleanrl/td3_continuous_action.py \
    --env-id PendulumBalance-v0 \
    --total-timesteps 2000000 \
    --save-model

INFERENCE:
==========
After training, copy the model to models/ and run:
    python Run_PendulumEnv.py

MONITORING:
===========
tensorboard --logdir runs/
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from pendulum_env import PendulumEnv

# Register environment
register(
    id="PendulumBalance-v0",
    entry_point="pendulum_env:PendulumEnv",
    max_episode_steps=3000,
)

if __name__ == "__main__":
    # Quick test
    env = gym.make("PendulumBalance-v0")
    obs, info = env.reset()
    print(f"Environment: PendulumBalance-v0")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    env.close()
    print("\nEnvironment registered successfully!")
