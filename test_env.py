"""
Test script to verify the pendulum environment works correctly.
"""

import numpy as np
from pendulum_env import PendulumEnv


def test_environment():
    """Test basic environment functionality."""
    print("=" * 50)
    print("Testing PendulumEnv")
    print("=" * 50)

    env = PendulumEnv()

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Frame skip: {env.frame_skip} (control freq: {1000/env.frame_skip}Hz)")

    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Info: {info}")

    # Test with zero action (hold still)
    print("\n--- Testing with zero action ---")
    total_reward = 0
    for i in range(100):  # 100 steps at 50Hz = 2 seconds
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Balance count: {info.get('balance_count', 0)}")
            print(f"  Sim time: {info.get('time', 0):.2f}s")
            break

    print(f"Total reward (100 steps): {total_reward:.2f}")

    # Test with random actions
    print("\n--- Testing with random actions ---")
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    for i in range(500):  # 500 steps at 50Hz = 10 seconds
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Balance count: {info.get('balance_count', 0)}")
            print(f"  Sim time: {info.get('time', 0):.2f}s")
            print(f"  Uprightness: {info.get('uprightness', 0):.3f}")
            break

    print(f"Total reward ({steps} steps): {total_reward:.2f}")
    print(f"Avg reward per step: {total_reward/steps:.3f}")

    env.close()
    print("\n" + "=" * 50)
    print("Environment test passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_environment()
