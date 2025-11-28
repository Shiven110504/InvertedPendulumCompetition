"""
Gymnasium environment wrapper for inverted pendulum balancing task.
Fixed version with correct physics stepping order and frame skipping.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class PendulumEnv(gym.Env):
    """
    Gymnasium environment for inverted pendulum balancing on 6-DOF robotic arm.

    Key fixes from original:
    1. Disturbance applied BEFORE mj_step (matching Run_PendulumEnv.py)
    2. qfrc_applied cleared each step
    3. Frame skip for more stable learning
    4. Simplified reward function
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, max_episode_steps=3000, render_mode=None, frame_skip=20):
        """
        Initialize the environment.

        Args:
            model_path: Path to MuJoCo XML file
            max_episode_steps: Max steps per episode (at control frequency, not physics)
            render_mode: Rendering mode
            frame_skip: Number of physics steps per control step (20 = 50Hz control)
        """
        super().__init__()

        # Get model path
        if model_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(dir_path, "Robot/miniArm_with_pendulum.xml")

        self.model_path = model_path
        self.frame_skip = frame_skip  # 20 physics steps per control step = 50Hz control

        # Load MuJoCo model
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)

        # Get body IDs
        self.pend_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")

        # Episode settings
        self.max_episode_steps = max_episode_steps  # 3000 steps * 20ms = 60 seconds
        self.render_mode = render_mode
        self.step_count = 0

        # Disturbance settings (matching Run_PendulumEnv.py exactly)
        self.pushing_trial_gap = 4.0  # seconds between pushes
        self.pushing_duration = 0.1   # duration of push
        self.initial_push_force = 0.0005
        self.push_force_increment = 0.001

        # Disturbance state
        self.next_pushing_time = 0.5
        self.push_force = self.initial_push_force
        self.balance_count = 0

        # Action space: 6D continuous control normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Actuator limits for scaling
        self.action_limits = np.array([
            [-10, 10],    # base_yaw
            [-25, 25],    # shoulder_pitch
            [-15, 15],    # shoulder_roll
            [-20, 20],    # elbow
            [-10, 10],    # wrist_pitch
            [-5, 5]       # wrist_roll
        ], dtype=np.float32)

        # Observation space: 15D (simplified for better learning)
        # - Pendulum local_z (3D)
        # - Pendulum angular velocity (3D)
        # - Joint positions (6D)
        # - Joint velocities (6D) - removed, too noisy
        # - Time phase (1D) - cos of push cycle
        # - Time phase (1D) - sin of push cycle
        obs_dim = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize
        self._reset_simulation()

    def _get_observation(self):
        """Extract simplified observation vector."""
        obs = np.zeros(15, dtype=np.float32)
        idx = 0

        # Get pendulum orientation
        pend_quat = self.d.body(self.pend_id).xquat.copy()
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, pend_quat)
        R = R_flat.reshape(3, 3)
        local_z = R[:, 2]

        # Pendulum local_z (3D) - unit vector, already normalized
        obs[idx:idx+3] = local_z
        idx += 3

        # Pendulum angular velocity (3D) - normalized
        pend_vel = self.d.body(self.pend_id).cvel
        angular_vel = pend_vel[3:].copy()
        obs[idx:idx+3] = np.clip(angular_vel / 10.0, -1.0, 1.0)
        idx += 3

        # Joint positions (6D) - normalized
        joint_pos = self.d.qpos[0:6].copy()
        obs[idx:idx+6] = np.clip(joint_pos / np.pi, -1.0, 1.0)
        idx += 6

        # Time phase encoding (2D) - helps predict when pushes come
        phase = 2 * np.pi * (self.d.time % self.pushing_trial_gap) / self.pushing_trial_gap
        obs[idx] = np.cos(phase)
        obs[idx+1] = np.sin(phase)
        idx += 2

        # Push strength indicator (1D) - normalized
        obs[idx] = min(self.push_force / 0.01, 1.0)  # Normalize by max expected force
        idx += 1

        return obs

    def _apply_disturbance(self):
        """
        Apply periodic disturbance forces BEFORE physics step.
        Matches Run_PendulumEnv.py logic exactly.
        """
        # Clear previous forces
        self.d.qfrc_applied[:] = 0

        point = np.zeros((3,))
        force = np.zeros((3,))
        torque = np.zeros((3,))

        current_time = self.d.time

        # First half of push: positive Y force
        if self.next_pushing_time < current_time < self.next_pushing_time + self.pushing_duration / 2:
            force[1] = self.push_force
            mujoco.mj_applyFT(self.m, self.d, force, torque, point, self.pend_id, self.d.qfrc_applied)

        # Second half of push: negative Y force
        elif (self.next_pushing_time + self.pushing_duration / 2 < current_time <
              self.next_pushing_time + self.pushing_duration):
            force[1] = -self.push_force
            mujoco.mj_applyFT(self.m, self.d, force, torque, point, self.pend_id, self.d.qfrc_applied)

        # Push completed - schedule next push
        elif current_time > self.next_pushing_time + self.pushing_duration + 0.5:
            self.balance_count += 1
            self.next_pushing_time += self.pushing_trial_gap
            self.push_force += self.push_force_increment

    def _compute_reward(self, local_z):
        """
        Simplified reward function focusing on the primary objective.
        """
        # Primary: keep pendulum upright (local_z[2] = 1 when vertical)
        uprightness = local_z[2]  # Range: [-1, 1]

        # Alive bonus - strong incentive to stay alive
        alive_bonus = 1.0

        # Combine: reward is positive when upright, with alive bonus
        reward = alive_bonus + 2.0 * uprightness

        # Small penalty for large angular velocity (encourages stability)
        pend_vel = self.d.body(self.pend_id).cvel
        angular_vel_mag = np.linalg.norm(pend_vel[3:])
        reward -= 0.01 * min(angular_vel_mag, 5.0)

        return reward

    def _reset_simulation(self):
        """Reset MuJoCo simulation to initial state."""
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)

        # Reset disturbance state
        self.next_pushing_time = 0.5
        self.push_force = self.initial_push_force
        self.balance_count = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self._reset_simulation()

        obs = self._get_observation()
        info = {"balance_count": self.balance_count, "time": self.d.time}

        return obs, info

    def step(self, action):
        """
        Execute one control step (multiple physics steps with frame_skip).

        Order matches Run_PendulumEnv.py:
        1. Apply control
        2. Apply disturbance
        3. Step physics
        """
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        action_scaled = np.zeros(6, dtype=np.float32)
        for i in range(6):
            low, high = self.action_limits[i]
            action_scaled[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        # Apply control signal
        self.d.ctrl[0:6] = action_scaled

        # Run multiple physics steps (frame skip)
        for _ in range(self.frame_skip):
            # Apply disturbance BEFORE physics step
            self._apply_disturbance()

            # Step physics
            mujoco.mj_step(self.m, self.d)

        # Get pendulum state
        pend_quat = self.d.body(self.pend_id).xquat.copy()
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, pend_quat)
        R = R_flat.reshape(3, 3)
        local_z = R[:, 2]

        # Compute reward
        reward = self._compute_reward(local_z)

        # Check termination (pendulum fell)
        terminated = local_z[2] < 0

        # Large penalty for falling
        if terminated:
            reward = -10.0

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_observation()

        info = {
            "balance_count": self.balance_count,
            "push_force": self.push_force,
            "time": self.d.time,
            "uprightness": local_z[2]
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        pass
