"""
RPO Controller for inverted pendulum balancing.
Loads trained RPO policies from CleanRL for inference.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import mujoco


# RPO Actor (CleanRL architecture - 256-256 hidden with tanh activations)
class RPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_mean(x)


class RLController:
    """
    RPO controller for CleanRL-trained RPO models.
    Loads RPO policies for real-time inference at 50Hz control frequency.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, model_path=None):
        self.m = m
        self.d = d

        # Body IDs
        self.pend_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")

        # Disturbance timing (for observation)
        self.pushing_trial_gap = 4.0
        self.initial_push_force = 0.0005
        self.push_force_increment = 0.001

        # Find model
        if model_path is None:
            model_path = self._find_model()

        # Dimensions
        self.obs_dim = 15
        self.action_dim = 6

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self._load_model(model_path)

        # Action limits
        self.action_limits = np.array([
            [-10, 10], [-25, 25], [-15, 15],
            [-20, 20], [-10, 10], [-5, 5]
        ], dtype=np.float32)

        # Control frequency (50Hz)
        self.last_action_time = 0.0
        self.control_dt = 0.02
        self.current_action = np.zeros(6, dtype=np.float32)

    def _find_model(self):
        """Find most recent RPO model in runs directory."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        runs_dir = os.path.join(dir_path, "runs")

        if not os.path.exists(runs_dir):
            raise FileNotFoundError(
                f"No runs directory found at {runs_dir}. "
                "Train with: python cleanrl/cleanrl/rpo_continuous_action.py --save-model"
            )

        # Find most recent RPO run
        rpo_runs = [d for d in os.listdir(runs_dir) if 'rpo_continuous_action' in d]
        if not rpo_runs:
            raise FileNotFoundError(
                f"No RPO runs found in {runs_dir}. "
                "Train with: python cleanrl/cleanrl/rpo_continuous_action.py --save-model"
            )

        # Sort by timestamp (last part of directory name)
        rpo_runs.sort(key=lambda x: x.split('__')[-1], reverse=True)
        latest_run = rpo_runs[0]

        # Look for .cleanrl_model file
        run_path = os.path.join(runs_dir, latest_run)
        model_files = [f for f in os.listdir(run_path) if f.endswith('.cleanrl_model')]

        if not model_files:
            raise FileNotFoundError(
                f"No model file found in {run_path}. "
                "Train with: python cleanrl/cleanrl/rpo_continuous_action.py --save-model"
            )

        return os.path.join(run_path, model_files[0])

    def _load_model(self, model_path):
        """Load RPO model from checkpoint."""
        print(f"Loading RPO model: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # RPO models are saved as state dicts with 'actor_mean.*' keys
        if not isinstance(checkpoint, dict) or 'actor_mean.0.weight' not in checkpoint:
            raise ValueError(
                f"Invalid RPO checkpoint format. Expected dict with 'actor_mean.0.weight', "
                f"got: {type(checkpoint)} with keys {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
            )

        # Verify architecture (should be 256 hidden units)
        hidden_size = checkpoint['actor_mean.0.weight'].shape[0]
        if hidden_size != 256:
            raise ValueError(
                f"Expected 256 hidden units for RPO, got {hidden_size}. "
                "This might be a PPO model. Please retrain with RPO."
            )

        # Load weights into RPO actor
        actor = RPOActor(self.obs_dim, self.action_dim).to(self.device)
        actor.fc1.weight.data = checkpoint['actor_mean.0.weight']
        actor.fc1.bias.data = checkpoint['actor_mean.0.bias']
        actor.fc2.weight.data = checkpoint['actor_mean.2.weight']
        actor.fc2.bias.data = checkpoint['actor_mean.2.bias']
        actor.fc_mean.weight.data = checkpoint['actor_mean.4.weight']
        actor.fc_mean.bias.data = checkpoint['actor_mean.4.bias']
        if 'actor_logstd' in checkpoint:
            actor.fc_logstd.data = checkpoint['actor_logstd']

        actor.eval()
        print(f"  ✓ RPO actor loaded (256-256 architecture)")
        return actor

    def _get_observation(self):
        """Extract 15D observation matching pendulum_env.py."""
        obs = np.zeros(15, dtype=np.float32)
        idx = 0

        # Pendulum orientation (local_z)
        pend_quat = self.d.body(self.pend_id).xquat.copy()
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, pend_quat)
        R = R_flat.reshape(3, 3)
        local_z = R[:, 2]
        obs[idx:idx+3] = local_z
        idx += 3

        # Angular velocity
        pend_vel = self.d.body(self.pend_id).cvel
        obs[idx:idx+3] = np.clip(pend_vel[3:].copy() / 10.0, -1.0, 1.0)
        idx += 3

        # Joint positions
        obs[idx:idx+6] = np.clip(self.d.qpos[0:6].copy() / np.pi, -1.0, 1.0)
        idx += 6

        # Time phase
        phase = 2 * np.pi * (self.d.time % self.pushing_trial_gap) / self.pushing_trial_gap
        obs[idx] = np.cos(phase)
        obs[idx+1] = np.sin(phase)
        idx += 2

        # Push force estimate
        num_pushes = max(0, int((self.d.time - 0.5) / self.pushing_trial_gap))
        force = self.initial_push_force + num_pushes * self.push_force_increment
        obs[idx] = min(force / 0.01, 1.0)

        return obs

    def CtrlUpdate(self):
        """Main control function called every physics step (1kHz)."""
        # Only compute new action at 50Hz
        if self.d.time - self.last_action_time >= self.control_dt:
            self.last_action_time = self.d.time

            obs = self._get_observation()
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action = self.actor(obs_t).cpu().numpy()[0]

            self.current_action = np.clip(action, -1.0, 1.0)

        # Scale to actuator limits
        action_scaled = np.zeros(6, dtype=np.float32)
        for i in range(6):
            low, high = self.action_limits[i]
            action_scaled[i] = low + (self.current_action[i] + 1.0) * 0.5 * (high - low)

        self.d.ctrl[0:6] = action_scaled
        return True
