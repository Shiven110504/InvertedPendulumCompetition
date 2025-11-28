import os
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import mujoco

from rl.baseline_ctrl import BaselineController


class PendulumBalanceEnvCont(gym.Env):
    """
    Continuous-action environment mirroring Run_PendulumEnv + YourControlCode.

    Observation: [qpos(6), qvel(6), head_pos(3), theta(1), qfrc_applied(nv)]
    Action: residual torques in [-1, 1]^nu scaled by actuator ctrlrange magnitudes.
    Reward: shaped to encourage uprightness and smooth control. Termination on large tilt or joint limits.
    Balance count increases when each disturbance cycle is rejected, matching the viewer's log.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 model_path: Optional[str] = None,
                 control_dt: Optional[float] = None,
                 reward_mode: str = "custom_reward",
                 action_scale: float = 1.0,
                 residual_scale: float = 0.05):
        super().__init__()
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.model_path = model_path or os.path.join(dir_path, "Robot", "miniArm_with_pendulum.xml")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = self.sim_dt if control_dt is None else float(control_dt)
        self.sim_steps_per_ctrl = max(1, int(round(self.control_dt / self.sim_dt)))

        ctrl_ranges = np.array(self.model.actuator_ctrlrange).reshape(self.model.nu, 2)
        self.ctrl_low = ctrl_ranges[:, 0]
        self.ctrl_high = ctrl_ranges[:, 1]
        ctrl_mag = np.maximum(np.abs(self.ctrl_low), np.abs(self.ctrl_high))
        self.ctrl_max = (action_scale * ctrl_mag).astype(np.float32)
        self.residual_scale = float(residual_scale)

        self.force_dim = int(self.model.nv)
        high_obs = np.ones(16 + self.force_dim, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self._geom_mass_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "mass")
        self._pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")

        self._next_push_time = 0.5
        self._push_force = 5e-4
        self._push_duration = 0.1
        self._push_gap = 4.0
        self._point = np.zeros(3, dtype=np.float64)
        self._force = np.zeros(3, dtype=np.float64)
        self._torque = np.zeros(3, dtype=np.float64)

        self.init_qpos = self.data.qpos.copy()
        jnt_range = np.array(self.model.jnt_range[:6], dtype=np.float32)
        jnt_limited = np.array(self.model.jnt_limited[:6], dtype=bool)
        self._joint_low = np.where(jnt_limited, jnt_range[:, 0], -np.inf)
        self._joint_high = np.where(jnt_limited, jnt_range[:, 1], np.inf)
        self._z_min = 0.0
        self._z_max = 1.5
        self.reward_mode = reward_mode
        self._prev_theta = 0.0
        self._prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self._balance_count = 0

        self.baseline = BaselineController(self.model)

    def _get_local_z(self) -> np.ndarray:
        quat = self.data.body(self._pend_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, quat)
        return R_flat.reshape(3, 3)[:, 2]

    def _get_theta(self) -> float:
        c = float(np.clip(self._get_local_z()[2], -1.0, 1.0))
        return float(np.arccos(c))

    def _get_head_pos(self) -> np.ndarray:
        return np.array(self.data.geom_xpos[self._geom_mass_id], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        p_head = self._get_head_pos()
        theta = self._get_theta()
        applied = np.array(self.data.qfrc_applied[:self.force_dim], dtype=np.float32)
        obs = np.concatenate([
            np.array(self.data.qpos[:6], dtype=np.float32),
            np.array(self.data.qvel[:6], dtype=np.float32),
            p_head.astype(np.float32),
            np.array([theta], dtype=np.float32),
            applied,
        ], axis=0)
        return obs

    def _apply_disturbance(self):
        t = self.data.time
        if self._next_push_time < t < self._next_push_time + self._push_duration / 2:
            self._force[:] = 0
            self._force[1] = self._push_force
            mujoco.mj_applyFT(self.model, self.data, self._force, self._torque, self._point,
                              self._pend_id, self.data.qfrc_applied)
        elif self._next_push_time + self._push_duration / 2 < t < self._next_push_time + self._push_duration:
            self._force[:] = 0
            self._force[1] = -self._push_force
            mujoco.mj_applyFT(self.model, self.data, self._force, self._torque, self._point,
                              self._pend_id, self.data.qfrc_applied)
        elif t > self._next_push_time + self._push_duration + 0.5:
            self._next_push_time += self._push_gap
            self._push_force += 0.001
            self._balance_count += 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.init_qpos = self.data.qpos.copy()
        eps = 0.01
        self.data.qpos[:6] += self.np_random.uniform(low=-eps, high=eps, size=6)
        self.data.qvel[:6] += self.np_random.uniform(low=-eps, high=eps, size=6)
        mujoco.mj_forward(self.model, self.data)
        self._next_push_time = 0.5
        self._push_force = 5e-4
        self._prev_theta = self._get_theta()
        self._prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self._balance_count = 0
        self.baseline.reset(self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        base_tau = self.baseline.compute_torques(self.data)

        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        residual = a * (self.residual_scale * self.ctrl_max)
        torques = base_tau + residual
        torques = np.clip(torques, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = torques

        for _ in range(self.sim_steps_per_ctrl):
            self._apply_disturbance()
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z_local = self._get_local_z()
        r_upright = float(np.clip(z_local[2], -1.0, 1.0))
        theta = float(np.arccos(np.clip(r_upright, -1.0, 1.0)))
        theta_dot = (theta - self._prev_theta) / max(self.control_dt, 1e-6)
        self._prev_theta = theta

        omega = np.array(self.data.cvel[self._pend_id][:3], dtype=np.float64)
        omega_mag = float(np.linalg.norm(omega))
        r_ang = 1.0 - np.tanh(omega_mag / 6.0)

        v_ee = np.array(self.data.cvel[self._ee_body_id][3:], dtype=np.float64)
        v_mag = float(np.linalg.norm(v_ee))
        r_ee = 1.0 - np.tanh(v_mag / 0.8)

        z_tip = float(self._get_head_pos()[2])
        r_z = np.clip((z_tip - self._z_min) / max(self._z_max - self._z_min, 1e-5), 0.0, 1.0)

        u = residual
        du = u - self._prev_action
        r_act_pen = float(np.dot(u, u))
        r_dact_pen = float(np.dot(du, du))

        reward = (
            1.0 * r_upright +
            0.3 * r_z +
            0.2 * r_ee +
            0.3 * r_ang -
            1e-2 * r_act_pen -
            5e-3 * r_dact_pen +
            0.01
        )
        reward = float(np.clip(reward, -2.0, 2.0))

        tilt = float(np.arccos(np.clip(r_upright, -1.0, 1.0)))
        joint_pos = np.array(self.data.qpos[:6], dtype=np.float32)
        hit_joint_limit = bool(np.any(joint_pos < self._joint_low) or np.any(joint_pos > self._joint_high))
        terminated = bool(tilt > np.deg2rad(75.0) or hit_joint_limit)
        if terminated:
            reward -= 1.0

        self._prev_action = u.copy()

        truncated = False
        info = {
            "reward": reward,
            "r_upright": r_upright,
            "r_z": r_z,
            "r_ee": r_ee,
            "r_ang": r_ang,
            "theta": theta,
            "theta_dot": theta_dot,
            "omega_mag": omega_mag,
            "ee_v_mag": v_mag,
            "z_tip": z_tip,
            "tilt": tilt,
            "hit_joint_limit": hit_joint_limit,
            "balance_count": self._balance_count,
            "residual_norm": float(np.linalg.norm(residual)),
        }
        return obs, reward, terminated, truncated, info
