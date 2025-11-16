import os
from typing import Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import mujoco


class PendulumBalanceEnv(gym.Env):
    """
    Custom env wrapping the MuJoCo model in Robot/miniArm_with_pendulum.xml.
    - Observation: [qpos(6), qvel(6), p_head_world(3), theta_from_global_z(1)] => 16-D
    - Action: Discrete(3**6) mapping to additive torque offsets on all 6 actuators,
      where per-joint offset bins are scaled to a fraction of that joint's forcerange.
    - Reward: time_alive * cos(theta) + z(pendulum head). Terminate if cos(theta) < -0.9.
    """

    metadata = {"render_modes": []}

    def __init__(self, model_path: Optional[str] = None,
                 ctrl_kp: float = 150.0, ctrl_kd: float = 5.2, control_dt: float = 0.01,
                 per_joint_scale: float = 0.2):
        super().__init__()
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.model_path = model_path or os.path.join(dir_path, "Robot", "miniArm_with_pendulum.xml")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.n_act = int(self.model.nu)  # 6
        self.kp = float(ctrl_kp)
        self.kd = float(ctrl_kd)
        self.control_dt = float(control_dt)
        self.sim_dt = float(self.model.opt.timestep)
        self.sim_steps_per_ctrl = max(1, int(round(self.control_dt / self.sim_dt)))

        # Target initial pose from keyframe 0
        self.init_qpos = self.data.qpos.copy()

        # Forcerange-based per-joint offsets {-alpha, 0, +alpha}
        fr = np.array(self.model.actuator_forcerange).reshape(self.n_act, 2)
        fr_mag = np.maximum(np.abs(fr[:, 0]), np.abs(fr[:, 1]))
        self.alpha = per_joint_scale * fr_mag.astype(np.float32)

        # Action space: 3 bins per joint -> 3**6 total combinations
        self.n_bins_per_joint = 3
        self.action_space = gym.spaces.Discrete(int(self.n_bins_per_joint ** self.n_act))

        # Geom id for pendulum mass (head)
        self._geom_mass_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "mass")
        # Body id for pendulum to compute local z-axis
        self._pend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")

        # Observation space: 16-D
        high = np.ones(16, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        # Disturbance schedule (matches viewer logic)
        self._next_push_time = 0.5
        self._push_force = 5e-4
        self._push_duration = 0.1
        self._push_gap = 4.0
        self._point = np.zeros(3, dtype=np.float64)
        self._force = np.zeros(3, dtype=np.float64)
        self._torque = np.zeros(3, dtype=np.float64)

    def _get_local_z(self) -> np.ndarray:
        quat = self.data.body(self._pend_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, quat)
        R = R_flat.reshape(3, 3)
        return R[:, 2]

    def _get_theta(self) -> float:
        # Angle from global +z axis (0 = upright)
        local_z = self._get_local_z()
        c = np.clip(local_z[2], -1.0, 1.0)
        return float(np.arccos(c))

    def _get_head_pos(self) -> np.ndarray:
        return np.array(self.data.geom_xpos[self._geom_mass_id], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        p_head = self._get_head_pos()
        theta = self._get_theta()
        obs = np.concatenate([
            np.array(self.data.qpos[:6], dtype=np.float32),
            np.array(self.data.qvel[:6], dtype=np.float32),
            p_head.astype(np.float32),
            np.array([theta], dtype=np.float32),
        ], axis=0)
        return obs

    def _apply_pd(self):
        for i in range(6):
            self.data.ctrl[i] = self.kp * (self.init_qpos[i] - self.data.qpos[i]) - self.kd * self.data.qvel[i]

    def _apply_disturbance(self):
        t = self.data.time
        if self._next_push_time < t < self._next_push_time + self._push_duration / 2:
            self._force[:] = 0
            self._force[1] = self._push_force
            mujoco.mj_applyFT(self.model, self.data, self._force, self._torque, self._point, self._pend_id, self.data.qfrc_applied)
        elif self._next_push_time + self._push_duration / 2 < t < self._next_push_time + self._push_duration:
            self._force[:] = 0
            self._force[1] = -self._push_force
            mujoco.mj_applyFT(self.model, self.data, self._force, self._torque, self._point, self._pend_id, self.data.qfrc_applied)
        elif t > self._next_push_time + self._push_duration + 0.5:
            self._next_push_time += self._push_gap
            self._push_force += 0.001

    def _decode_action(self, a: int) -> np.ndarray:
        # Convert integer a in [0, 3**n_act -1] to trits per joint in {-1, 0, +1}
        vals = np.zeros(self.n_act, dtype=np.int32)
        x = int(a)
        for i in range(self.n_act):
            vals[i] = x % 3
            x //= 3
        # map {0,1,2} -> {-1,0,+1}
        vals = vals - 1
        return vals.astype(np.int32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.init_qpos = self.data.qpos.copy()
        # randomize slight initial angles/vels
        eps = 0.01
        self.data.qpos[:6] += self.np_random.uniform(low=-eps, high=eps, size=6)
        self.data.qvel[:6] += self.np_random.uniform(low=-eps, high=eps, size=6)
        mujoco.mj_forward(self.model, self.data)
        # reset disturbances
        self._next_push_time = 0.5
        self._push_force = 5e-4
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        # PD baseline
        self._apply_pd()
        # RL additive torques on all actuators scaled by forcerange
        trits = self._decode_action(int(action))  # shape (6,), values in {-1,0,1}
        offsets = (self.alpha * trits.astype(np.float32))
        for i in range(self.n_act):
            self.data.ctrl[i] = self.data.ctrl[i] + float(offsets[i])

        # Simulate for control_dt
        for _ in range(self.sim_steps_per_ctrl):
            self._apply_disturbance()
            mujoco.mj_step(self.model, self.data)

        # Observation and termination
        obs = self._get_obs()
        local_z = self._get_local_z()
        cos_theta = float(np.clip(local_z[2], -1.0, 1.0))
        t = float(self.data.time)
        z_head = float(self._get_head_pos()[2])
        terminated = bool(cos_theta < -0.9)

        # Reward: time_alive * cos(theta) + z(head). No torque penalty.
        reward = float(t * cos_theta + z_head)

        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
