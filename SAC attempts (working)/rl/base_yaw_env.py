"""MuJoCo environment wrapper for SAC training on base-yaw residual torque."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import numpy as np

import YourControlCode


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class BaseYawEnv:
    def __init__(
        self,
        xml_path: str = "Robot/miniArm_with_pendulum.xml",
        frame_skip: int = 10,
        max_time: float = 12.0,
        seed: int | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_time = max_time
        self.max_steps = int(max_time / (self.model.opt.timestep * frame_skip))
        self.np_random = np.random.default_rng(seed)

        self.pend_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self._tmp_vel = np.zeros(6, dtype=np.float64)
        self.ctrl = YourControlCode.YourCtrl(self.model, self.data)

        self.push_gap = 4.0
        self.push_duration = 0.1
        self.balance_pause = 0.5
        self.default_push = 0.0005
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.point = np.zeros(3)

        self.obs_dim = self.ctrl.get_policy_observation(normalized=True).shape[0]
        self.action_dim = 1

        self.reset()

    def _apply_push(self) -> None:
        t = self.data.time
        if self.next_push_time < t < self.next_push_time + self.push_duration / 2:
            self.force[:] = 0.0
            self.force[1] = self.current_push
            mujoco.mj_applyFT(self.model, self.data, self.force, self.torque, self.point, self.pend_body_id, self.data.qfrc_applied)
        elif self.next_push_time + self.push_duration / 2 <= t < self.next_push_time + self.push_duration:
            self.force[:] = 0.0
            self.force[1] = -self.current_push
            mujoco.mj_applyFT(self.model, self.data, self.force, self.torque, self.point, self.pend_body_id, self.data.qfrc_applied)
        elif t >= self.next_push_time + self.push_duration + self.balance_pause:
            self.next_push_time += self.push_gap
            self.current_push += 0.001
            self.balance_count += 1

    def _fallen(self) -> bool:
        quat = self.data.body(self.pend_body_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, quat)
        local_z = R_flat.reshape(3, 3)[:, 2]
        return bool(local_z[2] < 0.0)

    def _reward(self, residual: float) -> float:
        obs = self.ctrl.get_policy_observation(normalized=False)
        q_p, qd_p, yaw, yawd, pend_y, pend_vy = obs
        angle_cost = (q_p / math.pi) ** 2
        vel_cost = 0.1 * (qd_p / 15.0) ** 2
        yaw_cost = 0.05 * (yaw / math.pi) ** 2
        act_cost = 0.01 * (residual / self.ctrl.residual_limit) ** 2
        pend_y_cost = 0.05 * pend_y ** 2
        pend_vy_cost = 0.02 * pend_vy ** 2
        reward = 1.0 - (angle_cost + vel_cost + yaw_cost + act_cost + pend_y_cost + pend_vy_cost)
        return float(max(reward, -10.0))

    def reset(self) -> np.ndarray:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.ctrl = YourControlCode.YourCtrl(self.model, self.data)
        noise = self.np_random.normal(scale=0.02, size=self.model.nq)
        self.data.qpos[:] = self.data.qpos + noise
        self.data.qvel[:] = self.np_random.normal(scale=0.05, size=self.model.nv)
        self.next_push_time = 0.5
        self.current_push = self.default_push
        self.balance_count = 0
        self.elapsed_steps = 0
        self.force[:] = 0.0
        self.torque[:] = 0.0
        self.point[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.ctrl.get_policy_observation(normalized=True)

    def step(self, action: np.ndarray | float) -> StepResult:
        scalar = float(np.clip(action, -1.0, 1.0))
        residual = scalar * self.ctrl.residual_limit
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            self.ctrl.set_residual_action(residual)
            self.ctrl.CtrlUpdate()
            self._apply_push()
            mujoco.mj_step(self.model, self.data)
            total_reward += self._reward(residual)
            if self._fallen():
                done = True
                break
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_steps:
            done = True
        obs = self.ctrl.get_policy_observation(normalized=True)
        info = {
            "time": self.data.time,
            "balance_count": float(self.balance_count),
            "residual": residual,
        }
        return StepResult(obs, total_reward, done, info)
