"""Environment that mirrors the viewer pushes/clock jitter for balance eval."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np

from rl.baseline_ctrl import BaselineController


@dataclasses.dataclass
class ViewerEnvConfig:
    model_path: str
    residual_joints: Sequence[str] = ("base_yaw", "shoulder_pitch", "elbow")
    frame_skip: int = 1
    frame_skip_choices: Sequence[int] | None = None
    max_episode_steps: int = 50_000
    action_scale: float = 0.35
    push_time_jitter_std: float = 0.0
    init_push_time: float = 0.5
    push_gap: float = 4.0
    push_gap_std: float = 0.0
    push_gap_min: float = 0.0
    push_duration: float = 0.1
    push_pause: float = 0.5
    push_force_start: float = 0.0005
    push_force_min: float = 0.0005
    push_force_max: float = 0.1 
    push_force_start_random: bool = False
    push_force_increment: float = 0.001
    randomize_state_std: float = 0.0


class ViewerResidualEnv:
    """Evaluation env that mimics Run_PendulumEnv pushes + timing jitter."""

    def __init__(self, cfg: ViewerEnvConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.baseline = BaselineController(self.model)

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.ctrl_indices = self._select_actuators(self.cfg.residual_joints)
        self.residual_scale = self._compute_residual_scale()
        self.action_dim = len(self.ctrl_indices)

        self.pend_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
        self.pend_qpos_adr = self.model.jnt_qposadr[self.pend_joint_id]
        self.pend_dof_adr = self.model.jnt_dofadr[self.pend_joint_id]
        self.tip_jac = np.zeros((3, self.model.nv))
        self.tip_rot = np.zeros((3, self.model.nv))
        self._last_baseline_tau = np.zeros(self.model.nu, dtype=np.float32)

        self.rng = np.random.default_rng()
        self.balance_count = 0
        self.step_count = 0
        self.next_push_time = self.cfg.init_push_time
        self.push_force = self.cfg.push_force_start
        self.push_gap = self.cfg.push_gap

    # ---------------------- setup helpers ---------------------- #
    def _select_actuators(self, joint_names: Sequence[str]) -> np.ndarray:
        indices: List[int] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            found = False
            for actuator_id in range(self.model.nu):
                trntype = self.model.actuator_trntype[actuator_id]
                trnid = self.model.actuator_trnid[actuator_id, 0]
                if trntype == mujoco.mjtTrn.mjTRN_JOINT and trnid == joint_id:
                    indices.append(actuator_id)
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing actuator for joint '{joint_name}'")
        return np.asarray(indices, dtype=np.int32)

    def _compute_residual_scale(self) -> np.ndarray:
        ranges = self.model.actuator_ctrlrange[self.ctrl_indices]
        span = (ranges[:, 1] - ranges[:, 0]) * 0.5
        return (self.cfg.action_scale * span).astype(np.float32)

    # ---------------------- observation ---------------------- #
    def _get_tip_velocity(self) -> np.ndarray:
        pend_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        mujoco.mj_jacBody(self.model, self.data, self.tip_jac, self.tip_rot, pend_body)
        return self.tip_jac @ self.data.qvel

    def _get_obs(self) -> np.ndarray:
        q_p = float(self.data.qpos[self.pend_qpos_adr])
        qd_p = float(self.data.qvel[self.pend_dof_adr])
        obs: List[float] = [np.sin(q_p), np.cos(q_p), qd_p]

        for idx, joint in enumerate(self.cfg.residual_joints):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            q_idx = self.model.jnt_qposadr[joint_id]
            v_idx = self.model.jnt_dofadr[joint_id]
            obs.append(float(self.data.qpos[q_idx]))
            obs.append(float(self.data.qvel[v_idx]))
            obs.append(float(self._last_baseline_tau[self.ctrl_indices[idx]] / (self.residual_scale[idx] + 1e-6)))

        tip_vel = self._get_tip_velocity()
        obs.append(float(tip_vel[1]))
        obs.append(float(tip_vel[2]))
        return np.asarray(obs, dtype=np.float32)

    # ---------------------- push + termination ---------------------- #
    def _apply_push(self) -> None:
        t = float(self.data.time)
        push_half = self.cfg.push_duration * 0.5
        pend_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")

        if self.next_push_time < t < self.next_push_time + push_half:
            force = np.zeros(3)
            force[1] = self.push_force
            mujoco.mj_applyFT(self.model, self.data, force, np.zeros(3), np.zeros(3), pend_body, self.data.qfrc_applied)
        elif self.next_push_time + push_half < t < self.next_push_time + self.cfg.push_duration:
            force = np.zeros(3)
            force[1] = -self.push_force
            mujoco.mj_applyFT(self.model, self.data, force, np.zeros(3), np.zeros(3), pend_body, self.data.qfrc_applied)
        elif t >= self.next_push_time + self.cfg.push_duration + self.cfg.push_pause:
            self.balance_count += 1
            self.next_push_time += self.push_gap + self._push_time_noise()
            self.push_force = min(self.push_force + self.cfg.push_force_increment, self.cfg.push_force_max)

    def _fallen(self) -> bool:
        quat = self.data.body(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, quat)  # type: ignore[attr-defined]
        local_z = R_flat.reshape(3, 3)[:, 2]
        return bool(local_z[2] < 0.0)

    def _push_time_noise(self) -> float:
        if self.cfg.push_time_jitter_std <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, self.cfg.push_time_jitter_std))

    # ---------------------- public API ---------------------- #
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        if self.cfg.randomize_state_std > 0.0:
            self.data.qpos[:] += self.rng.normal(0.0, self.cfg.randomize_state_std, size=self.model.nq)
            self.data.qvel[:] += self.rng.normal(0.0, self.cfg.randomize_state_std, size=self.model.nv)

        self.baseline.reset(self.data)
        self._last_baseline_tau = self.baseline.compute_torques(self.data)
        self.balance_count = 0
        self.step_count = 0
        self.next_push_time = self.cfg.init_push_time + self._push_time_noise()
        if self.cfg.push_force_start_random:
            self.push_force = float(self.rng.uniform(self.cfg.push_force_min, self.cfg.push_force_max))
        else:
            self.push_force = self.cfg.push_force_start
        if self.cfg.push_gap_std > 0.0:
            gap = float(self.rng.normal(self.cfg.push_gap, self.cfg.push_gap_std))
            self.push_gap = max(self.cfg.push_gap_min, gap)
        else:
            self.push_gap = self.cfg.push_gap
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        baseline_tau = self.baseline.compute_torques(self.data)
        self._last_baseline_tau = baseline_tau
        ctrl = baseline_tau.copy()
        ctrl[self.ctrl_indices] += action * self.residual_scale
        ctrl = np.clip(ctrl, self.ctrl_low, self.ctrl_high)
        self._apply_push()

        if self.cfg.frame_skip_choices:
            skip = int(self.rng.choice(self.cfg.frame_skip_choices))
        else:
            skip = self.cfg.frame_skip

        total_reward = 0.0
        terminated = False
        for _ in range(skip):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            if self._fallen():
                terminated = True
                break
        self.step_count += 1
        truncated = self.step_count >= self.cfg.max_episode_steps

        obs = self._get_obs()
        # Use the same reward shaping as training for consistency.
        reward, metrics = self._compute_reward(action)
        metrics["balance_count"] = float(self.balance_count)
        metrics["skip"] = float(skip)
        return obs, reward, terminated, truncated, metrics

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        q_p = float(self.data.qpos[self.pend_qpos_adr])
        qd_p = float(self.data.qvel[self.pend_dof_adr])
        base_yaw = float(self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")]])
        base_yaw_vel = float(self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")]])
        tip_vel = self._get_tip_velocity()
        lateral_speed = float(np.linalg.norm(tip_vel[:2]))

        upright = np.exp(-(q_p / 0.25) ** 2)
        vel_penalty = 0.15 * abs(qd_p)
        yaw_penalty = 0.5 * abs(base_yaw) + 0.1 * abs(base_yaw_vel)
        action_pen = 0.02 * float(np.sum(np.square(action)))
        tip_penalty = 0.1 * lateral_speed
        reward = 2.5 * upright - vel_penalty - yaw_penalty - action_pen - tip_penalty + 0.1

        metrics = {
            "upright": upright,
            "pole_angle": q_p,
            "pole_velocity": qd_p,
            "base_yaw": base_yaw,
            "base_yaw_rate": base_yaw_vel,
            "lateral_speed": lateral_speed,
            "action_penalty": action_pen,
            "reward": reward,
        }
        return reward, metrics

    def observation_size(self) -> int:
        return 3 + 3 * self.action_dim + 2

    def action_size(self) -> int:
        return self.action_dim
