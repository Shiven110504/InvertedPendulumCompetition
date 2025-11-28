import dataclasses
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np

from rl.baseline_ctrl import BaselineController


@dataclasses.dataclass
class ResidualEnvConfig:
    model_path: str
    residual_joints: Sequence[str] = (
        "base_yaw",
        "shoulder_pitch",
        "shoulder_roll",
        "elbow",
        "wrist_pitch",
    )
    frame_skip: int = 10
    frame_skip_choices: Sequence[int] | None = None
    max_episode_steps: int = 1500
    upright_tolerance: float = 0.25
    action_scale: float = 0.35
    observation_noise: float = 0.0
    tip_velocity_weight: float = 0.1
    pend_vel_weight: float = 0.15
    action_penalty: float = 0.02
    yaw_penalty: float = 0.25
    pend_vel_noise: float = 0.0
    base_yaw_pos_noise: float = 0.0
    push_enabled: bool = False
    push_force_start: float = 0.0005
    push_force_increment: float = 0.0
    push_force_min: float = 0.0005
    push_force_max: float = 0.0005
    push_force_start_random: bool = False
    push_force_resample_each: bool = False
    push_gap: float = 4.0
    push_gap_std: float = 0.0
    push_gap_min: float = 1.5
    push_duration: float = 0.1
    push_pause: float = 0.5
    push_time_jitter_std: float = 0.01


class MiniArmResidualEnv:
    """
    Residual RL environment for MiniArm + inverted pendulum.
    The agent learns torque offsets for a subset of joints on top of the deterministic baseline controller.
    """

    def __init__(self, config: ResidualEnvConfig):
        self.cfg = config
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.baseline = BaselineController(self.model)

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.joint_qpos_adr = self._build_qpos_adr()
        self.joint_dof_adr = self._build_dof_adr()
        self.ctrl_indices = self._select_actuators(self.cfg.residual_joints)

        self.residual_scale = self._compute_residual_scale()
        self.action_dim = len(self.ctrl_indices)

        self.pend_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
        self.pend_qpos_adr = self.model.jnt_qposadr[self.pend_joint_id]
        self.pend_dof_adr = self.model.jnt_dofadr[self.pend_joint_id]
        self.pend_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self.base_joint_adr = self.joint_qpos_adr["base_yaw"]
        self.base_dof_adr = self.joint_dof_adr["base_yaw"]
        self.tip_jac = np.zeros((3, self.model.nv))
        self.tip_rot = np.zeros((3, self.model.nv))
        self._last_baseline_tau = np.zeros(self.model.nu, dtype=np.float32)

        self.step_count = 0
        self.balance_count = 0
        self.obs_dim = self._compute_obs_dim()
        self.rng = np.random.default_rng()
        self.next_push_time = 0.0
        self.push_force = self.cfg.push_force_start
        self.push_force_max = self.cfg.push_force_max
        self.push_force_max_current = self.cfg.push_force_max
        self.push_gap = self.cfg.push_gap

    def _build_qpos_adr(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                mapping[name] = int(self.model.jnt_qposadr[joint_id])
        return mapping

    def _build_dof_adr(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                mapping[name] = int(self.model.jnt_dofadr[joint_id])
        return mapping

    def _select_actuators(self, joint_names: Sequence[str]) -> np.ndarray:
        indices: List[int] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            found = False
            for actuator_id in range(self.model.nu):
                # Use actuator_trntype to disambiguate how the actuator is attached.
                trntype = self.model.actuator_trntype[actuator_id]
                trnid = self.model.actuator_trnid[actuator_id, 0]
                if trntype == mujoco.mjtTrn.mjTRN_JOINT and trnid == joint_id:
                    indices.append(actuator_id)
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing actuator for joint '{joint_name}'")
        return np.asarray(indices, dtype=np.int32)

    def _compute_obs_dim(self) -> int:
        """Observation size with base_yaw position omitted (rotational symmetry)."""
        base_term = 2 if "base_yaw" in self.cfg.residual_joints else 0  # qvel + tau
        other_terms = 3 * (len(self.cfg.residual_joints) - (1 if "base_yaw" in self.cfg.residual_joints else 0))
        return 3 + base_term + other_terms + 2  # pend sin/cos/vel + joint terms + tip vel (y,z)

    def _compute_residual_scale(self) -> np.ndarray:
        ranges = self.model.actuator_ctrlrange[self.ctrl_indices]
        span = (ranges[:, 1] - ranges[:, 0]) * 0.5
        return (self.cfg.action_scale * span).astype(np.float32)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_pose")
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self._apply_randomization()
        mujoco.mj_forward(self.model, self.data)
        self.baseline.reset(self.data)
        self._last_baseline_tau = self.baseline.compute_torques(self.data)
        self.step_count = 0
        self.balance_count = 0
        self.push_force_max_current = self.cfg.push_force_max
        if self.cfg.push_force_start_random or self.cfg.push_force_resample_each:
            self.push_force = float(self.rng.uniform(self.cfg.push_force_min, self.push_force_max_current))
        else:
            self.push_force = self.cfg.push_force_start
        if self.cfg.push_gap_std > 0.0:
            gap = float(self.rng.normal(self.cfg.push_gap, self.cfg.push_gap_std))
            self.push_gap = max(self.cfg.push_gap_min, gap)
        else:
            self.push_gap = self.cfg.push_gap
        self.next_push_time = 0.5 + self._push_time_noise()
        return self._get_obs()

    def _apply_randomization(self) -> None:
        # Targeted noise: pendulum velocity and base yaw position.
        if self.cfg.pend_vel_noise > 0.0:
            self.data.qvel[self.pend_dof_adr] += self.rng.normal(0.0, self.cfg.pend_vel_noise)
        if self.cfg.base_yaw_pos_noise > 0.0:
            self.data.qpos[self.base_joint_adr] += self.rng.uniform(
                -self.cfg.base_yaw_pos_noise, self.cfg.base_yaw_pos_noise
            )

    def _get_tip_velocity(self) -> np.ndarray:
        pend_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        mujoco.mj_jacBody(self.model, self.data, self.tip_jac, self.tip_rot, pend_body)
        return self.tip_jac @ self.data.qvel

    def _apply_push(self) -> None:
        if not self.cfg.push_enabled:
            return
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
            # Increase current max and resample for next push if configured.
            self.push_force_max_current = min(
                self.push_force_max_current + self.cfg.push_force_increment,
                self.push_force_max,
            )
            if self.cfg.push_force_resample_each or self.cfg.push_force_start_random:
                self.push_force = float(self.rng.uniform(self.cfg.push_force_min, self.push_force_max_current))
            else:
                self.push_force = min(
                    self.push_force + self.cfg.push_force_increment,
                    self.push_force_max_current,
                )

    def _push_time_noise(self) -> float:
        if self.cfg.push_time_jitter_std <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, self.cfg.push_time_jitter_std))

    def _get_obs(self) -> np.ndarray:
        q_p = float(self.data.qpos[self.pend_qpos_adr])
        qd_p = float(self.data.qvel[self.pend_dof_adr])
        obs: List[float] = [np.sin(q_p), np.cos(q_p), qd_p]

        for idx, joint in enumerate(self.cfg.residual_joints):
            q_idx = self.joint_qpos_adr[joint]
            v_idx = self.joint_dof_adr[joint]
            # Omit base_yaw position for rotational symmetry.
            if joint != "base_yaw":
                obs.append(float(self.data.qpos[q_idx]))
            obs.append(float(self.data.qvel[v_idx]))
            obs.append(float(self._last_baseline_tau[self.ctrl_indices[idx]] / (self.residual_scale[idx] + 1e-6)))

        tip_vel = self._get_tip_velocity()
        obs.append(float(tip_vel[1]))
        obs.append(float(tip_vel[2]))

        if self.cfg.observation_noise > 0.0:
            obs = (np.asarray(obs) + self.rng.normal(0.0, self.cfg.observation_noise, size=len(obs))).tolist()

        self.obs_dim = len(obs)
        return np.asarray(obs, dtype=np.float32)

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

        for _ in range(skip):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward, metrics = self._compute_reward(action)
        terminated = self._check_termination(metrics["pole_angle"], metrics["base_yaw"])
        truncated = self.step_count >= self.cfg.max_episode_steps
        info = dict(metrics)
        info["time_limit_reached"] = truncated
        info["balance_count"] = float(self.balance_count)
        info["skip"] = float(skip)
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        q_p = float(self.data.qpos[self.pend_qpos_adr])
        qd_p = float(self.data.qvel[self.pend_dof_adr])
        base_yaw_vel = float(self.data.qvel[self.base_dof_adr])
        tip_vel = self._get_tip_velocity()
        lateral_speed = float(np.linalg.norm(tip_vel[:2]))

        upright = np.exp(- (q_p / self.cfg.upright_tolerance) ** 2)
        vel_penalty = self.cfg.pend_vel_weight * abs(qd_p)
        yaw_penalty = self.cfg.yaw_penalty * abs(base_yaw_vel)
        action_pen = self.cfg.action_penalty * float(np.sum(np.square(action)))
        tip_penalty = self.cfg.tip_velocity_weight * lateral_speed
        reward = 2.5 * upright - vel_penalty - yaw_penalty - action_pen - tip_penalty + 0.1

        metrics = {
            "upright": upright,
            "pole_angle": q_p,
            "pole_velocity": qd_p,
            "base_yaw": float(self.data.qpos[self.base_joint_adr]),
            "base_yaw_rate": base_yaw_vel,
            "lateral_speed": lateral_speed,
            "action_penalty": action_pen,
            "reward": reward,
        }
        return reward, metrics

    def _check_termination(self, pole_angle: float, base_yaw: float) -> bool:
        if not np.isfinite(pole_angle):
            return True
        quat = self.data.body(self.pend_body_id).xquat
        R_flat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_flat, quat)  # type: ignore[attr-defined]
        local_z = R_flat.reshape(3, 3)[:, 2]
        return bool(local_z[2] < 0.0)

    def observation_size(self) -> int:
        return self.obs_dim

    def action_size(self) -> int:
        return self.action_dim
