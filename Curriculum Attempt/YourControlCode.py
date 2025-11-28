import os
import mujoco
import numpy as np

try:
  import torch
except Exception:
  torch = None

from rl.residual_env import ResidualEnvConfig
from rl.sac_agent import SACAgent, SACConfig
from rl.baseline_ctrl import BaselineController


class YourCtrl:
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d

    self.baseline = BaselineController(self.m)
    self.baseline.reset(self.d)

    self.n_act = int(self.m.nu)
    ctrl_ranges = np.array(self.m.actuator_ctrlrange, dtype=np.float32)
    self.ctrl_low = ctrl_ranges[:, 0]
    self.ctrl_high = ctrl_ranges[:, 1]

    # Residual SAC settings (aligned with training defaults).
    self.cfg = ResidualEnvConfig(model_path="")
    self.residual_joints = self.cfg.residual_joints
    self.ctrl_indices = self._select_actuators(self.residual_joints)
    self.residual_scale = self._compute_residual_scale()
    self.pend_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
    self.pend_qpos_adr = self.m.jnt_qposadr[self.pend_joint_id]
    self.pend_dof_adr = self.m.jnt_dofadr[self.pend_joint_id]
    self.tip_jac = np.zeros((3, self.m.nv))
    self.tip_rot = np.zeros((3, self.m.nv))
    self._last_baseline_tau = np.zeros(self.m.nu, dtype=np.float32)

    self.agent = None
    self.use_agent = False
    self._maybe_load_agent()

  def _select_actuators(self, joint_names):
    indices = []
    for joint_name in joint_names:
      joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      found = False
      for actuator_id in range(self.m.nu):
        trntype = self.m.actuator_trntype[actuator_id]
        trnid = self.m.actuator_trnid[actuator_id, 0]
        if trntype == mujoco.mjtTrn.mjTRN_JOINT and trnid == joint_id:
          indices.append(actuator_id)
          found = True
          break
      if not found:
        raise ValueError(f"Missing actuator for joint '{joint_name}'")
    return np.asarray(indices, dtype=np.int32)

  def _compute_residual_scale(self):
    ranges = self.m.actuator_ctrlrange[self.ctrl_indices]
    span = (ranges[:, 1] - ranges[:, 0]) * 0.5
    return (self.cfg.action_scale * span).astype(np.float32)

  def _maybe_load_agent(self):
    if torch is None:
      print("[YourControlCode] PyTorch not available, skipping SAC agent load.")
      return
    sac_path = os.getenv("SAC_AGENT_PATH") or os.getenv("SAC_MODEL_PATH")
    if not sac_path or not os.path.isfile(sac_path):
      print("[YourControlCode] SAC_AGENT_PATH not set or file does not exist, skipping SAC agent load.")
      return
    obs_dim = 3 + 3 * len(self.residual_joints) + 2
    act_dim = len(self.residual_joints)
    cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)
    self.agent = SACAgent(cfg)
    try:
      self.agent.load(sac_path)
      self.use_agent = True
      print(f"[YourControlCode] Loaded SAC agent from {sac_path}")
    except Exception as exc:  # noqa: BLE001
      print(f"[YourControlCode] Failed to load SAC agent '{sac_path}': {exc}")
      self.agent = None
      self.use_agent = False

  def _get_tip_velocity(self):
    pend_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    mujoco.mj_jacBody(self.m, self.d, self.tip_jac, self.tip_rot, pend_body)
    return self.tip_jac @ self.d.qvel

  def _get_obs(self):
    q_p = float(self.d.qpos[self.pend_qpos_adr])
    qd_p = float(self.d.qvel[self.pend_dof_adr])
    obs = [np.sin(q_p), np.cos(q_p), qd_p]

    for idx, joint in enumerate(self.residual_joints):
      q_idx = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
      v_idx = self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
      obs.append(float(self.d.qpos[q_idx]))
      obs.append(float(self.d.qvel[v_idx]))
      obs.append(float(self._last_baseline_tau[self.ctrl_indices[idx]] / (self.residual_scale[idx] + 1e-6)))

    tip_vel = self._get_tip_velocity()
    obs.append(float(tip_vel[1]))
    obs.append(float(tip_vel[2]))
    return np.asarray(obs, dtype=np.float32)

  def CtrlUpdate(self):
    base_tau = self.baseline.compute_torques(self.d)
    self._last_baseline_tau = base_tau.copy()
    torques = base_tau.copy()

    if self.use_agent and self.agent is not None:
      obs = self._get_obs()
      residual = self.agent.select_action(obs, deterministic=True)
      torques[self.ctrl_indices] += residual * self.residual_scale

    self.d.ctrl[:] = np.clip(torques, self.ctrl_low, self.ctrl_high)
    return True
