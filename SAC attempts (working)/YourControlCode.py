import os
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

try:
  import torch
  from rl.models import ResidualPolicy
  _TORCH_AVAILABLE = True
except Exception:  # torch may be unavailable during baselines
  ResidualPolicy = None
  torch = None
  _TORCH_AVAILABLE = False


class YourCtrl:
  """Computed-torque stabilizer with extra lateral motion to reject pushes."""

  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    # Joint posture regulation.
    self.kp_joint = 80.0
    self.kd_joint = 6.0

    # Pendulum joint regulation with integral action.
    self.kp_pend = 280.0
    self.kd_pend = 55.0
    self.ki_pend = 35.0
    self.pend_int = 0.0
    self.int_limit = 0.4

    # Cartesian shaping so the arm intentionally overcorrects.
    self.cart_vel_gain_p = 25.0
    self.cart_vel_gain_d = 6.0
    self.cart_tracking_gain = 45.0
    self.cart_vel_clip = 2.5

    self.pend_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
    self.pend_qpos_adr = self.m.jnt_qposadr[self.pend_joint_id]
    self.pend_dof_adr = self.m.jnt_dofadr[self.pend_joint_id]
    self.pend_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    self.base_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
    self.base_qpos_adr = self.m.jnt_qposadr[self.base_joint_id]
    self.base_dof_adr = self.m.jnt_dofadr[self.base_joint_id]
    self.base_actuator_id = self._find_actuator_index(self.base_joint_id)
    self.actuated_dofs = np.arange(self.m.nu, dtype=int)

    self.M_full = np.zeros((self.m.nv, self.m.nv))
    self.Jp = np.zeros((3, self.m.nv))
    self.Jr = np.zeros((3, self.m.nv))
    self.body_vel_tmp = np.zeros(6, dtype=np.float64)

    self.residual_limit = 3.5
    self.external_residual = 0.0
    self.policy_residual_limit = self.residual_limit
    self.policy_obs_scale = np.array(
        [1 / np.pi, 0.1, 1 / np.pi, 0.1, 1.5, 0.3], dtype=np.float32
    )
    self.policy: Optional[ResidualPolicy] = None
    self.policy_enabled = False
    self.policy_path = Path(
        os.environ.get("SAC_POLICY_PATH", "models/working/sac_base_yaw.pt")
    )
    self._maybe_load_policy()

  def CtrlUpdate(self) -> bool:
    mujoco.mj_forward(self.m, self.d)
    mujoco.mj_fullM(self.m, self.M_full, self.d.qM)

    q_p = self.d.qpos[self.pend_qpos_adr]
    qd_p = self.d.qvel[self.pend_dof_adr]
    self.pend_int += q_p * self.m.opt.timestep
    self.pend_int = float(np.clip(self.pend_int, -self.int_limit, self.int_limit))
    qacc_p_des = -self.kp_pend * q_p - self.kd_pend * qd_p - self.ki_pend * self.pend_int

    p_idx = self.pend_dof_adr
    A_idx = self.actuated_dofs
    M_pA = self.M_full[p_idx, A_idx]
    M_pp = self.M_full[p_idx, p_idx]
    bias_p = self.d.qfrc_bias[p_idx]
    denom = float(np.dot(M_pA, M_pA))
    if denom < 1e-8:
      qacc_a_particular = np.zeros_like(M_pA)
    else:
      rhs = -(M_pp * qacc_p_des + bias_p)
      qacc_a_particular = (rhs / denom) * M_pA

    pos_error = self.init_qpos[A_idx] - self.d.qpos[A_idx]
    vel_error = -self.d.qvel[A_idx]
    qacc_posture = self.kp_joint * pos_error + self.kd_joint * vel_error

    mujoco.mj_jacBody(self.m, self.d, self.Jp, self.Jr, self.pend_body_id)
    Jp_act = self.Jp[:, A_idx]
    v_lat_des = -self.cart_vel_gain_p * q_p - self.cart_vel_gain_d * qd_p
    v_lat_des = float(np.clip(v_lat_des, -self.cart_vel_clip, self.cart_vel_clip))
    v_des = np.array([0.0, v_lat_des, 0.0])
    Jpinv = np.linalg.pinv(Jp_act, rcond=1e-4)
    qvel_cart_des = Jpinv @ v_des
    qvel_cart_des = np.clip(qvel_cart_des, -3.5, 3.5)
    cart_term = self.cart_tracking_gain * (qvel_cart_des - self.d.qvel[A_idx])
    qacc_posture += cart_term

    if denom < 1e-8:
      qacc_posture_proj = qacc_posture
    else:
      factor = np.dot(M_pA, qacc_posture) / denom
      qacc_posture_proj = qacc_posture - factor * M_pA

    qacc_a = qacc_a_particular + qacc_posture_proj

    qacc_des = np.zeros(self.m.nv)
    qacc_des[A_idx] = qacc_a
    qacc_des[p_idx] = qacc_p_des

    tau_full = self.M_full @ qacc_des + self.d.qfrc_bias
    tau_full[self.base_actuator_id] += self._residual_torque()

    for i in range(self.m.nu):
      low, high = self.m.actuator_ctrlrange[i]
      self.d.ctrl[i] = float(np.clip(tau_full[i], low, high))

    return True

  def set_residual_action(self, value: float) -> None:
    """External interface for RL training to specify a base-yaw residual torque."""
    self.external_residual = float(np.clip(value, -self.residual_limit, self.residual_limit))

  def get_policy_observation(self, normalized: bool = True) -> np.ndarray:
    obs = self._collect_policy_observation()
    if normalized:
      return obs * self.policy_obs_scale
    return obs

  def _collect_policy_observation(self) -> np.ndarray:
    mujoco.mj_objectVelocity(
        self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.pend_body_id, self.body_vel_tmp, 0
    )
    pend_y = self.d.body(self.pend_body_id).xpos[1]
    pend_vy = self.body_vel_tmp[1]
    obs = np.array(
        [
            self.d.qpos[self.pend_qpos_adr],
            self.d.qvel[self.pend_dof_adr],
            self.d.qpos[self.base_qpos_adr],
            self.d.qvel[self.base_dof_adr],
            pend_y,
            pend_vy,
        ],
        dtype=np.float32,
    )
    return obs

  def _residual_torque(self) -> float:
    torque = self.external_residual
    if self.policy_enabled and self.policy is not None and _TORCH_AVAILABLE:
      obs = self.get_policy_observation(normalized=True)
      with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.policy(tensor)[0, 0].item()
      torque += float(np.clip(action, -1.0, 1.0) * self.policy_residual_limit)
    return float(np.clip(torque, -self.residual_limit, self.residual_limit))

  def _maybe_load_policy(self) -> None:
    if not _TORCH_AVAILABLE or ResidualPolicy is None:
      return
    if not self.policy_path.is_file():
      return
    try:
      checkpoint = torch.load(self.policy_path, map_location="cpu", weights_only=False)
    except TypeError:
      checkpoint = torch.load(self.policy_path, map_location="cpu")
    obs_dim = int(checkpoint.get("obs_dim", 6))
    hidden_dims = tuple(checkpoint.get("hidden_dims", (256, 256)))
    action_dim = int(checkpoint.get("action_dim", 1))
    policy = ResidualPolicy(obs_dim, action_dim, hidden_dims)
    state_dict = checkpoint.get("policy_state_dict")
    if state_dict is None:
      return
    adapted = self._adapt_policy_state(state_dict, action_dim)
    try:
      policy.load_state_dict(adapted, strict=False)
    except RuntimeError as exc:
      print(f"Failed to load SAC policy: {exc}")
      return
    policy.eval()
    self.policy_obs_scale = np.array(
        checkpoint.get("obs_scale", self.policy_obs_scale), dtype=np.float32
    )
    self.policy_residual_limit = float(
        checkpoint.get("residual_limit", self.policy_residual_limit)
    )
    self.policy = policy
    self.policy_enabled = True

  def _adapt_policy_state(self, state_dict, action_dim: int):
    adapted = {}
    for key, value in state_dict.items():
      target_key = key
      if key.startswith("net."):
        target_key = key.replace("net.", "backbone.", 1)
      tensor = value
      if target_key.endswith("net.4.weight") and tensor.shape[0] == 2 * action_dim:
        tensor = tensor[:action_dim].clone()
      if target_key.endswith("net.4.bias") and tensor.shape[0] == 2 * action_dim:
        tensor = tensor[:action_dim].clone()
      adapted[target_key] = tensor
    return adapted

  def _find_actuator_index(self, joint_id: int) -> int:
    trnid = self.m.actuator_trnid.reshape(-1, 2)
    for i in range(self.m.nu):
      if trnid[i, 0] == joint_id:
        return i
    return 0
