import os
import mujoco
import numpy as np

try:
  import torch
  from rl.algos.td3 import Actor as TD3Actor
except Exception:
  torch = None
  TD3Actor = None

from rl.baseline_ctrl import BaselineController


class YourCtrl:
  """
  Baseline controller plus optional TD3 residual policy used by Run_PendulumEnv.
  """

  RESIDUAL_SCALE = 0.05

  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d

    self.baseline = BaselineController(self.m)
    self.baseline.reset(self.d)

    self.n_act = int(self.m.nu)
    ctrl_ranges = np.array(self.m.actuator_ctrlrange, dtype=np.float32)
    self.ctrl_low = ctrl_ranges[:, 0]
    self.ctrl_high = ctrl_ranges[:, 1]
    self.ctrl_mag = np.maximum(np.abs(self.ctrl_low), np.abs(self.ctrl_high))
    self.residual_scale = float(self.RESIDUAL_SCALE)

    self.obs_dim = 16 + self.m.nv
    self.obs_norm_mean = np.zeros(self.obs_dim, dtype=np.float32)
    self.obs_norm_std = np.ones(self.obs_dim, dtype=np.float32)

    ckpt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rl', 'checkpoints')
    td3_actor_path_best = os.path.join(ckpt_dir, 'td3_actor_best.pt')
    td3_actor_path = os.path.join(ckpt_dir, 'td3_actor.pt')
    norm_best = os.path.join(ckpt_dir, 'td3_obs_norm_best.npz')
    norm_last = os.path.join(ckpt_dir, 'td3_obs_norm.npz')

    self.use_td3 = False
    td3_path_to_use = td3_actor_path_best if os.path.isfile(td3_actor_path_best) else td3_actor_path
    norm_path = norm_best if os.path.isfile(norm_best) else norm_last

    if (torch is not None and TD3Actor is not None and os.path.isfile(td3_path_to_use)
            and os.path.isfile(norm_path)):
      try:
        self.td3_actor = TD3Actor(self.obs_dim, self.n_act, hidden=(256, 256))
        self.td3_actor.load_state_dict(torch.load(td3_path_to_use, map_location=self.td3_actor.device))
        self.td3_actor.eval()
        with np.load(norm_path) as norm_stats:
          self.obs_norm_mean = norm_stats["mean"].astype(np.float32)
          self.obs_norm_std = np.sqrt(norm_stats["var"].astype(np.float32) + 1e-8)
        self.use_td3 = True
        print('Using TD3 residual policy from', td3_path_to_use)
      except Exception:
        self.use_td3 = False
        print('Failed to load TD3 policy, running baseline only.')
    else:
      print('TD3 policy or normalizer missing, running baseline only.')

  def _get_local_z(self) -> np.ndarray:
    pend_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    quat = self.d.body(pend_id).xquat
    R_flat = np.empty(9, dtype=np.float64)
    mujoco._functions.mju_quat2Mat(R_flat, quat)
    return R_flat.reshape(3, 3)[:, 2]

  def _get_theta(self) -> float:
    local_z = self._get_local_z()
    c = np.clip(local_z[2], -1.0, 1.0)
    return float(np.arccos(c))

  def _get_head_pos(self) -> np.ndarray:
    geom_mass_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "mass")
    return np.array(self.d.geom_xpos[geom_mass_id], dtype=np.float32)

  def _get_obs(self) -> np.ndarray:
    p_head = self._get_head_pos()
    theta = self._get_theta()
    applied = np.array(self.d.qfrc_applied[:self.m.nv], dtype=np.float32)
    obs = np.concatenate([
      np.array(self.d.qpos[:6], dtype=np.float32),
      np.array(self.d.qvel[:6], dtype=np.float32),
      p_head.astype(np.float32),
      np.array([theta], dtype=np.float32),
      applied,
    ], axis=0)
    return obs

  def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
    return (obs - self.obs_norm_mean) / (self.obs_norm_std + 1e-8)

  def CtrlUpdate(self):
    base_tau = self.baseline.compute_torques(self.d)
    torques = base_tau.copy()

    if self.use_td3:
      obs = self._normalize_obs(self._get_obs())
      obs_t = torch.tensor(obs, dtype=torch.float32, device=self.td3_actor.device).unsqueeze(0)
      with torch.no_grad():
        action = self.td3_actor(obs_t).squeeze(0).cpu().numpy()
      residual = action.astype(np.float32) * (self.residual_scale * self.ctrl_mag)
      torques += residual

    self.d.ctrl[:] = np.clip(torques, self.ctrl_low, self.ctrl_high)
    return True
