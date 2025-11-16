import numpy as np
import mujoco


class BaselineController:
    """
    Deterministic controller used both for training and evaluation.
    Produces torque commands for all actuators.
    """

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.kp_joint = 80.0
        self.kd_joint = 6.0
        self.kp_pend = 280.0
        self.kd_pend = 55.0
        self.ki_pend = 35.0
        self.int_limit = 0.4
        self.cart_vel_gain_p = 25.0
        self.cart_vel_gain_d = 6.0
        self.cart_tracking_gain = 45.0
        self.cart_vel_clip = 2.5

        self.pend_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
        self.pend_qpos_adr = self.m.jnt_qposadr[self.pend_joint_id]
        self.pend_dof_adr = self.m.jnt_dofadr[self.pend_joint_id]
        self.pend_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
        self.actuated_dofs = np.arange(self.m.nu, dtype=int)

        self.M_full = np.zeros((self.m.nv, self.m.nv))
        self.Jp = np.zeros((3, self.m.nv))
        self.Jr = np.zeros((3, self.m.nv))

        ctrl_ranges = np.array(self.m.actuator_ctrlrange, dtype=np.float32)
        self.ctrl_low = ctrl_ranges[:, 0]
        self.ctrl_high = ctrl_ranges[:, 1]

        self.init_qpos = None
        self.pend_int = 0.0

    def reset(self, data: mujoco.MjData):
        self.init_qpos = data.qpos.copy()
        self.pend_int = 0.0

    def compute_torques(self, data: mujoco.MjData) -> np.ndarray:
        if self.init_qpos is None:
            self.reset(data)

        mujoco.mj_forward(self.m, data)
        mujoco.mj_fullM(self.m, self.M_full, data.qM)

        q_p = data.qpos[self.pend_qpos_adr]
        qd_p = data.qvel[self.pend_dof_adr]
        self.pend_int += q_p * self.m.opt.timestep
        self.pend_int = float(np.clip(self.pend_int, -self.int_limit, self.int_limit))
        qacc_p_des = -self.kp_pend * q_p - self.kd_pend * qd_p - self.ki_pend * self.pend_int

        p_idx = self.pend_dof_adr
        A_idx = self.actuated_dofs
        M_pA = self.M_full[p_idx, A_idx]
        M_pp = self.M_full[p_idx, p_idx]
        bias_p = data.qfrc_bias[p_idx]
        denom = float(np.dot(M_pA, M_pA))
        if denom < 1e-8:
            qacc_a_particular = np.zeros_like(M_pA)
        else:
            rhs = -(M_pp * qacc_p_des + bias_p)
            qacc_a_particular = (rhs / denom) * M_pA

        pos_error = self.init_qpos[A_idx] - data.qpos[A_idx]
        vel_error = -data.qvel[A_idx]
        qacc_posture = self.kp_joint * pos_error + self.kd_joint * vel_error

        mujoco.mj_jacBody(self.m, data, self.Jp, self.Jr, self.pend_body_id)
        Jp_act = self.Jp[:, A_idx]
        v_lat_des = -self.cart_vel_gain_p * q_p - self.cart_vel_gain_d * qd_p
        v_lat_des = float(np.clip(v_lat_des, -self.cart_vel_clip, self.cart_vel_clip))
        v_des = np.array([0.0, v_lat_des, 0.0])
        Jpinv = np.linalg.pinv(Jp_act, rcond=1e-4)
        qvel_cart_des = Jpinv @ v_des
        qvel_cart_des = np.clip(qvel_cart_des, -3.5, 3.5)
        cart_term = self.cart_tracking_gain * (qvel_cart_des - data.qvel[A_idx])
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

        tau_full = self.M_full @ qacc_des + data.qfrc_bias
        torques = np.zeros(self.m.nu, dtype=np.float32)
        torques[:] = 0.0
        torques[:] = np.clip(tau_full[:self.m.nu], self.ctrl_low, self.ctrl_high)
        return torques
