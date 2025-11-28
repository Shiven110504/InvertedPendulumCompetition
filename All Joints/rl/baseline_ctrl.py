import numpy as np
import mujoco


class BaselineController:
    """Model-based baseline controller for the miniArm + pendulum system.

    This controller uses MuJoCo's dynamics (mass matrix and bias forces) to
    construct an inverse-dynamics control law for an underactuated system:
    the first `m.nu` joints are actuated, while the pendulum joint is
    unactuated. The design has three main components:

    1. Pendulum PID in joint space -> desired pendulum acceleration.
    2. Underactuated constraint solve: choose arm accelerations that satisfy
       the pendulum dynamics row with the desired pendulum acceleration.
    3. Extra "posture" and "virtual cart" behaviour projected into the
       nullspace of the pendulum dynamics, so they don't fight the pendulum
       stabilization.

    The output is a vector of torques for all actuators (length m.nu).
    """

    def __init__(self, model: mujoco.MjModel):
        self.m = model

        # Joint-space posture regulation gains (for actuated joints).
        self.kp_joint = 80.0
        self.kd_joint = 6.0

        # Pendulum PID gains (upright at q_p = 0).
        self.kp_pend = 280.0
        self.kd_pend = 55.0
        self.ki_pend = 35.0
        self.int_limit = 0.4

        # "Virtual cart" behaviour: we treat the pendulum body's COM as if
        # it were a cart-pole and try to give it a lateral velocity that
        # stabilizes the upright configuration.
        self.cart_vel_gain_p = 25.0
        self.cart_vel_gain_d = 6.0
        self.cart_tracking_gain = 45.0
        self.cart_vel_clip = 2.5

        # Direction in world coordinates that we treat as the "cart" axis.
        # By default this is the world y-axis, consistent with the original
        # controller. You can change this if your environment orientation
        # differs.
        self.cart_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        # External force compensation: we can explicitly subtract the
        # generalized external forces (qfrc_applied) when converting desired
        # accelerations to torques. A gain of 1.0 fully cancels them; 0.0
        # ignores them.
        self.external_comp_gain = 2.75

        # Pendulum joint / body bookkeeping.
        self.pend_joint_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll"
        )
        self.pend_qpos_adr = self.m.jnt_qposadr[self.pend_joint_id]
        self.pend_dof_adr = self.m.jnt_dofadr[self.pend_joint_id]
        self.pend_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )

        # Map each actuator to the DOF it actuates using actuator_trnid.
        # This is more robust than assuming DOF indices [0..m.nu-1].
        self.actuated_dofs = np.array(
            [self.m.jnt_dofadr[self.m.actuator_trnid[i, 0]]
             for i in range(self.m.nu)],
            dtype=int,
        )

        # Buffers for mass matrix and Jacobians.
        self.M_full = np.zeros((self.m.nv, self.m.nv))
        self.Jp = np.zeros((3, self.m.nv))
        self.Jr = np.zeros((3, self.m.nv))

        # Actuator torque limits (ctrlrange in the XML).
        ctrl_ranges = np.array(self.m.actuator_ctrlrange, dtype=np.float32)
        self.ctrl_low = ctrl_ranges[:, 0]
        self.ctrl_high = ctrl_ranges[:, 1]

        # State for posture target and pendulum integral term.
        self.init_qpos = None
        self.pend_int = 0.0

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def reset(self, data: mujoco.MjData) -> None:
        """Reset posture target and integral term.

        Should be called at episode start.
        """
        self.init_qpos = data.qpos.copy()
        self.pend_int = 0.0

    # ---------------------------------------------------------------------
    # Main control law
    # ---------------------------------------------------------------------
    def compute_torques(self, data: mujoco.MjData) -> np.ndarray:
        """Compute actuator torques for the current state.

        Parameters
        ----------
        data : mujoco.MjData
            Current simulation data.

        Returns
        -------
        np.ndarray
            Torque commands of shape (m.nu,).
        """
        # Lazily initialize posture target on first call.
        if self.init_qpos is None:
            self.reset(data)

        # Update dynamics terms: qfrc_bias, qM, etc.
        mujoco.mj_forward(self.m, data)
        mujoco.mj_fullM(self.m, self.M_full, data.qM)

        # ------------------------------------------------------------------
        # 1) Pendulum PID -> desired pendulum acceleration
        # ------------------------------------------------------------------
        q_p = float(data.qpos[self.pend_qpos_adr])
        qd_p = float(data.qvel[self.pend_dof_adr])

        # Integral of pendulum angle, with simple clamping for anti-windup.
        self.pend_int += q_p * float(self.m.opt.timestep)
        self.pend_int = float(np.clip(self.pend_int, -self.int_limit, self.int_limit))

        # Desired pendulum angular acceleration about its joint.
        qacc_p_des = (
            -self.kp_pend * q_p
            - self.kd_pend * qd_p
            - self.ki_pend * self.pend_int
        )

        # ------------------------------------------------------------------
        # 2) Underactuated constraint solve for arm accelerations
        # ------------------------------------------------------------------
        p_idx = self.pend_dof_adr
        A_idx = self.actuated_dofs

        # Extract the pendulum row of the mass matrix and the scalar M_pp.
        M_pA = self.M_full[p_idx, A_idx]        # shape (n_act,)
        M_pp = float(self.M_full[p_idx, p_idx])  # scalar
        bias_p = float(data.qfrc_bias[p_idx])

        # Constraint from the pendulum dynamics row (no direct actuation):
        #   M_pA * qacc_a + M_pp * qacc_p_des + bias_p = 0
        # Solve for a particular qacc_a that satisfies this with minimum norm.
        denom = float(np.dot(M_pA, M_pA))
        if denom < 1e-8:
            qacc_a_particular = np.zeros_like(M_pA)
        else:
            rhs = -(M_pp * qacc_p_des + bias_p)
            qacc_a_particular = (rhs / denom) * M_pA

        # ------------------------------------------------------------------
        # 3) Posture PD term on the actuated joints
        # ------------------------------------------------------------------
        pos_error = self.init_qpos[A_idx] - data.qpos[A_idx]
        vel_error = -data.qvel[A_idx]
        qacc_posture = self.kp_joint * pos_error + self.kd_joint * vel_error

        # ------------------------------------------------------------------
        # 4) "Virtual cart" velocity control at the pendulum COM
        # ------------------------------------------------------------------
        # Compute translational Jacobian for the pendulum body COM.
        mujoco.mj_jacBody(self.m, data, self.Jp, self.Jr, self.pend_body_id)
        Jp_act = self.Jp[:, A_idx]

        # Normalize the chosen cart axis in world coordinates.
        axis_norm = float(np.linalg.norm(self.cart_axis))
        if axis_norm > 0.0:
            cart_axis = self.cart_axis / axis_norm
        else:
            # Fallback to world-y if somehow zero.
            cart_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        # Desired scalar COM velocity along the cart axis, analogous to the
        # cart velocity in cart-pole.
        v_cart_des = -self.cart_vel_gain_p * q_p - self.cart_vel_gain_d * qd_p
        v_cart_des = float(np.clip(v_cart_des, -self.cart_vel_clip, self.cart_vel_clip))

        # Full 3D desired COM velocity vector.
        v_des = v_cart_des * cart_axis

        # Least-squares joint velocities that realize v_des.
        # Small rcond for numerical damping.
        Jpinv = np.linalg.pinv(Jp_act, rcond=1e-4)
        qvel_cart_des = Jpinv @ v_des
        qvel_cart_des = np.clip(qvel_cart_des, -3.5, 3.5)

        # Acceleration term that drives actual joint velocities toward
        # qvel_cart_des.
        cart_term = self.cart_tracking_gain * (qvel_cart_des - data.qvel[A_idx])
        qacc_posture = qacc_posture + cart_term

        # ------------------------------------------------------------------
        # 5) Project posture/cart term into nullspace of M_pA
        # ------------------------------------------------------------------
        if denom < 1e-8:
            qacc_posture_proj = qacc_posture
        else:
            # Projection: v_proj = v - (M_pA·v / ||M_pA||^2) * M_pA
            factor = float(np.dot(M_pA, qacc_posture) / denom)
            qacc_posture_proj = qacc_posture - factor * M_pA

        # Total arm accelerations: particular solution + nullspace motion.
        qacc_a = qacc_a_particular + qacc_posture_proj

        # ------------------------------------------------------------------
        # 6) Assemble full desired accelerations
        # ------------------------------------------------------------------
        qacc_des = np.zeros(self.m.nv)
        qacc_des[A_idx] = qacc_a
        qacc_des[p_idx] = qacc_p_des

        # ------------------------------------------------------------------
        # 7) Inverse dynamics: convert desired accelerations to torques
        # ------------------------------------------------------------------
        # Base inverse-dynamics torques.
        tau_full = self.M_full @ qacc_des + data.qfrc_bias

        # Optionally compensate for external generalized forces, e.g. pushes
        # applied in Run_PendulumEnv via mj_applyFT. A gain of 1.0 tries to
        # cancel them exactly; smaller values partially compensate.
        if self.external_comp_gain != 0.0:
            tau_full = tau_full - self.external_comp_gain * data.qfrc_applied

        # Map DOF-space torques to actuator torques. For simple joint motors,
        # each actuator acts on a single joint DOF.
        torques = np.zeros(self.m.nu, dtype=np.float32)
        for act_id, dof_id in enumerate(self.actuated_dofs):
            torques[act_id] = tau_full[dof_id]

        # Enforce actuator torque limits.
        torques = np.clip(torques, self.ctrl_low, self.ctrl_high)
        return torques

