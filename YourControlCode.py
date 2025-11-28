"""
Control adapter for inverted pendulum balancing.
Loads trained RL model and exposes CtrlUpdate() interface for Run_PendulumEnv.py.
"""

import mujoco
from rl_controller import RLController


class YourCtrl:
    """
    Control adapter that wraps RPO RL controller for Run_PendulumEnv.py compatibility.
    Automatically loads the most recent RPO model from the runs directory.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, model_path=None):
        """
        Initialize the RPO RL controller.

        Args:
            m: MuJoCo model
            d: MuJoCo data
            model_path: Optional path to trained model (auto-detects latest RPO model if None)
        """
        self.controller = RLController(m, d, model_path=model_path)

    def CtrlUpdate(self):
        """Update control signal using RL policy."""
        return self.controller.CtrlUpdate()
