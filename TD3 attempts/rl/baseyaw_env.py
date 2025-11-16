import numpy as np
import gymnasium as gym

from rl.pendulum_env_cont import PendulumBalanceEnvCont


class BaseYawResidualEnv(gym.Env):
    """
    Thin wrapper around PendulumBalanceEnvCont that exposes a 1-D action space.
    The action controls only the base yaw actuator, while the other actuators
    remain driven solely by the baseline controller.
    """

    metadata = {"render_modes": []}

    def __init__(self, **pendulum_kwargs):
        super().__init__()
        self._env = PendulumBalanceEnvCont(**pendulum_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._nu = int(self._env.model.nu)
        if self._nu < 1:
            raise ValueError("Underlying environment must have at least one actuator.")

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 1:
            raise ValueError("Action must be 1-D for BaseYawResidualEnv.")
        full_action = np.zeros(self._nu, dtype=np.float32)
        full_action[0] = float(np.clip(action[0], -1.0, 1.0))
        return self._env.step(full_action)

    def close(self):
        self._env.close()

