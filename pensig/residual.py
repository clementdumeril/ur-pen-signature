"""Residual RL: PPO learns a correction on top of the scripted grasp.

The policy starts from the scripted success rate instead of from scratch. From scratch, PPO settles on
hovering next to the pen: closing the gripper rarely pays off until the alignment is already good.
"""

import gymnasium as gym
import numpy as np

from pensig.config import EnvConfig
from pensig.env import PenGraspEnv
from pensig.scripted import ScriptedGrasp

N_PHASES = 4


class ResidualGraspEnv(PenGraspEnv):
    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None,
                 scale=(0.5, 0.5, 0.5, 0.5, 1.0)):
        super().__init__(cfg, render_mode)
        self.base = ScriptedGrasp()
        self.scale = np.asarray(scale, float)
        n = self.observation_space.shape[0] + 5 + N_PHASES
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.base.reset()
        self.base_action = self.base(self)
        return self._augment(obs), info

    def step(self, action):
        total = np.clip(self.base_action + self.scale * np.asarray(action, float), -1.0, 1.0)
        obs, reward, terminated, truncated, info = super().step(total)
        self.base_action = self.base(self)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        phase = np.zeros(N_PHASES)
        phase[min(self.base.phase, N_PHASES - 1)] = 1.0
        return np.concatenate([obs, self.base_action, phase]).astype(np.float32)
