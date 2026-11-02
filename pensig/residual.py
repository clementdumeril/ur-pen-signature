"""Residual RL: PPO learns a correction on top of a scripted controller.

The policy starts from the scripted success rate instead of from scratch. From scratch, PPO settles on
hovering next to the pen: closing the gripper rarely pays off until the alignment is already good.
"""

import gymnasium as gym
import numpy as np

from pensig.env import PenGraspEnv
from pensig.scripted import ScriptedGrasp
from pensig.tower import TowerEnv
from pensig.tower_scripted import ScriptedStack


class _Residual:
    scripted_cls = None
    n_phases = 0

    def __init__(self, cfg=None, render_mode: str | None = None, scale=(0.5, 0.5, 0.5, 0.5, 1.0)):
        super().__init__(cfg, render_mode)
        self.base = self.scripted_cls()
        self.scale = np.asarray(scale, float)
        n = self.observation_space.shape[0] + 5 + self.n_phases
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
        phase = np.zeros(self.n_phases)
        phase[min(self.base.phase, self.n_phases - 1)] = 1.0
        return np.concatenate([obs, self.base_action, phase]).astype(np.float32)


class ResidualGraspEnv(_Residual, PenGraspEnv):
    scripted_cls = ScriptedGrasp
    n_phases = 4


class ResidualTowerEnv(_Residual, TowerEnv):
    scripted_cls = ScriptedStack
    n_phases = 8

    # A small gripper correction: exploring with the full range opened the fingers and dropped the block
    # in transit.
    def __init__(self, cfg=None, render_mode: str | None = None, scale=(0.5, 0.5, 0.5, 0.5, 0.3)):
        super().__init__(cfg, render_mode, scale)

    def next_stage(self):
        obs, info = super().next_stage()
        self.base.reset()
        self.base_action = self.base(self)
        return self._augment(obs), info
