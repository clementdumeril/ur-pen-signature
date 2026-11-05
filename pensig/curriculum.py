"""Tower episodes that start partway through the task, for an end-to-end policy (no script under it at run time).

PPO from the start of the task never finds a placement, so it never sees a success to learn from. Here the
scripted stack plays the first part of the episode and hands over at a given phase: at level 0 the policy only
has to let go and lift (phase 7), each level adds one earlier phase, and from level 8 some episodes start from
the plain reset, as at run time. Start states from demonstrations: Nair et al. 2018, Salimans and Chen 2018;
the curriculum on start states: Florensa et al. 2017.

Block sizes are drawn at every reset, so one policy has to handle any block within the gripper stroke.
"""

import gymnasium as gym
import numpy as np

from pensig.tower_scripted import ScriptedStack

N_PHASES = 8  # ScriptedStack phases 0..7
MAX_LEVEL = N_PHASES + 1  # 8: full starts join, 9: half of the starts are full starts


def random_sizes(rng) -> dict:
    """Square blocks, (width, height): the base wider, the held ones within the 50 mm Hand-E stroke."""
    return {"base_size": (rng.uniform(0.040, 0.060), rng.uniform(0.025, 0.080)),
            "mid_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.080)),
            "top_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.060))}


def frontier(level: int) -> int:
    """The earliest start phase at this level, -1 for the plain reset."""
    return N_PHASES - 1 - level if level < N_PHASES else -1


class ScriptStart(gym.Wrapper):
    """TowerEnv whose episodes start where the script hands over. info["start_phase"]: -1 for the plain reset."""

    def __init__(self, env, sizes: dict | None = None, noise=(0.0, 0.1), max_extra: int = 20, seed: int = 0):
        super().__init__(env)
        self.sizes = sizes  # None: random sizes at every reset
        self.noise = noise
        self.max_extra = max_extra
        self.rng = np.random.default_rng(seed)
        self.script = ScriptedStack()
        self.level = 0
        self.start_phase = -1

    def set_level(self, level: int):
        self.level = int(level)

    def _sample_phase(self) -> int:
        if self.level >= MAX_LEVEL and self.rng.random() < 0.5:
            return -1
        lo = frontier(min(self.level, N_PHASES))
        return int(self.rng.integers(lo, N_PHASES))  # lo = -1 includes the plain reset

    def reset(self, *, seed=None, options=None):
        e = self.env.unwrapped
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(10):
            for k, v in (self.sizes or random_sizes(self.rng)).items():
                setattr(e.cfg, k, v)
            obs, info = self.env.reset(seed=seed, options=options)
            seed = None
            p = self._sample_phase()
            if p < 0 or self._replay(p):
                break
        else:
            p = -1
            obs, info = self.env.reset(options=options)
        if p >= 0:
            self.start_phase = self.script.phase
            e.t = 0  # the policy gets the full step budget from the handover
            obs = e._obs()
        else:
            self.start_phase = -1
        info["start_phase"] = self.start_phase
        return obs, info

    def _replay(self, p: int) -> bool:
        """The script drives until phase p (plus a few steps into it). False if the episode ended first."""
        e = self.env.unwrapped
        self.script.reset()
        noise = self.rng.choice(self.noise)
        # Phase 7 only lifts away: a few steps more and the script has already succeeded.
        extra = int(self.rng.integers(0, (3 if p == N_PHASES - 1 else self.max_extra) + 1))
        inside = 0
        for _ in range(e.cfg.episode_steps):
            if self.script.phase > p:
                return True
            if self.script.phase == p:
                if inside >= extra:
                    return True
                inside += 1
            a = self.script(e)
            a[:4] = np.clip(a[:4] + noise * self.rng.standard_normal(4), -1, 1)
            _, _, term, trunc, _ = self.env.step(a)
            if term or trunc:
                return False
        return False

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["start_phase"] = self.start_phase
        return obs, r, term, trunc, info
