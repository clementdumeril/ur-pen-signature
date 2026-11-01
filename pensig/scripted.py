"""Hand-written grasp on the same estimate the policy sees: the baseline the RL policy has to beat."""

import numpy as np

from pensig.env import wrap_pi


class ScriptedGrasp:
    def __init__(self, approach: float = 0.08, lift: float = 0.15, settle_steps: int = 4, close_steps: int = 8):
        self.approach = approach
        self.lift = lift
        self.settle_steps = settle_steps
        self.close_steps = close_steps
        self.reset()

    def reset(self):
        self.phase = 0
        self.timer = 0

    def __call__(self, env) -> np.ndarray:
        e = env.unwrapped
        cfg = e.cfg
        # Tip on the TCP -y side, away from the wrist camera.
        yaw_err = wrap_pi(e.est_theta - np.pi / 2 - e.target_yaw)
        grasp = e.est_grasp

        def move(target, grip, turn=True):
            # Half gain: a full step per period oscillates when the arm answers one period late.
            dp = np.clip(0.5 * (target - e.target_pos) / cfg.max_dpos, -1, 1)
            dyaw = np.clip(0.5 * yaw_err / cfg.max_dyaw, -1, 1) if turn else 0.0
            return np.array([*dp, dyaw, grip])

        if self.phase == 0:
            target = grasp + [0, 0, self.approach]
            action = move(target, 1.0)
            if np.linalg.norm(target - e.target_pos) < 2e-3 and abs(yaw_err) < 0.02:
                self.phase = 1
        elif self.phase == 1:
            action = move(grasp, 1.0)
            if np.linalg.norm(grasp - e.target_pos) < 2e-3:
                self.timer += 1
                if self.timer >= self.settle_steps:
                    self.phase, self.timer = 2, 0
        elif self.phase == 2:
            action = move(grasp, -1.0, turn=False)
            self.timer += 1
            if self.timer >= self.close_steps:
                self.phase = 3
        else:
            action = move(grasp + [0, 0, self.lift], -1.0, turn=False)
        return action
