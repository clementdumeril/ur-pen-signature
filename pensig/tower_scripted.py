"""Hand-written pick and place on the same estimates the policy sees: the baseline the RL policy has to beat.
It goes down to a block or to the tower only on the wrist-camera estimate, taken from above."""

import numpy as np

from pensig.tower import TIP_BELOW_TCP, wrap_quarter


class ScriptedStack:
    """Phases: 0 above the block, 1 down to the grasp, 2 close, 3 lift, 4 over the tower,
    5 down until the wrist force says contact, 6 open, 7 back up."""

    def __init__(self, clearance: float = 0.03, settle_steps: int = 4, close_steps: int = 8, open_steps: int = 6,
                 contact_force: float = 3.0, search: float = 0.004):
        self.clearance = clearance
        self.settle_steps = settle_steps
        self.close_steps = close_steps
        self.open_steps = open_steps
        self.contact_force = contact_force
        self.search = search
        self.reset()

    def reset(self):
        self.phase = 0
        self.timer = 0
        self.hold_pos = None
        self.f_carry = None

    def __call__(self, env) -> np.ndarray:
        e = env.unwrapped
        cfg = e.cfg
        # Above everything: the fingertips clear the tower and the held block by `clearance`.
        safe_z = max(e.est_top[2], e.est_held[2] + e.est_size[1] / 2) + TIP_BELOW_TCP + self.clearance
        carry_z = e.est_place[2] + self.clearance

        def move(target, grip, yaw_err=0.0, gain=0.5):
            # Half gain: a full step per period oscillates when the arm answers one period late.
            dp = np.clip(gain * (target - e.target_pos) / cfg.max_dpos, -1, 1)
            return np.array([*dp, np.clip(0.5 * yaw_err / cfg.max_dyaw, -1, 1), grip])

        def near(target, tol=2e-3):
            return np.linalg.norm(target - e.target_pos) < tol

        grasp_yaw_err = wrap_quarter(e.est_held_yaw - e.target_yaw)
        place_yaw_err = wrap_quarter(e.est_target_yaw - e.target_yaw)

        if self.phase == 0:
            target = np.array([*e.est_grasp[:2], max(safe_z, e.est_grasp[2])])
            action = move(target, 1.0, grasp_yaw_err)
            if near(target) and abs(grasp_yaw_err) < 0.02 and (e.wrist_held or not cfg.wrist_camera):
                self.phase = 1
        elif self.phase == 1:
            action = move(e.est_grasp, 1.0, grasp_yaw_err)
            if near(e.est_grasp):
                self.timer += 1
                if self.timer >= self.settle_steps:
                    self.phase, self.timer = 2, 0
        elif self.phase == 2:
            action = move(e.est_grasp, -1.0)
            self.timer += 1
            if self.timer >= self.close_steps:
                self.phase, self.timer = 3, 0
        elif self.phase == 3:
            target = np.array([*e.target_pos[:2], carry_z])
            action = move(target, -1.0)
            if near(target, 5e-3):
                self.phase = 4
        elif self.phase == 4:
            target = np.array([*e.est_place[:2], max(carry_z, e.target_pos[2])])
            action = move(target, -1.0, place_yaw_err)
            if near(target) and abs(place_yaw_err) < 0.02 and (e.wrist_target or not cfg.wrist_camera):
                self.phase = 5
                self.f_carry = e.force.copy()
        elif self.phase == 5:
            # Down until contact, like moveUntilContact, at most `search` below the estimated top: at 6 cm/s,
            # then the last 12 mm above the estimate at 1 cm/s, and only there look at the wrist force, which also
            # feels the 1 kg gripper accelerate. Faster, the arm lags the command by centimeters and lands hard.
            target = e.est_place - [0, 0, self.search]
            action = move(target, -1.0, place_yaw_err)
            final = e.target_pos[2] - e.est_place[2] < 0.012
            action[2] = max(action[2], -(0.0005 if final else 0.003) / cfg.max_dpos)
            touched = final and np.linalg.norm(e.force - self.f_carry) > self.contact_force
            if touched or near(target, 5e-4):
                self.phase, self.timer = 6, 0
                self.hold_pos = e.target_pos.copy()
                action = move(self.hold_pos, -1.0)
        elif self.phase == 6:
            action = move(self.hold_pos, 1.0)
            self.timer += 1
            if self.timer >= self.open_steps:
                self.phase = 7
        else:
            action = move(self.hold_pos + [0, 0, 0.08], 1.0)
        return action
