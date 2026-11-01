from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvConfig:
    control_dt: float = 0.05
    sim_dt: float = 0.002
    episode_steps: int = 160

    # Action scaling: one step moves the TCP target by at most max_dpos and turns it by max_dyaw.
    max_dpos: float = 0.015
    max_dyaw: float = 0.15
    target_lo: tuple = (0.25, -0.35, 0.03)
    target_hi: tuple = (0.75, 0.35, 0.45)

    pen_xy_lo: tuple = (0.40, -0.15)
    pen_xy_hi: tuple = (0.58, 0.15)
    pen_radius: tuple = (0.0035, 0.0060)
    pen_length: tuple = (0.12, 0.16)
    pen_mass: tuple = (0.006, 0.025)

    support_probs: dict = field(default_factory=lambda: {"post": 0.45, "bridge": 0.35, "bottle": 0.20})
    support_height: tuple = (0.04, 0.12)
    post_width: tuple = (0.020, 0.030)
    post_depth: tuple = (0.020, 0.050)
    bridge_half_span: tuple = (0.035, 0.050)
    bottle_radius: tuple = (0.030, 0.045)
    finger_clearance: float = 0.012

    start_above: tuple = (0.10, 0.20)
    start_xy_spread: float = 0.10

    # Error of the pen pose estimate from the camera, drawn once per episode.
    pos_noise: float = 0.003
    yaw_noise: float = float(np.deg2rad(4.0))
    size_noise: float = 0.0005

    grip_force: tuple = (8.0, 20.0)
    pen_friction: tuple = (0.4, 1.0)
    pad_friction: tuple = (0.6, 1.2)
    latency_prob: float = 0.3

    stop_force: float = 25.0

    lift_height: float = 0.08
    hold_steps: int = 3

    def noiseless(self) -> "EnvConfig":
        cfg = EnvConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        cfg.pos_noise = cfg.yaw_noise = cfg.size_noise = 0.0
        cfg.latency_prob = 0.0
        return cfg
