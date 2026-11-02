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


@dataclass
class TowerConfig:
    control_dt: float = 0.05
    sim_dt: float = 0.002
    episode_steps: int = 160

    max_dpos: float = 0.015
    max_dyaw: float = 0.15
    target_lo: tuple = (0.25, -0.35, 0.02)
    target_hi: tuple = (0.75, 0.35, 0.45)

    # Lab blocks, bottom to top: (width, height) of a square footprint, estimated from a photo. Measure
    # yours. The base is wider than the 50 mm Hand-E stroke, so it stays where it is and the tower goes on it.
    base_size: tuple = (0.050, 0.050)
    mid_size: tuple = (0.040, 0.050)
    top_size: tuple = (0.035, 0.040)
    size_jitter: float = 0.003
    block_mass: tuple = (0.015, 0.060)
    block_friction: tuple = (0.4, 0.9)

    xy_lo: tuple = (0.38, -0.18)
    xy_hi: tuple = (0.60, 0.18)
    min_gap: float = 0.10

    # Stage 0 puts the middle block on the base, stage 1 the top block on the middle one. In stage 1 the
    # middle block starts where an imperfect first placement would have left it.
    stage_probs: tuple = (0.5, 0.5)
    stack_offset: float = 0.004
    stack_yaw: float = float(np.deg2rad(5.0))

    start_above: tuple = (0.08, 0.16)
    start_xy_spread: float = 0.10

    # Grasp height: the fingertips stay this far above the bottom of the block, and so above the table.
    finger_above_bottom: float = 0.015

    pos_noise: float = 0.003
    yaw_noise: float = float(np.deg2rad(4.0))
    size_noise: float = 0.0005
    force_noise: float = 0.3

    grip_force: tuple = (10.0, 20.0)
    pad_friction: tuple = (0.6, 1.2)
    latency_prob: float = 0.3

    stop_force: float = 25.0

    place_tol: float = 0.008
    place_yaw_tol: float = float(np.deg2rad(10.0))
    clear_height: float = 0.01
    hold_steps: int = 3

    def noiseless(self) -> "TowerConfig":
        cfg = TowerConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        cfg.pos_noise = cfg.yaw_noise = cfg.size_noise = cfg.force_noise = 0.0
        cfg.latency_prob = 0.0
        return cfg
