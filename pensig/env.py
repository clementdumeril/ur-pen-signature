"""Grasp a pen resting horizontally on a support with a UR arm and a Hand-E gripper, then lift it.

Action (5): TCP target displacement (x, y, z), yaw increment about the vertical, gripper (-1 close, +1 open).
The TCP points down; an inverse-kinematics step turns the target into joint position commands,
as servoL would on the real arm. The pen pose comes from a camera estimate drawn once per episode.
"""

import gymnasium as gym
import mujoco
import numpy as np

from pensig.config import EnvConfig
from pensig.ik import Q_SEED, DiffIK, down_rotation
from pensig.scene import MAX_OPEN, PAD, build_model

ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
SUPPORT_BODIES = ("post", "bridge_a", "bridge_b", "bottle")
OBS_CENTER = np.array([0.5, 0.0, 0.1])
GRIP_SPEED = 0.1  # m/s per finger, within the Hand-E range of 0.02 to 0.15
SUCCESS_PER_STEP = 4.0  # upper bound of the shaped reward per step


def quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_half_pi(angle: float) -> float:
    """Angle difference for a gripper that is symmetric under a half turn."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


class PenGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode
        self.model = build_model()
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(self.cfg.control_dt / self.cfg.sim_dt))
        self.ik = DiffIK(self.model)

        m = self.model
        self.finger_qadr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_left", "finger_right")]
        self.pen_qadr = m.jnt_qposadr[m.joint("pen_free").id]
        # A round pen on two flat supports rolls on its own in simulation; a little damping stops it.
        pen_dof = m.jnt_dofadr[m.joint("pen_free").id]
        m.dof_damping[pen_dof + 3:pen_dof + 6] = 2e-4
        self.pen_dof = pen_dof
        self.arm_act = np.array([m.actuator(n).id for n in ARM_ACTUATORS])
        # Menagerie's servo gains lag the target by centimeters; the real arm tracks servoL closely.
        for i in self.arm_act:
            kp, kv = (10000.0, 300.0) if m.actuator_forcerange[i, 1] > 100 else (2000.0, 60.0)
            m.actuator_gainprm[i, 0] = kp
            m.actuator_biasprm[i, 1:3] = (-kp, -kv)
        self.grip_act = m.actuator("gripper").id
        self.tcp_sid = m.site("tcp").id
        self.tip_sid = m.site("pen_tip").id
        self.gid = {n: m.geom(n).id for n in
                    ("pen", "pen_tip", "pad_left", "pad_right", "hande_body", "camera_mount", "table", *SUPPORT_BODIES)}
        self.bid = {n: m.body(n).id for n in ("pen", *SUPPORT_BODIES)}
        self.gripper_geoms = np.array([self.gid[n] for n in ("hande_body", "camera_mount", "pad_left", "pad_right")])
        self.obstacle_geoms = np.array([self.gid[n] for n in ("table", *SUPPORT_BODIES)])
        self._is_pressing = np.zeros(m.ngeom, bool)
        self._is_pressing[np.append(self.gripper_geoms, self.gid["pen"])] = True
        self._is_obstacle = np.zeros(m.ngeom, bool)
        self._is_obstacle[self.obstacle_geoms] = True

        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (21,), np.float32)
        self._renderer = None
        self._camera = None
        self.view_center = None
        self.ink = []

    def clear_supports(self):
        """Move the supports out of the way, e.g. to free the paper once the pen is lifted."""
        for k, name in enumerate(SUPPORT_BODIES):
            self._place(name, (5.0, float(k), 0.0), (1, 0, 0, 0))
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for _ in range(20):
            if self._randomize(self.np_random) and self._settle():
                break
        else:
            raise RuntimeError("could not place a stable pen in 20 tries")
        self.t = 0
        self.hold = 0
        self.prev_action = np.zeros(5)
        self.queued = np.zeros(5)
        self.latency = self.np_random.random() < self.cfg.latency_prob
        return self._obs(), self._info(False, False)

    def _set_box(self, name: str, half: tuple):
        g = self.gid[name]
        self.model.geom_size[g] = half
        self.model.geom_aabb[g] = (0, 0, 0, *half)
        self.model.geom_rbound[g] = np.linalg.norm(half)

    def _set_capsule(self, name: str, radius: float, half_len: float):
        g = self.gid[name]
        self.model.geom_size[g, :2] = (radius, half_len)
        self.model.geom_aabb[g] = (0, 0, 0, radius, radius, half_len + radius)
        self.model.geom_rbound[g] = half_len + radius

    def _place(self, name: str, pos, quat):
        self.model.body_pos[self.bid[name]] = pos
        self.model.body_quat[self.bid[name]] = quat

    def _randomize(self, rng: np.random.Generator) -> bool:
        cfg, m, d = self.cfg, self.model, self.data
        u = lambda r: rng.uniform(*r)  # noqa: E731

        kinds = list(cfg.support_probs)
        probs = np.array([cfg.support_probs[k] for k in kinds], dtype=float)
        kind = kinds[rng.choice(len(kinds), p=probs / probs.sum())]

        radius, length, mass = u(cfg.pen_radius), u(cfg.pen_length), u(cfg.pen_mass)
        half = length / 2
        # Capsule, not cylinder: MuJoCo's cylinder-box contacts misbehave when a pad pinches the rim.
        self._set_capsule("pen", radius, half - radius)
        m.geom_pos[self.gid["pen_tip"]] = (0, 0, half - 0.002)
        m.site_pos[self.tip_sid] = (0, 0, half)
        b = self.bid["pen"]
        m.body_mass[b] = mass
        # Floors keep a few-gram pen numerically stable when squeezed; they slow its rolling slightly.
        i_axial = max(mass * radius**2 / 2, 1e-6)
        i_trans = max(mass * (3 * radius**2 + length**2) / 12, 1e-5)
        m.body_inertia[b] = (i_trans, i_trans, i_axial)
        m.body_ipos[b] = 0
        m.body_iquat[b] = (1, 0, 0, 0)

        xy = rng.uniform(cfg.pen_xy_lo, cfg.pen_xy_hi)
        theta = rng.uniform(-np.pi, np.pi)
        axis = np.array([np.cos(theta), np.sin(theta), 0.0])
        q_yaw = quat_z(theta)
        for k, name in enumerate(SUPPORT_BODIES):
            self._place(name, (5.0, float(k), 0.0), (1, 0, 0, 0))

        if kind == "post":
            top, width, depth = u(cfg.support_height), u(cfg.post_width), u(cfg.post_depth)
            self._set_box("post", (width / 2, depth / 2, top / 2))
            self._place("post", (*xy, top / 2), q_yaw)
            offset = width / 2 + PAD[1] + cfg.finger_clearance
        elif kind == "bridge":
            top, depth = u(cfg.support_height), u(cfg.post_depth)
            span = min(u(cfg.bridge_half_span), half - 0.012)
            for sign, name in ((1, "bridge_a"), (-1, "bridge_b")):
                self._set_box(name, (0.006, depth / 2, top / 2))
                self._place(name, (*(xy + sign * span * axis[:2]), top / 2), q_yaw)
            offset = 0.0
        else:
            r_bottle = u(cfg.bottle_radius)
            top = 2 * r_bottle
            self._set_capsule("bottle", r_bottle, 0.10)
            q_across = quat_mul(q_yaw, np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0]))
            self._place("bottle", (*xy, r_bottle), q_across)
            # Fingertips reach 12 mm below the pen axis; slide out along the pen until the bottle has
            # dropped away by that much (plus 3 mm) under the inner edge of the finger.
            drop = 0.015 - radius
            offset = np.sqrt(max(2 * r_bottle * drop - drop**2, 0.0)) + PAD[1] + 0.002

        m.geom_friction[self.gid["pen"], 0] = u(cfg.pen_friction)
        for pad in ("pad_left", "pad_right"):
            m.geom_friction[self.gid[pad], 0] = u(cfg.pad_friction)
        force = u(cfg.grip_force)
        m.actuator_forcerange[self.grip_act] = (-force, force)

        mujoco.mj_resetData(m, d)
        center = np.array([*xy, top + radius])
        pen_quat = quat_mul(q_yaw, np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))
        d.qpos[self.pen_qadr:self.pen_qadr + 3] = center
        d.qpos[self.pen_qadr + 3:self.pen_qadr + 7] = pen_quat

        start = np.array([*(center[:2] + rng.uniform(-cfg.start_xy_spread, cfg.start_xy_spread, 2)),
                          center[2] + u(cfg.start_above)])
        start = np.clip(start, cfg.target_lo, cfg.target_hi)
        yaw = rng.uniform(-np.pi, np.pi)
        q = self.ik.solve(Q_SEED, start, down_rotation(yaw), iters=200, tol=1e-7)
        if np.linalg.norm(self.ik.fk(q)[0] - start) > 2e-3:
            return False
        d.qpos[self.ik.qadr] = q
        d.ctrl[self.arm_act] = q
        d.qpos[self.finger_qadr] = MAX_OPEN
        d.ctrl[self.grip_act] = GRIP_SPEED
        mujoco.mj_forward(m, d)

        self.q_cmd = q
        self.target_pos = start
        self.target_yaw = yaw
        self.kind = kind
        self.grasp_offset = offset

        err_pos = rng.normal(0, cfg.pos_noise, 3) if cfg.pos_noise else np.zeros(3)
        self.est_theta = theta + (rng.normal(0, cfg.yaw_noise) if cfg.yaw_noise else 0.0)
        est_axis = np.array([np.cos(self.est_theta), np.sin(self.est_theta), 0.0])
        self.est_center = center + err_pos
        self.est_grasp = self.est_center - offset * est_axis
        size_err = rng.normal(0, cfg.size_noise, 2) if cfg.size_noise else np.zeros(2)
        self.est_size = np.array([radius, length]) + size_err
        return True

    def _settle(self) -> bool:
        m, d = self.model, self.data
        z_start = d.qpos[self.pen_qadr + 2]
        for _ in range(150):
            mujoco.mj_step(m, d)
        pen_axis = d.xmat[self.bid["pen"]].reshape(3, 3)[:, 2]
        ok = abs(d.qpos[self.pen_qadr + 2] - z_start) < 0.003 and abs(pen_axis[2]) < 0.05
        self.pen_z0 = d.qpos[self.pen_qadr + 2]
        return bool(ok)

    # ------------------------------------------------------------------- step

    def step(self, action):
        cfg, m, d = self.cfg, self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied = self.queued if self.latency else a
        self.queued = a

        self.target_pos = np.clip(self.target_pos + applied[:3] * cfg.max_dpos, cfg.target_lo, cfg.target_hi)
        self.target_yaw = (self.target_yaw + applied[3] * cfg.max_dyaw + np.pi) % (2 * np.pi) - np.pi
        stopped = self.servo(self.target_pos, down_rotation(self.target_yaw), applied[4])
        self.t += 1

        if self._unstable():
            # MuJoCo has already reset the state; end the episode without a learning signal.
            info = self._info(False, False)
            info["unstable"] = True
            return self._obs(), 0.0, False, True, info

        reward, success, dropped = self._reward(a)
        self.prev_action = a
        if stopped:
            reward -= 5.0
            success = False
        terminated = success or dropped or stopped
        truncated = self.t >= cfg.episode_steps
        info = self._info(success, dropped)
        info["stopped"] = stopped
        return self._obs(), reward, terminated, truncated, info

    def servo(self, pos: np.ndarray, rot: np.ndarray, grip: float) -> bool:
        """One control period towards a TCP pose, like servoL. Returns True on a protective stop."""
        m, d = self.model, self.data
        q_new = self.ik.solve(self.q_cmd, pos, rot)
        d.ctrl[self.grip_act] = grip * GRIP_SPEED
        stopped = False
        for k in range(self.n_sub):
            d.ctrl[self.arm_act] = self.q_cmd + (k + 1) / self.n_sub * (q_new - self.q_cmd)
            mujoco.mj_step(m, d)
            if self._unstable():
                break
            if self._protective_stop():
                stopped = True
                break
        self.q_cmd = q_new
        return stopped

    def _protective_stop(self) -> bool:
        """Like the UR protective stop: the gripper or the pen pressed hard against the table or a
        support, or the pen knocked away. Without it the arm crushes the pen and the solver blows up."""
        d = self.data
        v = d.qvel[self.pen_dof:self.pen_dof + 3]
        w = d.qvel[self.pen_dof + 3:self.pen_dof + 6]
        if v @ v > 4.0 or w @ w > 1e4:
            return True
        n = d.ncon
        if n == 0:
            return False
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        pressing = (self._is_pressing[g1] & self._is_obstacle[g2]) | (self._is_pressing[g2] & self._is_obstacle[g1])
        adr = d.contact.efc_address[:n][pressing]
        adr = adr[adr >= 0]
        # With elliptic cones the first constraint row of a contact is its normal force.
        return bool(adr.size and d.efc_force[adr].max() > self.cfg.stop_force)

    def _unstable(self) -> bool:
        w = self.data.warning
        return any(w[i].number > 0 for i in (mujoco.mjtWarning.mjWARN_BADQACC,
                                             mujoco.mjtWarning.mjWARN_BADQVEL,
                                             mujoco.mjtWarning.mjWARN_BADQPOS))

    def contacts(self) -> tuple[bool, bool, bool]:
        d = self.data
        g1 = d.contact.geom1[:d.ncon]
        g2 = d.contact.geom2[:d.ncon]
        pen = self.gid["pen"]

        def touching(a, b):
            return bool(np.any(((g1 == a) & (g2 == b)) | ((g1 == b) & (g2 == a))))

        left = touching(pen, self.gid["pad_left"])
        right = touching(pen, self.gid["pad_right"])
        hit = bool(np.any((np.isin(g1, self.gripper_geoms) & np.isin(g2, self.obstacle_geoms))
                          | (np.isin(g2, self.gripper_geoms) & np.isin(g1, self.obstacle_geoms))))
        return left, right, hit

    def pen_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        return d.qpos[self.pen_qadr:self.pen_qadr + 3].copy(), d.xmat[self.bid["pen"]].reshape(3, 3)[:, 2].copy()

    def _reward(self, a: np.ndarray) -> tuple[float, bool, bool]:
        cfg, d = self.cfg, self.data
        tcp = d.site_xpos[self.tcp_sid]
        tool_y = d.site_xmat[self.tcp_sid].reshape(3, 3)[:, 1]
        pen_pos, pen_axis = self.pen_pose()
        grasp = pen_pos - self.grasp_offset * pen_axis
        dist = np.linalg.norm(tcp - grasp)
        left, right, hit = self.contacts()
        grasped = left and right
        lift = pen_pos[2] - self.pen_z0

        reward = 1.0 - np.tanh(10.0 * dist)
        # The tip must end up on the side opposite the wrist camera (TCP -y), or the camera mount hits the
        # paper when the pen is turned tip-down to write.
        reward += 0.25 * (1.0 - tool_y @ pen_axis) * (dist < 0.06)
        reward += 0.5 * grasped
        reward += 2.0 * np.clip(lift / cfg.lift_height, 0.0, 1.0) * grasped
        reward -= 0.2 * hit
        reward -= 0.01 * float(a @ a)

        tip_away = tool_y @ pen_axis < -0.5
        self.hold = self.hold + 1 if (grasped and tip_away and lift > cfg.lift_height) else 0
        success = self.hold >= cfg.hold_steps
        dropped = (not grasped) and lift < -0.03
        if success:
            # Pay the best per-step reward for every step left: ending the episode on a success must
            # never earn less than hovering next to the pen until the time limit.
            reward += SUCCESS_PER_STEP * (cfg.episode_steps - self.t + 1)
        if dropped:
            reward -= 5.0
        return float(reward), bool(success), bool(dropped)

    # ---------------------------------------------------------------- outputs

    def tcp_yaw(self) -> float:
        x = self.data.site_xmat[self.tcp_sid].reshape(3, 3)[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def _obs(self) -> np.ndarray:
        d = self.data
        tcp = d.site_xpos[self.tcp_sid]
        yaw = self.tcp_yaw()
        rel = yaw - (self.est_theta - np.pi / 2)
        opening = d.qpos[self.finger_qadr[0]] / MAX_OPEN
        obs = np.concatenate([
            tcp - OBS_CENTER,
            [np.cos(yaw), np.sin(yaw)],
            [opening],
            self.est_grasp - tcp,
            [np.cos(self.est_theta), np.sin(self.est_theta)],
            [np.cos(rel), np.sin(rel)],
            [self.est_size[0] * 100, self.est_size[1] * 10],
            self.prev_action,
            [self.t / self.cfg.episode_steps],
        ])
        return obs.astype(np.float32)

    def _info(self, success: bool, dropped: bool) -> dict:
        return {"is_success": success, "dropped": dropped, "support": self.kind}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640, max_geom=20000)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = (0.48, 0.0, 0.08)
        cam = self._camera
        if self.view_center is None:
            cam.lookat[:2], cam.lookat[2] = self.est_center[:2], 0.08
            cam.distance, cam.azimuth, cam.elevation = 0.8, 150, -22
        else:
            # From the side of the paper: the wrist is over the far edge, the base behind.
            cam.lookat[:2], cam.lookat[2] = self.view_center, 0.0
            cam.distance, cam.azimuth, cam.elevation = 0.5, 90, -62
        self._renderer.update_scene(self.data, cam)
        scene = self._renderer.scene
        for p in self.ink:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([0.0013, 0.0002, 0.0]), np.array([p[0], p[1], 0.0002]),
                                np.eye(3).ravel(), np.array([0.05, 0.12, 0.45, 1.0], np.float32))
            scene.ngeom += 1
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
