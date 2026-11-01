"""After the grasp: turn the pen tip-down and trace strokes on the table, regulating the pen force.

On the real UR the force loop is force_mode along z; here an admittance loop on the measured
pen-table contact force plays that role.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pensig.signature import resample


def pen_vertical_rotation(tip_along_plus_y: bool) -> np.ndarray:
    """TCP frame with the pen (the TCP y axis after a top grasp) vertical, tip down, and the gripper
    pointing back at the base (-x). Pointing it forward drives the forearm and wrist into the table."""
    s = 1.0 if tip_along_plus_y else -1.0
    return np.column_stack([(0.0, s, 0.0), (0.0, 0.0, -s), (-1.0, 0.0, 0.0)])


class Drawer:
    def __init__(self, env, force: float = 1.5, speed: float = 0.03, hover: float = 0.015,
                 gain: float = 0.0004, on_step=None):
        self.env = env
        self.force = force
        self.speed = speed
        self.hover = hover
        self.gain = gain
        self.on_step = on_step or (lambda: None)

    def tip_contact(self) -> tuple[float, np.ndarray | None]:
        e = self.env
        d = e.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        pen, table = e.gid["pen"], e.gid["table"]
        idx = np.flatnonzero(((g1 == pen) & (g2 == table)) | ((g2 == pen) & (g1 == table)))
        adr = d.contact.efc_address[idx]
        idx, adr = idx[adr >= 0], adr[adr >= 0]
        if not idx.size:
            return 0.0, None
        return float(d.efc_force[adr].sum()), d.contact.pos[idx].mean(0)

    def _servo(self, pos, rot) -> bool:
        stopped = self.env.servo(pos, rot, -1.0)
        self.on_step()
        return stopped

    def _tcp_for_tip(self, tip) -> np.ndarray:
        return np.asarray(tip, float) - self.rot @ self.tip_local

    def _move(self, pos_goal, rot_goal, steps: int):
        d = self.env.data
        p0 = d.site_xpos[self.env.tcp_sid].copy()
        r0 = d.site_xmat[self.env.tcp_sid].reshape(3, 3).copy()
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, rot_goal])))
        for i in range(1, steps + 1):
            a = i / steps
            self._servo(p0 + a * (np.asarray(pos_goal) - p0), slerp([a]).as_matrix()[0])

    def run(self, strokes: list[np.ndarray], paper_z: float = 0.0) -> dict:
        e = self.env
        d = e.data
        tcp = d.site_xpos[e.tcp_sid].copy()
        r_tcp = d.site_xmat[e.tcp_sid].reshape(3, 3).copy()
        # Where the tip sits in the gripper; on the real arm this comes from the grasp point and a touch-off.
        self.tip_local = r_tcp.T @ (d.site_xpos[e.tip_sid] - tcp)
        self.rot = pen_vertical_rotation(self.tip_local[1] > 0)

        start = strokes[0][0]
        self._move(tcp + (0, 0, 0.06), r_tcp, 20)
        self._move(self._tcp_for_tip((*start, paper_z + 0.06)), self.rot, 70)

        ink, forces, stopped = [], [], False
        e.ink = ink
        step = self.speed * e.cfg.control_dt
        for stroke in strokes:
            pts = resample(stroke, step)
            self._move(self._tcp_for_tip((*pts[0], paper_z + self.hover)), self.rot, 12)
            z = paper_z + 0.002
            for _ in range(60):
                z -= 0.0005
                stopped |= self._servo(self._tcp_for_tip((*pts[0], z)), self.rot)
                if self.tip_contact()[0] > 0.5 * self.force:
                    break
            for p in pts:
                f, pos = self.tip_contact()
                z = float(np.clip(z + self.gain * (f - self.force), paper_z - 0.004, paper_z + 0.003))
                stopped |= self._servo(self._tcp_for_tip((*p, z)), self.rot)
                f, pos = self.tip_contact()
                if pos is not None and f > 0.2:
                    ink.append(pos[:2].copy())
                    forces.append(f)
            self._move(self._tcp_for_tip((*pts[-1], paper_z + self.hover)), self.rot, 6)
        return {"ink": np.array(ink).reshape(-1, 2), "forces": np.array(forces), "stopped": bool(stopped)}
