import mujoco
import numpy as np

from pensig.scene import ARM_JOINTS

# Shoulder pan turned so the gripper points down over the workspace (x > 0).
Q_SEED = np.array([-np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def down_rotation(yaw: float) -> np.ndarray:
    """TCP frame pointing down; its x axis (the closing axis) at angle yaw in the table plane."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


class DiffIK:
    """Damped least-squares IK on the six arm joints, for the TCP site. Runs on its own MjData."""

    def __init__(self, model: mujoco.MjModel, site: str = "tcp", damping: float = 0.03):
        self.m = model
        self.d = mujoco.MjData(model)
        ids = [model.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([model.jnt_dofadr[i] for i in ids])
        self.lo = model.jnt_range[ids, 0]
        self.hi = model.jnt_range[ids, 1]
        self.sid = model.site(site).id
        self.lam2 = damping**2
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 3,
              tol: float = 1e-4) -> np.ndarray:
        q = q0.copy()
        for _ in range(iters):
            p, r = self.fk(q)
            err = np.concatenate([pos - p, 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            jac = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.lam2 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q
