import numpy as np
import pytest

from pensig.config import EnvConfig
from pensig.env import PenGraspEnv
from pensig.ik import Q_SEED, DiffIK, down_rotation
from pensig.scripted import ScriptedGrasp


@pytest.fixture(scope="module")
def env():
    return PenGraspEnv(EnvConfig().noiseless())


def test_ik_reaches_workspace(env):
    ik = DiffIK(env.model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = rng.uniform((0.35, -0.2, 0.05), (0.65, 0.2, 0.35))
        yaw = rng.uniform(-np.pi, np.pi)
        q = ik.solve(Q_SEED, pos, down_rotation(yaw), iters=200, tol=1e-8)
        p, r = ik.fk(q)
        assert np.linalg.norm(p - pos) < 1e-3
        assert np.degrees(np.arccos(np.clip((np.trace(r.T @ down_rotation(yaw)) - 1) / 2, -1, 1))) < 0.5


def test_pen_rests_on_support(env):
    for seed in range(15):
        env.reset(seed=seed)
        z0 = env.pen_pose()[0][2]
        for _ in range(20):
            env.step(np.zeros(5))
        pos, axis = env.pen_pose()
        assert abs(pos[2] - z0) < 0.003, env.kind
        assert abs(axis[2]) < 0.1, env.kind


def test_step_api(env):
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs)) and np.isfinite(r)
    assert {"is_success", "dropped", "support"} <= info.keys()


def test_scripted_grasp_without_noise(env):
    policy = ScriptedGrasp()
    wins = 0
    for seed in range(20):
        env.reset(seed=100 + seed)
        policy.reset()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(policy(env))
            done = term or trunc
        wins += info["is_success"]
    assert wins >= 16
