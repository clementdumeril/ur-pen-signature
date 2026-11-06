"""End-to-end tower policy: PPO on TowerEnv alone, episodes started partway by the script (pensig.curriculum).

The level goes up once the policy places from the earliest start phase in half of the last 200 such episodes.
Block sizes are random at every reset. At the end, full towers from the plain reset, the two stages chained and
no script anywhere, on the measured lab blocks and on random ones.

python scripts/train_tower_curriculum.py --out runs/tower_cur --timesteps 20000000
python scripts/train_tower_curriculum.py --resume runs/tower_cur --out runs/tower_cur2   # newest checkpoint, its level
python scripts/train_tower_curriculum.py --eval-only runs/tower_cur/final_model.zip
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_policy  # noqa: E402
from pensig.config import TowerConfig  # noqa: E402
from pensig.curriculum import MAX_LEVEL, ScriptStart, frontier, random_sizes  # noqa: E402
from pensig.tower import TowerEnv  # noqa: E402

MEASURED = {"base_size": (0.05, 0.05), "mid_size": (0.04, 0.08), "top_size": (0.035, 0.03)}


def make_env(rank: int, seed: int, level: int):
    def thunk():
        env = ScriptStart(TowerEnv(TowerConfig()), seed=seed + 1000 * rank)
        env.set_level(level)
        env.reset(seed=seed + 1000 * rank)
        return env
    return thunk


def resume_from(run: Path):
    """Newest checkpoint of a run (or its final model), its VecNormalize stats, and the level it had reached."""
    final = run / "final_model.zip"
    ckpts = sorted((run / "checkpoints").glob("ppo_*_steps.zip"), key=lambda f: int(f.stem.split("_")[1]))
    if final.exists():
        zip_, stats = final, run / "vecnormalize.pkl"
    elif ckpts:
        n = ckpts[-1].stem.split("_")[1]
        zip_, stats = ckpts[-1], run / "checkpoints" / f"ppo_vecnormalize_{n}_steps.pkl"
    else:
        raise FileNotFoundError(f"no model in {run}")
    log = json.loads((run / "curriculum.json").read_text()) if (run / "curriculum.json").exists() else []
    return zip_, stats, (log[-1]["level"] if log else 0)


def evaluate(path: Path, towers: int, seed: int = 10_000) -> dict:
    """Full towers from the plain reset, the policy alone, on the measured blocks and on random ones."""
    policy = load_policy(path)
    env = TowerEnv(TowerConfig())
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("measured", "random"):
        built, stage_ok, failures = [], [[], []], {}
        for ep in range(towers):
            for k, v in (MEASURED if mode == "measured" else random_sizes(rng)).items():
                setattr(env.cfg, k, v)
            obs, _ = env.reset(seed=seed + ep, options={"stage": 0})
            n = 0
            for stage in (0, 1):
                if stage == 1:
                    obs, _ = env.next_stage()
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(policy(env, obs))
                    done = term or trunc
                stage_ok[stage].append(bool(info["is_success"]))
                if not info["is_success"]:
                    why = f"stage{stage}:" + (info["failure"] or ("stop" if info.get("stopped") else "timeout"))
                    failures[why] = failures.get(why, 0) + 1
                    break
                n += 1
            built.append(n == 2)
        out[mode] = {"full_tower": float(np.mean(built)), "stage0": float(np.mean(stage_ok[0])),
                     "stage1": float(np.mean(stage_ok[1])) if stage_ok[1] else None, "failures": failures}
        print(f"eval {mode} blocks: {out[mode]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/tower_cur"))
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=os.cpu_count())
    ap.add_argument("--level", type=int, help="curriculum level to start at (default 0, or the resumed run's)")
    ap.add_argument("--resume", type=Path, help="run directory to continue from: newest checkpoint and its level")
    ap.add_argument("--threshold", type=float, default=0.5, help="success from the earliest start to level up")
    ap.add_argument("--window", type=int, default=200, help="episodes from the earliest start that count")
    ap.add_argument("--log-std-init", type=float, default=-1.2)
    ap.add_argument("--max-minutes", type=float, help="stop training after this long, then save and evaluate")
    ap.add_argument("--eval-towers", type=int, default=100)
    ap.add_argument("--eval-only", type=Path, help="model zip: only evaluate it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.eval_only:
        evaluate(args.eval_only, args.eval_towers)
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"env": asdict(TowerConfig()), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2, default=str))

    resume = resume_from(args.resume) if args.resume else None
    if args.level is None:
        args.level = resume[2] if resume else 0
    venv = SubprocVecEnv([make_env(i, args.seed, args.level) for i in range(args.n_envs)])
    if resume:
        print(f"resuming from {resume[0]} at level {args.level}", flush=True)
        venv = VecNormalize.load(str(resume[1]), venv)
        model = PPO.load(resume[0], env=venv, device="cpu")
        model.tensorboard_log = str(args.out)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.98)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=8, learning_rate=3e-4, gamma=0.98,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                    tensorboard_log=str(args.out),
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=args.log_std_init))

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.level, self.front, self.full = args.level, deque(maxlen=args.window), deque(maxlen=args.window)
            self.all, self.log, self.t0 = deque(maxlen=1000), [], time.time()

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if not done:
                    continue
                ok, p = bool(info.get("is_success")), info.get("start_phase", -1)
                self.all.append(ok)
                if p == -1:
                    self.full.append(ok)
                if p == frontier(min(self.level, MAX_LEVEL - 1)):
                    self.front.append(ok)
            if (self.level < MAX_LEVEL and len(self.front) == args.window
                    and np.mean(self.front) >= args.threshold):
                self.level += 1
                self.front.clear()
                venv.env_method("set_level", self.level)
                self._print(f"level up -> {self.level}")
            return not args.max_minutes or time.time() - self.t0 < 60 * args.max_minutes

        def _print(self, what=""):
            mean = lambda q: round(float(np.mean(q)), 3) if q else None  # noqa: E731
            entry = {"steps": self.num_timesteps, "level": self.level, "earliest_start_success": mean(self.front),
                     "all_success": mean(self.all), "full_start_success": mean(self.full),
                     "minutes": round((time.time() - self.t0) / 60, 1)}
            self.log.append(entry)
            print(json.dumps(entry), what, flush=True)
            (args.out / "curriculum.json").write_text(json.dumps(self.log, indent=1))

        def _on_rollout_end(self) -> None:
            if self.n_calls % (20 * 512) < 512:  # about every 20 rollouts
                self._print()
            venv.save(str(args.out / "vecnormalize.pkl"))

    cur = Curriculum()
    checkpoints = CheckpointCallback(save_freq=max(1_000_000 // args.n_envs, 1),
                                     save_path=str(args.out / "checkpoints"), name_prefix="ppo",
                                     save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoints, cur], reset_num_timesteps=not resume)
    cur._print("end of training")
    model.save(args.out / "final_model")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.close()

    result = {"curriculum": cur.log[-1], "eval": evaluate(args.out / "final_model.zip", args.eval_towers)}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
