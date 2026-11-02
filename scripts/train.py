"""Train the pen-grasp or tower policy with PPO on parallel MuJoCo environments.

python scripts/train.py --timesteps 8000000 --out runs/ppo
python scripts/train.py --task tower --residual --timesteps 4000000 --out runs/tower
tensorboard --logdir runs
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pensig.config import EnvConfig, TowerConfig  # noqa: E402
from pensig.env import PenGraspEnv  # noqa: E402
from pensig.residual import ResidualGraspEnv, ResidualTowerEnv  # noqa: E402
from pensig.tower import TowerEnv  # noqa: E402

TASKS = {"pen": (EnvConfig, PenGraspEnv, ResidualGraspEnv), "tower": (TowerConfig, TowerEnv, ResidualTowerEnv)}


def make_env(rank: int, seed: int, residual: bool, task: str = "pen"):
    def thunk():
        cfg_cls, env_cls, residual_cls = TASKS[task]
        env = (residual_cls if residual else env_cls)(cfg_cls())
        env.reset(seed=seed + 1000 * rank)
        return env
    return thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=8_000_000)
    ap.add_argument("--n-envs", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=Path("runs/ppo"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=list(TASKS), default="pen")
    ap.add_argument("--residual", action="store_true", help="learn a correction on top of the scripted controller")
    ap.add_argument("--resume", type=Path, help="model zip to continue from (vecnormalize.pkl next to it)")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"env": asdict(TASKS[args.task][0]()), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2, default=str))

    venv = VecMonitor(SubprocVecEnv([make_env(i, args.seed, args.residual, args.task) for i in range(args.n_envs)]),
                      info_keywords=("is_success", "dropped", "stopped"))
    if args.resume:
        venv = VecNormalize.load(str(args.resume.parent / "vecnormalize.pkl"), venv)
        model = PPO.load(args.resume, env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.98)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=512, batch_size=1024, n_epochs=8,
            learning_rate=3e-4, gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.2 if args.residual else -0.7),
            tensorboard_log=str(args.out), seed=args.seed, device="cpu", verbose=1,
        )

    class SaveNormalize(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            venv.save(str(args.out / "vecnormalize.pkl"))

    checkpoints = CheckpointCallback(save_freq=max(500_000 // args.n_envs, 1), save_path=str(args.out / "checkpoints"),
                                     name_prefix="ppo", save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoints, SaveNormalize()],
                reset_num_timesteps=not args.resume)
    model.save(args.out / "final_model")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.close()


if __name__ == "__main__":
    main()
