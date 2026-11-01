"""Success rate of the scripted grasp or of a trained policy, per support type.

python scripts/evaluate.py --scripted --episodes 200
python scripts/evaluate.py --policy runs/ppo/final_model.zip --episodes 200 --video results/grasp.mp4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pensig.config import EnvConfig  # noqa: E402
from pensig.env import PenGraspEnv  # noqa: E402
from pensig.residual import ResidualGraspEnv  # noqa: E402
from pensig.scripted import ScriptedGrasp  # noqa: E402


def load_policy(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    stats = path.parent / "vecnormalize.pkl"
    norm = None
    if stats.exists():
        import pickle

        with open(stats, "rb") as f:
            norm = pickle.load(f)

    def act(env, obs):
        if norm is not None:
            obs = np.clip((obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon),
                          -norm.clip_obs, norm.clip_obs)
        return model.predict(obs, deterministic=True)[0]

    act.reset = lambda: None
    return act


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scripted", action="store_true")
    src.add_argument("--policy", type=Path)
    ap.add_argument("--residual", action="store_true", help="the policy is a correction on the scripted grasp")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--noiseless", action="store_true", help="exact pen pose, no latency")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the pose estimate error")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--video", type=Path, help="mp4 of the first --video-episodes episodes")
    ap.add_argument("--video-episodes", type=int, default=4)
    ap.add_argument("--out", type=Path, help="json summary")
    args = ap.parse_args()

    cfg = EnvConfig().noiseless() if args.noiseless else EnvConfig()
    cfg.pos_noise *= args.noise_scale
    cfg.yaw_noise *= args.noise_scale
    env_cls = ResidualGraspEnv if args.residual else PenGraspEnv
    env = env_cls(cfg, render_mode="rgb_array" if args.video else None)

    if args.scripted:
        scripted = ScriptedGrasp()
        policy = lambda env, obs: scripted(env)  # noqa: E731
        policy.reset = scripted.reset
    else:
        policy = load_policy(args.policy)

    results = defaultdict(list)
    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
            if args.video and ep < args.video_episodes:
                frames.append(env.render())
        results[info["support"]].append((info["is_success"], info["dropped"], env.t))

    summary = {"source": "scripted" if args.scripted else str(args.policy), "episodes": args.episodes,
               "noiseless": args.noiseless, "noise_scale": args.noise_scale}
    allres = [r for v in results.values() for r in v]
    summary["success"] = float(np.mean([r[0] for r in allres]))
    summary["dropped"] = float(np.mean([r[1] for r in allres]))
    summary["per_support"] = {k: {"n": len(v), "success": float(np.mean([r[0] for r in v]))}
                              for k, v in sorted(results.items())}
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    if args.video:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=env.metadata["render_fps"])
        print("video:", args.video)


if __name__ == "__main__":
    main()
