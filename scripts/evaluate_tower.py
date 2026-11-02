"""Tower building: success of each placement, and of the full tower with the two placements chained.

python scripts/evaluate_tower.py --scripted --episodes 200
python scripts/evaluate_tower.py --residual --policy runs/tower/final_model.zip --episodes 200 --video results/tower.mp4
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_policy  # noqa: E402
from pensig.config import TowerConfig  # noqa: E402
from pensig.residual import ResidualTowerEnv  # noqa: E402
from pensig.tower import TowerEnv  # noqa: E402
from pensig.tower_scripted import ScriptedStack  # noqa: E402


def run_stage(env, policy, obs, frames):
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy(env, obs))
        done = term or trunc
        if frames is not None:
            frames.append(env.render())
    return info


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scripted", action="store_true")
    src.add_argument("--policy", type=Path)
    ap.add_argument("--residual", action="store_true", help="the policy is a correction on the scripted stack")
    ap.add_argument("--episodes", type=int, default=200, help="full towers to build")
    ap.add_argument("--noiseless", action="store_true", help="exact block poses, no latency")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the pose estimate error")
    ap.add_argument("--no-wrist", action="store_true", help="only the first camera estimate, no wrist close-up")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--video", type=Path, help="mp4 of the first --video-episodes towers")
    ap.add_argument("--video-episodes", type=int, default=3)
    ap.add_argument("--out", type=Path, help="json summary")
    args = ap.parse_args()

    cfg = TowerConfig().noiseless() if args.noiseless else TowerConfig()
    cfg.pos_noise *= args.noise_scale
    cfg.yaw_noise *= args.noise_scale
    cfg.wrist_pos_noise *= args.noise_scale
    cfg.wrist_yaw_noise *= args.noise_scale
    cfg.wrist_camera = not args.no_wrist
    env = (ResidualTowerEnv if args.residual else TowerEnv)(cfg, render_mode="rgb_array" if args.video else None)

    if args.scripted:
        scripted = ScriptedStack()
        policy = lambda env, obs: scripted(env)  # noqa: E731
        policy.reset = scripted.reset
    else:
        policy = load_policy(args.policy)

    stage_ok = [[], []]
    place_err = [[], []]
    failures = Counter()
    towers, heights = [], []
    frames = []
    for ep in range(args.episodes):
        rec = frames if args.video and ep < args.video_episodes else None
        obs, _ = env.reset(seed=args.seed + ep, options={"stage": 0})
        policy.reset()
        built = 0
        for stage in (0, 1):
            if stage == 1:
                obs, _ = env.next_stage()
                policy.reset()
            info = run_stage(env, policy, obs, rec)
            stage_ok[stage].append(info["is_success"])
            if info["is_success"]:
                place_err[stage].append(env.place_err)
                built += 1
            else:
                failures[f"stage{stage}:" + (info["failure"] or ("stop" if info.get("stopped") else "timeout"))] += 1
                break
        towers.append(built == 2)
        heights.append(env.tower_height())

    summary = {
        "source": "scripted" if args.scripted else str(args.policy), "towers": args.episodes,
        "noiseless": args.noiseless, "noise_scale": args.noise_scale, "wrist_camera": cfg.wrist_camera,
        "full_tower": float(np.mean(towers)),
        "stage0_success": float(np.mean(stage_ok[0])),
        "stage1_success": float(np.mean(stage_ok[1])) if stage_ok[1] else None,
        "place_err_mm": [round(1000 * float(np.mean(e)), 2) if e else None for e in place_err],
        "mean_height_cm": round(100 * float(np.mean(heights)), 2),
        "failures": dict(failures.most_common()),
    }
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
