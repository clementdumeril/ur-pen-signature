"""Full demo in simulation: grasp the pen (scripted or learned), turn it tip-down, sign, and score the result.

python scripts/draw_signature.py --strokes examples/signature_demo.json --video results/signature.mp4
python scripts/draw_signature.py --policy runs/ppo_v1/final_model.zip --strokes examples/my_signature.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pensig.config import EnvConfig  # noqa: E402
from pensig.draw import Drawer  # noqa: E402
from pensig.env import PenGraspEnv  # noqa: E402
from pensig.residual import ResidualGraspEnv  # noqa: E402
from pensig.scripted import ScriptedGrasp  # noqa: E402
from pensig.signature import compare, fit_to_box, load_strokes, to_page  # noqa: E402


def grasp(env, policy, seed: int, frames: list | None, max_tries: int = 10) -> int:
    for attempt in range(max_tries):
        obs, _ = env.reset(seed=seed + attempt)
        policy.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            done = term or trunc
            if frames is not None:
                frames.append(env.render())
        if info["is_success"]:
            return attempt + 1
    raise RuntimeError(f"no successful grasp in {max_tries} tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=Path, default=Path("examples/signature_demo.json"))
    ap.add_argument("--policy", type=Path, help="trained PPO model; scripted grasp if omitted")
    ap.add_argument("--residual", action="store_true", help="the policy is a correction on the scripted grasp")
    ap.add_argument("--width", type=float, default=0.08, help="signature width on paper, m")
    ap.add_argument("--paper", type=float, nargs=2, default=(0.42, -0.27), help="signature center x y, m")
    ap.add_argument("--force", type=float, default=1.5, help="pen force, N")
    ap.add_argument("--speed", type=float, default=0.03, help="drawing speed, m/s")
    ap.add_argument("--grip-force", type=float, default=40.0, help="gripper force while drawing, N")
    ap.add_argument("--pad-torsion", type=float, default=0.03, help="torsional friction of the pads, m")
    ap.add_argument("--noslip", type=int, default=10, help="MuJoCo noslip iterations while drawing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/signature"))
    ap.add_argument("--video", type=Path)
    args = ap.parse_args()

    env_cls = ResidualGraspEnv if args.residual else PenGraspEnv
    env = env_cls(EnvConfig(), render_mode="rgb_array" if args.video else None)
    if args.policy:
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import load_policy
        policy = load_policy(args.policy)
    else:
        scripted = ScriptedGrasp()
        policy = lambda env, obs: scripted(env)  # noqa: E731
        policy.reset = scripted.reset

    frames = [] if args.video else None
    tries = grasp(env, policy, args.seed, frames)
    env.clear_supports()

    # Ballpoint on paper slides more easily than the rubber-like default.
    env.model.geom_friction[env.gid["pen"], 0] = 0.3
    env.model.geom_friction[env.gid["table"], 0] = 0.3
    # Squeeze harder once the pen is lifted (the Hand-E goes up to 185 N), with grooved pads that
    # resist the pen pivoting under the paper drag.
    env.model.actuator_forcerange[env.grip_act] = (-args.grip_force, args.grip_force)
    for pad in ("pad_left", "pad_right"):
        env.model.geom_friction[env.gid[pad], 1] = args.pad_torsion
    # MuJoCo friction is slightly elastic and lets a held pen creep under sustained drag; noslip removes it.
    env.model.opt.noslip_iterations = args.noslip
    for _ in range(4):
        env.servo(env.data.site_xpos[env.tcp_sid].copy(), env.data.site_xmat[env.tcp_sid].reshape(3, 3).copy(), -1.0)
    reference = fit_to_box(load_strokes(args.strokes), args.width, args.paper)
    env.view_center = np.asarray(args.paper)
    on_step = (lambda: frames.append(env.render())) if frames is not None else None
    out = Drawer(env, force=args.force, speed=args.speed, on_step=on_step).run(reference)

    score = compare(out["ink"], reference)
    score.update({"grasp_tries": tries, "mean_force_N": float(out["forces"].mean()) if len(out["forces"]) else None,
                  "protective_stop": out["stopped"]})
    print(json.dumps(score, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(score, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for s in reference:
        page = to_page(s, args.paper) * 1e3
        ax.plot(page[:, 0], page[:, 1], color="0.75", lw=6, solid_capstyle="round", zorder=1)
    if len(out["ink"]):
        page = to_page(out["ink"], args.paper) * 1e3
        ax.scatter(page[:, 0], page[:, 1], s=4, color="#1f4fbf", zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    err = score["mean_error_mm"]
    ax.set_title(f"reference (grey) vs ink (blue): mean error {err:.2f} mm, coverage {score['coverage']:.0%}"
                 if err is not None else "no ink")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print("figure:", f"{args.out}.png")

    if args.video:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=env.metadata["render_fps"])
        print("video:", args.video)


if __name__ == "__main__":
    main()
