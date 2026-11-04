"""Export a PPO policy and its VecNormalize statistics to plain numpy arrays for the real arm.

The lab machine (ROS 2 Humble, numpy 1.x) then needs neither torch nor stable-baselines3, and does not have
to unpickle files written with numpy 2. tower_ros.policy.NumpyPolicy reads the result.

python scripts/export_tower_policy.py runs/tower_v6
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO


def export(run: Path, out: Path):
    model = PPO.load(run / "final_model.zip", device="cpu")
    policy = model.policy
    if policy.squash_output or not isinstance(policy.activation_fn(), nn.Tanh):
        raise SystemExit("expected an unsquashed tanh MLP, as scripts/train.py builds")
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [policy.action_net]
    arrays = {f"w{k}": m.weight.detach().numpy().astype(np.float64) for k, m in enumerate(linears)}
    arrays |= {f"b{k}": m.bias.detach().numpy().astype(np.float64) for k, m in enumerate(linears)}
    with open(run / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)
    np.savez(out, n_layers=len(linears), obs_mean=norm.obs_rms.mean, obs_var=norm.obs_rms.var,
             epsilon=norm.epsilon, clip_obs=norm.clip_obs, norm_obs=norm.norm_obs, **arrays)
    print(f"{out}: {len(linears)} layers, {linears[0].in_features} inputs, {linears[-1].out_features} outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="folder with final_model.zip and vecnormalize.pkl")
    ap.add_argument("--out", type=Path, help="default: <run>/policy.npz")
    args = ap.parse_args()
    export(args.run, args.out or args.run / "policy.npz")


if __name__ == "__main__":
    main()
