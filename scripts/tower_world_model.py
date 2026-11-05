"""Tower with a learned world model and evolutionary planning, no policy network (PETS / V-JEPA 2-AC style).

1. Data: towers in TowerEnv with block sizes and yaws drawn at random, driven by the scripted stack plus action
   noise (the script only explores; it is not used at test time).
2. World model: an ensemble of MLPs predicting the next 29 TowerEnv values and the reward from (state, action).
3. Control: at every step, the cross-entropy method (an evolution strategy) samples action sequences, rolls them
   out in the model, keeps the best and refits; the first action of the best sequence is executed (MPC).
4. Rounds: the planner's own towers are added to the data and the model refitted, so it learns where it is wrong.
Evaluation: full towers on the measured lab blocks and on random sizes, against the script's 91.5 %.

python scripts/tower_world_model.py --out runs/tower_wm --workers 8 --towers 64 --rounds 1 --eval-towers 8
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEASURED = {"base_size": (0.05, 0.05), "mid_size": (0.04, 0.08), "top_size": (0.035, 0.03)}


def random_sizes(rng) -> dict:
    """Square blocks, (width, height): the base wider, the held ones within the 50 mm Hand-E stroke."""
    return {"base_size": (rng.uniform(0.040, 0.060), rng.uniform(0.025, 0.080)),
            "mid_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.080)),
            "top_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.060))}


# ---------------------------------------------------------------- world model

def build_model(n_in, n_out, members, width, torch):
    class Ensemble(torch.nn.Module):
        """`members` MLPs run as one batched matmul: input (E, B, n_in) -> (E, B, n_out)."""

        def __init__(self):
            super().__init__()
            dims = [n_in, width, width, width, n_out]
            self.w = torch.nn.ParameterList()
            self.b = torch.nn.ParameterList()
            for a, b in zip(dims[:-1], dims[1:]):
                self.w.append(torch.nn.Parameter(torch.randn(members, a, b) * (1.0 / a) ** 0.5))
                self.b.append(torch.nn.Parameter(torch.zeros(members, 1, b)))

        def forward(self, x):
            for k, (w, b) in enumerate(zip(self.w, self.b)):
                x = torch.baddbmm(b, x, w)
                if k < len(self.w) - 1:
                    x = torch.nn.functional.silu(x)
            return x

    return Ensemble()


class WorldModel:
    """Normalized (state, action) -> normalized (state change, reward)."""

    def __init__(self, path=None, stats=None, members=5, width=256):
        import torch

        self.torch = torch
        if path is not None:
            ck = torch.load(path, weights_only=False)
            stats, members, width = ck["stats"], ck["members"], ck["width"]
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}
        self.members, self.width = members, width
        self.net = build_model(34, 30, members, width, torch)
        if path is not None:
            self.net.load_state_dict(ck["net"])

    def save(self, path):
        self.torch.save({"net": self.net.state_dict(), "members": self.members, "width": self.width,
                         "stats": {k: v.numpy() for k, v in self.stats.items()}}, path)

    def step(self, obs, act):
        """obs (E, B, 29), act (E, B, 5) tensors -> next obs, reward."""
        s = self.stats
        x = self.torch.cat([(obs - s["obs_mu"]) / s["obs_sd"], act], -1)
        y = self.net(x) * s["out_sd"] + s["out_mu"]
        return obs + y[..., :29], y[..., 29]


def fit(model: WorldModel, data: dict, steps: int, log):
    torch = model.torch
    x = torch.as_tensor(np.concatenate([data["obs"], data["act"]], 1), dtype=torch.float32)
    y = torch.as_tensor(np.concatenate([data["next"] - data["obs"], data["rew"][:, None]], 1), dtype=torch.float32)
    n = len(x)
    perm = torch.randperm(n)
    val, train = perm[: max(1, n // 20)], perm[max(1, n // 20):]
    s = model.stats
    xn = torch.cat([(x[:, :29] - s["obs_mu"]) / s["obs_sd"], x[:, 29:]], 1)
    yn = (y - s["out_mu"]) / s["out_sd"]
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    E = model.members
    for k in range(steps):
        idx = train[torch.randint(0, len(train), (E, 512))]  # a bootstrap batch per member
        loss = ((model.net(xn[idx]) - yn[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (k + 1) % max(1, steps // 5) == 0:
            with torch.no_grad():
                v = ((model.net(xn[val].expand(E, -1, -1)) - yn[val]) ** 2).mean(1)
                log(f"  fit step {k + 1}/{steps}: train {loss.item():.4f}, validation per output "
                    f"state {v[:, :29].mean().item():.4f} reward {v[:, 29].mean().item():.4f}")


def make_stats(data):
    obs, delta = data["obs"], data["next"] - data["obs"]
    out = np.concatenate([delta, data["rew"][:, None]], 1)
    return {"obs_mu": obs.mean(0), "obs_sd": obs.std(0) + 1e-3, "out_mu": out.mean(0), "out_sd": out.std(0) + 1e-3}


# ---------------------------------------------------------------- planner

class CEM:
    """Cross-entropy method over action sequences in the world model, warm-started from the last plan.
    Each candidate is rolled out by one ensemble member, so plans that only work in one model lose out."""

    def __init__(self, model: WorldModel, horizon=12, pop=400, elites=40, iters=6, rng=None):
        self.m, self.h, self.pop, self.k, self.iters = model, horizon, pop - pop % model.members, elites, iters
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.mu = np.zeros((self.h, 5))

    def __call__(self, obs) -> np.ndarray:
        torch, E = self.m.torch, self.m.members
        mu, sd = self.mu, np.full((self.h, 5), 0.5)
        o0 = torch.as_tensor(obs, dtype=torch.float32).expand(E, self.pop // E, 29)
        with torch.no_grad():
            for _ in range(self.iters):
                a = np.clip(mu + sd * self.rng.standard_normal((self.pop, self.h, 5)), -1, 1)
                at = torch.as_tensor(a, dtype=torch.float32).reshape(E, self.pop // E, self.h, 5)
                o, ret = o0, torch.zeros(E, self.pop // E)
                for t in range(self.h):
                    o, r = self.m.step(o, at[:, :, t])
                    ret += 0.98 ** t * r
                best = np.argsort(-ret.reshape(-1).numpy())[: self.k]
                mu, sd = a[best].mean(0), a[best].std(0) + 0.05
        self.mu = np.concatenate([mu[1:], np.zeros((1, 5))])
        return mu[0]


# ---------------------------------------------------------------- workers

def run_towers(job):
    """Towers driven by the noisy script (model_path None) or by the planner. Returns transitions and results."""
    seed, n, model_path, sizes_mode, plan_kw = job
    import torch

    torch.set_num_threads(1)
    from pensig.config import TowerConfig
    from pensig.tower import TowerEnv
    from pensig.tower_scripted import ScriptedStack

    rng = np.random.default_rng(seed)
    env = TowerEnv(TowerConfig())
    script = ScriptedStack()
    planner = CEM(WorldModel(model_path), rng=rng, **plan_kw) if model_path else None
    buf = {k: [] for k in ("obs", "act", "next", "rew")}
    towers, stage0 = 0, 0
    for ep in range(n):
        sizes = MEASURED if sizes_mode == "measured" else random_sizes(rng)
        for k, v in sizes.items():
            setattr(env.cfg, k, v)
        obs, _ = env.reset(seed=int(seed * 1000 + ep), options={"stage": 0})
        noise = rng.choice([0.0, 0.1, 0.3])
        built = 0
        while True:
            script.reset()
            if planner:
                planner.reset()
            done, info = False, {}
            while not done:
                if planner:
                    a = planner(obs)
                else:
                    a = np.clip(script(env) + noise * rng.standard_normal(5), -1, 1)
                nxt, r, term, trunc, info = env.step(a)
                for key, val in zip(buf, (obs, a, nxt, r)):
                    buf[key].append(val)
                obs, done = nxt, term or trunc
            if not info.get("is_success"):
                break
            built += 1
            if env.stage == 1:
                break
            obs, _ = env.next_stage()
        towers += built == 2
        stage0 += built >= 1
    out = {k: np.array(v, np.float32) for k, v in buf.items()}
    return out, towers, stage0, n


def collect(pool, workers, towers, seed, model_path, sizes_mode, plan_kw):
    per = -(-towers // workers)
    res = pool.map(run_towers, [(seed + w, per, model_path, sizes_mode, plan_kw) for w in range(workers)])
    data = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
    n = sum(r[3] for r in res)
    return data, sum(r[1] for r in res) / n, sum(r[2] for r in res) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/tower_wm"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--towers", type=int, default=3000, help="noisy-script towers for the first model")
    ap.add_argument("--rounds", type=int, default=3, help="planner rounds added to the data after the first fit")
    ap.add_argument("--round-towers", type=int, default=128)
    ap.add_argument("--fit-steps", type=int, default=30000)
    ap.add_argument("--eval-towers", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_kw = dict(horizon=args.horizon, pop=args.pop, iters=args.iters)
    model_path = args.out / "world_model.pt"
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "rounds": []}

    def log(msg):
        print(msg, flush=True)

    def save():
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    import torch

    torch.manual_seed(args.seed)
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        data, rate, _ = collect(pool, args.workers, args.towers, args.seed * 10_000, None, "random", plan_kw)
        log(f"script data: {len(data['obs'])} steps, noisy script built {rate:.1%} ({time.time() - t0:.0f} s)")
        summary["noisy_script_towers"] = rate
        for rnd in range(args.rounds + 1):
            t0 = time.time()
            wm = WorldModel(stats=make_stats(data))
            fit(wm, data, args.fit_steps, log)
            wm.save(model_path)
            log(f"round {rnd}: model fitted on {len(data['obs'])} steps ({time.time() - t0:.0f} s)")
            if rnd == args.rounds:
                break
            t0 = time.time()
            new, rate, s0 = collect(pool, args.workers, args.round_towers, (rnd + 1) * 100_000, model_path,
                                    "random", plan_kw)
            data = {k: np.concatenate([data[k], new[k]]) for k in data}
            summary["rounds"].append({"round": rnd, "planner_towers": rate, "planner_stage0": s0})
            log(f"round {rnd}: planner built {rate:.1%} of towers, stage 0 {s0:.1%} ({time.time() - t0:.0f} s)")
            save()

        for mode in ("measured", "random"):
            t0 = time.time()
            _, rate, s0 = collect(pool, args.workers, args.eval_towers, 9_000_000, model_path, mode, plan_kw)
            summary[f"eval_{mode}"] = {"full_tower": rate, "stage0": s0, "towers": args.eval_towers}
            log(f"eval {mode} blocks: {rate:.1%} full towers, stage 0 {s0:.1%} ({time.time() - t0:.0f} s)")
            save()


if __name__ == "__main__":
    main()
