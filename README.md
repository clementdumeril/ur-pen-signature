# ur-pen-signature

A UR7e picks up a pen lying on a support with a learned grasp, turns it tip-down and signs with it. Simulation in MuJoCo first, then the EECS 106A lab cell at UC Berkeley (UR7e, Robotiq Hand-E, Logitech C922 on the wrist).

Learning where it helps, control where control is better: the grasp is a PPO policy trained under camera error and randomized pens; the signature is a Cartesian trajectory with a regulated pen force.

## Results (simulation)

Grasp success over the same 300 episodes with camera error (3 mm, 4 deg, one-step latency), and 200 with the exact pen pose:

| | Camera error | Bottle | Post | Bridge | Exact pose |
|---|---|---|---|---|---|
| Scripted grasp (baseline) | 74 % | 48 % | 75 % | 89 % | 99 % |
| PPO residual on the scripted grasp | **84 %** | 60 % | 88 % | 92 % | 93.5 % |

A success is the pen lifted 8 cm with its tip on the side away from the wrist camera. Pens are 12 to 16 cm long, 7 to 12 mm wide and 6 to 25 g, on a post, a two-post bridge or a lying bottle. The residual policy (4M steps, PPO) cuts the failures under camera error by 40 %, and gives up a few points when the pose is exact: it learned to be careful.

Signature after the learned grasp (CD monogram, 8 cm wide, 3 cm/s): mean distance from the ink to the reference 0.11 mm, 95th percentile 0.19 mm, 100 % of the reference inked, pen force 1.51 N for a 1.5 N target.

## How it works

- `pensig/scene.py`: UR5e model from MuJoCo Menagerie (the UR7e has the same kinematics), Hand-E with the lab fingers, wrist camera mount, base 0.275 m above the table as in the lab.
- `pensig/env.py`: Gymnasium environment. Action: TCP displacement, yaw, gripper. Damped least-squares IK turns the TCP target into joint commands, as servoL would. A protective stop ends the episode when the gripper or the pen presses on the table or a support, as on the real arm.
- `pensig/scripted.py`: the hand-written grasp the policy has to beat, on the same pose estimate.
- `pensig/residual.py`: the environment where PPO learns a correction on top of the scripted grasp.
- `pensig/draw.py`: turns the pen tip-down and traces the strokes; an admittance loop on the pen force stands in for force_mode.
- `scripts/`: `train.py` (PPO, Stable-Baselines3), `evaluate.py`, `draw_signature.py`, `capture_signature.py` (sign with the mouse), `fetch_assets.py`.

## Run

```
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
python scripts/fetch_assets.py
python -m pytest -q
python scripts/evaluate.py --scripted --episodes 200
python scripts/train.py --residual --timesteps 4000000 --out runs/residual
python scripts/evaluate.py --residual --policy runs/residual/final_model.zip --episodes 300
python scripts/capture_signature.py --out examples/my_signature.json
python scripts/draw_signature.py --residual --policy runs/residual/final_model.zip --strokes examples/my_signature.json --video results/signature.mp4
```

## What the simulation taught me

- MuJoCo contacts are stiff in proportion to the object's mass: a 10 g pen squeezed at 40 N sinks millimeters into the pads. The grasp uses 8 to 20 N; the drawing phase squeezes at 40 N with noslip iterations, or the pen creeps 0.8 mm per stroke.
- Without rolling and torsional friction on the pads, a pen grasped off its center of mass pivots in the fingers as it lifts. Soft pads resist that rotation; modeling it took the scripted grasp under camera error from 60 % to 74 %.
- A scripted P controller with full gain oscillates when the arm answers one period late, and never reaches its tolerance: halving the gain took the baseline from 36 % to 60 %. A fair baseline matters before claiming that RL beats it.
- My first reward paid +25 for a success that ended the episode, while hovering next to the pen earned about 1.5 per step until the time limit. PPO from scratch (8M steps) learned to hover with the gripper open, and a residual policy started at 47 % and unlearned the grasp down to 1 %. Paying the best per-step reward for every step left after a success fixed it.
- Cylinder-box contacts misbehave when a pad pinches the rim; the pen is a capsule.
- With the gripper pointing away from the base, the forearm hits the table while drawing; pointing it back at the base keeps every paper position reachable.
- The wrist camera sits on one side of the gripper: the pen must be grasped with its tip on the other side, or the mount reaches the paper before the pen.

## Toward the real arm

The simulation world frame is the UR controller base frame shifted up by the pedestal height, so TCP targets map to `ur_rtde` servoL directly. Still to do: pen detection with the C922, a touch-off to measure the tip position in the gripper, force_mode for the pen-down phase, and grooved fingers for the pen.

## License

MIT. The UR5e model comes from MuJoCo Menagerie (BSD-3) and is downloaded by `scripts/fetch_assets.py`.
