# What the simulation taught me

Back to the [README](../README.md). The before/after success rates below come from evaluation runs that are not kept in `results/`; only the final evaluations are.

- MuJoCo contacts are stiff in proportion to the object's mass: a 10 g pen squeezed at 40 N sinks millimeters into the pads. The grasp uses 8 to 20 N; the drawing phase squeezes at 40 N with noslip iterations, or the pen creeps about 0.8 mm per stroke.
- Without rolling and torsional friction on the pads, a pen grasped off its center of mass pivots in the fingers as it lifts. Soft pads resist that rotation; modeling it took the scripted grasp under camera error from 60 % to 74 %.
- A scripted P controller with full gain oscillates when the arm answers one period late, and never reaches its tolerance: halving the gain took the baseline from 36 % to 60 %. A fair baseline matters before claiming that RL beats it.
- My first reward paid +25 for a success that ended the episode, while hovering next to the pen earned about 1.5 per step until the time limit. PPO from scratch (8M steps) learned to hover with the gripper open, and a residual policy started at 47 % and unlearned the grasp down to 1 %. Paying the best per-step reward for every step left after a success fixed it (`SUCCESS_PER_STEP` in [`env.py`](../pensig/env.py)).
- Cylinder-box contacts misbehave when a pad pinches the rim; the pen is a capsule.
- With the gripper pointing away from the base, the forearm hits the table while drawing; pointing it back at the base keeps every paper position reachable ([`draw.py`](../pensig/draw.py)).
- The wrist camera sits on one side of the gripper: the pen must be grasped with its tip on the other side, or the mount reaches the paper before the pen.
