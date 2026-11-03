# tower_ros

The scripted block tower from `pensig/tower_scripted.py` on the lab UR7e: black on green, then red on black.
It uses lab 5's interfaces (MoveIt IK and planning, `/toggle_gripper`, the wrist RealSense) and adds a
close-up estimate from the wrist camera before each descent and a touchdown that stops on the wrist force.

## Build

In the lab workspace, next to lab 5's `planning` package (its collision boxes are loaded if present):

```
ln -s ~/ur-pen-signature/ros2/tower_ros ~/<workspace>/src/tower_ros
cd ~/<workspace> && colcon build --packages-select tower_ros && source install/setup.bash
```

## Run

Robot driver up as in lab 5, then:

```
ros2 launch tower_ros tower_bringup.launch.py
ros2 run tower_ros tower --ros-args --params-file install/tower_ros/share/tower_ros/config/tower.yaml
```

The arm's pose when `tower` starts is the start pose: each stage estimates from there. Add `-p dry_run:=true`
to only log targets and check IK, `-p stages:=1` for black on green only, `-p step:=false` once step mode has
gone through cleanly. RViz: `/tower/blocks`, `/tower/block_points`, `/tower/target`.

## Safety

- Step mode first (the default): every motion and gripper toggle waits for Enter. Keep a hand on the e-stop and
  the pendant speed slider low.
- Measure first: `gripper_length` (flange to fingertips: 0.16 measured on the lab Hand-E, lab 5's code ended
  at 0.14; too short sends the fingers into the table), the block heights, and the black top against
  `/tower/black/top` (lab 5 saw it 2 cm low; set `black_top_offset`). In the sim a 3 mm height error took the
  tower from 94 % to 89 %, 10 mm to 83 %.
- The gripper must be open at startup: `/toggle_gripper` only toggles.
- Lab 5's gripper collision box reaches 18.5 cm below the flange: if IK fails near the table, check
  `obstacles.yaml`.

`colcon test` or `python -m pytest ros2/tower_ros` runs the geometry and phase tests without ROS.
