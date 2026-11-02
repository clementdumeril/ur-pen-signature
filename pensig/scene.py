"""MuJoCo model: UR5e arm (kinematic stand-in for the UR7e), Robotiq Hand-E, a pen and its supports."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
UR_DIR = ROOT / "assets" / "mujoco_menagerie" / "universal_robots_ur5e"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Lab cell (EECS 106A, UR7e): the table surface sits 0.275 m below the robot base, and the gripper
# envelope is 0.065 x 0.065 x 0.17 m centered 0.10 m from the flange. The UR7e has the UR5e kinematics.
BASE_HEIGHT = 0.275

# Robotiq Hand-E with the lab fingers: 120 mm body, 60 mm fingers, 2 x 25 mm stroke.
HANDE_BODY = 0.12
HANDE_RADIUS = 0.0325
FINGER_LEN = 0.06
MAX_OPEN = 0.025
TCP_Z = HANDE_BODY + FINGER_LEN - 0.012
PAD = (0.004, 0.009, FINGER_LEN / 2)

# Logitech C922 on the wrist: 0.13 m off the tool axis, looking along it (lab static TF).
CAMERA_POS = (-0.025, 0.13, 0.0)

# Geoms are compiled at their largest size and only shrunk at reset, so the
# compile-time bounding boxes used by the broadphase stay conservative.
PEN_MAX = (0.006, 0.08)
SUPPORT_MAX = {"post": (0.02, 0.03, 0.07), "bridge": (0.006, 0.03, 0.07), "bottle": (0.045, 0.10)}

GRIPPER_XML = f"""
<body name="hande" gravcomp="1">
  <inertial mass="1.0" pos="0 0 0.05" diaginertia="0.003 0.003 0.0015"/>
  <geom name="hande_body" type="cylinder" size="{HANDE_RADIUS} {HANDE_BODY / 2}" pos="0 0 {HANDE_BODY / 2}"
        rgba="0.08 0.08 0.08 1"/>
  <geom name="camera_mount" type="box" size="0.05 0.04 0.03" pos="0.025 0.10 0.0" rgba="0.2 0.2 0.2 1"
        contype="2" conaffinity="2"/>
  <site name="tcp" pos="0 0 {TCP_Z}" size="0.003" rgba="1 0 0 1" group="4"/>
  <site name="ft" size="0.002" group="4"/>
  <camera name="wrist_cam" pos="{' '.join(map(str, CAMERA_POS))}" xyaxes="1 0 0 0 -1 0" fovy="40"/>
  <body name="finger_left" pos="0 0 {HANDE_BODY}" gravcomp="1">
    <inertial mass="0.04" pos="{PAD[0]} 0 {PAD[2]}" diaginertia="1e-5 1e-5 1e-5"/>
    <joint name="finger_left" type="slide" axis="1 0 0" range="0 {MAX_OPEN}" damping="5" armature="0.01"/>
    <geom name="pad_left" type="box" size="{PAD[0]} {PAD[1]} {PAD[2]}" pos="{PAD[0]} 0 {PAD[2]}"
          friction="1 0.03 0.01" condim="6" solref="0.004 1" solimp="0.95 0.99 0.0005" rgba="0.55 0.55 0.55 1"/>
  </body>
  <body name="finger_right" pos="0 0 {HANDE_BODY}" gravcomp="1">
    <inertial mass="0.04" pos="{-PAD[0]} 0 {PAD[2]}" diaginertia="1e-5 1e-5 1e-5"/>
    <joint name="finger_right" type="slide" axis="-1 0 0" range="0 {MAX_OPEN}" damping="5" armature="0.01"/>
    <geom name="pad_right" type="box" size="{PAD[0]} {PAD[1]} {PAD[2]}" pos="{-PAD[0]} 0 {PAD[2]}"
          friction="1 0.03 0.01" condim="6" solref="0.004 1" solimp="0.95 0.99 0.0005" rgba="0.55 0.55 0.55 1"/>
  </body>
</body>
"""

WORLD_XML = f"""
<worldbody>
  <light name="top" pos="0.5 0 1.6" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
  <geom name="table" type="plane" size="1.5 1.5 0.05" material="table" conaffinity="3"/>
  <geom name="pedestal" type="cylinder" size="0.08 {BASE_HEIGHT / 2}" pos="0 0 {BASE_HEIGHT / 2}" rgba="0.3 0.3 0.32 1"/>
  <body name="post" pos="5 0 0">
    <geom name="post" type="box" conaffinity="3" size="{' '.join(map(str, SUPPORT_MAX['post']))}" rgba="0.12 0.12 0.12 1"/>
  </body>
  <body name="bridge_a" pos="5 1 0">
    <geom name="bridge_a" type="box" conaffinity="3" size="{' '.join(map(str, SUPPORT_MAX['bridge']))}" rgba="0.12 0.12 0.12 1"/>
  </body>
  <body name="bridge_b" pos="5 2 0">
    <geom name="bridge_b" type="box" conaffinity="3" size="{' '.join(map(str, SUPPORT_MAX['bridge']))}" rgba="0.12 0.12 0.12 1"/>
  </body>
  <body name="bottle" pos="5 3 0">
    <geom name="bottle" type="capsule" conaffinity="3" size="{' '.join(map(str, SUPPORT_MAX['bottle']))}" rgba="0.55 0.75 0.95 0.7"/>
  </body>
  <body name="pen" pos="0.5 0 0.2">
    <freejoint name="pen_free"/>
    <geom name="pen" type="capsule" size="{PEN_MAX[0]} {PEN_MAX[1]}" mass="0.012" rgba="0.1 0.25 0.75 1"
          friction="0.8 0.005 0.0001" condim="4" solref="0.004 1" solimp="0.95 0.99 0.0005"/>
    <geom name="pen_tip" type="cylinder" size="0.0015 0.004" pos="0 0 {PEN_MAX[1] + 0.004}" mass="0"
          contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
    <site name="pen_tip" pos="0 0 {PEN_MAX[1] + 0.008}" size="0.002"/>
  </body>
</worldbody>
"""

EXTRA_XML = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.78 0.66 0.48" rgb2="0.74 0.62 0.44"
             width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="10 10"/>
  </asset>
  <actuator>
    <velocity name="gripper" joint="finger_left" kv="300" forcerange="-20 20" ctrlrange="-0.1 0.1"/>
  </actuator>
  <equality>
    <joint joint1="finger_right" joint2="finger_left" polycoef="0 1 0 0 0" solref="0.004 1"/>
  </equality>
  <sensor>
    <force name="wrist_force" site="ft"/>
  </sensor>
</mujoco>
"""


def build_xml(world_xml: str = WORLD_XML) -> str:
    xml_path = UR_DIR / "ur5e.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"{xml_path} missing: run python scripts/fetch_assets.py")
    root = ET.parse(xml_path).getroot()

    root.find("compiler").set("meshdir", (UR_DIR / "assets").as_posix())
    option = root.find("option")
    option.attrib.update(
        {"timestep": "0.002", "integrator": "implicitfast", "cone": "elliptic", "impratio": "10",
         "noslip_iterations": "5"}
    )

    world = root.find("worldbody")
    for light in world.findall("light"):
        world.remove(light)
    for body in world.iter("body"):
        body.set("gravcomp", "1")
    next(b for b in world.iter("body") if b.get("name") == "base").set("pos", f"0 0 {BASE_HEIGHT}")

    wrist = next(b for b in world.iter("body") if b.get("name") == "wrist_3_link")
    site = next(s for s in wrist.findall("site") if s.get("name") == "attachment_site")
    gripper = ET.fromstring(GRIPPER_XML)
    gripper.set("pos", site.get("pos"))
    gripper.set("quat", site.get("quat"))
    wrist.append(gripper)

    for child in ET.fromstring(world_xml):
        world.append(child)

    extra = ET.fromstring(EXTRA_XML)
    for section in extra:
        existing = root.find(section.tag)
        if existing is None:
            root.append(section)
        else:
            existing.extend(list(section))

    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    return ET.tostring(root, encoding="unicode")


def build_model(world_xml: str = WORLD_XML) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(world_xml))
