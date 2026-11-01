"""Download the UR5e model from MuJoCo Menagerie (BSD-3) into assets/. Not redistributed here."""

import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "assets" / "mujoco_menagerie"


def main():
    if (DEST / "universal_robots_ur5e" / "ur5e.xml").exists():
        print("already there:", DEST)
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    "https://github.com/google-deepmind/mujoco_menagerie", str(DEST)], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "universal_robots_ur5e"], cwd=DEST, check=True)
    print("done:", DEST / "universal_robots_ur5e")


if __name__ == "__main__":
    main()
