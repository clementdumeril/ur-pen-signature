from glob import glob

from setuptools import find_packages, setup

package_name = "tower_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Clement Dumeril",
    maintainer_email="clementdumeril@gmail.com",
    description="Scripted three-block tower on the lab UR7e with a Robotiq Hand-E and a wrist RealSense",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception = tower_ros.perception_node:main",
            "tower = tower_ros.tower_node:main",
            "tower_policy = tower_ros.policy_node:main",
        ],
    },
)
