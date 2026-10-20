"""Frames of the UR7e + Robotiq Hand-E render assets.

``ur7e_cell.urdf`` is the kinematic truth. The official UR meshes are
modelled in the frames of Universal Robots' own description, which puts
several link origins elsewhere along the joint axes; ``mesh_in_link``
converts between the two. Import-light (xml + numpy) so that the Isaac
preview and plain CI can use it.
"""

import math
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np

URDF_PATH = pathlib.Path(__file__).with_name("ur7e_cell.urdf")
ARM_LINKS = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)
# Joint vector of the real cell at the start of every episode (rad).
REAL_HOME_RAD = (-0.1000, -1.6217, -1.0714, -2.0205, 1.5722, -1.6666)

_HALF_PI = math.pi / 2
# Universal_Robots_ROS2_Description d4042c1f, config/ur7e/default_kinematics.yaml:
# joint origin (xyz, rpy) of each link in its parent, from base_link_inertia,
# which is our `base`. The file's 2e-11 m residues are dropped.
ROS_JOINT_ORIGINS = {
    "shoulder_link": ((0.0, 0.0, 0.1625), (0.0, 0.0, 0.0)),
    "upper_arm_link": ((0.0, 0.0, 0.0), (_HALF_PI, 0.0, 0.0)),
    "forearm_link": ((-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "wrist_1_link": ((-0.3922, 0.0, 0.1333), (0.0, 0.0, 0.0)),
    "wrist_2_link": ((0.0, -0.0997, 0.0), (_HALF_PI, 0.0, 0.0)),
    "wrist_3_link": ((0.0, 0.0996, 0.0), (_HALF_PI, math.pi, math.pi)),
}
# Same commit, config/ur7e/visual_parameters.yaml: mesh file and its
# mesh_offset (xyz, rpy) in the ROS link. The ur7e entry uses the ur5e meshes.
ROS_MESHES = {
    "base": ("base.dae", (0.0, 0.0, 0.0), (0.0, 0.0, math.pi)),
    "shoulder_link": ("shoulder.dae", (0.0, 0.0, 0.0), (0.0, 0.0, math.pi)),
    "upper_arm_link": ("upperarm.dae", (0.0, 0.0, 0.138), (_HALF_PI, 0.0, -_HALF_PI)),
    "forearm_link": ("forearm.dae", (0.0, 0.0, 0.007), (_HALF_PI, 0.0, -_HALF_PI)),
    "wrist_1_link": ("wrist1.dae", (0.0, 0.0, -0.127), (_HALF_PI, 0.0, 0.0)),
    "wrist_2_link": ("wrist2.dae", (0.0, 0.0, -0.0997), (0.0, 0.0, 0.0)),
    "wrist_3_link": ("wrist3.dae", (0.0, -0.0005, -0.0989), (_HALF_PI, 0.0, 0.0)),
}

# robotiq_hande_description 5ae8b970, urdf/robotiq_hande_gripper.xacro (m).
HANDE_COUPLER_HEIGHT = 0.011
HANDE_BODY_HEIGHT = 0.099
HANDE_FINGER_LENGTH = 0.0465
HANDE_FINGER_STROKE = 0.025
# Flange to the end of the closed fingertips, from the gripper model. The
# URDF's 0.160 is an unmeasured placeholder.
FINGERTIP_Z = HANDE_COUPLER_HEIGHT + HANDE_BODY_HEIGHT + HANDE_FINGER_LENGTH

# The pipeline-wide cell root (glb_to_usdc.CELL_ROOT_PRIM), repeated here:
# importing that module loads the whole station config stack.
INSTANCES = "/vention_cell/Instances"
# Arm prims of the layer -> the URDF link each one is modelled in.
ARM_MEMBERS = {
    "ur7e_base": "base",
    "ur7e_shoulder": "shoulder_link",
    "ur7e_upper_arm": "upper_arm_link",
    "ur7e_forearm": "forearm_link",
    "ur7e_wrist_1": "wrist_1_link",
    "ur7e_wrist_2": "wrist_2_link",
    "ur7e_wrist_3": "wrist_3_link",
}
# The gripper has no link in the URDF: its prims are modelled in tool0. The
# names are the physics bodies of robot_profile.HAND_E, which a tape binds to
# these prims.
TOOL_MEMBERS = ("hande_body", "hande_finger_left", "hande_finger_right")
# Modelled closed; each finger slides this way along tool0 x to open.
FINGER_OPEN_DIRECTION = {"hande_finger_left": 1.0, "hande_finger_right": -1.0}


def transform(
    xyz: Sequence[float] = (0.0, 0.0, 0.0), rpy: Sequence[float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """4x4 from a translation and URDF fixed-axis roll, pitch, yaw (rad)."""
    (sr, sp, sy), (cr, cp, cy) = np.sin(rpy), np.cos(rpy)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    matrix[:3, 3] = xyz
    return matrix


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


def _attribute(
    joint: ET.Element, tag: str, name: str, default: str | None = None
) -> str:
    """A joint child's attribute; a missing one without a default is an error."""
    child = joint.find(tag)
    value = default if child is None else child.get(name, default)
    if value is None:
        raise ValueError(f"joint {joint.get('name')}: no {tag} {name}")
    return value


def link_poses(
    joints_rad: Sequence[float], urdf: pathlib.Path = URDF_PATH
) -> dict[str, np.ndarray]:
    """Pose of every link of the URDF in `base` at the six joint angles."""
    angles = iter(joints_rad)
    poses = {"base": np.eye(4)}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        pose = poses[_attribute(joint, "parent", "link")] @ transform(
            _floats(_attribute(joint, "origin", "xyz", "0 0 0")),
            _floats(_attribute(joint, "origin", "rpy", "0 0 0")),
        )
        if joint.get("type") == "revolute":
            if _floats(_attribute(joint, "axis", "xyz")) != (0.0, 0.0, 1.0):
                raise ValueError(f"{joint.get('name')}: only z-axis joints supported")
            pose = pose @ transform(rpy=(0.0, 0.0, next(angles)))
        poses[_attribute(joint, "child", "link")] = pose
    return poses


def ros_link_poses(joints_rad: Sequence[float]) -> dict[str, np.ndarray]:
    """Link poses of the official UR description in `base`."""
    poses = {"base": np.eye(4)}
    pose = np.eye(4)
    for (link, (xyz, rpy)), angle in zip(
        ROS_JOINT_ORIGINS.items(), joints_rad, strict=True
    ):
        pose = pose @ transform(xyz, rpy) @ transform(rpy=(0.0, 0.0, angle))
        poses[link] = pose
    return poses


def mesh_in_link(joints_rad: Sequence[float] = (0.0,) * 6) -> dict[str, np.ndarray]:
    """Transform taking each official mesh into its link frame of the URDF.

    Evaluated through both kinematic chains, so it is the same at every
    joint vector only if the two chains share their joint axes (unit test).
    """
    ours, ros = link_poses(joints_rad), ros_link_poses(joints_rad)
    return {
        link: np.linalg.inv(ours[link]) @ ros[link] @ transform(xyz, rpy)
        for link, (_, xyz, rpy) in ROS_MESHES.items()
    }


def arm_poses(joints_rad: Sequence[float]) -> dict[str, np.ndarray]:
    """Pose in `base` of every arm prim."""
    links = link_poses(joints_rad)
    return {member: links[link] for member, link in ARM_MEMBERS.items()}


def tool_pose(member: str, finger_opening_m: float = 0.0) -> np.ndarray:
    """Pose of a gripper prim in tool0; the opening is per finger."""
    if not 0.0 <= finger_opening_m <= HANDE_FINGER_STROKE:
        raise ValueError(f"finger opening {finger_opening_m} outside the stroke")
    slide = FINGER_OPEN_DIRECTION.get(member, 0.0) * finger_opening_m
    return transform((slide, 0.0, 0.0))
