"""UR7e and Hand-E parameters for the assembly simulation."""

import dataclasses
import os
import pathlib
import types
from typing import Any

from kinematics import ur7e_kinematics

_ROOT = pathlib.Path(__file__).resolve().parents[4]
ENVIRONMENT_VARIABLE = "GANTRY_ROBOT"


@dataclasses.dataclass(frozen=True)
class Gripper:
    """A two-jaw gripper on the flange: +z is the tool axis, the jaws slide on x.

    Millimetres, in the flange frame. A jaw joint reads 0 at ``rest_half_gap_mm``
    between the tool axis and its pad face and grows as the jaw opens.
    """

    name: str
    # Physics bodies: the fixed body, the jaw on -x, the jaw on +x.
    bodies: tuple[str, str, str]
    rest_half_gap_mm: float
    open_half_gap_mm: float
    # One motor moves both jaws by equal and opposite amounts.
    coupled: bool
    # Pad faces along the tool axis: nearest and farthest edge from the flange.
    pad_z_mm: tuple[float, float]
    # Flange above the main body's rest plane and above the actuator's ear
    # plane at the grasp; grasp.py and episode_plan.py derive the rest.
    body_grasp_height_mm: float
    actuator_grasp_height_mm: float


# Robotiq Hand-E with the fingertip holder (HND-TIP-HLD-KIT) and the flat
# silicone fingertip (HND-TIP-SLC-KIT), from Robotiq's dimension drawings
# (Hand-E instruction manual, section 6.1.2): 11 mm coupler, 99 mm body of
# 75 mm diameter, 6.5 mm holder plate on the rack, then the 40 mm fingertip:
# an 8 mm base and a 32 mm upright, 21 mm wide, with a silicone pad over the
# whole 40 mm. The fingertip ends 46.5 mm from the body, as the basic
# aluminium finger does. 25 mm of travel per finger.
# Main body: the fingertip base, which stands out behind the pad down to
# 124.5 mm from the flange, 5 mm above the ribs and wing flanges (90 mm above
# the rest plane); the pads then lie on the hub tube from 63 mm to its top
# at 80 mm.
# Actuator: the fingertips 2.2 mm above the dome of the simulator's housing
# collider at the motor radius (a capsule; the real shoulder is 12.8 mm
# lower), which leaves 14.65 mm of pad on the 32 mm motor.
HAND_E = Gripper(
    name="hande",
    bodies=("hande_body", "hande_finger_right", "hande_finger_left"),
    rest_half_gap_mm=0.0,
    open_half_gap_mm=25.0,
    coupled=True,
    pad_z_mm=(116.5, 156.5),
    body_grasp_height_mm=124.5 + 90.0 + 5.0,
    actuator_grasp_height_mm=198.0,
)


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    name: str
    # fk, flange_pose, flange_jacobian, ik, vertical_flange_pose,
    # HOME_JOINTS_DEG and URDF_PATH (chain, limits, inertias, collision).
    kinematics: types.ModuleType
    joint_names: tuple[str, ...]
    # Base first, then each joint's child: also the physics body order.
    link_names: tuple[str, ...]
    flange: str
    # Fixed URDF joints from the last link to the flange frame.
    flange_joints: tuple[str, ...]
    # Controller J3 = URDF J3 - coupling * J2: the FANUC reports J3 from the
    # horizontal, a UR reports each joint against its parent link.
    j2_j3_coupling: float
    # Gains, armature, effort limits, home, tool bodies, collision meshes.
    native_profile: pathlib.Path
    gripper: Gripper

    @property
    def last_link(self) -> str:
        return self.link_names[-1]

    @property
    def tool_bodies(self) -> tuple[str, ...]:
        """Bodies after the arm links, in physics order."""
        return (*self.gripper.bodies, "wrist_camera")


UR7E = RobotProfile(
    name="ur7e",
    kinematics=ur7e_kinematics,
    joint_names=ur7e_kinematics.JOINT_NAMES,
    link_names=(
        ur7e_kinematics.BASE_LINK,
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ),
    flange=ur7e_kinematics.FLANGE,
    flange_joints=("flange",),
    j2_j3_coupling=0.0,
    native_profile=_ROOT / "env/configs/ur7e_native_profile.json",
    gripper=HAND_E,
)
_PROFILES = {profile.name: profile for profile in (UR7E,)}


def get(name: str) -> RobotProfile:
    if name not in _PROFILES:
        raise ValueError(f"unknown robot {name!r}; known: {sorted(_PROFILES)}")
    return _PROFILES[name]


def of(native_profile: dict[str, Any]) -> RobotProfile:
    """The arm a native physics profile was written for."""
    return get(native_profile.get("robot", UR7E.name))


def active() -> RobotProfile:
    """The arm this process runs the gantry scene with."""
    return get(os.environ.get(ENVIRONMENT_VARIABLE, UR7E.name))


def kinematics() -> types.ModuleType:
    return active().kinematics
