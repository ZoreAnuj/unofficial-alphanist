"""Import-light URDF parse + FK composition for the FANUC CRX-5iA.

The URDF chain is consumed on two very different paths: the cell builder
(pxr/trimesh-backed, build-host only) and the runtime planning stack
(``crx_kinematics``/``stage_builder``, imported per episode and in plain
CI). Parsing the URDF and composing link transforms is pure xml + numpy,
so it lives here as the shared public seam: neither ``pxr`` nor
``trimesh`` may appear at module level in this file, keeping the
planning import path free of the build chain's heavy dependencies.
"""

import dataclasses
import math
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from spatial import math_helpers

JOINT_CHAIN = ("J1", "J2", "J3", "J4", "J5", "J6")


@dataclasses.dataclass(frozen=True)
class UrdfJoint:
    xyz: tuple[float, ...]
    rpy: tuple[float, ...]
    axis: tuple[float, ...] | None
    limits: tuple[float, float] | None
    child: str


def _floats(text: str | None, default: str = "0 0 0") -> tuple[float, ...]:
    return tuple(float(value) for value in (text or default).split())


def origin_transform(
    xyz: tuple[float, ...], rpy: tuple[float, ...] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """4x4 from translation + URDF extrinsic-XYZ rpy (RADIANS)."""
    matrix = np.eye(4)
    matrix[:3, :3] = math_helpers.extrinsic_xyz_rotation_matrix(*rpy)
    matrix[:3, 3] = xyz
    return matrix


def parse_urdf(
    urdf_path: pathlib.Path,
) -> tuple[dict[str, UrdfJoint], dict[str, tuple[np.ndarray, str]]]:
    """Joints by name + per-link (visual-origin transform, mesh filename)."""
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, UrdfJoint] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints[str(joint.get("name"))] = UrdfJoint(
            xyz=_floats(origin.get("xyz") if origin is not None else None),
            rpy=_floats(origin.get("rpy") if origin is not None else None),
            axis=_floats(axis.get("xyz")) if axis is not None else None,
            limits=(
                (float(str(limit.get("lower"))), float(str(limit.get("upper"))))
                if limit is not None
                else None
            ),
            child=str(joint.find("child").get("link")),  # type: ignore[union-attr]
        )
    visuals: dict[str, tuple[np.ndarray, str]] = {}
    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        origin = link.find("visual/origin")
        if mesh is None:
            continue
        visuals[str(link.get("name"))] = (
            origin_transform(
                _floats(origin.get("xyz") if origin is not None else None),
                _floats(origin.get("rpy") if origin is not None else None),
            ),
            str(mesh.get("filename")),
        )
    return joints, visuals


def fk_link_transforms(
    joints: dict[str, UrdfJoint], sample_deg: tuple[float, ...]
) -> dict[str, np.ndarray]:
    """base_link-frame world transform per link at the joint sample."""
    transforms = {"base_link": np.eye(4)}
    chain = np.eye(4)
    for name, angle_deg in zip(JOINT_CHAIN, sample_deg, strict=True):
        joint = joints[name]
        angle = math.radians(angle_deg)
        lower, upper = joint.limits or (-math.inf, math.inf)
        if not lower <= angle <= upper:
            raise ValueError(
                f"joint {name} sample {angle_deg} deg = {angle:.3f} rad is "
                f"outside the URDF limits [{lower:.3f}, {upper:.3f}] rad — "
                f"the capture sample must be degrees within limits"
            )
        assert joint.axis is not None
        # Origin rpy converts to DEGREES (create_rpy_transform convention
        # inside); the joint value stays radians.
        chain = chain @ math_helpers.create_rotary_joint_transform(
            *joint.xyz,
            *(math.degrees(value) for value in joint.rpy),
            *joint.axis,
            angle,
        )
        transforms[joint.child] = chain
    return transforms
