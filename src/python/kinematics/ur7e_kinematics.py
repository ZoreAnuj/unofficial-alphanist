"""UR7e kinematics on the real cell's chain: FK, Jacobian, closed-form IK.

Same public surface as ``crx_kinematics`` for the functions the gantry scene
uses, so a robot profile can name either module. Joints are degrees at every
public boundary, in the controller's order (shoulder pan, shoulder lift,
elbow, wrist 1, wrist 2, wrist 3); poses are 4x4 in the controller ``base``
frame; the flange is ``tool0`` with +z out of the flange.
"""

import functools
import math
import pathlib
from collections.abc import Sequence

import numpy as np

from asset_pipeline.fanuc_cell import urdf_fk
from spatial import math_helpers

URDF_PATH = pathlib.Path(urdf_fk.__file__).parents[1] / "ur7e_cell" / "ur7e_sim.urdf"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
BASE_LINK = "base"
FLANGE = "tool0"

# Tool straight down, wrist 2 at 90 deg and the elbow up like the real cell's
# home (rad -0.1000, -1.6217, -1.0714, -2.0205, 1.5722, -1.6666), turned to
# the FANUC home's flange point on this mount (base 0.402, 0.415, 0.393 m,
# yaw 32.3 deg) so the gripper starts above the pick area.
HOME_JOINTS_DEG: tuple[float, float, float, float, float, float] = (
    59.251,
    -100.867,
    -91.991,
    -77.142,
    90.0,
    -63.049,
)

# Both flanges put +z out of the flange, so one "tool down at this yaw" holds.
def vertical_flange_pose(position_xyz: Sequence[float], yaw_deg: float) -> np.ndarray:
    """Flange target with tool +Z pointing world-down and the given yaw."""
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4)
    # Rz(yaw) @ diag(1, -1, -1): flange +Z down, x-axis at yaw about world z.
    pose[:3, :3] = np.array(
        [[cos_y, sin_y, 0.0], [sin_y, -cos_y, 0.0], [0.0, 0.0, -1.0]]
    )
    pose[:3, 3] = position_xyz
    return pose

_SINGULAR_SIN = 1e-9


@functools.lru_cache(maxsize=1)
def _chain() -> tuple[
    tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]
]:
    """Joint origins, flange origin, limits (rad) and link names of the URDF."""
    joints = urdf_fk.parse_urdf(URDF_PATH)[0]
    limits = np.array([joints[name].limits for name in JOINT_NAMES], dtype=float)
    return (
        tuple(
            urdf_fk.origin_transform(joints[name].xyz, joints[name].rpy)
            for name in JOINT_NAMES
        ),
        urdf_fk.origin_transform(joints["flange"].xyz, joints["flange"].rpy),
        limits[:, 0],
        limits[:, 1],
        tuple(joints[name].child for name in JOINT_NAMES),
    )


@functools.lru_cache(maxsize=1)
def _dh() -> tuple[float, float, float, float, float, float]:
    """d1, a2, a3, d4, d5, d6 of the standard UR table, read from the URDF.

    The closed-form IK holds only for that table; any other chain raises.
    """
    origins, flange = _chain()[:2]
    d1, a2, a3 = origins[1][2, 3], origins[2][0, 3], origins[3][0, 3]
    d4, d5, d6 = origins[4][2, 3], origins[5][2, 3], flange[2, 3]
    expected = (
        np.eye(4),
        math_helpers.dh_transform(0.0, d1, 0.0, math.pi / 2),
        math_helpers.dh_transform(0.0, 0.0, a2, 0.0),
        math_helpers.dh_transform(0.0, 0.0, a3, 0.0),
        math_helpers.dh_transform(0.0, d4, 0.0, math.pi / 2),
        math_helpers.dh_transform(0.0, d5, 0.0, -math.pi / 2),
        math_helpers.dh_transform(0.0, d6, 0.0, 0.0),
    )
    if not np.allclose((*origins, flange), expected, atol=1e-9):
        raise ValueError(f"{URDF_PATH} is not the standard UR link-length table")
    return d1, a2, a3, d4, d5, d6


def _rz(angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    result = np.eye(4)
    result[0, 0] = result[1, 1] = cos
    result[0, 1], result[1, 0] = -sin, sin
    return result


def _link_frames(q_rad: np.ndarray) -> list[np.ndarray]:
    origins = _chain()[0]
    frames, chain = [], np.eye(4)
    for origin, angle in zip(origins, q_rad, strict=True):
        chain = chain @ origin @ _rz(angle)
        frames.append(chain)
    return frames


def _radians(joints_deg: Sequence[float]) -> np.ndarray:
    values = np.asarray(joints_deg, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("UR7e kinematics requires six finite joint angles")
    return np.deg2rad(values)


def fk(joints_deg: Sequence[float]) -> dict[str, np.ndarray]:
    """Joints (degrees) -> {link name: 4x4} in the base frame, flange included.

    Raises ValueError outside the URDF joint limits."""
    q = _radians(joints_deg)
    _, flange, lower, upper, links = _chain()
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError(f"UR7e joints {tuple(joints_deg)} deg exceed the URDF limits")
    frames = _link_frames(q)
    return {
        BASE_LINK: np.eye(4),
        **dict(zip(links, frames, strict=True)),
        FLANGE: frames[-1] @ flange,
    }


def flange_pose(joints_deg: Sequence[float]) -> np.ndarray:
    return _link_frames(_radians(joints_deg))[-1] @ _chain()[1]


def flange_jacobian(joints_deg: Sequence[float]) -> np.ndarray:
    """Base-frame linear/angular flange velocity per joint radian."""
    frames = _link_frames(_radians(joints_deg))
    point = (frames[-1] @ _chain()[1])[:3, 3]
    jacobian = np.empty((6, 6))
    for index, frame in enumerate(frames):
        jacobian[:3, index] = np.cross(frame[:3, 2], point - frame[:3, 3])
        jacobian[3:, index] = frame[:3, 2]
    return jacobian


def _branches(pose: np.ndarray, wrist_3_at_singularity: float) -> list[np.ndarray]:
    """Every closed-form solution (rad, each angle in (-pi, pi]); up to eight."""
    d1, a2, a3, d4, d5, d6 = _dh()
    wrist = pose @ np.array([0.0, 0.0, -d6, 1.0])
    reach = math.hypot(wrist[0], wrist[1])
    if reach < abs(d4):
        return []
    solutions = []
    for shoulder in (1.0, -1.0):
        q1 = math.atan2(wrist[1], wrist[0]) + shoulder * math.acos(d4 / reach)
        q1 += math.pi / 2
        cos5 = (pose[0, 3] * math.sin(q1) - pose[1, 3] * math.cos(q1) - d4) / d6
        if abs(cos5) > 1.0 + 1e-12:
            continue
        for wrist_sign in (1.0, -1.0):
            q5 = wrist_sign * math.acos(min(1.0, max(-1.0, cos5)))
            sin5 = math.sin(q5)
            inverse = np.linalg.inv(pose)
            q6 = (
                wrist_3_at_singularity
                if abs(sin5) < _SINGULAR_SIN
                else math.atan2(
                    (-inverse[1, 0] * math.sin(q1) + inverse[1, 1] * math.cos(q1))
                    / sin5,
                    (inverse[0, 0] * math.sin(q1) - inverse[0, 1] * math.cos(q1))
                    / sin5,
                )
            )
            # Frame of joint 2 to frame of joint 5: the planar three-joint part.
            planar = (
                np.linalg.inv(math_helpers.dh_transform(q1, d1, 0.0, math.pi / 2))
                @ pose
                @ np.linalg.inv(
                    math_helpers.dh_transform(q5, d5, 0.0, -math.pi / 2)
                    @ math_helpers.dh_transform(q6, d6, 0.0, 0.0)
                )
            )
            x, y = planar[0, 3], planar[1, 3]
            cos3 = (x * x + y * y - a2 * a2 - a3 * a3) / (2 * a2 * a3)
            if abs(cos3) > 1.0 + 1e-12:
                continue
            for elbow in (1.0, -1.0):
                q3 = elbow * math.acos(min(1.0, max(-1.0, cos3)))
                q2 = math.atan2(y, x) - math.atan2(
                    a3 * math.sin(q3), a2 + a3 * math.cos(q3)
                )
                q4 = math.atan2(planar[1, 0], planar[0, 0]) - q2 - q3
                solutions.append(np.array([q1, q2, q3, q4, q5, q6]))
    return solutions


def ik(
    pose6d: np.ndarray,
    seed_joints_deg: Sequence[float],
    *,
    pos_tol_m: float = 1e-4,
    rot_tol_rad: float = 1e-3,
    limit_margin_deg: float = 0.0,
) -> tuple[float, ...] | None:
    """Closed-form IK: the solution nearest ``seed_joints_deg``, or None.

    All eight arm branches and every whole turn inside the limits (minus
    ``limit_margin_deg``) are candidates; nearest is the smallest joint-space
    distance from the seed. A branch must reproduce the pose within the
    tolerances, so an unreachable pose gives None, never a near miss."""
    pose = np.asarray(pose6d, dtype=float)
    seed = _radians(seed_joints_deg)
    _, flange, lower, upper, _ = _chain()
    margin = math.radians(limit_margin_deg)
    best: np.ndarray | None = None
    for branch in _branches(pose, float(seed[5])):
        reached = _link_frames(branch)[-1] @ flange
        if (
            np.linalg.norm(reached[:3, 3] - pose[:3, 3]) > pos_tol_m
            or np.linalg.norm(math_helpers.log_SO3(pose[:3, :3] @ reached[:3, :3].T))
            > rot_tol_rad
        ):
            continue
        # Each joint takes its own nearest whole turn; that minimises the norm.
        turn = 2 * math.pi
        candidates = [
            [
                value
                for value in (
                    angle + turn * (round((start - angle) / turn) + k)
                    for k in (0, -1, 1)
                )
                if low + margin <= value <= high - margin
            ]
            for angle, start, low, high in zip(branch, seed, lower, upper, strict=True)
        ]
        if not all(candidates):
            continue
        q = np.array(
            [
                values[int(np.argmin(np.abs(np.subtract(values, start))))]
                for values, start in zip(candidates, seed, strict=True)
            ]
        )
        if best is None or np.linalg.norm(q - seed) < np.linalg.norm(best - seed):
            best = q
    return None if best is None else tuple(float(v) for v in np.rad2deg(best))


def joint_limit_margin_deg(joints_deg: Sequence[float]) -> float:
    """Smallest distance (deg) from any joint angle to its URDF limit."""
    q = _radians(joints_deg)
    _, _, lower, upper, _ = _chain()
    return float(np.degrees(min(np.min(q - lower), np.min(upper - q))))


# Folded past this the wrist and tool come close to the upper arm and the
# shoulder; the reach sweep of the gantry task needs 132 deg.
ELBOW_FOLD_LIMIT_DEG = 150.0


def elbow_bend_deg(joints_deg: Sequence[float]) -> float:
    """Elbow margin (deg) with the planner's meaning: 0 at the straight-arm
    singularity, and falling again from 20 at the fold limit so that one
    threshold keeps the elbow between "20 deg bent" and the fold limit."""
    bend = abs(float(joints_deg[2]))
    return min(bend, 20.0 + ELBOW_FOLD_LIMIT_DEG - bend)


def turned_seed(seed_deg: Sequence[float], pose6d: np.ndarray) -> tuple[float, ...]:
    """The seed with the shoulder pan turned to face the target.

    The nearest solution to a seed taken as it is can flip the shoulder when
    the target is far round the base (the nest from the pick side); turned
    about the base axis first, the arm keeps its shape.
    """
    seed = [float(v) for v in seed_deg]
    here = flange_pose(seed)
    turn = math.atan2(pose6d[1, 3], pose6d[0, 3]) - math.atan2(here[1, 3], here[0, 3])
    seed[0] += math.degrees((turn + math.pi) % (2 * math.pi) - math.pi)
    return tuple(seed)


def _shoulder_reach_m(joints_deg: Sequence[float]) -> float:
    """How far wrist 1 stands in front of the base axis, along the direction
    the shoulder pan faces; negative when the arm reaches back over its base."""
    wrist = fk(joints_deg)["wrist_1_link"][:3, 3]
    pan = math.radians(joints_deg[0])
    return float(wrist[0] * math.cos(pan) + wrist[1] * math.sin(pan))


def same_arm_shape(joints_deg: Sequence[float], other_deg: Sequence[float]) -> bool:
    """Shoulder, elbow and wrist 2 on the same side in both joint vectors.

    The shoulder counts: when the natural solution would turn wrist 3 past its
    limit, the nearest one left reaches back over the base, and the path to it
    carried the main body into the arm (seeds 100041, 100048).
    """
    return _shoulder_reach_m(joints_deg) * _shoulder_reach_m(other_deg) > 0 and all(
        math.sin(math.radians(joints_deg[i])) * math.sin(math.radians(other_deg[i])) > 0
        for i in (2, 4)
    )
