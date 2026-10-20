"""Generate and physically validate assembly resets; never infer acceptance from IK.

Reset families and consecutive stability checks follow OmniReset's released
UWLab implementation (isaaclab2, 825cfce364f7e34b4b29aab73d37c5a1464f4375).
Thresholds below are explicit UR7 engineering choices, not AlphaNIST constants.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import dataclasses
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
from scipy import optimize as scipy_optimize
from scipy.spatial import ConvexHull
from scipy.spatial import transform as scipy_transform

from isaac.assembly_rl import assembly_motion, generate_bank, reset_sampling

FAMILIES = ("loose", "near_grasp", "held_transport", "near_mating", "partial_assembly")
BASE_TRANSLATION = np.array([-0.199, -0.4186, 0.8749])
PAD_CENTER_Z_M = 0.1365  # robot_profile.HAND_E.pad_z_mm midpoint, flange frame.
HOME_RAD = np.radians([59.251, -100.867, -91.991, -77.142, 90.0, -63.049])
SCREW_CHALLENGE_ANGLE_RAD = np.pi / 4
SCREW_CHALLENGE_AXIAL_TOLERANCE_M = 0.000025


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    stable_steps: int = 5
    max_settle_steps: int = 200
    close_steps: int = 25
    max_linear_speed_m_s: float = 0.01
    max_angular_speed_rad_s: float = 0.1
    max_joint_speed_rad_s: float = 0.05
    max_position_change_m: float = 0.0005
    max_angle_change_rad: float = 0.01
    max_penetration_m: float = 0.00005
    max_held_pad_penetration_m: float = 0.0005
    transient_max_penetration_m: float = 0.001
    transient_table_tolerance_m: float = 0.001
    table_tolerance_m: float = 0.0001
    retention_position_m: float = 0.003
    retention_angle_rad: float = 0.05
    perturb_force_n: float = 0.01
    perturb_steps: int = 25
    partial_perturb_steps: int = 1
    lift_height_m: float = 0.04
    near_mating_lift_height_m: float = 0.01
    candidate_attempts_per_sample: int = 12
    grasp_trials: int = 40
    grasp_axial_offset_m: float = 0.018
    gripper_spawn_clearance_m: float = 0.0001
    near_grasp_distance_m: float = 0.03
    near_mating_distance_m: float = 0.08
    near_mating_min_offset_m: float = 0.015
    near_mating_max_offset_m: float = 0.06
    partial_goal_min_distance_m: float = 0.001
    partial_goal_max_distance_m: float = 0.025
    partial_grasp_probability: float = 0.5
    partial_grasp_min_offset_m: float = 0.003
    partial_grasp_max_offset_m: float = 0.015
    partial_grasp_lift_height_m: float = 0.002
    dislodge_acceleration_m_s2: float = 8.0
    max_batches: int = 1000

    def __post_init__(self) -> None:
        values = dataclasses.asdict(self)
        if any(
            not np.isfinite(value)
            or (value <= 0 and name != "partial_grasp_probability")
            for name, value in values.items()
        ):
            raise ValueError(
                "Validation thresholds and attempt counts must be finite and positive"
            )
        if self.max_settle_steps < self.stable_steps:
            raise ValueError("Settling budget must cover consecutive stability checks")
        if self.grasp_axial_offset_m > 0.020:
            raise ValueError("Grasp contact must remain on the physical Hand-E pad")
        if not 0 <= self.partial_grasp_probability <= 1 or not (
            self.partial_goal_min_distance_m
            <= self.partial_grasp_min_offset_m
            < self.partial_grasp_max_offset_m
            < self.partial_goal_max_distance_m - self.partial_grasp_lift_height_m
        ):
            raise ValueError("Grasp-supported partial proposal must fit partial bounds")


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    result = np.zeros(pose.shape[:-1] + (4, 4))
    result[..., :3, :3] = (
        scipy_transform.Rotation.from_quat(pose[..., 3:].reshape(-1, 4))
        .as_matrix()
        .reshape(pose.shape[:-1] + (3, 3))
    )
    result[..., :3, 3] = pose[..., :3]
    result[..., 3, 3] = 1
    return result


def matrix_pose(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            matrix[..., :3, 3],
            scipy_transform.Rotation.from_matrix(matrix[..., :3, :3].reshape(-1, 3, 3))
            .as_quat()
            .reshape(matrix.shape[:-2] + (4,)),
        ),
        axis=-1,
    )


def quaternion_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # q and -q encode the same rotation; numerical norm drift is not a rotation.
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def sample_antipodal_grasps(
    vertices: np.ndarray,
    faces: np.ndarray,
    rng: np.random.Generator,
    count: int = 64,
    max_width_m: float = 0.047,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Surface contact pairs, including narrow lips of otherwise wide components."""
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(normals, axis=1)
    valid = areas > 1e-14
    triangles, normals, areas = triangles[valid], normals[valid], areas[valid]
    if not len(triangles):
        return []
    normals /= areas[:, None]
    ids = rng.choice(len(triangles), size=512, p=areas / areas.sum())
    u, v = rng.random((2, len(ids)))
    u = np.sqrt(u)
    points = (
        (1 - u)[:, None] * triangles[ids, 0]
        + (u * (1 - v))[:, None] * triangles[ids, 1]
        + (u * v)[:, None] * triangles[ids, 2]
    )
    normals = normals[ids]
    # Test pairs in randomized order, avoiding nearest-neighbour dependencies.
    pairs = rng.integers(0, len(points), size=(30000, 2))
    delta = points[pairs[:, 1]] - points[pairs[:, 0]]
    width = np.linalg.norm(delta, axis=1)
    axis = delta / np.maximum(width[:, None], 1e-12)
    valid = (
        (width > 0.002)
        & (width < max_width_m)
        & (np.sum(normals[pairs[:, 0]] * axis, axis=1) < -0.85)
        & (np.sum(normals[pairs[:, 1]] * axis, axis=1) > 0.85)
    )
    chosen = np.flatnonzero(valid)[:count]
    return [
        ((points[pairs[i, 0]] + points[pairs[i, 1]]) / 2, axis[i], float(width[i]))
        for i in chosen
    ]


def sample_normal_grasps(
    vertices: np.ndarray, faces: np.ndarray, rng: np.random.Generator, count: int = 64
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Opposed surface-normal rays avoid tilting flat pads through a thin plate."""
    triangles = vertices[faces]
    edges1, edges2 = (
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.cross(edges1, edges2)
    areas = np.linalg.norm(normals, axis=1)
    valid = areas > 1e-14
    triangles, edges1, edges2, normals, areas = (
        value[valid] for value in (triangles, edges1, edges2, normals, areas)
    )
    if not len(triangles):
        return []
    normals /= areas[:, None]
    ids = rng.choice(len(triangles), 512, p=areas / areas.sum())
    u, v = np.sqrt(rng.random(512)), rng.random(512)
    points = (
        (1 - u)[:, None] * triangles[ids, 0]
        + (u * (1 - v))[:, None] * triangles[ids, 1]
        + (u * v)[:, None] * triangles[ids, 2]
    )
    contacts = []
    for point, normal in zip(points, normals[ids], strict=True):
        axis = -normal
        cross = np.cross(axis, edges2)
        determinant = np.sum(edges1 * cross, axis=-1)
        inverse = np.divide(
            1.0,
            determinant,
            out=np.zeros_like(determinant),
            where=np.abs(determinant) > 1e-12,
        )
        relative = point + axis * 1e-7 - triangles[:, 0]
        other_cross = np.cross(relative, edges1)
        first = inverse * np.sum(relative * cross, axis=-1)
        second = inverse * np.sum(axis * other_cross, axis=-1)
        distances = inverse * np.sum(edges2 * other_cross, axis=-1)
        hits = (
            (np.abs(determinant) > 1e-12)
            & (first >= -1e-7)
            & (second >= -1e-7)
            & (first + second <= 1 + 1e-7)
            & (distances > 1e-6)
        )
        if not hits.any():
            continue
        nearest = int(np.argmin(np.where(hits, distances, np.inf)))
        width = float(distances[nearest] + 1e-7)
        if 0.001 < width < 0.047 and np.dot(normals[nearest], axis) > 0.99:
            contacts.append((point + axis * width / 2, axis, width))
            if len(contacts) == count:
                break
    return contacts


def grasp_flange(
    object_transform: np.ndarray,
    contact: tuple[np.ndarray, np.ndarray, float],
    roll: float = 0.0,
    *,
    contact_z_m: float = PAD_CENTER_Z_M,
) -> np.ndarray:
    midpoint, closing_axis, _ = contact
    x = object_transform[:3, :3] @ closing_axis
    desired = np.array([0.0, 0.0, -1.0])
    z = desired - x * np.dot(x, desired)
    if np.linalg.norm(z) < 1e-5:
        desired = np.array([0.0, 1.0, 0.0])
        z = desired - x * np.dot(x, desired)
    z /= np.linalg.norm(z)
    z = scipy_transform.Rotation.from_rotvec(x * roll).apply(z)
    result = np.eye(4)
    result[:3, :3] = np.stack((x, np.cross(z, x), z), axis=1)
    center = object_transform[:3, :3] @ midpoint + object_transform[:3, 3]
    result[:3, 3] = center - z * contact_z_m
    return result


def solve_ik(
    world_flange: np.ndarray, seed_rad: np.ndarray = HOME_RAD
) -> np.ndarray | None:
    from kinematics import ur7e_kinematics as kin

    local = world_flange.copy()
    local[:3, 3] -= BASE_TRANSLATION
    joints_deg = kin.ik(local, np.degrees(seed_rad), limit_margin_deg=5)
    if joints_deg is None or not np.allclose(
        kin.flange_pose(joints_deg), local, atol=2e-5
    ):
        return None
    return np.radians(joints_deg)


def flange_from_joints(joints: np.ndarray) -> np.ndarray:
    from kinematics import ur7e_kinematics as kin

    result = kin.flange_pose(np.degrees(joints[:6]))
    result[:3, 3] += BASE_TRANSLATION
    return result


def load_task_geometry(assets_path: pathlib.Path, tasks: list[dict]) -> list[dict]:
    result = []
    for task in tasks:
        parts, meshes, grasps, pad_grasps, collision_hulls = [], [], [], [], []
        finite_pads = task["id"] in ("05_enclosure", "09_keyed_pegs")
        for part in task["parts"]:
            with np.load(assets_path / part["mesh_cache"], allow_pickle=False) as mesh:
                vertices, faces = mesh["vertices"], mesh["faces"]
                if finite_pads:
                    hulls = [
                        mesh[f"hull_{i}"].astype(np.float32).astype(float)
                        for i in range(part["hull_count"])
                    ]
                    collision_hulls.append(
                        [(points, ConvexHull(points).equations) for points in hulls]
                    )
            nominal = np.asarray(part["assembled_transform"], dtype=float)
            left, _, right = np.linalg.svd(nominal[:3, :3])
            nominal[:3, :3] = left @ right
            parts.append(
                reset_sampling.PartSpec(
                    part["name"], vertices, nominal[:3, :3], part["fixed"]
                )
            )
            meshes.append(vertices)
            salt = int.from_bytes(
                hashlib.sha256(part["name"].encode()).digest()[:4], "little"
            )
            grasps.append(
                sample_antipodal_grasps(vertices, faces, np.random.default_rng(salt))
            )
            if finite_pads:
                pad_grasps.append(
                    sample_normal_grasps(vertices, faces, np.random.default_rng(salt))
                )
        row = {"parts": parts, "vertices": meshes, "grasps": grasps}
        if finite_pads:
            row.update(pad_grasps=pad_grasps, collision_hulls=collision_hulls)
        result.append(row)
    return result


def gripper_spawn_clear(
    flange: np.ndarray,
    joints: np.ndarray,
    transforms: np.ndarray,
    geometry: dict,
    clearance: float,
) -> bool:
    """Reject provable tool/table or tool/part penetration before simulation.

    Vertex containment is a cheap negative filter, not a complete collision test;
    edge crossings and the rest of the arm still require native contact checks.
    """
    solids = geometry.get("gripper_solids", [])
    if not solids:
        return True
    part_points = np.concatenate(
        [
            points @ transform[:3, :3].T + transform[:3, 3]
            for points, transform in zip(geometry["vertices"], transforms, strict=True)
        ]
    )
    local_points = (part_points - flange[:3, 3]) @ flange[:3, :3]
    for points, equations, jaw_index, sign in solids:
        shift = np.array(
            [0.0 if jaw_index is None else sign * joints[jaw_index], 0.0, 0.0]
        )
        world = (points + shift) @ flange[:3, :3].T + flange[:3, 3]
        if world[:, 2].min() < generate_bank.TABLE_HEIGHT_M + clearance:
            return False
        distances = (local_points - shift) @ equations[:, :3].T + equations[:, 3]
        if np.any(np.all(distances < -clearance, axis=1)):
            return False
    return True


def transform_collision_hulls(geometry: dict, transforms: np.ndarray) -> dict:
    lower, upper, equations = [], [], []
    for hulls, transform in zip(geometry["collision_hulls"], transforms, strict=True):
        for points, planes in hulls:
            world = points @ transform[:3, :3].T + transform[:3, 3]
            normals = planes[:, :3] @ transform[:3, :3].T
            lower.append(world.min(0))
            upper.append(world.max(0))
            equations.append(np.c_[normals, planes[:, 3] - normals @ transform[:3, 3]])
    return {
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "equations": equations,
    }


def gripper_collision_clear(
    flange: np.ndarray, joints: np.ndarray, hulls: dict, solids: list, clearance: float
) -> bool:
    """Reject intersections of full convex solids, including crossing edges/faces."""
    for points, planes, jaw, sign in solids:
        shift = np.array([0 if jaw is None else sign * joints[jaw], 0, 0])
        origin = flange[:3, 3] + flange[:3, :3] @ shift
        world = points @ flange[:3, :3].T + origin
        if world[:, 2].min() < generate_bank.TABLE_HEIGHT_M + clearance:
            return False
        nearby = np.flatnonzero(
            np.all(world.min(0) < hulls["upper"] + clearance, axis=-1)
            & np.all(hulls["lower"] < world.max(0) + clearance, axis=-1)
        )
        normals = planes[:, :3] @ flange[:3, :3].T
        tool_planes = np.c_[normals, planes[:, 3] - normals @ origin]
        for index in nearby:
            both = np.concatenate([tool_planes, hulls["equations"][index]])
            # Inflated unit-normal halfspaces conservatively enforce the spawn gap.
            result = scipy_optimize.linprog(
                np.zeros(3),
                A_ub=both[:, :3],
                b_ub=-both[:, 3] + clearance / 2,
                bounds=[(None, None)] * 3,
                method="highs",
            )
            if result.status == 0:
                return False
            if result.status != 2:
                raise RuntimeError("Convex spawn-clearance query failed")
    return True


def assembly_dependencies(task: dict) -> list[set[int]]:
    names = {part.get("name", str(i)): i for i, part in enumerate(task["parts"])}
    dependencies: dict[int, set[int]] = {}

    def resolve(index: int, visiting: set[int]) -> set[int]:
        if index in visiting:
            raise ValueError("Assembly prerequisite cycle")
        if index not in dependencies:
            result = set()
            for name in task["parts"][index].get("requires_assembled", []):
                if name not in names:
                    raise ValueError(f"Unknown assembly prerequisite: {name}")
                parent = names[name]
                result.add(parent)
                result.update(resolve(parent, visiting | {index}))
            dependencies[index] = result
        return dependencies[index]

    return [resolve(i, set()) for i in range(len(task["parts"]))]


def candidate(  # noqa: PLR0912 -- explicit family and physical mating constraints
    task: dict,
    geometry: dict,
    family: str,
    rng: np.random.Generator,
    max_parts: int,
    *,
    config: ValidationConfig,
    active_part: int | None = None,
) -> dict:
    if family not in FAMILIES:
        raise ValueError("Unknown reset family")
    parts = geometry["parts"]
    receiver = next(i for i, part in enumerate(parts) if part.fixed)
    movable = [i for i, part in enumerate(parts) if not part.fixed]
    dependencies = assembly_dependencies(task)
    if active_part is not None and (
        not isinstance(active_part, (int, np.integer)) or active_part not in movable
    ):
        raise ValueError("Requested active part must be a movable part")
    active = int(rng.choice(movable) if active_part is None else active_part)
    placements = reset_sampling.propose_placements(
        parts,
        generate_bank.PROPOSAL_XY_BOUNDS_M,
        generate_bank.TABLE_HEIGHT_M,
        int(rng.integers(2**32)),
        exclusion_aabbs_m=(generate_bank.ROBOT_BASE_EXCLUSION_M,),
    )
    transforms = np.asarray(
        [placements[part.name].transform_world_from_local for part in parts]
    )
    assembled = np.asarray(
        [part["assembled_transform"] for part in task["parts"]], dtype=float
    )
    receiver_delta = transforms[receiver] @ np.linalg.inv(assembled[receiver])
    goals = receiver_delta @ assembled
    grasped_partial = (
        family == "partial_assembly" and rng.random() < config.partial_grasp_probability
    )
    q = np.r_[HOME_RAD, 0.025, 0.025]
    flange = flange_from_joints(q)
    required_assembled = set()
    if family in ("near_mating", "partial_assembly"):
        # Partial multi-part completion is a reset distribution, never a policy stage.
        required_assembled = dependencies[active]
        seated = set(required_assembled)
        for index in movable:
            if (
                index != active
                and active not in dependencies[index]
                and rng.random() < 0.5
            ):
                seated.update({index} | dependencies[index])
        for index in seated:
            transforms[index] = goals[index]
        transforms[active] = goals[active].copy()
        screw_partial = (
            family == "partial_assembly" and "screw_pitch_m" in task["parts"][active]
        )
        if family == "near_mating" or grasped_partial or screw_partial:
            local_axis = task["parts"][active].get("disassembly_axis_local")
            direction = (
                goals[active, :3, :3] @ np.asarray(local_axis, dtype=float)
                if local_axis is not None
                else rng.normal(size=3)
            )
            if np.linalg.norm(direction) < 1e-8:
                raise ValueError("Invalid disassembly axis")
            direction /= np.linalg.norm(direction)
            bounds = (
                (config.partial_grasp_min_offset_m, config.partial_grasp_max_offset_m)
                if family == "partial_assembly"
                else (config.near_mating_min_offset_m, config.near_mating_max_offset_m)
            )
            if task.get("id") == "04_linear_guide" and family == "near_mating":
                # Clear the 50 mm rail end plus the carriage's 11.45 mm half-length.
                bounds = (0.065, 0.069)
            if family == "near_mating":
                bounds = task["parts"][active].get("near_mating_offset_range_m", bounds)
                if (
                    len(bounds) != 2
                    or not 0 < bounds[0] < bounds[1] < config.near_mating_distance_m
                ):
                    raise ValueError("Invalid near-mating withdrawal range")
            distance = rng.uniform(*bounds)
            if "screw_pitch_m" in task["parts"][active]:
                transforms[active] = assembly_motion.withdrawal_transform(
                    goals[active], task["parts"][active], distance
                )
            else:
                transforms[active, :3, 3] += direction * distance
        if (
            task.get("id") == "05_enclosure"
            and family == "partial_assembly"
            and not grasped_partial
        ):
            # An aligned vertical kick simply reseats this lid. Start slightly
            # offset above the rim; native gravity must establish real support.
            center = np.mean(
                [
                    geometry["vertices"][active].min(axis=0),
                    geometry["vertices"][active].max(axis=0),
                ],
                axis=0,
            )
            delta = np.eye(4)
            yaw = rng.uniform(np.radians(3), np.radians(12)) * rng.choice([-1, 1])
            delta[:3, :3] = scipy_transform.Rotation.from_euler("z", yaw).as_matrix()
            lateral = np.zeros(3)
            lateral[int(rng.integers(2))] = rng.uniform(0.004, 0.008) * rng.choice(
                [-1, 1]
            )
            lateral[2] = rng.uniform(0.004, 0.008)
            delta[:3, 3] = center - delta[:3, :3] @ center + lateral
            transforms[active] = goals[active] @ delta
        # Unheld straight-fit partials are physically dislodged below. Threaded
        # partials start on the helix, then face the same perturbation/settling gates.
        # Grasp-supported
        # partials compose collision-free partial poses with real grasp sampling,
        # as in OmniReset; gravity/perturbation/retention still validate the result.
    elif family == "held_transport":
        transforms[active, 2, 3] += rng.uniform(0.08, 0.18)
    if family not in ("loose", "partial_assembly") or grasped_partial:
        finite_pads = task.get("id") in ("05_enclosure", "09_keyed_pegs")
        contacts = geometry["pad_grasps" if finite_pads else "grasps"][active]
        world_hulls = (
            transform_collision_hulls(geometry, transforms) if finite_pads else None
        )
        if not contacts:
            raise RuntimeError("no_antipodal_grasp")
        for _ in range(config.grasp_trials):
            contact = contacts[int(rng.integers(len(contacts)))]
            # OmniReset varies gripper standoff. Sample real positions along
            # the 40 mm pad, including a shallow fingertip grasp for low parts.
            flange = grasp_flange(
                transforms[active],
                contact,
                rng.uniform(-0.4, 0.4),
                contact_z_m=PAD_CENTER_Z_M
                + rng.uniform(0.0, config.grasp_axial_offset_m),
            )
            if family == "near_grasp":
                flange[:3, 3] -= flange[:3, 2] * rng.uniform(0.003, 0.02)
            arm = solve_ik(flange)
            if arm is None:
                continue
            openings = (
                np.unique(
                    np.minimum(
                        0.025,
                        contact[2] / 2 + np.array([0.0005, 0.0015, 0.003, 0.006, 0.01]),
                    )
                )
                if finite_pads
                else [0.025]
            )
            for opening in openings:
                q = np.r_[arm, np.full(2, opening)]
                if gripper_spawn_clear(
                    flange, q, transforms, geometry, config.gripper_spawn_clearance_m
                ) and (
                    not finite_pads
                    or gripper_collision_clear(
                        flange,
                        q,
                        world_hulls,
                        geometry["gripper_solids"],
                        config.gripper_spawn_clearance_m,
                    )
                ):
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("unreachable_or_colliding_grasp")
    padded = np.tile(np.eye(4), (max_parts, 1, 1))
    padded_goals = padded.copy()
    padded[: len(parts)], padded_goals[: len(parts)] = transforms, goals
    return {
        "part_poses": matrix_pose(padded),
        "goals": matrix_pose(padded_goals),
        "joint_q": q,
        "family": family,
        "grasped_partial": grasped_partial,
        "active": active,
        "required_assembled": sorted(required_assembled),
        "flange": flange,
        "gripper_open": float(q[6] / 0.025),
    }


def numpy_snapshot(snapshot: dict) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().copy()
        if hasattr(value, "detach")
        else np.asarray(value).copy()
        for key, value in snapshot.items()
    }


def rejection_reason(
    state: dict,
    row: int,
    geometry: dict,
    config: ValidationConfig,
    previous: dict | None = None,
    *,
    settling: bool = False,
    held_part: int | None = None,
) -> str | None:
    for key in (
        "part_poses",
        "joint_q",
        "joint_qd",
        "part_velocities",
        "max_penetration_m",
    ):
        if not np.isfinite(state[key][row]).all():
            return "nonfinite"
    if state["safety_failure"][row]:
        return "safety_failure"
    penetration_limit = (
        config.transient_max_penetration_m if settling else config.max_penetration_m
    )
    table_tolerance = (
        config.transient_table_tolerance_m if settling else config.table_tolerance_m
    )
    pair_keys = ("max_pad_part_penetration_m", "max_rigid_penetration_m")
    if held_part is not None and all(key in state for key in pair_keys):
        pad_depth = np.asarray(state[pair_keys[0]][row])
        rigid_depth = state[pair_keys[1]][row]
        if not np.isfinite(pad_depth).all() or not np.isfinite(rigid_depth):
            return "nonfinite"
        if (
            not 0 <= held_part < len(geometry["parts"])
            or geometry["parts"][held_part].fixed
        ):
            return "invalid_held_part"
        other_depth = np.delete(pad_depth, held_part)
        held_limit = max(penetration_limit, config.max_held_pad_penetration_m)
        if (
            rigid_depth > penetration_limit
            or (other_depth > penetration_limit).any()
            or pad_depth[held_part] > held_limit
        ):
            return "penetration"
    elif state["max_penetration_m"][row] > penetration_limit:
        return "penetration"
    poses = state["part_poses"][row, : len(geometry["parts"])]
    if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4):
        return "invalid_quaternion"
    matrices = pose_matrix(poses)
    for _part, vertices, transform in zip(
        geometry["parts"], geometry["vertices"], matrices, strict=True
    ):
        xyz = vertices @ transform[:3, :3].T + transform[:3, 3]
        if (xyz[:, :2] < generate_bank.TABLE_XY_BOUNDS_M[:, 0]).any() or (
            xyz[:, :2] > generate_bank.TABLE_XY_BOUNDS_M[:, 1]
        ).any():
            return "table_escape"
        if xyz[:, 2].min() < generate_bank.TABLE_HEIGHT_M - table_tolerance:
            return "below_table"
    velocity = state["part_velocities"][row, : len(geometry["parts"])]
    if (
        np.linalg.norm(velocity[:, :3], axis=1) > config.max_linear_speed_m_s
    ).any() or (
        np.linalg.norm(velocity[:, 3:], axis=1) > config.max_angular_speed_rad_s
    ).any():
        return "moving"
    if np.abs(state["joint_qd"][row]).max() > config.max_joint_speed_rad_s:
        return "moving"
    if previous is None:
        return "needs_history"
    old = previous["part_poses"][row, : len(geometry["parts"])]
    if (
        np.linalg.norm(poses[:, :3] - old[:, :3], axis=1) > config.max_position_change_m
    ).any() or (
        quaternion_distance(poses[:, 3:], old[:, 3:]) > config.max_angle_change_rad
    ).any():
        return "moving"
    return None


def retained(before: np.ndarray, after: np.ndarray, config: ValidationConfig) -> bool:
    return bool(
        np.linalg.norm(before[:3, 3] - after[:3, 3]) <= config.retention_position_m
        and quaternion_distance(matrix_pose(before)[3:], matrix_pose(after)[3:])
        <= config.retention_angle_rad
    )


def summarize_contacts(
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    forces: np.ndarray,
    distances: np.ndarray,
    world_ids: np.ndarray,
    *,
    labels: list[str],
    num_envs: int,
) -> list[list[dict]]:
    """Final-step contact evidence; summed magnitudes are not net grasp force."""
    rows: list[dict] = [{} for _ in range(num_envs)]
    for a, b, force, distance, world in zip(
        shape_a, shape_b, forces, distances, world_ids, strict=True
    ):
        if a < 0 or b < 0 or not 0 <= world < num_envs:
            continue
        label_a, label_b = (
            labels[index % len(labels)].split("/hull_")[0] for index in (a, b)
        )
        pair = tuple(sorted((label_a, label_b)))
        row = rows[world].setdefault(
            pair,
            {
                "pair": pair,
                "count": 0,
                "depth_m": 0.0,
                "sum_force_magnitudes_n": 0.0,
                "net_force_on_first_n": [0.0, 0.0, 0.0],
            },
        )
        row["count"] += 1
        row["depth_m"] = max(row["depth_m"], float(max(0.0, -distance)))
        row["sum_force_magnitudes_n"] += float(np.linalg.norm(force[:3]))
        # Newton reports world-space force on shape0; use one pair ordering.
        sign = 1.0 if label_a == pair[0] else -1.0
        row["net_force_on_first_n"] = (
            np.asarray(row["net_force_on_first_n"]) + sign * force[:3]
        ).tolist()
    return [sorted(row.values(), key=lambda item: item["pair"]) for row in rows]


def family_rejection(
    state: dict, row: int, record: dict, geometry: dict, config: ValidationConfig
) -> str | None:
    for index in record.get("required_assembled", []):
        current = pose_matrix(state["part_poses"][row, index])
        goal = pose_matrix(state["goals"][row, index])
        vertices = geometry["vertices"][index]
        delta = (
            vertices @ (current[:3, :3] - goal[:3, :3]).T + current[:3, 3] - goal[:3, 3]
        )
        if np.sqrt(np.mean(np.sum(delta**2, axis=-1))) >= 0.001:
            return "prerequisite_not_assembled"
    active = record["active"]
    distance = np.linalg.norm(
        state["part_poses"][row, active, :3] - state["goals"][row, active, :3]
    )
    if (
        record["family"] == "partial_assembly"
        and distance < config.partial_goal_min_distance_m
    ):
        return "partial_not_dislodged"
    if (
        record["family"] == "partial_assembly"
        and distance > config.partial_goal_max_distance_m
    ):
        return "partial_no_longer_near_goal"
    if record["family"] == "near_mating" and distance > config.near_mating_distance_m:
        return "no_longer_near_mating"
    if record["family"] == "near_grasp":
        flange = flange_from_joints(state["joint_q"][row])
        pinch = flange[:3, 3] + flange[:3, 2] * PAD_CENTER_Z_M
        transform = pose_matrix(state["part_poses"][row, active])
        surface = geometry["vertices"][active] @ transform[:3, :3].T + transform[:3, 3]
        if np.linalg.norm(surface - pinch, axis=1).min() > config.near_grasp_distance_m:
            return "no_longer_near_grasp"
    return None


def closing_openings(initial, target, step: int, steps: int) -> np.ndarray:
    """Smooth reset-generation close avoids a full-speed jaw target transient.

    This is an explicit UR7 generation choice, not a policy action modification.
    One-second smoothstep traverses 25 mm at a peak 37.5 mm/s, below the
    native 100 mm/s jaw limit without changing that physical acceptance guard.
    """
    fraction = np.clip(step / steps, 0.0, 1.0)
    blend = fraction * fraction * (3 - 2 * fraction)
    return np.asarray(initial) + blend * (np.asarray(target) - initial)


def requires_grasp(record: dict) -> bool:
    return record["family"] in ("held_transport", "near_mating") or bool(
        record.get("grasped_partial", False)
    )


def grasp_challenge_displacement(
    task: dict, record: dict, part_pose: np.ndarray, config: ValidationConfig
) -> np.ndarray:
    distance = (
        config.partial_grasp_lift_height_m
        if record.get("grasped_partial", False)
        else config.near_mating_lift_height_m
        if record["family"] == "near_mating"
        else config.lift_height_m
    )
    if record.get("grasped_partial", False):
        # A stationary supported part must exceed the grasp-slip tolerance.
        distance = max(distance, 2 * config.retention_position_m)
    direction = np.array([0.0, 0.0, 1.0])
    part = task["parts"][record["active"]]
    axis = part.get("grasp_challenge_axis_local")
    if "screw_pitch_m" in part and record["family"] == "near_mating":
        # Called only after the threaded part has fully cleared its receiver.
        distance = 2 * config.retention_position_m
        axis = part["disassembly_axis_local"]
    if axis is None and task.get("id") == "04_linear_guide":
        axis = part["disassembly_axis_local"]
    if axis is not None and record["family"] in ("near_mating", "partial_assembly"):
        # Engaged carriages translate along the raceway; vertical lifting jams them.
        direction = pose_matrix(part_pose)[:3, :3] @ np.asarray(axis, dtype=float)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError("Invalid grasp challenge axis")
        direction /= np.linalg.norm(direction)
    return direction * distance


def grasp_challenge_completed(before, after, displacement, config) -> bool:
    distance = np.linalg.norm(displacement)
    progress = np.dot(after[:3] - before[:3], displacement / distance)
    return bool(progress >= distance - config.retention_position_m)


def requires_screw_challenge(task: dict, record: dict, part_pose: np.ndarray) -> bool:
    part = task["parts"][record["active"]]
    if "screw_pitch_m" not in part or record["family"] not in (
        "partial_assembly",
        "near_mating",
    ):
        return False
    if record["family"] == "near_mating" and "screw_disengaged_offset_m" in part:
        goal = pose_matrix(record["goals"][record["active"]])
        axis = goal[:3, :3] @ np.asarray(part["disassembly_axis_local"], dtype=float)
        withdrawal = np.dot(part_pose[:3] - goal[:3, 3], axis / np.linalg.norm(axis))
        return bool(withdrawal < float(part["screw_disengaged_offset_m"]))
    # Without an authored clearance threshold, assume the part may still engage.
    return True


def screw_challenge_completed(before, after, part, config) -> bool:
    initial, final = pose_matrix(before), pose_matrix(after)
    axis = np.asarray(part["disassembly_axis_local"], dtype=float)
    axis /= np.linalg.norm(axis)
    relative_rotation = scipy_transform.Rotation.from_matrix(
        initial[:3, :3].T @ final[:3, :3]
    ).as_rotvec()
    angle = float(np.dot(relative_rotation, axis))
    advance = float(np.dot(final[:3, 3] - initial[:3, 3], initial[:3, :3] @ axis))
    pitch_per_radian = float(part["screw_pitch_m"]) / (2 * np.pi)
    return bool(
        angle >= 0.8 * SCREW_CHALLENGE_ANGLE_RAD
        and np.linalg.norm(relative_rotation - angle * axis)
        <= config.retention_angle_rad
        and abs(advance - SCREW_CHALLENGE_ANGLE_RAD * pitch_per_radian)
        <= SCREW_CHALLENGE_AXIAL_TOLERANCE_M
        and abs(advance - angle * pitch_per_radian) <= SCREW_CHALLENGE_AXIAL_TOLERANCE_M
    )


def physically_validate(  # noqa: PLR0912 -- explicit physical validation phases
    env: Any,
    records: list[dict],
    geometries: list[dict],
    config: ValidationConfig,
    rng: np.random.Generator,
    *,
    diagnostics: list[dict] | None = None,
) -> tuple[dict, list[str | None]]:
    """Close, challenge and settle real contacts; never attach objects virtually.

    OmniReset checks collision/stability acceptance after settling. Bounded soft
    contact transients are recorded separately from the unchanged final gates.
    Invalid spawn, forbidden robot contacts and solver failures remain failures.
    """
    import torch

    device = env.device
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(
        value, dtype=dtype, device=device
    )
    task_ids = np.asarray([record["task_id"] for record in records])
    q = np.asarray([record["joint_q"] for record in records]).copy()
    env.initialize(
        tensor(task_ids, torch.int64),
        tensor(np.asarray([r["part_poses"] for r in records])),
        tensor(q),
        tensor(np.asarray([r["goals"] for r in records])),
    )
    openings = np.asarray(
        [
            0.0 if requires_grasp(record) else record["gripper_open"]
            for record in records
        ]
    )
    terminal = [None] * len(records)
    initial_openings = np.asarray([record["gripper_open"] for record in records])
    relative = {}
    grasp_challenges = {}
    screw_challenges = {}
    parked = set()
    diagnostics = [] if diagnostics is None else diagnostics
    diagnostics.extend(
        {
            "task": env.task_names[task_ids[i]],
            "family": record["family"],
            "grasp_supported": requires_grasp(record),
            "held_part": record["active"] if requires_grasp(record) else None,
            "phases": {},
            "rejection": None,
        }
        for i, record in enumerate(records)
    )

    def observe(phase):
        state = numpy_snapshot(env.snapshot())
        newly_rejected = []
        for i, record in enumerate(records):
            if i in parked:
                continue
            row = diagnostics[i]
            phase_row = row["phases"].setdefault(
                phase,
                {
                    "checks": 0,
                    "peak_penetration_m": 0.0,
                    "peak_part_speed_m_s": 0.0,
                    "peak_joint_speed_rad_s": 0.0,
                },
            )
            phase_row["checks"] += 1
            phase_row["peak_penetration_m"] = max(
                phase_row["peak_penetration_m"], float(state["max_penetration_m"][i])
            )
            for kind in ("pad", "rigid"):
                field = f"max_{kind}_penetration_m"
                if field in state:
                    key = f"peak_{kind}_penetration_m"
                    phase_row[key] = max(
                        phase_row.get(key, 0.0), float(state[field][i])
                    )
            phase_row["peak_part_speed_m_s"] = max(
                phase_row["peak_part_speed_m_s"],
                float(np.linalg.norm(state["part_velocities"][i, :, :3], axis=1).max()),
            )
            phase_row["peak_joint_speed_rad_s"] = max(
                phase_row["peak_joint_speed_rad_s"],
                float(np.abs(state["joint_qd"][i]).max()),
            )
            reason = rejection_reason(
                state,
                i,
                geometries[task_ids[i]],
                config,
                state,
                settling=phase != "spawn",
                held_part=record["active"]
                if phase != "spawn" and requires_grasp(record)
                else None,
            )
            if reason not in (None, "moving"):
                terminal[i] = terminal[i] or reason
            if i in relative and terminal[i] is None:
                now = np.linalg.inv(
                    flange_from_joints(state["joint_q"][i])
                ) @ pose_matrix(state["part_poses"][i, record["active"]])
                phase_row["peak_grasp_translation_m"] = max(
                    phase_row.get("peak_grasp_translation_m", 0.0),
                    float(np.linalg.norm(relative[i][:3, 3] - now[:3, 3])),
                )
                phase_row["peak_grasp_rotation_rad"] = max(
                    phase_row.get("peak_grasp_rotation_rad", 0.0),
                    float(
                        quaternion_distance(
                            matrix_pose(relative[i])[3:], matrix_pose(now)[3:]
                        )
                    ),
                )
                if not retained(relative[i], now, config):
                    terminal[i] = "grasp_lost"
            if terminal[i] is not None:
                row["rejection"] = terminal[i]
                row["rejected_phase"] = phase
                if "failure_flags" in state:
                    row["failure_flags"] = int(state["failure_flags"][i])
                newly_rejected.append(i)
        if newly_rejected:
            # A rejected lane must not continue forcing an invalid grasp while
            # the other lanes settle. This prevents cross-lane capacity failures.
            if any(reason is None for reason in terminal):
                park = getattr(env, "park_worlds", None)
                if park is None:
                    raise RuntimeError(
                        "Mixed reset validation requires env.park_worlds"
                    )
                park(tensor(newly_rejected, torch.int64))
            for i in newly_rejected:
                parked.add(i)
                q[i] = np.r_[HOME_RAD, 0.025, 0.025]
                openings[i] = 1.0
        return state

    def advance(phase, action):
        try:
            action()
            return observe(phase)
        except RuntimeError as error:
            for row in diagnostics:
                row["physics_exception"] = str(error)
                row["exception_phase"] = phase
            print(json.dumps({"reset_validation_exception": diagnostics}), flush=True)
            raise

    inspect = getattr(env, "inspect_contacts", None)
    if inspect is not None:
        inspect()
    state = observe("spawn")
    perturb = getattr(env, "perturb_objects", None)
    closing_forces = np.zeros((len(records), env.max_parts, 3))
    for i, record in enumerate(records):
        if requires_grasp(record):
            part = env.tasks[task_ids[i]]["parts"][record["active"]]
            closing_forces[i, record["active"], 2] = float(part["mass_kg"]) * 9.81
    for close_step in range(1, config.close_steps + 1):
        if all(reason is not None for reason in terminal):
            break
        close_command = closing_openings(
            initial_openings, openings, close_step, config.close_steps
        )
        close_command[list(parked)] = 1.0
        if perturb is not None and np.any(closing_forces):
            # Cited OmniReset disables gravity for the first second of grasp
            # sampling. Compensate only the sampled held object here, then remove
            # compensation for the real gravity/perturbation/retention tests.
            closing_forces[list(parked)] = 0.0
            env.hold(tensor(q[:, :6]), tensor(close_command), steps=0)
            state = advance(
                "close_gravity_compensated",
                lambda: perturb(tensor(closing_forces), steps=1),
            )
        else:
            state = advance(
                "close",
                lambda command=close_command: env.hold(
                    tensor(q[:, :6]), tensor(command), steps=1
                ),
            )
    held = [
        i for i, r in enumerate(records) if requires_grasp(r) and terminal[i] is None
    ]
    partial = [
        i
        for i, r in enumerate(records)
        if r["family"] == "partial_assembly"
        and not requires_grasp(r)
        and terminal[i] is None
    ]
    if held or partial:
        perturb = getattr(env, "perturb_objects", None)
        if perturb is None:
            for i in held + partial:
                terminal[i] = "missing_perturbation_validation"
        else:
            relative = {
                i: np.linalg.inv(flange_from_joints(state["joint_q"][i]))
                @ pose_matrix(state["part_poses"][i, records[i]["active"]])
                for i in held
            }
            forces = np.zeros((len(records), env.max_parts, 3))
            for i in held + partial:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                forces[i, records[i]["active"]] = direction * config.perturb_force_n
                if i in partial:
                    part = env.tasks[task_ids[i]]["parts"][records[i]["active"]]
                    local_axis = part.get("disassembly_axis_local")
                    if local_axis is not None:
                        direction = pose_matrix(
                            state["part_poses"][i, records[i]["active"]]
                        )[:3, :3] @ np.asarray(local_axis, dtype=float)
                        direction /= np.linalg.norm(direction)
                    forces[i, records[i]["active"]] = float(part["mass_kg"]) * (
                        direction * config.dislodge_acceleration_m_s2
                        + np.array([0.0, 0.0, 9.81])
                    )
            for challenge_step in range(
                config.perturb_steps if held else config.partial_perturb_steps
            ):
                forces[list(parked)] = 0.0
                if challenge_step >= config.partial_perturb_steps:
                    forces[partial] = 0.0
                if all(reason is not None for reason in terminal):
                    break
                state = advance("perturb", lambda: perturb(tensor(forces), steps=1))
            for i in held:
                if terminal[i] is not None:
                    continue
                lift = flange_from_joints(state["joint_q"][i])
                part_pose = state["part_poses"][i, records[i]["active"]].copy()
                task = env.tasks[task_ids[i]]
                if requires_screw_challenge(task, records[i], part_pose):
                    part = task["parts"][records[i]["active"]]
                    before = pose_matrix(part_pose)
                    screw_challenges[i] = (
                        part_pose,
                        part,
                        np.linalg.inv(before) @ lift,
                    )
                    diagnostics[i]["grasp_challenge"] = {
                        "protocol": "helical_waypoints_45deg",
                        "rotation_rad": SCREW_CHALLENGE_ANGLE_RAD,
                        "axial_advance_m": float(part["screw_pitch_m"]) / 8,
                        "axial_and_pitch_tolerance_m": SCREW_CHALLENGE_AXIAL_TOLERANCE_M,
                        # Helical command motion alone does not prove contact.
                        "thread_contact_coupling_claimed": False,
                    }
                    continue
                offset = grasp_challenge_displacement(
                    task, records[i], part_pose, config
                )
                lift[:3, 3] += offset
                grasp_challenges[i] = (part_pose, offset)
                diagnostics[i]["grasp_challenge"] = {
                    "protocol": "projected_translation"
                }
                arm = solve_ik(lift, state["joint_q"][i, :6])
                if arm is None:
                    terminal[i] = "lift_unreachable"
                else:
                    q[i, :6] = arm
            start_q = state["joint_q"][:, :6].copy()
            for alpha in np.linspace(0, 1, 25)[1:]:
                if all(reason is not None for reason in terminal):
                    break
                targets = start_q + alpha * (q[:, :6] - start_q)
                for i, (before, part, flange_in_part) in screw_challenges.items():
                    if terminal[i] is not None:
                        continue
                    target_part = assembly_motion.withdrawal_transform(
                        pose_matrix(before),
                        part,
                        float(part["screw_pitch_m"]) * alpha / 8,
                    )
                    # Solve every helical waypoint; joint interpolation to the
                    # endpoint would cut across the engaged thread surfaces.
                    arm = solve_ik(target_part @ flange_in_part, q[i, :6])
                    if arm is None:
                        terminal[i] = "screw_waypoint_unreachable"
                        targets[i] = state["joint_q"][i, :6]
                    else:
                        targets[i] = q[i, :6] = arm
                targets[list(parked)] = HOME_RAD
                state = advance(
                    "lift",
                    lambda targets=targets: env.hold(
                        tensor(targets), tensor(openings), steps=1
                    ),
                )
    stable = np.zeros(len(records), dtype=int)
    for _ in range(config.max_settle_steps):
        if all(reason is not None for reason in terminal):
            break
        previous = state
        state = advance(
            "settle", lambda: env.hold(tensor(q[:, :6]), tensor(openings), steps=1)
        )
        for i in range(len(records)):
            if terminal[i] is not None:
                continue
            reason = rejection_reason(
                state,
                i,
                geometries[task_ids[i]],
                config,
                previous,
                held_part=records[i]["active"] if requires_grasp(records[i]) else None,
            )
            stable[i] = stable[i] + 1 if reason is None else 0
        if np.all(
            (stable >= config.stable_steps)
            | np.asarray([r is not None for r in terminal])
        ):
            break
    for i, record in enumerate(records):
        if stable[i] < config.stable_steps:
            terminal[i] = terminal[i] or "settling_timeout"
        if terminal[i] is None:
            terminal[i] = family_rejection(
                state, i, record, geometries[task_ids[i]], config
            )
        if terminal[i] is None and i in grasp_challenges:
            before, offset = grasp_challenges[i]
            if not grasp_challenge_completed(
                before, state["part_poses"][i, record["active"]], offset, config
            ):
                terminal[i] = "grasp_challenge_incomplete"
        if terminal[i] is None and i in screw_challenges:
            before, part, _ = screw_challenges[i]
            if not screw_challenge_completed(
                before, state["part_poses"][i, record["active"]], part, config
            ):
                terminal[i] = "screw_challenge_incomplete"
        diagnostics[i].update(
            rejection=terminal[i],
            final_penetration_m=float(state["max_penetration_m"][i]),
            consecutive_strict_stable_checks=int(stable[i]),
        )
        if i not in parked:
            diagnostics[i]["final_instantaneous_rejection"] = rejection_reason(
                state,
                i,
                geometries[task_ids[i]],
                config,
                state,
                held_part=record["active"] if requires_grasp(record) else None,
            )
            for kind in ("pad", "rigid"):
                key = f"max_{kind}_penetration_m"
                if key in state:
                    diagnostics[i][f"final_{kind}_penetration_m"] = float(state[key][i])
    if hasattr(env, "contacts"):
        data = env.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        contact_rows = summarize_contacts(
            env.contacts.rigid_contact_shape0.numpy()[:count],
            env.contacts.rigid_contact_shape1.numpy()[:count],
            env.contacts.force.numpy()[:count],
            data.contact.dist.numpy()[:count],
            data.contact.worldid.numpy()[:count],
            labels=env.maps["shape_labels"],
            num_envs=env.num_envs,
        )
        for i, row in enumerate(contact_rows):
            if i not in parked:
                diagnostics[i]["final_contacts"] = row
    print(json.dumps({"reset_validation": diagnostics}), flush=True)
    return state, terminal


def entry_content_id(entry: dict, task_name: str, part_count: int) -> str:
    """Stable physical-state ID, ignoring family labels and quaternion signs.

    Quantization at 1e-6 in SI/radians absorbs serialization noise. This is
    conservative duplicate detection, not an assertion of dynamical equivalence.
    """
    if not isinstance(part_count, (int, np.integer)) or not 0 < part_count <= len(
        entry["part_poses"]
    ):
        raise ValueError("Invalid task part count")
    digest = hashlib.sha256(task_name.encode("utf-8"))
    fields = (
        "part_poses",
        "goals",
        "joint_q",
        "joint_qd",
        "joint_targets",
        "jaw_servo",
        "part_velocities",
        "gripper_open",
    )
    if "arm_servo" in entry or "arm_servo_velocity" in entry:
        fields += ("arm_servo", "arm_servo_velocity")
    for key in fields:
        if key not in entry:
            raise ValueError(f"Missing bank state field {key}")
        value = np.asarray(entry[key], dtype=np.float64).copy()
        if key in ("part_poses", "goals", "part_velocities"):
            value = value[:part_count]
        if not np.isfinite(value).all() or (np.abs(value) > 1e9).any():
            raise ValueError("Cannot identify nonfinite or unbounded reset state")
        if key in ("part_poses", "goals"):
            norms = np.linalg.norm(value[:, 3:], axis=1, keepdims=True)
            if np.any(norms < 1e-10):
                raise ValueError("Invalid bank quaternion")
            quaternion = value[:, 3:] / norms
            greatest = np.argmax(np.abs(quaternion), axis=1)
            sign = np.sign(quaternion[np.arange(len(quaternion)), greatest])
            value[:, 3:] = quaternion * sign[:, None]
        quantized = np.rint(value / 1e-6).astype("<i8")
        digest.update(key.encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(quantized.tobytes())
    return digest.hexdigest()


def register_entry(
    entry: dict, task_name: str, part_count: int, seen: set[str]
) -> bool:
    content_id = entry_content_id(entry, task_name, part_count)
    if content_id in seen:
        return False
    seen.add(content_id)
    entry["entry_ids"] = content_id
    return True


def validate_bank_entries(arrays: dict, metadata: dict) -> list[str]:
    """Recompute IDs before trusting metadata for train/evaluation separation."""
    count = len(arrays.get("task_ids", []))
    fields = (
        "part_poses",
        "goals",
        "joint_q",
        "joint_qd",
        "joint_targets",
        "jaw_servo",
        "part_velocities",
        "gripper_open",
        "family",
        "entry_ids",
    )
    schema = metadata.get("content_id_schema", 1)
    if schema == 2:
        fields += ("arm_servo", "arm_servo_velocity")
    elif schema != 1 or "arm_servo" in arrays or "arm_servo_velocity" in arrays:
        raise ValueError("Inconsistent bank content-ID schema")
    if not count or any(
        key not in arrays or len(arrays[key]) != count for key in fields
    ):
        raise ValueError("Incomplete bank content-ID schema")
    names, part_counts = (
        metadata.get("task_names", []),
        metadata.get("task_part_counts", []),
    )
    if len(names) != len(part_counts) or not names:
        raise ValueError("Missing bank task geometry schema")
    ids = []
    for row, index in enumerate(arrays["task_ids"]):
        if index != int(index) or not 0 <= int(index) < len(names):
            raise ValueError("Invalid bank task identity")
        index = int(index)
        entry = {key: arrays[key][row] for key in fields}
        ids.append(entry_content_id(entry, names[index], part_counts[index]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate physical reset states in bank")
    if ids != list(arrays["entry_ids"]) or ids != metadata.get("entry_ids"):
        raise ValueError("Reset bank content IDs do not match physical state")
    return ids


def assert_disjoint_bank_entries(train_metadata: dict, eval_metadata: dict) -> None:
    if train_metadata.get("content_id_schema", 1) != eval_metadata.get(
        "content_id_schema", 1
    ):
        raise ValueError("Training/evaluation banks use different content-ID schemas")
    train_ids, eval_ids = (
        train_metadata.get("entry_ids", []),
        eval_metadata.get("entry_ids", []),
    )
    if not train_ids or not eval_ids:
        raise ValueError("Both banks require verified entry content IDs")
    overlap = set(train_ids).intersection(eval_ids)
    if overlap:
        raise ValueError(
            f"Training/evaluation reset banks overlap in {len(overlap)} physical states"
        )


def request_schedule(
    task_count: int, samples_per_task: int, families: tuple[str, ...]
) -> list[tuple[int, str]]:
    if (
        task_count < 1
        or not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES for f in families)
    ):
        raise ValueError("Invalid task/family selection")
    if samples_per_task < len(families):
        raise ValueError("At least one sample per selected family is required")
    # Interleave tasks and rotate family order to keep small physical batches mixed.
    return [
        (task, families[(sample + task) % len(families)])
        for sample in range(samples_per_task)
        for task in range(task_count)
    ]


def part_request_schedule(
    tasks: list[dict], samples_per_task: int, families: tuple[str, ...], seed: int
) -> list[tuple[int, str, int]]:
    """Balance each family by movable part; rotate remainder quotas across seeds."""
    movable = [
        [i for i, part in enumerate(task["parts"]) if not part["fixed"]]
        for task in tasks
    ]
    if any(not parts for parts in movable):
        raise ValueError("Every task must have a movable part")
    counts = collections.Counter()
    requests = []
    for task, family in request_schedule(len(tasks), samples_per_task, families):
        offset = seed + task + FAMILIES.index(family)
        active = movable[task][(offset + counts[task, family]) % len(movable[task])]
        requests.append((task, family, active))
        counts[task, family] += 1
    return requests


def _json_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported failure-capture value: {type(value).__name__}")


@contextlib.contextmanager
def capture_failed_batch(
    output: pathlib.Path, records: list[dict], rng: np.random.Generator, metadata: dict
):
    """Save one portable replay input without masking the backend exception."""
    captured = copy.deepcopy(
        {**metadata, "records": records, "rng_state": rng.bit_generator.state}
    )
    try:
        yield
    except Exception as error:
        try:
            captured["exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            (output / "failed_batch.json").write_text(
                json.dumps(captured, default=_json_numpy, allow_nan=False) + "\n"
            )
        except Exception as capture_error:
            error.add_note(f"Failed to save reset batch: {capture_error}")
        raise


def build(
    env: Any,
    assets_path: pathlib.Path,
    output: pathlib.Path,
    samples_per_task: int,
    seed: int,
    *,
    config: ValidationConfig,
    families: tuple[str, ...] = FAMILIES,
    max_wall_seconds: float | None = None,
) -> dict:
    if samples_per_task < len(families) or seed < 0:
        raise ValueError(
            "Request at least one sample per selected family and a nonnegative seed"
        )
    if max_wall_seconds is not None and (
        not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0
    ):
        raise ValueError("Wall-time budget must be positive and finite")
    deadline = None if max_wall_seconds is None else time.monotonic() + max_wall_seconds
    wall_budget_exhausted = False
    if (output / "bank.npz").exists() or (output / "bank.json").exists():
        raise FileExistsError(
            "Use a new output directory; validated banks are immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    validation_log = output / "validation.jsonl"
    geometry = load_task_geometry(assets_path, env.tasks)
    from isaac.fanuc_sim import native_model, robot_profile

    gripper = robot_profile.of(env.profile).gripper
    solids = []
    for body, _label, points, _role in native_model.gripper_shapes(env.profile)[0]:
        jaw = (
            6 if body == gripper.bodies[1] else 7 if body == gripper.bodies[2] else None
        )
        solids.append(
            (points, ConvexHull(points).equations, jaw, -1.0 if jaw == 6 else 1.0)
        )
    for task_geometry in geometry:
        task_geometry["gripper_solids"] = solids
    rng = np.random.default_rng(seed)
    wanted = part_request_schedule(env.tasks, samples_per_task, families, seed)
    accepted = []
    seen_ids = set()
    batch_count = 0
    rejects = collections.Counter()
    attempts = 0
    proposal_index = 0
    limit = len(wanted) * config.candidate_attempts_per_sample
    while wanted and attempts < limit and batch_count < config.max_batches:
        if deadline is not None and time.monotonic() >= deadline:
            wall_budget_exhausted = True
            break
        records = []
        requested = []
        while len(records) < env.num_envs and attempts < limit:
            # Spare lanes test independent alternatives for outstanding quotas.
            # Repeating one state across every lane wastes most of the physics
            # budget when only a difficult family remains.
            task_id, family, active_part = wanted[proposal_index % len(wanted)]
            proposal_index += 1
            attempts += 1
            try:
                proposal = candidate(
                    env.tasks[task_id],
                    geometry[task_id],
                    family,
                    rng,
                    env.max_parts,
                    config=config,
                    active_part=active_part,
                )
            except RuntimeError as error:
                rejects[f"{env.task_names[task_id]}/{family}/{error}"] += 1
                continue
            proposal["task_id"] = task_id
            records.append(proposal)
            requested.append((task_id, family, active_part))
        if not records:
            break
        actual_count = len(records)
        # Physical batch has fixed size; padding is never added to the bank.
        records += [records[-1]] * (env.num_envs - len(records))
        diagnostics = []
        physics_started = time.perf_counter()
        try:
            with capture_failed_batch(
                output,
                records,
                rng,
                {
                    "schema_version": 1,
                    "batch": batch_count,
                    "seed": seed,
                    "actual_count": actual_count,
                    "task_names": env.task_names,
                    "assets_path": str(assets_path.resolve()),
                    "validation": dataclasses.asdict(config),
                    "environment_config": env.config,
                    "physics_identity": getattr(env, "physics_identity", None),
                },
            ):
                state, reasons = physically_validate(
                    env, records, geometry, config, rng, diagnostics=diagnostics
                )
        finally:
            backend_error = sys.exception()
            try:
                with validation_log.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "batch": batch_count,
                                "physics_seconds": time.perf_counter()
                                - physics_started,
                                "entries": diagnostics,
                            }
                        )
                        + "\n"
                    )
            except (OSError, TypeError, ValueError) as log_error:
                if backend_error is None:
                    raise
                backend_error.add_note(f"Failed to save validation log: {log_error}")
        batch_count += 1
        for i in range(actual_count):
            task_id, family, active_part = requested[i]
            if reasons[i] is not None:
                rejects[f"{env.task_names[task_id]}/{family}/{reasons[i]}"] += 1
                continue
            if (task_id, family, active_part) not in wanted:
                # Another independent candidate already filled this quota.
                continue
            saved = {
                key: state[key][i]
                for key in (
                    "task_ids",
                    "part_poses",
                    "joint_q",
                    "joint_qd",
                    "joint_targets",
                    "jaw_servo",
                    "arm_servo",
                    "arm_servo_velocity",
                    "part_velocities",
                    "goals",
                    "gripper_open",
                )
            }
            if not register_entry(
                saved,
                env.task_names[task_id],
                len(env.tasks[task_id]["parts"]),
                seen_ids,
            ):
                rejects[f"{env.task_names[task_id]}/{family}/duplicate_state"] += 1
                continue
            saved["family"] = family
            saved["active_part"] = records[i]["active"]
            saved["held_part"] = (
                records[i]["active"] if requires_grasp(records[i]) else -1
            )
            saved["generation_variant"] = (
                "grasp_supported_partial"
                if records[i].get("grasped_partial", False)
                else family
            )
            saved["settled_orientation"] = [
                reset_sampling.classify_orientation(
                    matrix[:3, :3], part.nominal_rotation
                )
                for matrix, part in zip(
                    pose_matrix(saved["part_poses"][: len(geometry[task_id]["parts"])]),
                    geometry[task_id]["parts"],
                    strict=True,
                )
            ]
            accepted.append(saved)
            wanted.remove((task_id, family, active_part))
        print(
            json.dumps(
                {
                    "accepted": len(accepted),
                    "remaining": len(wanted),
                    "attempts": attempts,
                }
            ),
            flush=True,
        )
    coverage = {
        name: dict(
            collections.Counter(
                row["family"] for row in accepted if row["task_ids"] == index
            )
        )
        for index, name in enumerate(env.task_names)
    }
    source_paths = {assets_path / (name + ".json") for name in env.task_names}
    source_paths.update(
        assets_path / part["mesh_cache"] for task in env.tasks for part in task["parts"]
    )
    sources = {
        str(path.relative_to(assets_path)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_paths)
    }
    metadata = {
        "schema_version": 2,
        "content_id_schema": 2,
        "env_schema_version": getattr(env, "schema_version", 1),
        "seed": seed,
        "sample_count": len(accepted),
        "physics_validated": bool(accepted),
        "training_ready": bool(accepted)
        and not wanted
        and set(families) == set(FAMILIES)
        and all(coverage.values())
        and all(t.get("collision_geometry_ready", False) for t in env.tasks),
        "requested_samples_per_task": samples_per_task,
        "complete_requested_coverage": not wanted,
        "max_wall_seconds": max_wall_seconds,
        "wall_budget_exhausted": wall_budget_exhausted,
        "task_names": env.task_names,
        "task_part_counts": [len(task["parts"]) for task in env.tasks],
        "entry_ids": [row["entry_ids"] for row in accepted],
        "entry_id_quantization": "1e-6 SI/radians, normalized sign-canonical quaternion",
        "requested_families": list(families),
        "physical_batches": batch_count,
        "coverage": coverage,
        "rejections": dict(rejects),
        "validation": dataclasses.asdict(config),
        "validation_log": validation_log.name,
        "reset_generator_sha256": hashlib.sha256(
            b"".join(
                path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
                for path in (
                    pathlib.Path(__file__),
                    pathlib.Path(reset_sampling.__file__),
                    pathlib.Path(generate_bank.__file__),
                    pathlib.Path(assembly_motion.__file__),
                )
            )
        ).hexdigest(),
        "grasp_challenge_protocol": (
            "Declared part-local axes replace vertical lifts for constrained near/partial "
            "grasps. Straight partial challenges cover at least twice the grasp retention "
            "tolerance and require measured projected part motion. Engaged or uncertain "
            "thread near/partial grasps instead follow 24 helical waypoints over 45 degrees; "
            "measured rotation must reach 80% and both axial command error and measured "
            "pitch residual must remain within 25 um. This verifies motion, not thread "
            "contact coupling. Proven disengaged threads use a straight 6 mm challenge. "
            "Gravity, perturbation, grasp retention and contact gates remain active."
        ),
        "enclosure_partial_grasp_proposals": (
            "Opposed surface-normal rays; initial aperture selected from contact width "
            "plus bounded clearance only after full convex gripper/part separation. "
            "Unheld lids start 4–8 mm above the rim with 4–8 mm lateral offset and "
            "3–12 degree yaw about their center; gravity must establish support. "
            "Native spawn, real closing, gravity and grasp challenge gates still apply."
        ),
        "validation_log_sha256": hashlib.sha256(validation_log.read_bytes()).hexdigest()
        if validation_log.exists()
        else None,
        "source_hashes": sources,
        "reference": "https://github.com/UW-Lab/UWLab/tree/825cfce364f7e34b4b29aab73d37c5a1464f4375",
        "threshold_provenance": "UR7 engineering thresholds; five consecutive strict final checks. Cited OmniReset grasp generation uses 1 s gravity-off closing, then 1 s of 0.01 N perturbation with gravity. Implemented by temporary object mg compensation, removed before acceptance; no attachment. UR7 close commands use a one-second smoothstep to avoid native jaw speed overshoot. Final rigid/table/other-object and all spawn contacts retain the 50 um gate. Only silicone-pad contacts with the active held moving part permit 500 um: pinned OmniReset config/ur5e_robotiq_2f85/grasp_sampling_cfg.py:127 and reset_states_cfg.py:411 use min_dist=-0.0005 for robot versus grasped insertive object. Our narrower pair-specific application is an engineering choice, not measured HandE rubber calibration. The existing 1 mm transient settling bound does not replace final gates, gravity/lift retention, or velocity checks.",
        "settled_orientations": [row.pop("settled_orientation") for row in accepted],
    }
    if getattr(env, "physics_identity", None) is not None:
        metadata["physics_identity"] = env.physics_identity
    output.mkdir(parents=True, exist_ok=True)
    if accepted:
        arrays = {
            key: np.asarray([row[key] for row in accepted]) for key in accepted[0]
        }
        validate_bank_entries(arrays, metadata)
        np.savez_compressed(output / "bank.npz", **arrays)
    (output / "bank.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not accepted or not all(coverage.values()):
        raise RuntimeError(
            f"Reset bank lacks task coverage; inspect {output / 'bank.json'}"
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", action="append", help="Task ID; repeat for a subset")
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--max-attempts-per-sample", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        help="Stop starting new batches after this operational budget; finish the current batch and save accepted states.",
    )
    parser.add_argument(
        "--max-settle-steps", type=int, default=ValidationConfig().max_settle_steps
    )
    parser.add_argument(
        "--partial-grasp-probability",
        type=float,
        default=ValidationConfig().partial_grasp_probability,
        help="Fraction of partial proposals held by a physically validated grasp [0,1]",
    )
    args = parser.parse_args()
    from isaac.assembly_rl.environment import AssemblyEnvironment

    env = AssemblyEnvironment(
        assets_path=args.assets,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        config={} if args.task is None else {"task_names": args.task},
    )
    try:
        build(
            env,
            args.assets,
            args.out,
            args.samples_per_task,
            args.seed,
            config=ValidationConfig(
                candidate_attempts_per_sample=args.max_attempts_per_sample,
                max_batches=args.max_batches,
                max_settle_steps=args.max_settle_steps,
                partial_grasp_probability=args.partial_grasp_probability,
            ),
            families=FAMILIES if args.family is None else tuple(args.family),
            max_wall_seconds=args.max_wall_seconds,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
