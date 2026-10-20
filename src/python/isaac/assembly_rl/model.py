"""UR7 workshop articulation and per-task source CAD for batched Newton training."""

import collections
import dataclasses
import hashlib
import json
import operator
import pathlib
import weakref
from typing import Any

import newton
import numpy as np

from isaac.fanuc_sim import native_model, robot_profile

TABLE_HEIGHT_M = 0.86365
BASE_MOUNT = np.eye(4)
BASE_MOUNT[:3, 3] = (-0.199, -0.4186, 0.8749)


def _template_filter_graph(model, selected_shapes):
    """Return exact cross-body filters for one selected replicated template."""
    data = getattr(model, "_assembly_filter_template", None)
    if (
        data is None
        or not len(selected_shapes)
        or np.any(np.diff(selected_shapes) <= 0)
    ):
        return None
    count, local_pairs = data
    world = int(selected_shapes[0]) // count
    if not np.all(selected_shapes // count == world):
        return None
    selected = np.zeros(count, dtype=bool)
    selected[selected_shapes % count] = True
    pairs = local_pairs[selected[local_pairs[:, 0]] & selected[local_pairs[:, 1]]]
    pairs = pairs + world * count
    bodies = model.shape_body.numpy()
    pairs = pairs[bodies[pairs[:, 0]] != bodies[pairs[:, 1]]]
    return bodies, pairs


def _whole_body_excludes(bodies, pairs, selected_shapes, selected_bodies):
    shape_counts = collections.Counter(bodies[selected_shapes].tolist())
    pair_counts = collections.Counter(
        tuple(sorted((int(bodies[a]), int(bodies[b])))) for a, b in pairs
    )
    return [
        (a, b)
        for i, a in enumerate(selected_bodies)
        for b in selected_bodies[i + 1 :]
        if pair_counts[tuple(sorted((int(a), int(b))))]
        == shape_counts[a] * shape_counts[b]
    ]


class AssemblySolverMuJoCo(newton.solvers.SolverMuJoCo):
    """Compile exact native filters without enumerating all assembly hull pairs."""

    @staticmethod
    def _find_body_collision_filter_pairs(model, selected_bodies, colliding_shapes):
        graph = _template_filter_graph(model, colliding_shapes)
        if graph is None:
            return newton.solvers.SolverMuJoCo._find_body_collision_filter_pairs(
                model, selected_bodies, colliding_shapes
            )
        return _whole_body_excludes(*graph, colliding_shapes, selected_bodies)

    @staticmethod
    def _compile_newton_collision_masks(model, selected_shapes):
        from newton._src.solvers.mujoco.collision_masks import (
            compile_newton_collision_graph,
        )

        graph = _template_filter_graph(model, selected_shapes)
        groups = model.shape_collision_group.numpy()[selected_shapes]
        if graph is not None and np.all(groups == groups[0]) and groups[0] > 0:
            bodies, pairs = graph
            selected_bodies = np.unique(bodies[selected_shapes])
            # Static world shapes have body -1 and cannot receive an MjSpec
            # body exclude; their pair filters must remain in the masks.
            selected_bodies = selected_bodies[selected_bodies >= 0]
            excluded_bodies = set(
                _whole_body_excludes(bodies, pairs, selected_shapes, selected_bodies)
            )
            residual = np.asarray(
                [
                    (a, b)
                    for a, b in pairs
                    if tuple(sorted((bodies[a], bodies[b]))) not in excluded_bodies
                ],
                dtype=np.int64,
            ).reshape((-1, 2))
            affected = np.unique(residual)
            # Only shapes participating in partial cross-body exclusions need
            # individual masks. All other shapes share one universal-contact bit.
            compiled = compile_newton_collision_graph(
                np.ones(len(affected), dtype=np.int32),
                excluded_pairs=np.searchsorted(affected, residual),
                max_bits=31,
            )
            if compiled.exact:
                universal = np.uint32(1 << 31)
                types = np.full(len(selected_shapes), universal, dtype=np.uint32)
                affinities = types.copy()
                locations = np.searchsorted(selected_shapes, affected)
                types[locations] = compiled.collision_type
                affinities[locations] = compiled.collision_affinity | universal
                return dataclasses.replace(
                    compiled,
                    collision_type=types,
                    collision_affinity=affinities,
                    bit_count=compiled.bit_count + 1,
                )
        # Unsupported groups or complex partial-filter graphs retain Newton's
        # existing compiler; never approximate away an allowed contact.
        return newton.solvers.SolverMuJoCo._compile_newton_collision_masks(
            model, selected_shapes
        )


class MuJoCoNativeContactBuilder(newton.ModelBuilder):
    """Keep Newton's filters without its unused quadratic collision-pair table."""

    def _find_shape_contact_pairs(self, model, *, allow_filter_blocks):
        # Newton 1.6 private hook: MuJoCo reads packed filters, never this
        # quadratic candidate table. Keep flags/filters and all validations.
        if not allow_filter_blocks:
            raise RuntimeError("This assembly model requires MuJoCo native contacts")
        model.shape_contact_pairs = None


def _compact_same_body_filters(template):
    """Store exact same-body exclusions once, before Newton expands replicas."""
    pairs = template._iter_validated_shape_collision_filter_pairs(
        template.shape_collision_filter_pairs
    )
    codes = np.fromiter(((a << 32) | b for a, b in pairs), dtype=np.int64)
    codes = np.unique(codes)
    a, b = codes >> 32, codes & 0xFFFFFFFF
    bodies = np.asarray(template.shape_body)
    worlds = np.asarray(template.shape_world)
    redundant = (a != b) & (bodies[a] == bodies[b]) & (worlds[a] == worlds[b])
    # MuJoCo intrinsically excludes same-body contacts. Preserve these exact
    # pairs for its mask compiler/colorer; do not infer absent user filters.
    template.shape_collision_filter_pairs = list(
        zip(a[~redundant].tolist(), b[~redundant].tolist(), strict=True)
    )
    return codes[redundant]


def _install_template_filter_queries(native, local_codes, shape_count, world_count):
    """Native MuJoCo only: compact queries retain the original allowed graph."""
    if native.shape_count != shape_count * world_count:
        raise ValueError("Replicated collision filter shape layout changed")
    native_ref = weakref.ref(native)

    def mask(pairs):
        # Keep Newton's dtype/overflow/shape validation and explicit filters.
        result = newton.Model.shape_collision_filter_mask(native_ref(), pairs)
        if not local_codes.size:
            return result
        query = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
        valid = np.all((query >= 0) & (query < shape_count * world_count), axis=1)
        valid &= query[:, 0] // shape_count == query[:, 1] // shape_count
        indices = np.flatnonzero(valid)
        local = np.sort(query[indices] % shape_count, axis=1)
        codes = (local[:, 0] << 32) | local[:, 1]
        positions = np.searchsorted(local_codes, codes)
        result[indices] |= (positions < local_codes.size) & (
            local_codes[np.minimum(positions, local_codes.size - 1)] == codes
        )
        return result

    def contains(a, b):
        a, b = operator.index(a), operator.index(b)
        if newton.Model.shape_collision_filter_contains(native_ref(), a, b):
            return True
        if not (
            0 <= a < native_ref().shape_count and 0 <= b < native_ref().shape_count
        ):
            return False
        return bool(mask(np.asarray([[a, b]], dtype=np.int64))[0])

    native.shape_collision_filter_mask = mask
    native.shape_collision_filter_contains = contains
    # The native solver's sole array consumer explicitly ignores same-body
    # pairs. Generic Newton broad phases remain unsupported by build_model.


def load_tasks(
    assets_path: pathlib.Path,
    task_ids: list[str] | None = None,
    *,
    allow_unvalidated: bool = False,
) -> list[dict[str, Any]]:
    """Require the independent precision-collision gate, before dynamic validation."""
    root = pathlib.Path(assets_path).resolve()
    if task_ids is None:
        task_ids = [
            path.stem
            for path in sorted(root.glob("*.json"))
            if "parts" in json.loads(path.read_text())
        ]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Need nonempty unique assembly task IDs")
    tasks = []
    for task_id in task_ids:
        path = (root / f"{task_id}.json").resolve()
        if path.parent != root:
            raise ValueError("Task path must remain inside asset cache")
        task = json.loads(path.read_text())
        if not allow_unvalidated and not task.get("collision_geometry_ready", False):
            raise ValueError(
                f"{task_id}: precision collision geometry is not validated"
            )
        if task.get("id") != task_id or not task.get("parts"):
            raise ValueError("Task identity/parts are missing")
        if not any(part.get("fixed") for part in task["parts"]) or all(
            part.get("fixed") for part in task["parts"]
        ):
            raise ValueError(
                "Each task needs fixed receivers and dynamic assembly parts"
            )
        names = [part["name"] for part in task["parts"]]
        if len(set(names)) != len(names):
            raise ValueError("Task part names must be unique")
        for part in task["parts"]:
            mesh_path = (root / part["mesh_cache"]).resolve()
            if mesh_path.parent != root:
                raise ValueError("Mesh path must remain inside asset cache")
            with np.load(mesh_path, allow_pickle=False) as mesh:
                vertices = np.asarray(mesh["vertices"], dtype=np.float32)
                faces = np.asarray(mesh["faces"], dtype=np.int32)
                hulls = [
                    np.asarray(mesh[f"hull_{index}"], dtype=np.float32)
                    for index in range(part["hull_count"])
                ]
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.isfinite(vertices).all()
                or faces.ndim != 2
                or faces.shape[1] != 3
                or not len(faces)
                or faces.min() < 0
                or faces.max() >= len(vertices)
                or not hulls
            ):
                raise ValueError(f"{task_id}/{part['name']}: invalid cached geometry")
            for hull in hulls:
                if (
                    hull.ndim != 2
                    or hull.shape[1] != 3
                    or len(hull) < 4
                    or not np.isfinite(hull).all()
                ):
                    raise ValueError("Invalid convex collision component")
            native_model.validate_inertia(
                part["mass_kg"], part["com_m"], part["inertia_kg_m2"]
            )
            part.update(
                vertices=vertices,
                faces=faces,
                hulls=hulls,
                mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            )
        tasks.append(task)
    return tasks


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int = 0
) -> np.ndarray:
    """Area-weighted local XYZ samples with a fixed reproducible point order."""
    if count < 1:
        raise ValueError("Point count must be positive")
    triangles = np.asarray(vertices)[np.asarray(faces)]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Surface must contain nondegenerate finite triangles")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    first = np.sqrt(rng.random((count, 1)))
    second = rng.random((count, 1))
    return (
        (1 - first) * selected[:, 0]
        + first * (1 - second) * selected[:, 1]
        + first * second * selected[:, 2]
    ).astype(np.float32)


def robot_surface_points(
    profile: dict[str, Any], mappings: dict[str, Any], count_per_body: int = 32
) -> list[dict[str, Any]]:
    """Sample source robot collision triangles and per-solid gripper CAD hulls."""
    import trimesh

    robot = robot_profile.of(profile)
    urdf = native_model.read_urdf(robot=robot)
    results = []
    for name, row in urdf["links"].items():
        mesh = trimesh.load_mesh(
            pathlib.Path(profile["robot_collision_path"]) / row["mesh"], process=False
        )
        vertices = np.asarray(mesh.vertices) * row["scale"]
        matrix = row["collision_origin"]
        vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        results.append(
            {
                "name": name,
                "body": mappings["bodies"][name],
                "points": sample_surface(
                    vertices,
                    np.asarray(mesh.faces),
                    count_per_body,
                    mappings["bodies"][name],
                ),
            }
        )
    parts, _ = native_model.gripper_shapes(profile)
    for name in robot.gripper.bodies:
        solids = [
            trimesh.convex.convex_hull(points)
            for body, _, points, _ in parts
            if body == name
        ]
        if solids:
            mesh = trimesh.util.concatenate(solids)
            results.append(
                {
                    "name": name,
                    "body": mappings["bodies"][name],
                    "points": sample_surface(
                        mesh.vertices,
                        mesh.faces,
                        count_per_body,
                        mappings["bodies"][name],
                    ),
                }
            )
    return results


def add_assembly_convex_shape(
    template: Any,
    mappings: dict[str, Any],
    profile: dict[str, Any],
    *,
    body: int,
    label: str,
    points: Any,
    role: str,
    max_hull_vertices: int = -1,
) -> int:
    """Preserve skinny CAD hull facets instead of trimesh's lossy hull cleanup."""
    import newton

    from isaac.assembly_rl.collision_sections import convex_mesh

    vertices = np.asarray(points, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
        or np.max(np.ptp(vertices, axis=0)) > 5
    ):
        raise ValueError(f"Invalid metre-scale CAD geometry: {label}")
    if label in mappings["shape_labels"] or not isinstance(role, str) or not role:
        raise ValueError("Collision shape labels must be unique and roles nonempty")
    if type(max_hull_vertices) is not int or (
        max_hull_vertices != -1 and max_hull_vertices < 4
    ):
        raise ValueError("Convex vertex limit must be -1 or an integer of at least 4")
    # Build topology at Newton's actual float32 precision, so rounding cannot
    # collapse a facet after validation. Winding repair never deletes facets.
    vertices = np.unique(vertices.astype(np.float32).astype(float), axis=0)
    mesh = convex_mesh(vertices)
    native = newton.Mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        indices=np.asarray(mesh.faces, dtype=np.int32).flatten(),
        compute_inertia=False,
        maxhullvert=max_hull_vertices,
    )
    config = newton.ModelBuilder.ShapeConfig(
        density=0,
        has_shape_collision=True,
        **{
            k: v
            for k, v in profile["contact"].items()
            if k not in ("provenance", "condim")
        },
    )
    shape_id = template.add_shape_convex_hull(
        body,
        mesh=native,
        cfg=config,
        label=label,
        custom_attributes={"mujoco:condim": profile["contact"]["condim"]},
    )
    if shape_id != len(mappings["shape_labels"]):
        raise ValueError(
            "Every collision shape must be registered in the scene mappings"
        )
    mappings["shape_labels"].append(label)
    mappings["shape_roles"].append(role)
    return shape_id


def build_template(
    assets_path: pathlib.Path, tasks: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import newton
    import warp as wp

    profile = native_model.load_profile(robot_profile.UR7E.native_profile)
    profile["jaw"]["kd"] = 400.0
    profile["jaw"]["assembly_servo_provenance"] = (
        "Engineering choice: 400 N s/m damping and measured-state spring-error "
        "anti-windup; 0.04 m/s reference permits 16N holding force below the "
        "unchanged 20N actuator cap. Not hardware-calibrated or paper-specified."
    )
    template, mappings = native_model.build_robot_template(
        profile,
        base_mount=BASE_MOUNT,
        gripper_pose_flange=np.eye(4),
        wrist_pose_flange=np.eye(4),
    )
    # The approved workshop has no wrist camera. Keep the shared topology but
    # remove the collision-free camera placeholder's unauthored mass/inertia.
    camera = mappings["bodies"]["wrist_camera"]
    template.body_mass[camera] = template.body_inv_mass[camera] = 0.0
    template.body_inertia[camera] = template.body_inv_inertia[camera] = wp.mat33()
    mappings["absent_scene_bodies"] = ["wrist_camera"]
    # Native articulation ordinarily uses externally computed torques. Policy
    # targets instead use MuJoCo's implicit joint impedance actuators.
    for index in range(8):
        arm = index < 6
        template.joint_target_mode[index] = int(newton.JointTargetMode.POSITION)
        template.joint_target_ke[index] = (
            profile["actuator"]["kp"][index] if arm else profile["jaw"]["kp"]
        )
        template.joint_target_kd[index] = (
            profile["actuator"]["kd"][index] if arm else profile["jaw"]["kd"]
        )
        template.joint_target_q[index] = template.joint_q[index]
        template.joint_target_qd[index] = 0.0
    # MuJoCo includes gravity feedforward in joint actuator force saturation.
    template.custom_attributes["mujoco:gravcomp"].values = {
        index: profile["actuator"]["gravity_compensation"]
        for index in mappings["bodies"].values()
    }
    template.custom_attributes["mujoco:jnt_actgravcomp"].values = {
        index: True for index in range(8)
    }
    mappings["gravity_compensation"] = (
        "MuJoCo body gravcomp routed through limited joint actuator forces"
    )
    mappings["robot_point_clouds"] = robot_surface_points(profile, mappings)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0,
        **{
            key: value
            for key, value in profile["contact"].items()
            if key not in ("condim", "provenance")
        },
    )
    table = template.add_shape_box(
        -1,
        hx=0.825,
        hy=0.520,
        hz=0.020,
        xform=wp.transform((0.0, -0.045, TABLE_HEIGHT_M - 0.020), wp.quat_identity()),
        cfg=cfg,
        label="workshop/table",
        custom_attributes={"mujoco:condim": profile["contact"]["condim"]},
    )
    mappings["shape_labels"].append("workshop/table")
    mappings["shape_roles"].append("support")
    mappings["table_shape"] = table
    mappings["table_height_m"] = TABLE_HEIGHT_M
    mappings["task_parts"] = []
    mappings["task_ids"] = [task["id"] for task in tasks]
    for task_index, task in enumerate(tasks):
        task_parts = []
        for part_index, part in enumerate(task["parts"]):
            label = f"{task['id']}/{part['name']}"
            parked = wp.transform(
                (10.0 + task_index * 2.0, part_index * 2.0, 1.0), wp.quat_identity()
            )
            kwargs = {
                "xform": parked,
                "mass": part["mass_kg"],
                "com": part["com_m"],
                "inertia": part["inertia_kg_m2"],
                "lock_inertia": True,
                "label": label,
            }
            if part["fixed"]:
                body = template.add_link(**kwargs, is_kinematic=True)
                joint = template.add_joint_fixed(
                    -1, body, parent_xform=parked, label=label + "/fixed"
                )
                template.add_articulation([joint], label=label + "/articulation")
                q_start = qd_start = None
            else:
                body = template.add_body(**kwargs)
                joint = template.joint_count - 1
                q_start, qd_start = (
                    template.joint_q_start[joint],
                    template.joint_qd_start[joint],
                )
            mappings["bodies"][label] = body
            mappings["joints"][label] = joint
            shapes = [
                add_assembly_convex_shape(
                    template,
                    mappings,
                    profile,
                    body=body,
                    label=f"{label}/hull_{index}",
                    points=hull,
                    role="receiver" if part["fixed"] else "part",
                    max_hull_vertices=-1,
                )
                for index, hull in enumerate(part["hulls"])
            ]
            mappings["body_shapes"][label] = shapes
            task_parts.append(
                {
                    "name": part["name"],
                    "label": label,
                    "body": body,
                    "joint": joint,
                    "q_start": q_start,
                    "qd_start": qd_start,
                    "fixed": bool(part["fixed"]),
                    "shapes": shapes,
                    "vertices": part["vertices"],
                    "faces": part["faces"],
                    "point_cloud_local": sample_surface(
                        part["vertices"], part["faces"], 128, 1000 + body
                    ),
                    "assembled_transform": np.asarray(
                        part["assembled_transform"], dtype=float
                    ),
                }
            )
            mappings["asset_sha256"][
                str(pathlib.Path(assets_path) / part["mesh_cache"])
            ] = part["mesh_sha256"]
        mappings["task_parts"].append(task_parts)
    template.rigid_gap = 0.0
    template.shape_gap[:] = [0.0] * template.shape_count
    template.shape_margin[:] = [0.0] * template.shape_count
    mappings.update(
        body_count_per_world=template.body_count,
        shape_count_per_world=template.shape_count,
        joint_count_per_world=template.joint_count,
        joint_coord_count_per_world=template.joint_coord_count,
        joint_dof_count_per_world=template.joint_dof_count,
    )
    return template, mappings, profile


def build_model(
    assets_path: pathlib.Path,
    tasks: list[dict[str, Any]],
    *,
    world_count: int,
    use_mujoco_contacts: bool,
    device: str = "cuda:0",
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if use_mujoco_contacts is not True:
        raise ValueError("Assembly models require MuJoCo native contacts")
    if type(world_count) is not int or world_count < 1:
        raise ValueError("world_count must be a positive integer")

    template, mappings, profile = build_template(assets_path, tasks)
    if (
        len(mappings["shape_labels"]) != template.shape_count
        or len(mappings["shape_roles"]) != template.shape_count
    ):
        raise ValueError("Collision shape mappings must cover the entire template")
    builder = MuJoCoNativeContactBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    local_filters = _compact_same_body_filters(template)
    builder.replicate(template, world_count)
    mappings = {
        **mappings,
        "body_count_per_world": template.body_count,
        "shape_count_per_world": template.shape_count,
        "joint_coord_count_per_world": template.joint_coord_count,
        "joint_dof_count_per_world": template.joint_dof_count,
        "body_labels": list(template.body_label),
        "joint_labels": list(template.joint_label),
        "object_initial_state": {},
        "compact_same_body_filters_per_world": int(local_filters.size),
        "explicit_collision_filters_per_world": len(
            template.shape_collision_filter_pairs
        ),
    }
    native = builder.finalize(device=device)
    _install_template_filter_queries(
        native, local_filters, template.shape_count, world_count
    )
    native._assembly_filter_template = (
        template.shape_count,
        np.asarray(template.shape_collision_filter_pairs, dtype=np.int64).reshape(
            (-1, 2)
        ),
    )
    return native, mappings, profile
