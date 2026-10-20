"""Native arm articulation with explicit physical parameters and CAD provenance.

The arm is the one the profile names in ``robot`` (``robot_profile.of``).
"""

import copy
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from isaac.fanuc_sim import robot_profile

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DEFAULT = robot_profile.UR7E


def raw_to_urdf(values: Any, coupling: float = _DEFAULT.j2_j3_coupling) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result[..., 2] += coupling * result[..., 1]
    return result


def urdf_to_raw(values: Any, coupling: float = _DEFAULT.j2_j3_coupling) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result[..., 2] -= coupling * result[..., 1]
    return result


def raw_to_urdf_effort(
    values: Any, coupling: float = _DEFAULT.j2_j3_coupling
) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result[..., 1] -= coupling * result[..., 2]
    return result


def urdf_to_raw_effort(
    values: Any, coupling: float = _DEFAULT.j2_j3_coupling
) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result[..., 1] += coupling * result[..., 2]
    return result


def validate_inertia(mass: float, com: Any, inertia: Any) -> None:
    inertia = np.asarray(inertia, dtype=float)
    com = np.asarray(com, dtype=float)
    if (
        not np.isfinite(mass)
        or mass <= 0
        or com.shape != (3,)
        or not np.isfinite(com).all()
        or inertia.shape != (3, 3)
        or not np.isfinite(inertia).all()
        or not np.allclose(inertia, inertia.T, atol=1e-12)
    ):
        raise ValueError(
            "Body requires positive mass, finite COM and symmetric inertia"
        )
    principal = np.linalg.eigvalsh(inertia)
    if principal[0] <= 0 or principal[2] > principal[0] + principal[1] + 1e-12:
        raise ValueError(
            "Inertia must be positive definite and satisfy the triangle inequality"
        )


def _origin(element: Any) -> np.ndarray:
    from scipy.spatial import transform

    matrix = np.eye(4)
    if element is not None:
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = transform.Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return matrix


def _required_element(parent: ET.Element, path: str) -> ET.Element:
    element = parent.find(path)
    if element is None:
        raise ValueError(f"URDF is missing required {path}")
    return element


def read_urdf(
    path: pathlib.Path | None = None, *, robot: robot_profile.RobotProfile = _DEFAULT
) -> dict[str, Any]:
    path = robot.kinematics.URDF_PATH if path is None else path
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(_required_element(inertial, "mass").attrib["value"])
        origin = _origin(inertial.find("origin"))
        attrs = _required_element(inertial, "inertia").attrib
        tensor = np.array(
            [[float(attrs[f"i{min(a, b)}{max(a, b)}"]) for b in "xyz"] for a in "xyz"]
        )
        tensor = origin[:3, :3] @ tensor @ origin[:3, :3].T
        validate_inertia(mass, origin[:3, 3], tensor)
        mesh = _required_element(link, "collision/geometry/mesh")
        links[link.attrib["name"]] = {
            "mass": mass,
            "com": origin[:3, 3],
            "inertia": tensor,
            "collision_origin": _origin(link.find("collision/origin")),
            "mesh": pathlib.Path(mesh.attrib["filename"]).name,
            "scale": np.fromstring(mesh.get("scale", "1 1 1"), sep=" "),
        }
    joints = []
    for name in robot.joint_names:
        element = _required_element(root, f"joint[@name='{name}']")
        limit = _required_element(element, "limit")
        joints.append(
            {
                "name": name,
                "parent": _required_element(element, "parent").attrib["link"],
                "child": _required_element(element, "child").attrib["link"],
                "origin": _origin(element.find("origin")),
                "axis": np.fromstring(
                    _required_element(element, "axis").attrib["xyz"], sep=" "
                ),
                **{
                    key: float(limit.attrib[key])
                    for key in ["lower", "upper", "effort", "velocity"]
                },
            }
        )
    flange = np.eye(4)
    for name in robot.flange_joints:
        flange = flange @ _origin(root.find(f"joint[@name='{name}']/origin"))
    return {
        "links": links,
        "joints": joints,
        "j6_to_flange": flange,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_profile(path: pathlib.Path | str | dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0912
    profile = (
        copy.deepcopy(path)
        if isinstance(path, dict)
        else json.loads(pathlib.Path(path).read_text())
    )
    try:
        if (
            type(profile["schema_version"]) is not int
            or profile["schema_version"] != 1
            or profile["physics_hz"] not in (240, 750, 1500)
        ):
            raise ValueError("Unsupported profile version or physics rate")

        def vector(values: Any, n: int, positive: bool = False) -> None:
            array = np.asarray(values, dtype=float)
            if (
                array.shape != (n,)
                or not np.isfinite(array).all()
                or (positive and np.any(array <= 0))
            ):
                raise ValueError("Invalid finite profile vector")

        vector(profile["home_joints_deg"], 6)
        for key in ("kp", "kd", "armature", "effort_limit_nm"):
            vector(profile["actuator"][key], 6, True)
        if not 0 <= profile["actuator"]["gravity_compensation"] <= 1:
            raise ValueError("Invalid gravity compensation")
        for row in [
            profile["actuator"],
            profile["jaw"],
            profile["contact"],
            profile["solver"],
            *profile["tool_bodies"].values(),
            *profile["objects"].values(),
        ]:
            if not isinstance(row["provenance"], str) or not row["provenance"].strip():
                raise ValueError("Physical parameters require provenance")
        tool_bodies = robot_profile.of(profile).tool_bodies
        if set(profile["tool_bodies"]) != set(tool_bodies):
            raise ValueError(
                "Shared articulation requires the four declared tool bodies"
            )
        for name in tool_bodies:
            row = profile["tool_bodies"][name]
            validate_inertia(row["mass_kg"], row["com_m"], row["inertia_kg_m2"])
        if set(profile["contact"]) != {
            "mu",
            "ke",
            "kd",
            "mu_torsional",
            "mu_rolling",
            "condim",
            "provenance",
        }:
            raise ValueError("Unsupported material property")
        for key, value in profile["contact"].items():
            if key == "provenance":
                continue
            if (
                not np.isfinite(value)
                or value < 0
                or (key in ("ke", "kd", "condim") and value == 0)
            ):
                raise ValueError(
                    "Friction must be nonnegative and contact stiffness/damping positive"
                )
        if profile["contact"]["condim"] != 6:
            raise ValueError("Declared torsional and rolling friction require condim=6")
        jaw = profile["jaw"]
        for key in ("kp", "kd", "effort_limit_n", "velocity_limit_m_s", "armature_kg"):
            if not np.isfinite(jaw[key]) or jaw[key] <= 0:
                raise ValueError("Invalid jaw parameter")
        if (
            not 0
            <= jaw["limit_lower_m"]
            <= jaw["closed_m"]
            < jaw["open_m"]
            <= jaw["limit_upper_m"]
        ):
            raise ValueError("Invalid jaw travel")
        for key in ("nconmax", "njmax", "iterations"):
            if (
                not isinstance(profile["solver"][key], int)
                or profile["solver"][key] <= 0
            ):
                raise ValueError("Invalid solver capacity")
        if set(profile["solver"]) != {
            "nconmax",
            "njmax",
            "iterations",
            "tolerance",
            "cone",
            "impratio",
            "provenance",
        }:
            raise ValueError("Unsupported solver property")
        if (
            not np.isfinite(profile["solver"]["tolerance"])
            or profile["solver"]["tolerance"] <= 0
        ):
            raise ValueError("Invalid solver tolerance")
        if profile["solver"]["cone"] not in ("pyramidal", "elliptic"):
            raise ValueError("Unsupported friction cone")
        if (
            not np.isfinite(profile["solver"]["impratio"])
            or profile["solver"]["impratio"] <= 0
        ):
            raise ValueError("Invalid friction impedance ratio")
        if not isinstance(profile["objects"], dict) or not profile["objects"]:
            raise ValueError("Native profile requires named objects")
        for name, row in profile["objects"].items():
            if not isinstance(name, str) or not name or name in profile["tool_bodies"]:
                raise ValueError(
                    "Object names must be nonempty and distinct from tool bodies"
                )
            if "com_m" in row or "inertia_kg_m2" in row:
                validate_inertia(row["mass_kg"], row["com_m"], row["inertia_kg_m2"])
            if not np.isfinite(row["mass_kg"]) or row["mass_kg"] <= 0:
                raise ValueError("Invalid object mass")
            for key, width in (
                ("pose_xyzw", 7),
                ("velocity_mps", 3),
                ("angular_velocity_radps", 3),
            ):
                if key not in row:
                    continue
                values = np.asarray(row[key], dtype=float)
                if (
                    values.ndim not in (1, 2)
                    or values.shape[-1] != width
                    or values.size == 0
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"Object {key} requires finite ({width},) or (N,{width}) values"
                    )
                if key == "pose_xyzw" and not np.allclose(
                    np.linalg.norm(values[..., 3:], axis=-1), 1, atol=1e-5
                ):
                    raise ValueError(
                        "Object initial pose requires a unit xyzw quaternion"
                    )
        for key in ("robot_collision_path",):
            value = pathlib.Path(profile[key])
            profile[key] = str(value if value.is_absolute() else _ROOT / value)
        robot = robot_profile.of(profile)
        urdf = read_urdf(robot=robot)
        home = np.deg2rad(raw_to_urdf(profile["home_joints_deg"], robot.j2_j3_coupling))
        if any(
            not j["lower"] <= q <= j["upper"]
            for j, q in zip(urdf["joints"], home, strict=True)
        ):
            raise ValueError("Home violates coupled joint limits")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Incomplete native profile: {exc}") from exc
    return profile


def add_convex_shape(
    template: Any,
    mappings: dict[str, Any],
    profile: dict[str, Any],
    *,
    body: int,
    label: str,
    points: Any,
    role: str,
    has_shape_collision: bool = True,
    max_hull_vertices: int | None = None,
    pose_xyzw: Any = None,
) -> int:
    """Append a convex component; optional deferred collisions avoid quadratic pairs.

    A scene using deferred collisions must enable the shape flags and supply
    equivalent collision filtering before finalizing its model. A vertex limit
    of -1 preserves every convex vertex; None retains Newton's default limit.
    ``pose_xyzw`` places points given in a local frame on their body.
    """
    import newton
    import trimesh

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
    if max_hull_vertices is not None and (
        type(max_hull_vertices) is not int
        or (max_hull_vertices != -1 and max_hull_vertices < 4)
    ):
        raise ValueError("Convex vertex limit must be -1 or an integer of at least 4")
    mesh = trimesh.convex.convex_hull(vertices)
    native = newton.Mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        indices=np.asarray(mesh.faces, dtype=np.int32).flatten(),
        compute_inertia=False,
        **({} if max_hull_vertices is None else {"maxhullvert": max_hull_vertices}),
    )
    placement = {}
    if pose_xyzw is not None:
        import warp as wp

        placement["xform"] = wp.transform(pose_xyzw[:3], pose_xyzw[3:])
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0,
        has_shape_collision=has_shape_collision,
        **{
            k: v
            for k, v in profile["contact"].items()
            if k not in ("provenance", "condim")
        },
    )
    shape_id = template.add_shape_convex_hull(
        body,
        mesh=native,
        cfg=cfg,
        label=label,
        custom_attributes={"mujoco:condim": profile["contact"]["condim"]},
        **placement,
    )
    if shape_id != len(mappings["shape_labels"]):
        raise ValueError(
            "Every collision shape must be registered in the scene mappings"
        )
    mappings["shape_labels"].append(label)
    mappings["shape_roles"].append(role)
    return shape_id


def _box(low: Any, high: Any) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )


def gripper_shapes(
    profile: dict[str, Any],
) -> tuple[list[tuple[str, str, np.ndarray, str]], dict[str, str]]:
    """Convex parts of the profile's gripper: (body, label, points, role) in
    the gripper frame at jaw joint 0, metres, and the files they came from.

    The jaw on -x carries ``pad_a``, the jaw on +x ``pad_b``.
    """
    gripper = robot_profile.of(profile).gripper
    body, jaw_a, jaw_b = gripper.bodies
    if gripper is robot_profile.HAND_E:
        # Fingertip holder and flat silicone fingertip from Robotiq's drawings
        # (see robot_profile.HAND_E): the 40 mm fingertip face with its pad
        # (6 + 1.59 mm thick, 21 wide), the rest of the fingertip base behind
        # it (to 15.2 mm deep, 8 mm high) and the holder plate under the
        # fingertip (15 x 29.74 x 6.5 mm, reaching 20.24 mm toward its rack).
        # Pads touch at joint 0. The corner radii are not in the colliders.
        near, far = (value * 0.001 for value in gripper.pad_z_mm)
        base = near - 0.0065
        ring = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        circle = 0.0375 * np.c_[np.cos(ring), np.sin(ring)]
        parts = [
            (
                body,
                f"{body}/shell",
                np.r_[np.c_[circle, np.zeros(32)], np.c_[circle, np.full(32, base)]],
                "robot",
            )
        ]
        for name, sign, role in ((jaw_a, -1.0, "pad_a"), (jaw_b, 1.0, "pad_b")):
            side = np.array([sign, 1.0, 1.0])
            parts.append(
                (
                    name,
                    f"{name}/pad",
                    side * _box((0.0, -0.0105, near), (0.00759, 0.0105, far)),
                    role,
                )
            )
            parts.append(
                (
                    name,
                    f"{name}/base",
                    side
                    * _box((0.00759, -0.0105, near), (0.01679, 0.0105, near + 0.008)),
                    "robot",
                )
            )
            # Each holder reaches toward its own rack: a half turn about the
            # tool axis maps one finger onto the other.
            parts.append(
                (
                    name,
                    f"{name}/holder",
                    np.array([sign, sign, 1.0])
                    * _box((0.00179, -0.0095, base), (0.01679, 0.02024, near)),
                    "robot",
                )
            )
        return parts, {}
    raise ValueError("Only the UR7e Hand-E gripper is supported")


def build_robot_template(
    profile: dict[str, Any],
    *,
    base_mount: Any,
    gripper_pose_flange: Any,
    wrist_pose_flange: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build the shared torque-driven arm and physical jaws before scene objects."""
    import newton
    import trimesh
    import warp as wp
    from scipy.spatial import transform

    profile = load_profile(profile)
    for value in (base_mount, gripper_pose_flange, wrist_pose_flange):
        matrix = np.asarray(value, dtype=float)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-7)
        ):
            raise ValueError("Robot mounts must be finite rigid transforms")
    base = np.asarray(base_mount, dtype=float)
    robot = robot_profile.of(profile)
    base_link, last_link = robot.link_names[0], robot.last_link
    gripper_body, *jaws = robot.gripper.bodies
    urdf = read_urdf(robot=robot)
    assets = {str(robot.kinematics.URDF_PATH): urdf["sha256"]}
    zero_fk = {base_link: np.eye(4)}
    for joint in urdf["joints"]:
        zero_fk[joint["child"]] = zero_fk[joint["parent"]] @ joint["origin"]
    mount = urdf["j6_to_flange"] @ np.asarray(gripper_pose_flange)
    wrist_mount = urdf["j6_to_flange"] @ np.asarray(wrist_pose_flange)

    def pose(matrix: np.ndarray) -> Any:
        return wp.transform(
            matrix[:3, 3], transform.Rotation.from_matrix(matrix[:3, :3]).as_quat()
        )

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    bodies: dict[str, int] = {}
    shape_labels: list[str] = []
    shape_roles: list[str] = []
    body_shapes: dict[str, list[int]] = {}
    shapes = {"shape_labels": shape_labels, "shape_roles": shape_roles}

    def add_hull(body: int, label: str, points: np.ndarray, role: str) -> int:
        return add_convex_shape(
            template, shapes, profile, body=body, label=label, points=points, role=role
        )

    arm_joints = []
    joint_ids = {}
    for name, row in urdf["links"].items():
        bodies[name] = template.add_link(
            xform=pose(base @ zero_fk[name]),
            mass=row["mass"],
            com=row["com"],
            inertia=row["inertia"],
            lock_inertia=True,
            label=name,
        )
    arm_joints.append(
        template.add_joint_fixed(
            -1, bodies[base_link], parent_xform=pose(base), label="base_fixed"
        )
    )
    home = np.deg2rad(raw_to_urdf(profile["home_joints_deg"], robot.j2_j3_coupling))
    for index, joint in enumerate(urdf["joints"]):
        jid = template.add_joint_revolute(
            bodies[joint["parent"]],
            bodies[joint["child"]],
            parent_xform=pose(joint["origin"]),
            axis=joint["axis"],
            limit_lower=joint["lower"],
            limit_upper=joint["upper"],
            effort_limit=joint["effort"],
            velocity_limit=joint["velocity"],
            armature=profile["actuator"]["armature"][index],
            actuator_mode=newton.JointTargetMode.NONE,
            label=joint["name"],
        )
        arm_joints.append(jid)
        joint_ids[joint["name"]] = jid
        template.joint_q[template.joint_q_start[jid]] = float(home[index])
    for name in robot.tool_bodies:
        row = profile["tool_bodies"][name]
        bodies[name] = template.add_link(
            mass=row["mass_kg"],
            com=row["com_m"],
            inertia=row["inertia_kg_m2"],
            lock_inertia=True,
            label=name,
        )
    arm_joints.append(
        template.add_joint_fixed(
            bodies[last_link],
            bodies[gripper_body],
            parent_xform=pose(mount),
            label="tool_fixed",
        )
    )
    slide_sign = dict(zip(jaws, (-1.0, 1.0), strict=True))
    for name in jaws:
        jaw = profile["jaw"]
        jid = template.add_joint_prismatic(
            bodies[gripper_body],
            bodies[name],
            axis=(slide_sign[name], 0, 0),
            limit_lower=jaw["limit_lower_m"],
            limit_upper=jaw["limit_upper_m"],
            effort_limit=jaw["effort_limit_n"],
            velocity_limit=jaw["velocity_limit_m_s"],
            armature=jaw["armature_kg"],
            actuator_mode=newton.JointTargetMode.NONE,
            label=name,
        )
        arm_joints.append(jid)
        joint_ids[name] = jid
        template.joint_q[template.joint_q_start[jid]] = jaw["open_m"]
    arm_joints.append(
        template.add_joint_fixed(
            bodies[last_link],
            bodies["wrist_camera"],
            parent_xform=pose(wrist_mount),
            label="camera_fixed",
        )
    )
    template.add_articulation(arm_joints, label="fanuc")
    for name, row in urdf["links"].items():
        path = pathlib.Path(profile["robot_collision_path"]) / row["mesh"]
        mesh = trimesh.load_mesh(path, process=False)
        points = np.asarray(mesh.vertices) * row["scale"]
        matrix = row["collision_origin"]
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        body_shapes[name] = [
            add_hull(bodies[name], f"{name}/collision", points, "robot")
        ]
        assets[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    pad_geometry = {}
    group_vertices: dict[str, list[list[float]]] = {}
    parts, gripper_assets = gripper_shapes(profile)
    assets.update(gripper_assets)
    for name, label, points, role in parts:
        body_shapes.setdefault(name, []).append(
            add_hull(bodies[name], label, points, role)
        )
        group_vertices.setdefault(name, []).extend(points.tolist())
        if role in ("pad_a", "pad_b"):
            pad_geometry[name] = {
                "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                "center_m": ((points.min(axis=0) + points.max(axis=0)) / 2).tolist(),
                "inner_face_center_m": [
                    float(
                        points[:, 0].max() if role == "pad_a" else points[:, 0].min()
                    ),
                    float((points[:, 1].min() + points[:, 1].max()) / 2),
                    float((points[:, 2].min() + points[:, 2].max()) / 2),
                ],
                "slide_axis": [slide_sign[name], 0, 0],
            }
    exclusions = []
    if robot.gripper.coupled:
        template.add_constraint_mimic(joint_ids[jaws[0]], joint_ids[jaws[1]])
        # The fingers meet face to face at joint 0; their one motor, not a
        # contact between them, sets the gap.
        pairs = [(a, b) for a in body_shapes[jaws[0]] for b in body_shapes[jaws[1]]]
        reason = "coupled fingers meet at the closed stop"
        exclusions.append([*jaws, reason])
    else:
        # Convexification fills the two interlocking rack recesses (their hull
        # volume is 2.33x CAD volume), creating a 14 mm false overlap at the
        # authored open pose. Only this internal rack pair is excluded; pad
        # contacts remain live.
        racks = ["carriage_a/solid_11", "carriage_b/solid_20"]
        pairs = [(shape_labels.index(racks[0]), shape_labels.index(racks[1]))]
        exclusions.append(
            [
                *racks,
                "interlocking concave racks; convex-hull approximation fills recesses",
            ]
        )
    for a, b in pairs:
        template.add_shape_collision_filter_pair(a, b)
    # The two jaws overlap mechanically with the fixed gripper body/last link
    # assembly; adjacent joints are filtered by Newton. The rigid camera has
    # no collision mesh.
    for name in jaws:
        for a in body_shapes[last_link]:
            for b in body_shapes[name]:
                template.add_shape_collision_filter_pair(a, b)
        exclusions.append(
            [last_link, name, "fixed tool mount and internal carriage assembly"]
        )
    maps = {
        "bodies": bodies,
        "joints": joint_ids,
        "body_count_per_world": template.body_count,
        "shape_count_per_world": template.shape_count,
        "joint_coord_count_per_world": template.joint_coord_count,
        "joint_dof_count_per_world": template.joint_dof_count,
        "arm_q": [template.joint_q_start[joint_ids[n]] for n in robot.joint_names],
        "arm_qd": [template.joint_qd_start[joint_ids[n]] for n in robot.joint_names],
        "jaw_q": [template.joint_q_start[joint_ids[name]] for name in jaws],
        "jaw_qd": [template.joint_qd_start[joint_ids[name]] for name in jaws],
        "body_labels": template.body_label,
        "shape_labels": shape_labels,
        "shape_roles": shape_roles,
        "body_shapes": body_shapes,
        "joint_labels": template.joint_label,
        "joint_axes": [j["axis"].tolist() for j in urdf["joints"]],
        "joint_lower_rad": [j["lower"] for j in urdf["joints"]],
        "joint_upper_rad": [j["upper"] for j in urdf["joints"]],
        "joint_velocity_deg_s": np.rad2deg(
            [j["velocity"] for j in urdf["joints"]]
        ).tolist(),
        "base_mount": base.tolist(),
        "j6_to_fanuc_flange": urdf["j6_to_flange"].tolist(),
        "gripper_pose_flange": np.asarray(gripper_pose_flange).tolist(),
        "wrist_pose_flange": np.asarray(wrist_pose_flange).tolist(),
        "object_velocity_convention": "world-frame center-of-mass linear metres/s and angular radians/s; initialized and reset only",
        "pad_geometry": pad_geometry,
        "group_vertices_m": {
            name: np.unique(points, axis=0).tolist()
            for name, points in group_vertices.items()
        },
        "asset_sha256": assets,
        "collision_approximation": "convex hull per robot mesh/gripper solid; URDF collision meshes scaled from mm; scene collision components recorded separately",
        "collision_exclusions": exclusions,
        "adjacent_link_exclusions": [
            [
                template.body_label[parent],
                template.body_label[child],
                "adjacent joint attachment",
            ]
            for parent, child in zip(
                template.joint_parent, template.joint_child, strict=True
            )
            if parent >= 0
        ],
        "within_body_exclusion": "shapes on the same rigid body do not self-collide",
    }
    maps["robot_body_order"] = [*robot.link_names, *robot.tool_bodies]
    return template, maps


def finalize_model(
    template: Any,
    mappings: dict[str, Any],
    *,
    world_count: int,
    device: str,
    object_initial: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Replicate a completed scene and set initial free-body states only at construction."""
    import newton
    import warp as wp

    if type(world_count) is not int or world_count < 1:
        raise ValueError("world_count must be a positive integer")
    if (
        len(mappings["shape_labels"]) != template.shape_count
        or len(mappings["shape_roles"]) != template.shape_count
    ):
        raise ValueError("Collision shape mappings must cover the entire template")
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, world_count)
    states = {} if object_initial is None else object_initial
    resolved = {}
    for name, initial in states.items():
        if name not in mappings["bodies"] or name not in mappings["joints"]:
            raise ValueError("Initial state must name a registered free body")
        joint = mappings["joints"][name]
        q_end = (
            template.joint_q_start[joint + 1]
            if joint + 1 < template.joint_count
            else template.joint_coord_count
        )
        qd_end = (
            template.joint_qd_start[joint + 1]
            if joint + 1 < template.joint_count
            else template.joint_dof_count
        )
        if (
            q_end - template.joint_q_start[joint] != 7
            or qd_end - template.joint_qd_start[joint] != 6
        ):
            raise ValueError("Initial object state requires a free body joint")
        values_by_key = {}
        for key, width in (
            ("pose_xyzw", 7),
            ("velocity_mps", 3),
            ("angular_velocity_radps", 3),
        ):
            values = np.asarray(initial[key], dtype=float)
            if (
                values.shape not in ((width,), (world_count, width))
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Object {key} shape must match world_count")
            values_by_key[key] = np.broadcast_to(values, (world_count, width)).copy()
        if not np.allclose(
            np.linalg.norm(values_by_key["pose_xyzw"][:, 3:], axis=-1), 1, atol=1e-5
        ):
            raise ValueError("Object initial pose requires a unit xyzw quaternion")
        resolved[name] = values_by_key
    for world in range(world_count):
        for name, initial in resolved.items():
            joint = mappings["joints"][name]
            q_start = world * template.joint_coord_count + template.joint_q_start[joint]
            qd_start = world * template.joint_dof_count + template.joint_qd_start[joint]
            values = initial["pose_xyzw"][world]
            builder.joint_q[q_start : q_start + 7] = values.tolist()
            builder.joint_qd[qd_start : qd_start + 6] = np.r_[
                initial["velocity_mps"][world], initial["angular_velocity_radps"][world]
            ].tolist()
            builder.body_q[world * template.body_count + mappings["bodies"][name]] = (
                wp.transform(values[:3], values[3:])
            )
    maps = {
        **mappings,
        "body_count_per_world": template.body_count,
        "shape_count_per_world": template.shape_count,
        "joint_coord_count_per_world": template.joint_coord_count,
        "joint_dof_count_per_world": template.joint_dof_count,
        "body_labels": list(template.body_label),
        "joint_labels": list(template.joint_label),
        "object_initial_state": {
            name: {key: values.tolist() for key, values in initial.items()}
            for name, initial in resolved.items()
        },
    }
    return builder.finalize(device=device), maps
