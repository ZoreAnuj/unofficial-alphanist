"""Newton UR7 assembly environment, with physical resets and sparse assembly reward.

Unpublished settings are explicit engineering choices in DEFAULT_CONFIG. Rendering
is not required: sampled CAD points are transformed by current physical body poses.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
from typing import Any

import newton
import numpy as np
import torch
import warp as wp
from scipy.spatial import transform as scipy_transform

from isaac.assembly_rl import (
    ccd_workspace,
    compact_allocation,
    geometry,
    hessian_precision,
    initialization,
    physics_step,
    reset_bank_builder,
    sap_wide,
)
from isaac.assembly_rl import model as scene_model

DEFAULT_CONFIG = {
    "horizon_steps": 375,  # Steps per assembly pair: AlphaNIST allows 15 s per pair.
    "num_points": 1024,  # Unpublished in AlphaNIST.
    "control_cost": 1e-6,  # Unpublished coefficient; no dense pose/progress reward.
    "relative_velocity_fraction": 0.5,  # Unpublished action scale.
    "jaw_command_margin_m": 0.0002,  # Engineering soft-stop; physical guards unchanged.
    "gripper_velocity_fraction": 0.4,  # 16N spring force leaves actuator headroom.
    "arm_servo_tau_s": 0.02,
    "arm_acceleration_rad_s2": 2 * np.pi,
    "arm_spring_effort_fraction": 0.5,  # Reserve headroom for gravity/damping.
    "policy_hz": 25,
    "point_frame": "gripper",  # Cited SGS implementation convention.
    "point_allocation": "half_robot_quarter_receiver_quarter_moving",
    "pose_metric": "maximum_part_RMS_corresponding_surface_displacement",
    "success_threshold_m": 0.001,
    "deterministic_solver": False,
    "use_cuda_graph": True,
    "solver_nconmax": 16384,  # Per world; threaded task sets receive a larger pool.
    "solver_njmax": 65536,  # Per world; Gantry reset peaked at41023 rows.
    "solver_jacobian": "sparse",
}


def assembly_success(
    errors: torch.Tensor, moving: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    error = errors.masked_fill(~moving, 0).amax(-1)
    return (error < 0.001) & moving.any(-1), error


def sparse_reward(
    success: torch.Tensor,
    actions: torch.Tensor,
    velocity: torch.Tensor,
    torque: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    cost = (
        actions.square().sum(-1) + velocity.square().sum(-1) + torque.square().sum(-1)
    )
    return success.float() - coefficient * cost


def valid_contact_geometry(
    positions: torch.Tensor, frames: torch.Tensor, distances: torch.Tensor
) -> torch.Tensor:
    """Finite depth alone misses collapsed normals from rounded CCD witnesses."""
    gram = frames @ frames.transpose(-1, -2)
    identity = torch.eye(3, device=frames.device, dtype=frames.dtype)
    handedness = (torch.cross(frames[:, 0], frames[:, 1], dim=-1) * frames[:, 2]).sum(
        -1
    )
    return (
        torch.isfinite(positions).all(-1)
        & torch.isfinite(distances)
        & torch.isfinite(frames).all(dim=(-1, -2))
        & ((gram - identity).abs().amax(dim=(-1, -2)) < 1e-4)
        & ((handedness - 1).abs() < 1e-4)
    )


def load_bank(path: pathlib.Path, device: torch.device) -> tuple[dict, dict]:
    path = pathlib.Path(path)
    path = path.parent if path.is_file() else path
    metadata = json.loads((path / "bank.json").read_text())
    if not metadata.get("training_ready") or not metadata.get("physics_validated"):
        raise ValueError(
            "Training requires a physically validated, complete reset bank"
        )
    with np.load(path / "bank.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    reset_bank_builder.validate_bank_entries(raw, metadata)
    arrays = {
        key: torch.as_tensor(value, device=device)
        if value.dtype.kind not in "US"
        else value.tolist()
        for key, value in raw.items()
    }
    required = {
        "task_ids",
        "part_poses",
        "joint_q",
        "joint_qd",
        "goals",
        "gripper_open",
        "family",
    }
    if not required.issubset(arrays) or not len(arrays["task_ids"]):
        raise ValueError("Incomplete reset bank schema")
    for value in arrays.values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite reset bank")
    return arrays, metadata


def _matrix_pose(matrix: np.ndarray) -> np.ndarray:
    return np.r_[
        matrix[:3, 3], scipy_transform.Rotation.from_matrix(matrix[:3, :3]).as_quat()
    ]


class AssemblyEnvironment:
    schema_version = 1
    max_parts = 4
    proprio_dim = 140
    action_dim = 7

    @property
    def max_horizon_steps(self) -> int:
        return int(self.config["horizon_steps"]) * max(
            sum(not part["fixed"] for part in task["parts"]) for task in self.tasks
        )

    def __init__(
        self,
        assets_path: pathlib.Path,
        bank_path: pathlib.Path | None = None,
        num_envs: int = 1,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        config: dict | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.num_points = int(self.config["num_points"])
        if num_envs < 1 or self.num_points < 16 or self.config["horizon_steps"] < 1:
            raise ValueError(
                "Positive environment count, horizon, and at least16 points required"
            )
        if self.config["policy_hz"] != 25:
            raise ValueError("AlphaNIST policy rate is25Hz")
        if self.config["solver_jacobian"] not in ("dense", "sparse"):
            raise ValueError("Solver Jacobian must be dense or sparse")
        for key in (
            "point_frame",
            "point_allocation",
            "pose_metric",
            "success_threshold_m",
        ):
            if self.config[key] != DEFAULT_CONFIG[key]:
                raise ValueError(f"Unsupported {key}: {self.config[key]}")
        if (
            not 0 < self.config["relative_velocity_fraction"] <= 1
            or not 0 < self.config["gripper_velocity_fraction"] <= 1
            or not 0 <= self.config["jaw_command_margin_m"] < 0.0125
            or not 0 < self.config["arm_spring_effort_fraction"] < 1
            or not 0 < self.config["arm_servo_tau_s"] < float("inf")
            or not 0 < self.config["arm_acceleration_rad_s2"] < float("inf")
            or self.config["control_cost"] < 0
        ):
            raise ValueError("Invalid action scale/control cost")
        self.assets_path = pathlib.Path(assets_path)
        self.bank, self.bank_metadata = (
            ({}, {}) if bank_path is None else load_bank(bank_path, self.device)
        )
        task_names = self.bank_metadata.get("task_names", self.config.get("task_names"))
        self.tasks = scene_model.load_tasks(
            self.assets_path,
            task_names,
            allow_unvalidated=bool(self.config.get("allow_unvalidated_assets", False)),
        )
        if "solver_nconmax" not in (config or {}) and any(
            "screw_pitch_m" in part for task in self.tasks for part in task["parts"]
        ):
            # Engaged threads reached 48,896 broadphase pairs/world; source
            # fixture/nut bounds can exceed 74k with both nuts engaged.
            self.config["solver_nconmax"] = 131072
        self.task_names = [task["id"] for task in self.tasks]
        if self.bank_metadata and self.bank_metadata.get("task_part_counts") != [
            len(task["parts"]) for task in self.tasks
        ]:
            raise ValueError("Reset bank part schema differs from physical task assets")
        if any(len(task["parts"]) > self.max_parts for task in self.tasks):
            raise ValueError("The observation schema supports at most4 parts")
        for name, digest in self.bank_metadata.get("source_hashes", {}).items():
            if (
                hashlib.sha256((self.assets_path / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("Reset bank asset hash mismatch")
        self.bank_size = len(self.bank.get("task_ids", []))
        self.bank_task_ids = self.bank.get(
            "task_ids", torch.empty(0, dtype=torch.long, device=self.device)
        ).long()
        self.bank_family_names = self.bank.get("family", [])
        wp.init()
        if self.device.type == "cuda":
            # CUDA's legacy null stream cannot be captured by Warp. Keep Torch
            # and Newton on one owned, non-default stream, including PPO work.
            current = torch.cuda.current_stream(self.device)
            if current.cuda_stream == 0:
                self._stream = wp.Stream(str(self.device))
                self._torch_stream = wp.stream_to_torch(self._stream)
                self._torch_stream.wait_stream(current)
                torch.cuda.set_stream(self._torch_stream)
            else:
                self._stream = wp.stream_from_torch(current)
            wp.set_stream(self._stream)
        use_mujoco_contacts = True
        workspace_fingerprint = ccd_workspace.install()
        broadphase_fingerprint = sap_wide.install()
        allocation_fingerprint = compact_allocation.install()
        initialization_fingerprint = initialization.install()
        hessian_fingerprint = hessian_precision.install()
        self.model, self.maps, self.profile = scene_model.build_model(
            self.assets_path,
            self.tasks,
            world_count=num_envs,
            device=str(self.device),
            use_mujoco_contacts=use_mujoco_contacts,
        )
        physical = {
            "profile": {
                key: value
                for key, value in self.profile.items()
                if not key.endswith("_path")
            },
            "assets": sorted(self.maps["asset_sha256"].values()),
            "base_mount": self.maps["base_mount"],
            "absent_scene_bodies": self.maps.get("absent_scene_bodies", []),
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("newton", "warp-lang", "mujoco", "mujoco-warp")
            },
            "ccd_workspace": workspace_fingerprint,
            "sap_wide": broadphase_fingerprint,
            "compact_allocation": allocation_fingerprint,
            "initialization": initialization_fingerprint,
            "hessian_precision": hessian_fingerprint,
            "environment_source_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
            "physics_settings": {
                key: self.config[key]
                for key in (
                    "policy_hz",
                    "gripper_velocity_fraction",
                    "jaw_command_margin_m",
                    "relative_velocity_fraction",
                    "arm_servo_tau_s",
                    "arm_acceleration_rad_s2",
                    "arm_spring_effort_fraction",
                    "solver_nconmax",
                    "solver_njmax",
                    "solver_jacobian",
                    "deterministic_solver",
                )
            },
            "physics_source": {
                module.__name__: hashlib.sha256(
                    pathlib.Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (scene_model, physics_step)
            },
        }
        self.physics_identity = hashlib.sha256(
            json.dumps(physical, sort_keys=True).encode()
        ).hexdigest()
        if self.bank_metadata.get("physics_identity") not in (
            None,
            self.physics_identity,
        ):
            raise ValueError(
                "Reset bank physics identity differs from current model/runtime"
            )
        options = {
            key: value
            for key, value in self.profile["solver"].items()
            if key != "provenance"
        }
        options["nconmax"] = int(self.config["solver_nconmax"])
        options["njmax"] = int(self.config["solver_njmax"])
        options["jacobian"] = self.config["solver_jacobian"]
        self.solver = scene_model.AssemblySolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            use_mujoco_contacts=use_mujoco_contacts,
            solver="newton",
            integrator="implicitfast",
            enable_multiccd=True,
            deterministic=wp.DeterministicMode.RUN_TO_RUN
            if self.config["deterministic_solver"]
            else wp.DeterministicMode.NOT_GUARANTEED,
            **options,
        )
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.nq = self.maps["joint_coord_count_per_world"]
        self.nv = self.maps["joint_dof_count_per_world"]
        self.nb = self.maps["body_count_per_world"]
        self.nj = self.maps["joint_count_per_world"]
        self.ns = self.maps["shape_count_per_world"]
        self.dt = 1 / self.profile["physics_hz"]
        self.substeps = self.profile["physics_hz"] // 25
        self.contacts = newton.Contacts(
            self.solver.mjw_data.naconmax,
            0,
            device=str(self.device),
            requested_attributes={"force"},
        )
        self._initial_q = (
            wp.to_torch(self.model.joint_q).view(num_envs, self.nq).clone()
        )
        self._initial_roots = wp.to_torch(self.model.joint_X_p).clone()
        self.task_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.bank_indices = torch.full_like(self.task_ids, -1)
        self.elapsed = torch.zeros_like(self.task_ids)
        self.failure = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.parked = torch.zeros_like(self.failure)
        self.penetration = torch.zeros(num_envs, device=self.device)
        self.pad_penetration = torch.zeros_like(self.penetration)
        self.rigid_penetration = torch.zeros_like(self.penetration)
        self.pad_part_penetration = torch.zeros(
            (num_envs, self.max_parts), device=self.device
        )
        self.goals = torch.zeros((num_envs, self.max_parts, 7), device=self.device)
        self.goals[..., 6] = 1
        self.history = torch.zeros((num_envs, 5, 28), device=self.device)
        self.last_actions = torch.zeros((num_envs, 7), device=self.device)
        self.targets = self._initial_q[:, :8].clone()
        self.jaw_servo = self.targets[:, 6:].clone()
        self.arm_servo = self.targets[:, :6].clone()
        self.arm_servo_velocity = torch.zeros_like(self.arm_servo)
        if (
            self.profile["jaw"]["kd"]
            * self.profile["jaw"]["velocity_limit_m_s"]
            * self.config["gripper_velocity_fraction"]
            >= self.profile["jaw"]["effort_limit_n"]
        ):
            raise ValueError(
                "Gripper servo requires spring force below actuator saturation"
            )
        self.gripper_open = torch.ones(num_envs, device=self.device)
        self.lower = self._tensor([*self.maps["joint_lower_rad"], 0.0, 0.0])
        self.upper = self._tensor([*self.maps["joint_upper_rad"], 0.025, 0.025])
        self.vmax = self._tensor(
            [*np.radians(self.maps["joint_velocity_deg_s"]), 0.1, 0.1]
        )
        self._flange_body = self.maps["bodies"][self.maps["robot_body_order"][6]]
        self._flange_offset = self._tensor(
            _matrix_pose(np.asarray(self.maps["j6_to_fanuc_flange"]))
        )
        self._base_inverse = geometry.inverse_pose(
            self._tensor(_matrix_pose(scene_model.BASE_MOUNT))
        )
        self._point_templates(seed)
        # Only fingers/pads may touch task objects; robot/robot and robot/table
        # collisions fail the episode (adjacent links excluded by URDF template).
        self._shape_is_robot = torch.tensor(
            [
                role not in ("part", "receiver", "support") and "pad" not in role
                for role in self.maps["shape_roles"]
            ],
            device=self.device,
        )
        self._shape_is_pad = torch.tensor(
            ["pad" in role for role in self.maps["shape_roles"]], device=self.device
        )
        self._shape_part_slot = torch.full(
            (self.ns,), -1, dtype=torch.int32, device=self.device
        )
        for task_index, parts in enumerate(self.maps["task_parts"]):
            for part_index, part in enumerate(parts):
                if not part["fixed"]:
                    self._shape_part_slot[part["shapes"]] = (
                        task_index * self.max_parts + part_index
                    )
        self._mocap_joints = (
            None
            if self.solver.mjc_mocap_to_newton_jnt is None
            else wp.to_torch(self.solver.mjc_mocap_to_newton_jnt).long()
        )
        self._stepper = physics_step.Stepper(self)
        self._fk()
        self._refresh_history(torch.arange(num_envs, device=self.device), reset=True)

    def _tensor(self, value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _fk(self, ids: torch.Tensor | None = None) -> None:
        mask = None
        if ids is not None:
            selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            selected[ids] = True
            mask = wp.from_torch(
                selected.repeat_interleave(
                    self.model.articulation_count // self.num_envs
                ),
                dtype=wp.bool,
            )
        for state in (self.state, self.next_state):
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state, mask=mask)

    def _point_templates(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        robot = self.maps["robot_point_clouds"]
        robot_points = np.concatenate([row["points"] for row in robot])
        robot_bodies = np.concatenate(
            [np.full(len(row["points"]), row["body"]) for row in robot]
        )
        robot_ids = rng.choice(len(robot_points), self.num_points // 2, replace=True)
        all_points, all_bodies, metric_points, moving, body_slots = [], [], [], [], []
        for parts in self.maps["task_parts"]:
            points, bodies = [robot_points[robot_ids]], [robot_bodies[robot_ids]]
            for fixed, count in (
                (True, self.num_points // 4),
                (False, self.num_points - self.num_points // 2 - self.num_points // 4),
            ):
                group = [row for row in parts if row["fixed"] == fixed]
                for index, row in enumerate(group):
                    n = count // len(group) + int(index < count % len(group))
                    points.append(
                        scene_model.sample_surface(
                            row["vertices"], row["faces"], n, seed + row["body"]
                        )
                    )
                    bodies.append(np.full(n, row["body"]))
            all_points.append(np.concatenate(points))
            all_bodies.append(np.concatenate(bodies))
            metric_points.append(
                np.stack(
                    [row["point_cloud_local"] for row in parts]
                    + [np.zeros((128, 3))] * (self.max_parts - len(parts))
                )
            )
            moving.append(
                [not row["fixed"] for row in parts]
                + [False] * (self.max_parts - len(parts))
            )
            body_slots.append(
                [row["body"] for row in parts]
                + [parts[0]["body"]] * (self.max_parts - len(parts))
            )
        self.local_points = self._tensor(np.stack(all_points))
        self.point_bodies = self._tensor(np.stack(all_bodies), torch.long)
        self.metric_points = self._tensor(np.stack(metric_points))
        self.moving = self._tensor(moving, torch.bool)
        self.part_body_slots = self._tensor(body_slots, torch.long)

    def _body_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            wp.to_torch(self.state.body_q).view(self.num_envs, self.nb, 7),
            wp.to_torch(self.state.body_qd).view(self.num_envs, self.nb, 6),
        )

    def _proprio(self) -> torch.Tensor:
        body, velocity = self._body_state()
        link = body[:, self._flange_body]
        flange = geometry.compose(link, self._flange_offset.expand(self.num_envs, -1))
        com = wp.to_torch(self.model.body_com).view(self.num_envs, self.nb, 3)[
            :, self._flange_body
        ]
        com_world = geometry.rotate(link[:, 3:], com) + link[:, :3]
        twist = velocity[:, self._flange_body]
        linear = twist[:, :3] + torch.cross(
            twist[:, 3:], flange[:, :3] - com_world, dim=-1
        )
        base = self._base_inverse.expand(self.num_envs, -1)
        pose_base = geometry.compose(base, flange)
        twist_base = torch.cat(
            (
                geometry.rotate(base[:, 3:], linear),
                geometry.rotate(base[:, 3:], twist[:, 3:]),
            ),
            -1,
        )
        return torch.cat(
            (
                pose_base,
                twist_base,
                wp.to_torch(self.state.joint_q).view(self.num_envs, self.nq)[:, :8],
                self.last_actions,
            ),
            -1,
        )

    def _refresh_history(self, ids: torch.Tensor, *, reset: bool = False) -> None:
        current = self._proprio()[ids]
        if reset:
            self.history[ids] = current[:, None, :]
        else:
            self.history[ids, :-1] = self.history[ids, 1:].clone()
            self.history[ids, -1] = current

    def observe(self) -> dict[str, torch.Tensor]:
        body, _ = self._body_state()
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        poses = body[rows, self.point_bodies[self.task_ids]]
        world_points = (
            geometry.rotate(poses[..., 3:], self.local_points[self.task_ids])
            + poses[..., :3]
        )
        flange = geometry.compose(
            body[:, self._flange_body], self._flange_offset.expand(self.num_envs, -1)
        )
        points = geometry.transform_points(geometry.inverse_pose(flange), world_points)
        return {"points": points, "proprio": self.history.flatten(1).clone()}

    def _sync_mocap(self) -> None:
        if self._mocap_joints is not None:
            poses = wp.to_torch(self.model.joint_X_p)[self._mocap_joints]
            wp.to_torch(self.solver.mjw_data.mocap_pos).copy_(poses[..., :3])
            wp.to_torch(self.solver.mjw_data.mocap_quat).copy_(poses[..., [6, 3, 4, 5]])

    def _initialize(
        self,
        ids: torch.Tensor,
        task_ids: torch.Tensor,
        part_poses: torch.Tensor,
        robot_q: torch.Tensor,
        goals: torch.Tensor,
        robot_qd: torch.Tensor | None = None,
        part_velocities: torch.Tensor | None = None,
    ) -> None:
        ids = ids.to(self.device).long()
        task_ids = task_ids.to(self.device).long()
        if (
            ids.ndim != 1
            or len(ids.unique()) != len(ids)
            or (ids < 0).any()
            or (ids >= self.num_envs).any()
        ):
            raise ValueError("Reset environment indices must be unique and in range")
        if (
            len(ids) != len(task_ids)
            or (task_ids < 0).any()
            or (task_ids >= len(self.tasks)).any()
        ):
            raise ValueError("Invalid reset indices/task identities")
        part_poses, robot_q, goals = (
            self._tensor(value) for value in (part_poses, robot_q, goals)
        )
        if not all(
            torch.isfinite(value).all() for value in (part_poses, robot_q, goals)
        ):
            raise ValueError("Nonfinite reset state")
        if (
            part_poses.shape != (len(ids), self.max_parts, 7)
            or goals.shape != part_poses.shape
            or robot_q.shape != (len(ids), 8)
        ):
            raise ValueError("Incorrect reset state dimensions")
        for poses in (part_poses, goals):
            if not torch.allclose(
                poses[..., 3:].norm(dim=-1), torch.ones_like(poses[..., 0]), atol=1e-5
            ):
                raise ValueError("Reset poses require unit XYZW quaternions")
        if ((robot_q < self.lower) | (robot_q > self.upper)).any():
            raise ValueError("Reset robot joints violate limits")
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[ids] = True
        for state in (self.state, self.next_state):
            self.solver.reset(state, world_mask=wp.from_torch(mask, dtype=wp.bool))
            q = wp.to_torch(state.joint_q).view(self.num_envs, self.nq)
            qd = wp.to_torch(state.joint_qd).view(self.num_envs, self.nv)
            q[ids] = self._initial_q[ids]
            q[ids, :8] = robot_q
            qd[ids] = 0
            if robot_qd is not None:
                qd[ids, :8] = self._tensor(robot_qd)
            for task_index, parts in enumerate(self.maps["task_parts"]):
                selected = torch.nonzero(task_ids == task_index).flatten()
                worlds = ids[selected]
                for part_index, row in enumerate(parts):
                    if row["fixed"]:
                        roots = wp.to_torch(self.model.joint_X_p)
                        global_joints = ids * self.nj + row["joint"]
                        roots[global_joints] = self._initial_roots[global_joints]
                        roots[worlds * self.nj + row["joint"]] = part_poses[
                            selected, part_index
                        ]
                    else:
                        start, dof = row["q_start"], row["qd_start"]
                        q[worlds, start : start + 7] = part_poses[selected, part_index]
                        if part_velocities is not None:
                            qd[worlds, dof : dof + 6] = self._tensor(part_velocities)[
                                selected, part_index
                            ]
        self.task_ids[ids] = task_ids
        self.goals[ids] = goals
        self.targets[ids] = robot_q
        self.jaw_servo[ids] = robot_q[:, 6:]
        self.arm_servo[ids] = robot_q[:, :6]
        self.arm_servo_velocity[ids] = (
            0 if robot_qd is None else self._tensor(robot_qd)[:, :6]
        )
        self.gripper_open[ids] = (
            robot_q[:, 6:8].mean(-1) / self.profile["jaw"]["open_m"]
        )
        self.elapsed[ids] = 0
        self.failure[ids] = False
        self.parked[ids] = False
        self.penetration[ids] = 0
        self.pad_penetration[ids] = 0
        self.rigid_penetration[ids] = 0
        self.pad_part_penetration[ids] = 0
        self.last_actions[ids] = 0
        self._sync_mocap()
        self._fk(ids)
        self._refresh_history(ids, reset=True)

    def park_worlds(self, env_ids: torch.Tensor) -> None:
        """Quarantine rejected bank candidates without disturbing valid lanes."""
        ids = self._tensor(env_ids, torch.long)
        selected = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        selected[ids] = True
        for state in (self.state, self.next_state):
            self.solver.reset(state, world_mask=wp.from_torch(selected, dtype=wp.bool))
            state.clear_forces()
        roots = wp.to_torch(self.model.joint_X_p).view(self.num_envs, self.nj, 7)
        roots[ids] = self._initial_roots.view(self.num_envs, self.nj, 7)[ids]
        self.parked[ids] = True
        self.targets[ids] = self._initial_q[ids, :8]
        self.jaw_servo[ids] = self._initial_q[ids, 6:8]
        self.arm_servo[ids] = self._initial_q[ids, :6]
        self.arm_servo_velocity[ids] = 0
        self.gripper_open[ids] = 1
        self._sync_mocap()
        self._fk(ids)

    def inspect_contacts(self) -> dict[str, torch.Tensor]:
        """Query spawn overlap without solving constraints or advancing time."""
        import mujoco_warp

        data = self.solver.mjw_data
        self.solver._update_mjc_data(data, self.model, self.state)
        mujoco_warp.fwd_kinematics(self.solver.mjw_model, data)
        mujoco_warp.collision(self.solver.mjw_model, data)
        count = int(wp.to_torch(data.nacon)[0])
        if count > data.naconmax:
            raise RuntimeError("Spawn collision capacity exceeded")
        self.penetration.zero_()
        self.pad_penetration.zero_()
        self.rigid_penetration.zero_()
        self.pad_part_penetration.zero_()
        if count:
            world = wp.to_torch(data.contact.worldid)[:count].long()
            geoms = wp.to_torch(data.contact.geom)[:count].long()
            geometry_valid = valid_contact_geometry(
                wp.to_torch(data.contact.pos)[:count],
                wp.to_torch(data.contact.frame)[:count],
                wp.to_torch(data.contact.dist)[:count],
            )
            malformed = (world >= 0) & (world < self.num_envs) & ~geometry_valid
            self.failure[world[malformed]] = True
            depth = (-wp.to_torch(data.contact.dist)[:count]).clamp_min(0)
            valid = (world >= 0) & (world < self.num_envs) & (geoms >= 0).all(-1)
            world, geoms, depth = world[valid], geoms[valid], depth[valid]
            mapped = wp.to_torch(self.solver.mjc_geom_to_newton_shape).long()
            shapes = mapped[world[:, None], geoms]
            self.penetration.scatter_reduce_(
                0, world, depth, reduce="amax", include_self=True
            )
            pad = self._shape_is_pad[shapes % self.ns].any(-1)
            self.pad_penetration.scatter_reduce_(
                0, world[pad], depth[pad], reduce="amax", include_self=True
            )
            # Spawn validation permits no compliant-pad exception.
            self.rigid_penetration.copy_(self.penetration)
            forbidden = self._shape_is_robot[shapes % self.ns].any(-1) & (depth > 1e-5)
            self.failure[world[forbidden]] = True
        return self.snapshot()

    def initialize(
        self,
        task_ids: torch.Tensor,
        part_poses: torch.Tensor,
        robot_q: torch.Tensor,
        goals: torch.Tensor,
    ) -> None:
        self._initialize(
            torch.arange(self.num_envs, device=self.device),
            task_ids,
            part_poses,
            robot_q,
            goals,
        )

    def reset(
        self, env_ids: torch.Tensor, bank_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not self.bank_size:
            raise ValueError("reset requires a validated reset bank")
        bank_indices = bank_indices.to(self.device).long()
        if (bank_indices < 0).any() or (bank_indices >= self.bank_size).any():
            raise ValueError("Invalid reset-bank index")
        self._initialize(
            env_ids,
            self.bank_task_ids[bank_indices],
            self.bank["part_poses"][bank_indices],
            self.bank["joint_q"][bank_indices],
            self.bank["goals"][bank_indices],
            self.bank["joint_qd"][bank_indices],
            None
            if "part_velocities" not in self.bank
            else self.bank["part_velocities"][bank_indices],
        )
        self.bank_indices[env_ids] = bank_indices
        for name in ("jaw_servo", "arm_servo", "arm_servo_velocity"):
            if name in self.bank:
                getattr(self, name)[env_ids] = self.bank[name][bank_indices].float()
        self.gripper_open[env_ids] = self.bank["gripper_open"][bank_indices].float()
        self.targets[env_ids, 6:] = (
            self.gripper_open[env_ids, None] * self.profile["jaw"]["open_m"]
        )
        if "joint_targets" in self.bank:
            self.targets[env_ids] = self.bank["joint_targets"][bank_indices].float()
        return self.observe()

    def _park_inactive(self) -> None:
        q = wp.to_torch(self.state.joint_q).view(self.num_envs, self.nq)
        qd = wp.to_torch(self.state.joint_qd).view(self.num_envs, self.nv)
        for task_index, parts in enumerate(self.maps["task_parts"]):
            inactive = (self.task_ids != task_index) | self.parked
            for row in parts:
                if not row["fixed"]:
                    start, dof = row["q_start"], row["qd_start"]
                    q[inactive, start : start + 7] = self._initial_q[
                        inactive, start : start + 7
                    ]
                    qd[inactive, dof : dof + 6] = 0

    def _guards(self) -> None:
        flags = wp.to_torch(self._stepper.flags)
        if (flags & 3).any():
            data = self.solver.mjw_data
            raise RuntimeError(
                f"Newton nonfinite state or capacity overflow; training aborted: "
                f"flags={flags.tolist()}, nefc={wp.to_torch(data.nefc).tolist()}, "
                f"njmax={data.njmax}, nacon={wp.to_torch(data.nacon).tolist()}, "
                f"naconmax={data.naconmax}, overflow={wp.to_torch(data.overflow).tolist()}"
            )
        self.failure |= flags != 0
        body, _ = self._body_state()
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        poses = body[rows, self.part_body_slots[self.task_ids]]
        outside = (
            (poses[..., 0].abs() > 0.825)
            | (poses[..., 1] < -0.565)
            | (poses[..., 1] > 0.475)
            | (poses[..., 2] < scene_model.TABLE_HEIGHT_M - 0.1)
        )
        self.failure |= (outside & self.moving[self.task_ids]).any(-1)

    def _advance(self, forces: torch.Tensor | None = None) -> None:
        self._park_inactive()
        wp.to_torch(self.control.joint_target_qd).zero_()
        for state in (self.state, self.next_state):
            state.clear_forces()
            if forces is not None:
                applied = wp.to_torch(state.body_f).view(self.num_envs, self.nb, 6)
                for task_index, parts in enumerate(self.maps["task_parts"]):
                    worlds = torch.nonzero(self.task_ids == task_index).flatten()
                    for part_index, row in enumerate(parts):
                        if not row["fixed"]:
                            applied[worlds, row["body"], :3] = forces[
                                worlds, part_index
                            ]
        self._stepper.advance()
        self._guards()

    def hold(
        self,
        joint_targets_rad: torch.Tensor,
        gripper_open: torch.Tensor,
        steps: int = 1,
    ) -> None:
        targets, opened = self._tensor(joint_targets_rad), self._tensor(gripper_open)
        if (
            (targets < self.lower[:6]) | (targets > self.upper[:6])
        ).any() or not torch.isfinite(targets).all():
            raise ValueError("Invalid hold targets")
        if ((opened < 0) | (opened > 1)).any() or not torch.isfinite(opened).all():
            raise ValueError("Invalid gripper opening")
        self.targets[:, :6] = targets
        self.targets[:, 6:] = opened[:, None] * self.profile["jaw"]["open_m"]
        self.gripper_open.copy_(opened)
        self.targets[self.parked] = self._initial_q[self.parked, :8]
        self.gripper_open[self.parked] = 1
        self.penetration.zero_()
        self.pad_penetration.zero_()
        self.rigid_penetration.zero_()
        self.pad_part_penetration.zero_()
        for _ in range(steps):
            self._advance()
        self._refresh_history(torch.arange(self.num_envs, device=self.device))

    def perturb_objects(self, forces_world: torch.Tensor, steps: int = 1) -> None:
        forces = self._tensor(forces_world)
        if (
            forces.shape != (self.num_envs, self.max_parts, 3)
            or not torch.isfinite(forces).all()
        ):
            raise ValueError("Invalid perturbation forces")
        self.penetration.zero_()
        self.pad_penetration.zero_()
        self.rigid_penetration.zero_()
        self.pad_part_penetration.zero_()
        for _ in range(steps):
            self._advance(forces)
        self.state.clear_forces()
        self.next_state.clear_forces()

    def snapshot(self) -> dict[str, torch.Tensor]:
        body, velocity = self._body_state()
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        slots = self.part_body_slots[self.task_ids]
        return {
            key: value.clone()
            for key, value in {
                "task_ids": self.task_ids,
                "part_poses": body[rows, slots],
                "part_velocities": velocity[rows, slots],
                "goals": self.goals,
                "joint_q": wp.to_torch(self.state.joint_q).view(self.num_envs, self.nq)[
                    :, :8
                ],
                "joint_qd": wp.to_torch(self.state.joint_qd).view(
                    self.num_envs, self.nv
                )[:, :8],
                "safety_failure": self.failure,
                "failure_flags": wp.to_torch(self._stepper.flags),
                "max_penetration_m": self.penetration,
                "max_pad_penetration_m": self.pad_penetration,
                "max_rigid_penetration_m": self.rigid_penetration,
                "max_pad_part_penetration_m": self.pad_part_penetration,
                "gripper_open": self.gripper_open,
                "joint_targets": self.targets,
                "jaw_servo": self.jaw_servo,
                "arm_servo": self.arm_servo,
                "arm_servo_velocity": self.arm_servo_velocity,
                "joint_q_min": wp.to_torch(self._stepper.minimum),
                "joint_q_max": wp.to_torch(self._stepper.maximum),
                "joint_qd_peak": wp.to_torch(self._stepper.peak_velocity),
                "joint_effort_peak": wp.to_torch(self._stepper.peak_effort),
            }.items()
        }

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, 7) or not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite[N,7]")
        self.last_actions.copy_(actions)
        q = wp.to_torch(self.state.joint_q).view(self.num_envs, self.nq)
        applied = actions.clamp(-1, 1)
        targets = (
            q[:, :6]
            + applied[:, :6]
            * self.vmax[:6]
            * self.config["relative_velocity_fraction"]
            / 25
        )
        self.hold(
            targets.clamp(self.lower[:6], self.upper[:6]), (applied[:, 6] + 1) / 2
        )
        self.elapsed += 1
        state = self.snapshot()
        errors = geometry.pose_error(
            state["part_poses"], self.goals, self.metric_points[self.task_ids]
        )
        success, error = assembly_success(errors, self.moving[self.task_ids])
        success &= ~self.failure
        torque = wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, :8]
        reward = sparse_reward(
            success,
            actions,
            state["joint_qd"],
            torque,
            float(self.config["control_cost"]),
        )
        terminated = success | self.failure
        horizon = int(self.config["horizon_steps"]) * self.moving[self.task_ids].sum(-1)
        truncated = (self.elapsed >= horizon) & ~terminated
        return (
            self.observe(),
            reward,
            terminated,
            truncated,
            {
                "success": success,
                "safety_failure": self.failure.clone(),
                "task_id": self.task_ids.clone(),
                "bank_index": self.bank_indices.clone(),
                "assembly_error_m": error,
            },
        )

    def state_dict(self) -> dict[str, Any]:
        # Warm-start and actuator state are required for numerical continuation.
        names = (
            "qpos",
            "qvel",
            "qacc_warmstart",
            "act",
            "ctrl",
            "time",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "eq_active",
            "history",
            "userdata",
        )
        return {
            "schema_version": self.schema_version,
            "task_names": self.task_names,
            "physics_identity": self.physics_identity,
            "config": dict(self.config),
            "local_points": self.local_points.clone(),
            "point_bodies": self.point_bodies.clone(),
            "q": wp.to_torch(self.state.joint_q).clone(),
            "body_q": wp.to_torch(self.state.body_q).clone(),
            "body_qd": wp.to_torch(self.state.body_qd).clone(),
            "solver_step": self.solver._step,
            "qd": wp.to_torch(self.state.joint_qd).clone(),
            "roots": wp.to_torch(self.model.joint_X_p).clone(),
            "solver": {
                name: wp.to_torch(getattr(self.solver.mjw_data, name)).clone()
                for name in names
                if isinstance(getattr(self.solver.mjw_data, name, None), wp.array)
            },
            **{
                name: getattr(self, name).clone()
                for name in (
                    "task_ids",
                    "bank_indices",
                    "elapsed",
                    "failure",
                    "parked",
                    "penetration",
                    "pad_penetration",
                    "rigid_penetration",
                    "pad_part_penetration",
                    "goals",
                    "history",
                    "last_actions",
                    "targets",
                    "jaw_servo",
                    "arm_servo",
                    "arm_servo_velocity",
                    "gripper_open",
                )
            },
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        if (
            data["schema_version"] != self.schema_version
            or data["task_names"] != self.task_names
            or data["physics_identity"] != self.physics_identity
            or data["config"] != self.config
        ):
            raise ValueError("Incompatible environment checkpoint")
        for state in (self.state, self.next_state):
            wp.to_torch(state.joint_q).copy_(data["q"])
            wp.to_torch(state.joint_qd).copy_(data["qd"])
            wp.to_torch(state.body_q).copy_(data["body_q"])
            wp.to_torch(state.body_qd).copy_(data["body_qd"])
        wp.to_torch(self.model.joint_X_p).copy_(data["roots"])
        self.local_points.copy_(data["local_points"])
        self.point_bodies.copy_(data["point_bodies"])
        for name in (
            "task_ids",
            "bank_indices",
            "elapsed",
            "failure",
            "parked",
            "penetration",
            "pad_penetration",
            "rigid_penetration",
            "pad_part_penetration",
            "goals",
            "history",
            "last_actions",
            "targets",
            "jaw_servo",
            "arm_servo",
            "arm_servo_velocity",
            "gripper_open",
        ):
            getattr(self, name).copy_(data[name])
        self.solver._step = data["solver_step"]
        for name, value in data["solver"].items():
            wp.to_torch(getattr(self.solver.mjw_data, name)).copy_(value)

    def close(self) -> None:
        wp.synchronize_device(str(self.device))
