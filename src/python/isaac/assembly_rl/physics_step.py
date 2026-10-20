"""Captured policy intervals with every-substep physics guards on the GPU."""
# Warp kernels require runtime array annotations.
# mypy: disable-error-code="valid-type"

import math
from typing import Any

import warp as wp


@wp.kernel
def joint_guards(  # noqa: PLR0917 -- Warp arrays are explicit kernel arguments
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    nq: int,
    nv: int,
    lower: wp.array(dtype=float),
    upper: wp.array(dtype=float),
    vmax: wp.array(dtype=float),
    nacon: wp.array(dtype=int),
    naconmax: int,
    nefc: wp.array(dtype=int),
    njmax: int,
    overflow: wp.array(dtype=int),
    flags: wp.array(dtype=int),
):
    world = wp.tid()
    error = wp.int32(0)
    for axis in range(nq):
        if not wp.isfinite(q[world * nq + axis]):
            error |= 1
    for axis in range(nv):
        if not wp.isfinite(qd[world * nv + axis]):
            error |= 1
    for axis in range(8):
        tolerance = wp.float32(1e-4)
        if axis >= 6:
            tolerance = 1e-5
        if (
            q[world * nq + axis] < lower[axis] - tolerance
            or q[world * nq + axis] > upper[axis] + tolerance
        ):
            error |= 4
        if wp.abs(qd[world * nv + axis]) > vmax[axis] + tolerance:
            error |= 16
    if nacon[0] > naconmax or nefc[world] > njmax or overflow[world] != 0:
        error |= 2
    wp.atomic_or(flags, world, error)


@wp.kernel
def contact_guards(  # noqa: PLR0913, PLR0917 -- Explicit Warp array arguments
    count: wp.array(dtype=int),
    a: wp.array(dtype=int),
    b: wp.array(dtype=int),
    force: wp.array(dtype=wp.spatial_vector),
    dist: wp.array(dtype=float),
    world_id: wp.array(dtype=int),
    is_robot: wp.array(dtype=wp.bool),
    is_pad: wp.array(dtype=wp.bool),
    part_slots: wp.array(dtype=int),
    task_ids: wp.array(dtype=wp.int64),
    max_parts: int,
    shapes_per_world: int,
    worlds: int,
    flags: wp.array(dtype=int),
    penetration: wp.array(dtype=float),
    pad_penetration: wp.array(dtype=float),
    rigid_penetration: wp.array(dtype=float),
    pad_part_penetration: wp.array2d(dtype=float),
):
    index = wp.tid()
    if index >= count[0]:
        return
    world = world_id[index]
    if world < 0 or world >= worlds:
        return
    wp.atomic_max(penetration, world, wp.max(0.0, -dist[index]))
    first, second = a[index], b[index]
    if first < 0 or second < 0:
        wp.atomic_max(rigid_penetration, world, wp.max(0.0, -dist[index]))
        return
    if is_pad[first % shapes_per_world] or is_pad[second % shapes_per_world]:
        wp.atomic_max(pad_penetration, world, wp.max(0.0, -dist[index]))
    if first // shapes_per_world != second // shapes_per_world:
        wp.atomic_or(flags, world, 8)
        wp.atomic_max(rigid_penetration, world, wp.max(0.0, -dist[index]))
        return
    slot = wp.int32(-1)
    if is_pad[first % shapes_per_world]:
        slot = part_slots[second % shapes_per_world]
    elif is_pad[second % shapes_per_world]:
        slot = part_slots[first % shapes_per_world]
    if slot >= 0 and slot // max_parts == task_ids[world]:
        wp.atomic_max(
            pad_part_penetration, world, slot % max_parts, wp.max(0.0, -dist[index])
        )
    else:
        # Includes pad/table, pad/receiver, and inactive-task contacts.
        wp.atomic_max(rigid_penetration, world, wp.max(0.0, -dist[index]))
    forbidden = (
        is_robot[first % shapes_per_world] or is_robot[second % shapes_per_world]
    )
    if forbidden and wp.length(wp.spatial_top(force[index])) > 0.01:
        wp.atomic_or(flags, world, 8)


@wp.kernel
def jaw_targets(  # noqa: PLR0917 -- Warp kernel arguments
    commands: wp.array2d(dtype=float),
    servo: wp.array2d(dtype=float),
    q: wp.array(dtype=float),
    target: wp.array(dtype=float),
    nq: int,
    stride: int,
    max_step: float,
    alpha: float,
    max_error: float,
    command_lower: float,
    command_upper: float,
):
    world, jaw = wp.tid()
    previous = servo[world, jaw]
    goal = wp.clamp(commands[world, jaw + 6], command_lower, command_upper)
    value = wp.clamp(
        previous + wp.clamp(alpha * (goal - previous), -max_step, max_step),
        command_lower,
        command_upper,
    )
    servo[world, jaw] = value
    position = q[world * nq + jaw + 6]
    target[world * stride + jaw + 6] = position + wp.clamp(
        value - position, -max_error, max_error
    )


@wp.kernel
def arm_targets(  # noqa: PLR0913, PLR0917 -- Warp kernel arguments
    commands: wp.array2d(dtype=float),
    servo: wp.array2d(dtype=float),
    velocity: wp.array2d(dtype=float),
    q: wp.array(dtype=float),
    target: wp.array(dtype=float),
    vmax: wp.array(dtype=float),
    kp: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    effort: wp.array(dtype=float),
    nq: int,
    stride: int,
    dt: float,
    tau: float,
    acceleration: float,
    speed_fraction: float,
    spring_fraction: float,
):
    world, axis = wp.tid()
    position = q[world * nq + axis]
    limit = spring_fraction * effort[axis] / kp[axis]
    # Anti-windup when contact blocks the servo's requested trajectory.
    previous = wp.clamp(servo[world, axis], position - limit, position + limit)
    error = commands[world, axis] - previous
    speed = wp.min(
        vmax[axis] * speed_fraction, wp.sqrt(2.0 * acceleration * wp.abs(error))
    )
    desired = wp.clamp(error / tau, -speed, speed)
    v = velocity[world, axis] + wp.clamp(
        desired - velocity[world, axis], -acceleration * dt, acceleration * dt
    )
    value = previous + v * dt
    if error * (commands[world, axis] - value) < 0.0:
        value = commands[world, axis]
        v = 0.0
    servo[world, axis] = value
    velocity[world, axis] = v
    # POSITION ignores target_qd: encode velocity feedforward in its position.
    correction = value - position + kd[axis] * v / kp[axis]
    target[world * stride + axis] = position + wp.clamp(correction, -limit, limit)


@wp.kernel
def joint_extrema(  # noqa: PLR0917 -- Warp kernel arguments
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    torque: wp.array2d(dtype=float),
    nq: int,
    nv: int,
    minimum: wp.array2d(dtype=float),
    maximum: wp.array2d(dtype=float),
    peak_velocity: wp.array2d(dtype=float),
    peak_effort: wp.array2d(dtype=float),
):
    world, axis = wp.tid()
    value = q[world * nq + axis]
    minimum[world, axis] = wp.min(minimum[world, axis], value)
    maximum[world, axis] = wp.max(maximum[world, axis], value)
    peak_velocity[world, axis] = wp.max(
        peak_velocity[world, axis], wp.abs(qd[world * nv + axis])
    )
    peak_effort[world, axis] = wp.max(
        peak_effort[world, axis], wp.abs(torque[world, axis])
    )


class Stepper:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.flags = wp.zeros(env.num_envs, dtype=int, device=str(env.device))
        self.lower = wp.from_torch(env.lower)
        self.upper = wp.from_torch(env.upper)
        self.vmax = wp.from_torch(env.vmax)
        self.robot = wp.from_torch(env._shape_is_robot, dtype=wp.bool)
        self.pad = wp.from_torch(env._shape_is_pad, dtype=wp.bool)
        self.part_slots = wp.from_torch(env._shape_part_slot, dtype=wp.int32)
        self.task_ids = wp.from_torch(env.task_ids, dtype=wp.int64)
        self.penetration = wp.from_torch(env.penetration)
        self.pad_penetration = wp.from_torch(env.pad_penetration)
        self.rigid_penetration = wp.from_torch(env.rigid_penetration)
        self.pad_part_penetration = wp.from_torch(env.pad_part_penetration)
        self.commands = wp.from_torch(env.targets)
        self.jaw_servo = wp.from_torch(env.jaw_servo)
        self.arm_servo = wp.from_torch(env.arm_servo)
        self.arm_velocity = wp.from_torch(env.arm_servo_velocity)
        self.kp = wp.array(
            env.profile["actuator"]["kp"], dtype=float, device=str(env.device)
        )
        self.kd = wp.array(
            env.profile["actuator"]["kd"], dtype=float, device=str(env.device)
        )
        self.effort = wp.array(
            env.profile["actuator"]["effort_limit_nm"],
            dtype=float,
            device=str(env.device),
        )
        self.minimum = wp.zeros((env.num_envs, 8), dtype=float, device=str(env.device))
        self.maximum = wp.zeros_like(self.minimum)
        self.peak_velocity = wp.zeros_like(self.minimum)
        self.peak_effort = wp.zeros_like(self.minimum)
        self.target_stride = len(env.control.joint_target_q) // env.num_envs
        self.graph = None
        self.warmed = False
        if (
            env.substeps % 2
            or env.solver.update_data_interval != 1
            or env.solver.enable_sleeping
        ):
            raise ValueError(
                "Captured stepping requires even substeps, state sync every step, and no sleeping"
            )

    def _advance(self) -> None:
        env = self.env
        data = env.solver.mjw_data
        for _ in range(env.substeps):
            wp.launch(
                arm_targets,
                dim=(env.num_envs, 6),
                inputs=[
                    self.commands,
                    self.arm_servo,
                    self.arm_velocity,
                    env.state.joint_q,
                    env.control.joint_target_q,
                    self.vmax,
                    self.kp,
                    self.kd,
                    self.effort,
                    env.nq,
                    self.target_stride,
                    env.dt,
                    env.config["arm_servo_tau_s"],
                    env.config["arm_acceleration_rad_s2"],
                    env.config["relative_velocity_fraction"],
                    env.config["arm_spring_effort_fraction"],
                ],
                device=str(env.device),
            )
            wp.launch(
                jaw_targets,
                dim=(env.num_envs, 2),
                inputs=[
                    self.commands,
                    self.jaw_servo,
                    env.state.joint_q,
                    env.control.joint_target_q,
                    env.nq,
                    self.target_stride,
                    env.profile["jaw"]["velocity_limit_m_s"]
                    * env.dt
                    * env.config["gripper_velocity_fraction"],
                    1 - math.exp(-env.dt / 0.02),
                    env.profile["jaw"]["kd"]
                    * env.profile["jaw"]["velocity_limit_m_s"]
                    * env.config["gripper_velocity_fraction"]
                    / env.profile["jaw"]["kp"],
                    env.profile["jaw"]["closed_m"] + env.config["jaw_command_margin_m"],
                    env.profile["jaw"]["open_m"] - env.config["jaw_command_margin_m"],
                ],
                device=str(env.device),
            )
            env.solver.step(env.state, env.next_state, env.control, None, env.dt)
            env.state, env.next_state = env.next_state, env.state
            wp.launch(
                joint_extrema,
                dim=(env.num_envs, 8),
                inputs=[
                    env.state.joint_q,
                    env.state.joint_qd,
                    data.qfrc_actuator,
                    env.nq,
                    env.nv,
                    self.minimum,
                    self.maximum,
                    self.peak_velocity,
                    self.peak_effort,
                ],
                device=str(env.device),
            )
            wp.launch(
                joint_guards,
                dim=env.num_envs,
                inputs=[
                    env.state.joint_q,
                    env.state.joint_qd,
                    env.nq,
                    env.nv,
                    self.lower,
                    self.upper,
                    self.vmax,
                    data.nacon,
                    data.naconmax,
                    data.nefc,
                    data.njmax,
                    data.overflow,
                    self.flags,
                ],
                device=str(env.device),
            )
            env.solver.update_contacts(env.contacts, env.state)
            wp.launch(
                contact_guards,
                dim=data.naconmax,
                inputs=[
                    data.nacon,
                    env.contacts.rigid_contact_shape0,
                    env.contacts.rigid_contact_shape1,
                    env.contacts.force,
                    data.contact.dist,
                    data.contact.worldid,
                    self.robot,
                    self.pad,
                    self.part_slots,
                    self.task_ids,
                    env.max_parts,
                    env.ns,
                    env.num_envs,
                    self.flags,
                    self.penetration,
                    self.pad_penetration,
                    self.rigid_penetration,
                    self.pad_part_penetration,
                ],
                device=str(env.device),
            )

    def advance(self) -> None:
        self.flags.zero_()
        self.minimum.fill_(float("inf"))
        self.maximum.fill_(float("-inf"))
        self.peak_velocity.zero_()
        self.peak_effort.zero_()
        if (
            not self.warmed
            or self.env.device.type != "cuda"
            or not self.env.config["use_cuda_graph"]
        ):
            self._advance()
            self.warmed = True
            return
        if self.graph is None:
            counter = self.env.solver._step
            with wp.ScopedCapture(device=str(self.env.device)) as capture:
                self._advance()
            self.graph = capture.graph
            self.env.solver._step = counter
        wp.capture_launch(self.graph)
        self.env.solver._step += self.env.substeps
