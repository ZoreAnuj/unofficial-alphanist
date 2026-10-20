import numpy as np
import warp as wp

from isaac.assembly_rl import physics_step


def test_gripper_servo_limits_transient_command_speed() -> None:
    commands = wp.array([[0.0] * 8], dtype=float, device="cpu")
    servo = wp.array([[0.025, 0.025]], dtype=float, device="cpu")
    target = wp.zeros(8, dtype=float, device="cpu")
    wp.launch(
        physics_step.jaw_targets,
        dim=(1, 2),
        inputs=[
            commands,
            servo,
            wp.array([0.0] * 6 + [0.025, 0.025], dtype=float, device="cpu"),
            target,
            8,
            8,
            0.1 / 750,
            0.05,
            0.0013333333,
            0.0001,
            0.0249,
        ],
        device="cpu",
    )
    np.testing.assert_allclose(servo.numpy(), [[0.025 - 0.1 / 750] * 2], atol=1e-8)
    np.testing.assert_allclose(target.numpy()[6:], servo.numpy()[0])


def test_guards_latch_transient_violation_and_detect_capacity() -> None:
    wp.init()
    q = np.zeros((2, 8), dtype=np.float32)
    q[0, 0] = 2
    flags = wp.zeros(2, dtype=int, device="cpu")

    def run(values, capacity):
        wp.launch(
            physics_step.joint_guards,
            dim=2,
            inputs=[
                wp.array(values.ravel(), dtype=float, device="cpu"),
                wp.zeros(16, dtype=float, device="cpu"),
                8,
                8,
                wp.array([-1.0] * 8, dtype=float, device="cpu"),
                wp.array([1.0] * 8, dtype=float, device="cpu"),
                wp.array([1.0] * 8, dtype=float, device="cpu"),
                wp.array([capacity], dtype=int, device="cpu"),
                4,
                wp.zeros(2, dtype=int, device="cpu"),
                8,
                wp.zeros(2, dtype=int, device="cpu"),
                flags,
            ],
            device="cpu",
        )

    run(q, 0)
    run(q * 0, 0)
    assert flags.numpy().tolist() == [4, 0]
    run(q * 0, 5)
    assert flags.numpy().tolist() == [6, 2]


def test_arm_reference_acceleration_and_effort_budget():
    def array(value):
        return wp.array(value, dtype=float, device="cpu")

    commands = array([[1.0] * 6 + [0.0, 0.0]])
    servo, velocity = array([[0.0] * 6]), array([[0.0] * 6])
    q, target = array([0.0] * 8), array([0.0] * 8)
    kp, kd, effort = array([1000.0] * 6), array([8.0] * 6), array([28.0] * 6)
    previous = np.zeros(6)
    for _ in range(80):
        wp.launch(
            physics_step.arm_targets,
            dim=(1, 6),
            inputs=[
                commands,
                servo,
                velocity,
                q,
                target,
                array([np.pi] * 8),
                kp,
                kd,
                effort,
                8,
                8,
                1 / 750,
                0.02,
                2 * np.pi,
                0.5,
                0.5,
            ],
            device="cpu",
        )
        current = velocity.numpy()[0]
        assert np.max(np.abs(current - previous)) <= 2 * np.pi / 750 + 1e-6
        assert np.max(np.abs(current)) <= 0.5 * np.pi
        # Simulated blocked joint: no arbitrary position-spring accumulation.
        assert np.max(np.abs(target.numpy()[:6])) <= 0.014 + 1e-7
        previous = current


def test_blocked_jaw_release_preserves_velocity_and_joint_limit():
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco>
    <option timestep="0.001333333333" integrator="implicitfast" gravity="0 0 0"/>
    <worldbody><body><joint name="jaw" type="slide" axis="1 0 0"
      range="0 .025" armature=".05"/>
      <geom type="box" size=".01 .01 .01" mass=".02955"/></body></worldbody>
    <actuator><position joint="jaw" kp="12000" kv="400" forcerange="-20 20"/>
    </actuator></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[0] = 0.01
    commands = wp.zeros((1, 8), dtype=float, device="cpu")
    servo = wp.zeros((1, 2), dtype=float, device="cpu")
    target = wp.zeros(8, dtype=float, device="cpu")
    q = wp.zeros(8, dtype=float, device="cpu")
    min_q, max_speed = 1.0, 0.0
    for _ in range(400):
        q.assign(np.asarray([0.0] * 6 + [data.qpos[0]] * 2, dtype=np.float32))
        wp.launch(
            physics_step.jaw_targets,
            dim=(1, 2),
            inputs=[
                commands,
                servo,
                q,
                target,
                8,
                8,
                0.04 / 750,
                0.05,
                400 * 0.04 / 12000,
                0.0001,
                0.0249,
            ],
            device="cpu",
        )
        data.ctrl[0] = target.numpy()[6]
        mujoco.mj_step(model, data)
        min_q = min(min_q, data.qpos[0])
        max_speed = max(max_speed, abs(data.qvel[0]))
    assert min_q >= -1e-5
    assert max_speed < 0.041
    assert 0.00009 < data.qpos[0] < 0.00011


def test_pad_penetration_diagnostics_preserve_global_contact_maximum():
    def floats(values):
        return wp.array(values, dtype=float, device="cpu")

    def ints(values):
        return wp.array(values, dtype=int, device="cpu")

    global_depth, pad_depth, rigid_depth = [floats([0.0]) for _ in range(3)]
    flags = ints([0])
    part_depth = wp.zeros((1, 4), dtype=float, device="cpu")
    wp.launch(
        physics_step.contact_guards,
        dim=5,
        inputs=[
            ints([5]),
            ints([0, 0, 0, 1, 0]),
            ints([1, 2, 3, 2, 5]),
            wp.zeros(5, dtype=wp.spatial_vector, device="cpu"),
            floats([-0.00008, -0.00012, -0.00015, -0.00003, -0.0002]),
            ints([0] * 5),
            wp.array([False] * 6, dtype=wp.bool, device="cpu"),
            wp.array(
                [True, False, False, False, False, False], dtype=wp.bool, device="cpu"
            ),
            ints([-1, 1, 2, -1, -1, 5]),
            wp.array([0], dtype=wp.int64, device="cpu"),
            4,
            6,
            1,
            flags,
            global_depth,
            pad_depth,
            rigid_depth,
            part_depth,
        ],
        device="cpu",
    )
    np.testing.assert_allclose(global_depth.numpy(), [0.0002])
    np.testing.assert_allclose(pad_depth.numpy(), [0.0002])
    # Table, receiver and inactive-task contacts never get the pad exception.
    np.testing.assert_allclose(rigid_depth.numpy(), [0.0002])
    np.testing.assert_allclose(part_depth.numpy(), [[0, 0.00008, 0.00012, 0]])
    assert flags.numpy().tolist() == [0]
