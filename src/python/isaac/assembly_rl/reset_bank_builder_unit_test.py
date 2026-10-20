"""Geometry and acceptance invariants independent of a simulator installation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import transform as scipy_transform

from isaac.assembly_rl import reset_bank_builder as bank
from isaac.assembly_rl import reset_sampling


@pytest.mark.parametrize("rare_part", [False, True])
def test_wall_budget_saves_accepted_batch_without_marking_complete(
    tmp_path, monkeypatch, rare_part
):
    from isaac.fanuc_sim import native_model, robot_profile

    geometry, state = fixture()
    part_count = 2 if rare_part else 1
    if rare_part:
        original = geometry["parts"][0]
        geometry["parts"].append(
            reset_sampling.PartSpec("second", original.vertices_m, np.eye(3), False)
        )
        state["part_poses"] = np.repeat(state["part_poses"], 2, axis=1)
        state["part_velocities"] = np.repeat(state["part_velocities"], 2, axis=1)
    task = {
        "collision_geometry_ready": True,
        "parts": [{"mesh_cache": "part.npz", "fixed": False}] * part_count,
    }
    (tmp_path / "task.json").write_text(json.dumps(task))
    (tmp_path / "part.npz").write_bytes(b"source geometry identity")
    env = SimpleNamespace(
        tasks=[task],
        task_names=["task"],
        num_envs=1,
        max_parts=part_count,
        profile={},
        config={},
        physics_identity="unchanged",
    )
    now, calls = [0.0], []
    monkeypatch.setattr(bank.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(bank, "load_task_geometry", lambda *_: [geometry])
    monkeypatch.setattr(native_model, "gripper_shapes", lambda *_: ([], []))
    monkeypatch.setattr(
        robot_profile,
        "of",
        lambda *_: SimpleNamespace(gripper=SimpleNamespace(bodies=(0, 1, 2))),
    )
    monkeypatch.setattr(
        bank,
        "candidate",
        lambda _task, _geometry, family, *_args, **kwargs: {
            "active": kwargs["active_part"],
            "family": family,
        },
    )
    state.update(
        task_ids=np.zeros(1, dtype=int),
        goals=state["part_poses"].copy(),
        joint_targets=np.zeros((1, 8)),
        jaw_servo=np.zeros((1, 2)),
        arm_servo=np.zeros((1, 6)),
        arm_servo_velocity=np.zeros((1, 6)),
        gripper_open=np.ones(1),
    )

    def validate(_env, records, *_args, **_kwargs):
        active = records[0]["active"]
        calls.append(active)
        now[0] += 2.0 if rare_part else 11.0
        state["part_poses"][0, active, 0] = len(calls) * 0.01
        return {key: value.copy() for key, value in state.items()}, [
            "rare_part_rejected" if rare_part and active == 1 else None
        ]

    monkeypatch.setattr(bank, "physically_validate", validate)
    output = tmp_path / "bank"
    metadata = bank.build(
        env,
        tmp_path,
        output,
        4 if rare_part else 5,
        0,
        config=bank.ValidationConfig(),
        families=("partial_assembly",) if rare_part else bank.FAMILIES,
        max_wall_seconds=10,
    )
    assert calls == ([0, 0, 1, 1, 1] if rare_part else [0])
    assert metadata["sample_count"] == (2 if rare_part else 1)
    assert metadata["physics_validated"]
    assert metadata["wall_budget_exhausted"]
    assert (
        not metadata["training_ready"] and not metadata["complete_requested_coverage"]
    )
    assert "max_wall_seconds" not in metadata["validation"]
    assert metadata["physics_identity"] == "unchanged"
    with np.load(output / "bank.npz", allow_pickle=False) as archive:
        bank.validate_bank_entries(dict(archive), metadata)
    assert json.loads((output / "bank.json").read_text()) == metadata


def test_failed_batch_capture_preserves_inputs_and_prevalidation_rng(tmp_path):
    rng = np.random.default_rng(342)
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = rng.bit_generator.state
    values = np.array([[np.nextafter(1.0, 2.0), -0.0], [1e-200, 1e200]])
    record = {
        "part_poses": values.copy(),
        "active": np.int64(3),
        "held": np.bool_(False),
    }
    records = [record, record]
    error = RuntimeError("native solver failed")
    with (
        pytest.raises(RuntimeError) as caught,
        bank.capture_failed_batch(tmp_path, records, rng, {"batch": 4, "seed": 342}),
    ):
        rng.random(20)
        record["part_poses"][:] = 0
        raise error
    assert caught.value is error
    saved = json.loads((tmp_path / "failed_batch.json").read_text())
    assert saved["batch"] == 4 and saved["seed"] == 342
    assert len(saved["records"]) == 2
    for row in saved["records"]:
        np.testing.assert_array_equal(
            np.asarray(row["part_poses"]).view(np.uint64), values.view(np.uint64)
        )
        assert row["active"] == 3 and row["held"] is False
    restored = np.random.default_rng()
    restored.bit_generator.state = saved["rng_state"]
    np.testing.assert_array_equal(restored.random(20), expected_rng.random(20))
    assert saved["exception"]["type"] == "RuntimeError"
    assert list(tmp_path.iterdir()) == [tmp_path / "failed_batch.json"]


def test_capture_write_failure_preserves_original_exception(tmp_path):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied")
    error = RuntimeError("original physics error")
    with (
        pytest.raises(RuntimeError) as caught,
        bank.capture_failed_batch(blocked, [], np.random.default_rng(1), {}),
    ):
        raise error
    assert caught.value is error
    assert "Failed to save reset batch" in error.__notes__[0]
    with bank.capture_failed_batch(tmp_path, [], np.random.default_rng(1), {}):
        pass
    assert not (tmp_path / "failed_batch.json").exists()


def fixture():
    vertices = np.array(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (0.0, 0.02)]
    )
    part = reset_sampling.PartSpec("part", vertices, np.eye(3), False)
    geometry = {"parts": [part], "vertices": [vertices]}
    state = {
        "part_poses": np.array(
            [[[0.0, 0.0, bank.generate_bank.TABLE_HEIGHT_M, 0.0, 0.0, 0.0, 1.0]]]
        ),
        "joint_q": np.zeros((1, 8)),
        "joint_qd": np.zeros((1, 8)),
        "part_velocities": np.zeros((1, 1, 6)),
        "max_penetration_m": np.zeros(1),
        "safety_failure": np.zeros(1, dtype=bool),
    }
    return geometry, state


@pytest.mark.parametrize(
    ("task_id", "offset_range", "distance_to_end"),
    [("04_linear_guide", None, 0.05), ("10_dual_carriage", [0.038, 0.055], 0.025)],
)
def test_rail_near_mating_proposal_clears_rail_end(
    monkeypatch, task_id, offset_range, distance_to_end
):
    geometry, _ = fixture()
    part = geometry["parts"][0]
    fixed = reset_sampling.PartSpec("receiver", part.vertices_m, np.eye(3), True)
    geometry.update(
        parts=[fixed, part],
        vertices=[part.vertices_m, part.vertices_m],
        grasps=[[], [(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.02)]],
    )
    task = {
        "id": task_id,
        "parts": [
            {"fixed": True, "assembled_transform": np.eye(4).tolist()},
            {
                "fixed": False,
                "assembled_transform": np.eye(4).tolist(),
                "disassembly_axis_local": [0, 1, 0],
            },
        ],
    }
    if offset_range is not None:
        task["parts"][1]["near_mating_offset_range_m"] = offset_range
    low, high = offset_range or (0.065, 0.069)
    monkeypatch.setattr(bank, "solve_ik", lambda *_: np.zeros(6))
    monkeypatch.setattr(bank, "flange_from_joints", lambda *_: np.eye(4))
    for seed in range(8):
        record = bank.candidate(
            task,
            geometry,
            "near_mating",
            np.random.default_rng(seed),
            2,
            config=bank.ValidationConfig(),
        )
        goal = bank.pose_matrix(record["goals"][1])
        delta = record["part_poses"][1, :3] - goal[:3, 3]
        local = goal[:3, :3].T @ delta
        assert low <= local[1] <= high
        np.testing.assert_allclose(local[[0, 2]], 0, atol=1e-10)
        assert local[1] - 0.01145 > distance_to_end


@pytest.mark.parametrize(
    ("task_id", "family", "grasped_partial", "expected"),
    [
        ("04_linear_guide", "near_mating", False, [-0.01, 0, 0]),
        ("04_linear_guide", "partial_assembly", True, [-0.006, 0, 0]),
        ("04_linear_guide", "held_transport", False, [0, 0, 0.04]),
        ("03_shaft_coupling", "partial_assembly", True, [0, 0, 0.006]),
    ],
)
def test_grasp_challenge_respects_rail_axis_and_preserves_other_lifts(
    task_id, family, grasped_partial, expected
):
    pose = np.r_[
        [0.2, -0.1, 0.9],
        scipy_transform.Rotation.from_euler("z", 90, degrees=True).as_quat(),
    ]
    task = {"id": task_id, "parts": [{"disassembly_axis_local": [0, 1, 0]}]}
    record = {"family": family, "active": 0, "grasped_partial": grasped_partial}
    config = bank.ValidationConfig()
    offset = bank.grasp_challenge_displacement(task, record, pose, config)
    np.testing.assert_allclose(offset, expected, atol=1e-12)
    if grasped_partial:
        assert np.linalg.norm(offset) > config.retention_position_m
        assert (
            np.linalg.norm(offset) + config.partial_grasp_max_offset_m
            < config.partial_goal_max_distance_m
        )


def test_declared_horizontal_challenge_and_actual_motion_required():
    task = {"parts": [{"grasp_challenge_axis_local": [1, 0, 0]}]}
    record = {"family": "partial_assembly", "active": 0, "grasped_partial": True}
    pose = np.array([0, 0, 1, 0, 0, 0, 1.0])
    config = bank.ValidationConfig()
    offset = bank.grasp_challenge_displacement(task, record, pose, config)
    np.testing.assert_allclose(offset, [0.006, 0, 0])
    assert not bank.grasp_challenge_completed(pose, pose, offset, config)
    final = pose.copy()
    final[1] += 0.006
    assert not bank.grasp_challenge_completed(pose, final, offset, config)
    final[:3] = pose[:3] + offset
    assert bank.grasp_challenge_completed(pose, final, offset, config)


def test_assembly_dependencies_are_transitive_and_reject_invalid_graphs():
    task = {
        "parts": [
            {"name": "base"},
            {"name": "coupling", "requires_assembled": ["base"]},
            {"name": "shaft", "requires_assembled": ["coupling"]},
        ]
    }
    assert bank.assembly_dependencies(task) == [set(), {0}, {0, 1}]
    task["parts"][0]["requires_assembled"] = ["shaft"]
    with pytest.raises(ValueError, match="cycle"):
        bank.assembly_dependencies(task)
    task["parts"][0]["requires_assembled"] = ["missing"]
    with pytest.raises(ValueError, match="Unknown"):
        bank.assembly_dependencies(task)


def test_dependency_proposals_assemble_prerequisites_but_not_active_dependents(
    monkeypatch,
):
    geometry, _ = fixture()
    vertices = geometry["vertices"][0]
    names = ["base", "coupling", "shaft"]
    geometry = {
        "parts": [
            reset_sampling.PartSpec(name, vertices, np.eye(3), i == 0)
            for i, name in enumerate(names)
        ],
        "vertices": [vertices] * 3,
        "grasps": [[(np.zeros(3), np.array([1.0, 0, 0]), 0.02)]] * 3,
    }
    task = {
        "parts": [
            {
                "name": name,
                "fixed": i == 0,
                "assembled_transform": np.eye(4).tolist(),
                "disassembly_axis_local": [0, 0, 1],
            }
            for i, name in enumerate(names)
        ]
    }
    task["parts"][2]["requires_assembled"] = ["coupling"]
    monkeypatch.setattr(bank, "solve_ik", lambda *_: np.zeros(6))
    monkeypatch.setattr(bank, "flange_from_joints", lambda *_: np.eye(4))
    seen = set()
    for seed in range(12):
        record = bank.candidate(
            task,
            geometry,
            "near_mating",
            np.random.default_rng(seed),
            3,
            config=bank.ValidationConfig(),
        )
        seen.add(record["active"])
        if record["active"] == 2:
            np.testing.assert_allclose(record["part_poses"][1], record["goals"][1])
            assert record["required_assembled"] == [1]
        else:
            assert not np.allclose(record["part_poses"][2], record["goals"][2])
    assert seen == {1, 2}
    for active in (1, 2):
        record = bank.candidate(
            task,
            geometry,
            "near_mating",
            np.random.default_rng(0),
            3,
            config=bank.ValidationConfig(),
            active_part=active,
        )
        assert record["active"] == active
    with pytest.raises(ValueError, match="movable"):
        bank.candidate(
            task,
            geometry,
            "near_mating",
            np.random.default_rng(0),
            3,
            config=bank.ValidationConfig(),
            active_part=0,
        )


def test_settled_prerequisite_must_still_be_assembled():
    geometry, state = fixture()
    state["part_poses"] = np.repeat(state["part_poses"], 2, axis=1)
    state["goals"] = state["part_poses"].copy()
    geometry["vertices"] *= 2
    record = {"active": 1, "family": "near_mating", "required_assembled": [0]}
    assert (
        bank.family_rejection(state, 0, record, geometry, bank.ValidationConfig())
        is None
    )
    state["part_poses"][0, 0, 0] += 0.005
    assert (
        bank.family_rejection(state, 0, record, geometry, bank.ValidationConfig())
        == "prerequisite_not_assembled"
    )


def test_pose_roundtrip_and_quaternion_sign():
    pose = np.r_[
        [0.1, 0.2, 0.3],
        scipy_transform.Rotation.from_euler("xyz", [0.5, 1.0, -2.0]).as_quat(),
    ]
    np.testing.assert_allclose(
        bank.pose_matrix(bank.matrix_pose(bank.pose_matrix(pose))),
        bank.pose_matrix(pose),
    )
    assert bank.quaternion_distance(pose[3:], -pose[3:]) < 1e-7


def test_antipodal_pairs_and_flange_geometry():
    import trimesh

    mesh = trimesh.creation.box([0.02, 0.025, 0.03])
    contacts = bank.sample_antipodal_grasps(
        mesh.vertices, mesh.faces, np.random.default_rng(3)
    )
    assert len(contacts) >= 10
    world = np.eye(4)
    world[:3, :3] = scipy_transform.Rotation.from_euler(
        "xyz", [0.4, 0.8, 1.1]
    ).as_matrix()
    world[:3, 3] = [0.2, -0.1, 1.0]
    for contact in contacts:
        midpoint, axis, width = contact
        assert 0.002 < width < 0.047
        flange = bank.grasp_flange(world, contact)
        np.testing.assert_allclose(
            flange[:3, :3].T @ flange[:3, :3], np.eye(3), atol=1e-12
        )
        assert np.linalg.det(flange[:3, :3]) > 0.999
        np.testing.assert_allclose(flange[:3, 0], world[:3, :3] @ axis)
        np.testing.assert_allclose(
            flange[:3, 3] + flange[:3, 2] * bank.PAD_CENTER_Z_M,
            world[:3, :3] @ midpoint + world[:3, 3],
        )


def test_normal_ray_grasps_align_finite_pads_with_thin_opposing_faces():
    import trimesh

    mesh = trimesh.creation.box([0.09, 0.09, 0.0015])
    contacts = bank.sample_normal_grasps(
        mesh.vertices, mesh.faces, np.random.default_rng(3)
    )
    assert len(contacts) == 64
    for midpoint, axis, width in contacts:
        assert abs(axis[2]) == pytest.approx(1)
        assert width == pytest.approx(0.0015)
        assert midpoint[2] == pytest.approx(0, abs=1e-10)


def test_full_gripper_geometry_rejects_crossings_and_clears_narrow_opening():
    def box(low, high):
        points = np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        return points, bank.ConvexHull(points).equations

    height = bank.generate_bank.TABLE_HEIGHT_M
    receiver = box([-0.02, -0.02, height], [0.02, 0.02, height + 0.04])
    lid = box([-0.02, -0.02, height + 0.049], [0.02, 0.02, height + 0.051])
    geometry = {"collision_hulls": [[receiver], [lid]]}
    hulls = bank.transform_collision_hulls(geometry, np.tile(np.eye(4), (2, 1, 1)))
    pad = box([0, -0.01, -0.01], [0.00759, 0.01, 0.01])
    other = box([-0.00759, -0.01, -0.01], [0, 0.01, 0.01])
    solids = [(pad[0], pad[1], 7, 1), (other[0], other[1], 6, -1)]
    flange = np.eye(4)
    flange[:3, :3] = scipy_transform.Rotation.from_euler(
        "y", -90, degrees=True
    ).as_matrix()
    flange[:3, 3] = [0, 0, height + 0.05]
    assert not bank.gripper_collision_clear(
        flange, np.r_[np.zeros(6), 0.025, 0.025], hulls, solids, 0.0001
    )
    assert bank.gripper_collision_clear(
        flange, np.r_[np.zeros(6), 0.0015, 0.0015], hulls, solids, 0.0001
    )

    # Neither long crossing box contains a vertex of the other.
    horizontal = box([-0.02, -0.002, height + 0.1], [0.02, 0.002, height + 0.104])
    vertical = box([-0.002, -0.02, height + 0.1], [0.002, 0.02, height + 0.104])
    hulls = bank.transform_collision_hulls(
        {"collision_hulls": [[vertical]]}, np.eye(4)[None]
    )
    assert not bank.gripper_collision_clear(
        np.eye(4), np.zeros(8), hulls, [(horizontal[0], horizontal[1], None, 1)], 0.0001
    )


def test_acceptance_rejects_physics_faults_and_requires_history():
    geometry, state = fixture()
    config = bank.ValidationConfig()
    assert bank.rejection_reason(state, 0, geometry, config) == "needs_history"
    assert bank.rejection_reason(state, 0, geometry, config, state) is None
    previous = {k: v.copy() for k, v in state.items()}
    state["part_poses"][0, 0, 0] += 0.01
    assert bank.rejection_reason(state, 0, geometry, config, previous) == "moving"
    state["max_penetration_m"][0] = 0.0001
    assert bank.rejection_reason(state, 0, geometry, config, previous) == "penetration"
    state["safety_failure"][0] = True
    assert (
        bank.rejection_reason(state, 0, geometry, config, previous) == "safety_failure"
    )
    state["joint_q"][0, 0] = np.nan
    assert bank.rejection_reason(state, 0, geometry, config, previous) == "nonfinite"


def test_mesh_bounds_not_origin_determine_table_escape():
    geometry, state = fixture()
    state["part_poses"][0, 0, 0] = bank.generate_bank.TABLE_XY_BOUNDS_M[0, 1] - 0.001
    assert (
        bank.rejection_reason(state, 0, geometry, bank.ValidationConfig(), state)
        == "table_escape"
    )
    state["part_poses"][0, 0, 0] = 0
    state["part_poses"][0, 0, 2] -= 0.002
    assert (
        bank.rejection_reason(state, 0, geometry, bank.ValidationConfig(), state)
        == "below_table"
    )


def test_only_active_held_pad_contact_uses_cited_grasp_depth_allowance():
    geometry, state = fixture()
    config = bank.ValidationConfig()
    state["max_penetration_m"][0] = 0.000125
    # Missing pair diagnostics fail closed, including legacy snapshots.
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, held_part=0)
        == "penetration"
    )
    state["max_pad_part_penetration_m"] = np.array([[0.000125, 0.0]])
    state["max_rigid_penetration_m"] = np.zeros(1)
    assert bank.rejection_reason(state, 0, geometry, config, state, held_part=0) is None
    # Spawn and unheld callers do not supply an eligible held part.
    assert bank.rejection_reason(state, 0, geometry, config, state) == "penetration"
    # The strict channel includes pad-table, pad-receiver, and rigid contacts.
    state["max_rigid_penetration_m"][0] = 0.000125
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, held_part=0)
        == "penetration"
    )
    state["max_rigid_penetration_m"][0] = 0
    state["max_pad_part_penetration_m"][0, 1] = 0.000125
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, held_part=0)
        == "penetration"
    )
    state["max_pad_part_penetration_m"][0] = [0.000501, 0]
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, held_part=0)
        == "penetration"
    )
    state["max_pad_part_penetration_m"][0, 0] = np.nan
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, held_part=0)
        == "nonfinite"
    )


def test_retention_rejects_slip_and_rotation():
    before, after = np.eye(4), np.eye(4)
    config = bank.ValidationConfig()
    assert bank.retained(before, after, config)
    after[0, 3] = 0.01
    assert not bank.retained(before, after, config)
    after = np.eye(4)
    after[:3, :3] = scipy_transform.Rotation.from_euler("z", 0.1).as_matrix()
    assert not bank.retained(before, after, config)


def test_invalid_thresholds_fail():
    with pytest.raises(ValueError):
        bank.ValidationConfig(max_penetration_m=-0.1)
    with pytest.raises(ValueError):
        bank.ValidationConfig(max_settle_steps=1)


def test_partial_and_near_goal_labels_require_final_geometry():
    geometry, state = fixture()
    state["goals"] = state["part_poses"].copy()
    config = bank.ValidationConfig()
    partial = {"active": 0, "family": "partial_assembly"}
    assert (
        bank.family_rejection(state, 0, partial, geometry, config)
        == "partial_not_dislodged"
    )
    state["part_poses"][0, 0, 0] = 0.01
    assert bank.family_rejection(state, 0, partial, geometry, config) is None
    state["part_poses"][0, 0, 0] = 0.04
    assert (
        bank.family_rejection(state, 0, partial, geometry, config)
        == "partial_no_longer_near_goal"
    )
    state["part_poses"][0, 0, 0] = 0.1
    assert (
        bank.family_rejection(
            state, 0, {"active": 0, "family": "near_mating"}, geometry, config
        )
        == "no_longer_near_mating"
    )


def physical_entry():
    _, state = fixture()
    entry = {key: value[0].copy() for key, value in state.items()}
    entry.update(
        goals=entry["part_poses"].copy(),
        joint_targets=entry["joint_q"].copy(),
        jaw_servo=np.zeros(2),
        gripper_open=np.asarray(1.0),
        family="loose",
        task_ids=0,
    )
    return entry


def identified_bank(entries):
    ids = [bank.entry_content_id(entry, "motor", 1) for entry in entries]
    arrays = {key: np.stack([entry[key] for entry in entries]) for key in entries[0]}
    arrays["entry_ids"] = np.asarray(ids)
    return arrays, {"task_names": ["motor"], "task_part_counts": [1], "entry_ids": ids}


def test_content_identity_ignores_labels_and_quaternion_sign_but_preserves_commands():
    first = physical_entry()
    second = physical_entry()
    second["family"] = "held_transport"
    second["part_poses"][:, 3:] *= -1
    second["goals"][:, 3:] *= -1
    first_id = bank.entry_content_id(first, "motor", 1)
    assert bank.entry_content_id(second, "motor", 1) == first_id
    second["joint_targets"][0] += 0.01
    assert bank.entry_content_id(second, "motor", 1) != first_id
    second = physical_entry()
    second["part_poses"][0, 0] += 0.001
    assert bank.entry_content_id(second, "motor", 1) != first_id
    assert bank.entry_content_id(first, "other_task", 1) != first_id


def test_duplicate_registration_and_untrusted_bank_ids_are_rejected():
    first, duplicate = physical_entry(), physical_entry()
    duplicate["family"] = "near_grasp"
    seen = set()
    assert bank.register_entry(first, "motor", 1, seen)
    assert not bank.register_entry(duplicate, "motor", 1, seen)
    arrays, metadata = identified_bank([first])
    assert bank.validate_bank_entries(arrays, metadata) == metadata["entry_ids"]
    arrays["joint_targets"][0, 0] = 0.01
    with pytest.raises(ValueError, match="do not match"):
        bank.validate_bank_entries(arrays, metadata)
    arrays, metadata = identified_bank([physical_entry(), duplicate])
    with pytest.raises(ValueError, match="Duplicate physical"):
        bank.validate_bank_entries(arrays, metadata)
    del arrays["entry_ids"]
    with pytest.raises(ValueError, match="Incomplete"):
        bank.validate_bank_entries(arrays, metadata)


def test_held_out_banks_require_disjoint_physical_states():
    first = physical_entry()
    _, train = identified_bank([first])
    _, evaluation = identified_bank([first])
    evaluation["seed"] = 12345
    with pytest.raises(ValueError, match="overlap"):
        bank.assert_disjoint_bank_entries(train, evaluation)
    distinct = physical_entry()
    distinct["part_poses"][0, 1] = 0.1
    _, evaluation = identified_bank([distinct])
    bank.assert_disjoint_bank_entries(train, evaluation)
    with pytest.raises(ValueError, match="require verified"):
        bank.assert_disjoint_bank_entries(train, {})


def test_request_schedule_balances_tasks_and_families_even_in_small_batches():
    from collections import Counter

    schedule = bank.request_schedule(6, 12, bank.FAMILIES)
    assert len(schedule) == 72
    assert {task for task, _ in schedule[:6]} == set(range(6))
    for task in range(6):
        counts = Counter(family for index, family in schedule if index == task)
        assert set(counts) == set(bank.FAMILIES)
        assert max(counts.values()) - min(counts.values()) <= 1
    with pytest.raises(ValueError):
        bank.request_schedule(1, 4, bank.FAMILIES)
    with pytest.raises(ValueError):
        bank.request_schedule(1, 5, ("loose", "loose"))


def test_part_request_schedule_balances_each_family_and_rotates_remainders():
    from collections import Counter

    tasks = [
        {"parts": [{"fixed": True}] + [{"fixed": False}] * count} for count in (1, 2, 3)
    ]
    schedules = [
        bank.part_request_schedule(tasks, 35, bank.FAMILIES, seed) for seed in range(3)
    ]
    for schedule in schedules:
        assert len(schedule) == 105
        assert {task for task, _, _ in schedule[:3]} == {0, 1, 2}
        for task, count in enumerate((1, 2, 3)):
            for family in bank.FAMILIES:
                counts = Counter(
                    active
                    for index, kind, active in schedule
                    if index == task and kind == family
                )
                assert set(counts) == set(range(1, count + 1))
                assert sum(counts.values()) == 7
                assert max(counts.values()) - min(counts.values()) <= 1
    extras = []
    for schedule in schedules:
        counts = Counter(
            active
            for task, family, active in schedule
            if task == 2 and family == "loose"
        )
        extras.append(max(counts, key=counts.get))
    assert set(extras) == {1, 2, 3}


def test_soft_contact_transient_is_bounded_but_final_penetration_gate_stays_strict():
    geometry, state = fixture()
    config = bank.ValidationConfig()
    state["max_penetration_m"][0] = 0.000232
    state["part_poses"][0, 0, 2] -= 0.000232
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, settling=True) is None
    )
    assert bank.rejection_reason(state, 0, geometry, config, state) == "penetration"
    state["max_penetration_m"][0] = 0.0011
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, settling=True)
        == "penetration"
    )
    state["max_penetration_m"][0] = 0.000004
    state["part_poses"][0, 0, 2] += 0.000232
    assert bank.rejection_reason(state, 0, geometry, config, state) is None
    state["safety_failure"][0] = True
    assert (
        bank.rejection_reason(state, 0, geometry, config, state, settling=True)
        == "safety_failure"
    )


def test_native_gripper_prefilter_rejects_table_and_solid_vertex_overlap():
    points = np.asarray(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (0.1, 0.2)]
    )
    geometry = {
        "gripper_solids": [(points, bank.ConvexHull(points).equations, None, 1.0)],
        "vertices": [np.array([[0.0, 0.0, 0.15], [0.005, 0.005, 0.16]])],
    }
    flange = np.eye(4)
    flange[2, 3] = bank.generate_bank.TABLE_HEIGHT_M
    transforms = np.eye(4)[None]
    transforms[0, 2, 3] = bank.generate_bank.TABLE_HEIGHT_M
    assert not bank.gripper_spawn_clear(
        flange, np.zeros(8), transforms, geometry, 0.0001
    )
    transforms[0, 0, 3] = 0.1
    assert bank.gripper_spawn_clear(flange, np.zeros(8), transforms, geometry, 0.0001)
    flange[2, 3] -= 0.15
    assert not bank.gripper_spawn_clear(
        flange, np.zeros(8), transforms, geometry, 0.0001
    )


def test_gripper_close_ramp_respects_native_speed_and_endpoints():
    previous = bank.closing_openings(np.array([1.0, 0.3]), np.array([0.0, 0.3]), 0, 25)
    np.testing.assert_allclose(previous, [1.0, 0.3])
    speeds = []
    for step in range(1, 26):
        current = bank.closing_openings(
            np.array([1.0, 0.3]), np.array([0.0, 0.3]), step, 25
        )
        assert 0 <= current[0] <= previous[0]
        assert current[1] == 0.3
        speeds.append(abs(current[0] - previous[0]) * 0.025 * 25)
        previous = current
    np.testing.assert_allclose(previous, [0.0, 0.3])
    assert max(speeds) < 0.04


def test_shallow_grasp_uses_real_fingertip_contact_without_changing_contact_pose():
    contact = (np.array([0.01, 0.02, 0.005]), np.array([1.0, 0.0, 0.0]), 0.01)
    world = np.eye(4)
    middle = bank.grasp_flange(world, contact)
    tip_z = bank.PAD_CENTER_Z_M + 0.018
    shallow = bank.grasp_flange(world, contact, contact_z_m=tip_z)
    np.testing.assert_allclose(shallow[:3, :3], middle[:3, :3])
    np.testing.assert_allclose(shallow[:3, 3] + tip_z * shallow[:3, 2], contact[0])
    assert shallow[2, 3] > middle[2, 3]


def test_partial_grasp_variant_preserves_partial_distance_and_real_grasp_requirement():
    geometry, _ = fixture()
    part = geometry["parts"][0]
    fixed = reset_sampling.PartSpec("receiver", part.vertices_m, np.eye(3), True)
    geometry = {
        "parts": [fixed, part],
        "vertices": [part.vertices_m, part.vertices_m],
        "grasps": [[], [(np.array([0.0, 0.0, 0.01]), np.array([1.0, 0.0, 0.0]), 0.02)]],
    }
    goal = np.eye(4)
    goal[2, 3] = 0.02
    task = {
        "parts": [
            {
                "name": "receiver",
                "fixed": True,
                "assembled_transform": np.eye(4).tolist(),
            },
            {
                "name": "part",
                "fixed": False,
                "assembled_transform": goal.tolist(),
                "disassembly_axis_local": [0.0, 0.0, 1.0],
            },
        ]
    }
    config = bank.ValidationConfig(partial_grasp_probability=1.0)
    record = bank.candidate(
        task, geometry, "partial_assembly", np.random.default_rng(4), 2, config=config
    )
    assert bank.requires_grasp(record)
    delta = record["part_poses"][1, :3] - record["goals"][1, :3]
    assert (
        config.partial_grasp_min_offset_m
        <= np.linalg.norm(delta)
        <= config.partial_grasp_max_offset_m
    )
    np.testing.assert_allclose(delta[:2], 0, atol=1e-10)
    np.testing.assert_allclose(
        bank.flange_from_joints(record["joint_q"]), record["flange"], atol=2e-5
    )
    assert not bank.requires_grasp(
        {"family": "partial_assembly", "grasped_partial": False}
    )
    assert bank.requires_grasp({"family": "held_transport"})


def test_arm_servo_state_is_identified_and_mixed_identity_schemas_fail_closed():
    old = physical_entry()
    current = physical_entry()
    current.update(arm_servo=np.zeros(6), arm_servo_velocity=np.zeros(6))
    current_id = bank.entry_content_id(current, "motor", 1)
    current["arm_servo_velocity"][2] = 0.01
    assert bank.entry_content_id(current, "motor", 1) != current_id
    arrays, metadata = identified_bank([current])
    metadata["content_id_schema"] = 2
    assert bank.validate_bank_entries(arrays, metadata) == metadata["entry_ids"]
    _, legacy = identified_bank([old])
    with pytest.raises(ValueError, match="schema"):
        bank.assert_disjoint_bank_entries(legacy, metadata)
    del arrays["arm_servo_velocity"]
    with pytest.raises(ValueError, match="Incomplete"):
        bank.validate_bank_entries(arrays, metadata)


def test_contact_diagnostics_aggregate_hulls_but_preserve_worlds_and_pad_pairs():
    rows = bank.summarize_contacts(
        np.array([0, 2, 4, -1]),
        np.array([1, 0, 7, 0]),
        np.array(
            [
                [3, 4, 0, 99, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [9, 9, 9, 0, 0, 0],
            ]
        ),
        np.array([-0.00012, -0.00008, -0.000001, -1]),
        np.array([0, 0, 1, 0]),
        labels=["finger/pad", "part/hull_0", "part/hull_1", "table"],
        num_envs=2,
    )
    assert rows[0] == [
        {
            "pair": ("finger/pad", "part"),
            "count": 2,
            "depth_m": 0.00012,
            "sum_force_magnitudes_n": 7.0,
            "net_force_on_first_n": [3.0, 4.0, -2.0],
        }
    ]
    assert rows[1][0]["pair"] == ("finger/pad", "table")
    assert rows[1][0]["sum_force_magnitudes_n"] == 1.0


def test_partial_grasp_probability_endpoints_and_invalid_values():
    assert (
        bank.ValidationConfig(partial_grasp_probability=0).partial_grasp_probability
        == 0
    )
    assert (
        bank.ValidationConfig(partial_grasp_probability=1).partial_grasp_probability
        == 1
    )
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bank.ValidationConfig(partial_grasp_probability=value)


def threaded_part():
    return {
        "screw_pitch_m": 0.002,
        "disassembly_axis_local": [0, 0, 1],
        "screw_axis_origin_local_m": [0.01, -0.02, 0],
        "screw_disengaged_offset_m": 0.023,
    }


def test_thread_withdrawal_rotates_about_declared_axis_in_world_frame():
    from isaac.assembly_rl.assembly_motion import withdrawal_transform

    part = threaded_part()
    pose = np.eye(4)
    pose[:3, :3] = scipy_transform.Rotation.from_euler(
        "xyz", [0.3, -0.7, 0.4]
    ).as_matrix()
    pose[:3, 3] = [0.2, -0.1, 0.9]
    moved = withdrawal_transform(pose, part, 0.0005)
    pivot = np.r_[part["screw_axis_origin_local_m"], 1]
    np.testing.assert_allclose(
        moved @ pivot, pose @ pivot + np.r_[pose[:3, 2] * 0.0005, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        pose[:3, :3].T @ moved[:3, :3],
        scipy_transform.Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        withdrawal_transform(moved, part, -0.0005), pose, atol=1e-12
    )


def test_thread_challenge_requires_actual_rotation_axial_motion_and_pitch_coupling():
    from isaac.assembly_rl.assembly_motion import withdrawal_transform

    part = threaded_part()
    config = bank.ValidationConfig()
    before = bank.matrix_pose(np.eye(4))
    after = bank.matrix_pose(withdrawal_transform(np.eye(4), part, 0.00025))
    assert bank.screw_challenge_completed(before, after, part, config)
    assert not bank.screw_challenge_completed(before, before, part, config)
    translation_only = before.copy()
    translation_only[:3] = after[:3]
    assert not bank.screw_challenge_completed(before, translation_only, part, config)
    rotation_only = after.copy()
    rotation_only[:3] = before[:3]
    assert not bank.screw_challenge_completed(before, rotation_only, part, config)
    wrong_pitch = after.copy()
    wrong_pitch[2] += 0.00004
    assert not bank.screw_challenge_completed(before, wrong_pitch, part, config)
    low_rotation = bank.matrix_pose(withdrawal_transform(np.eye(4), part, 0.00019))
    low_rotation[2] = 0.00025
    assert not bank.screw_challenge_completed(before, low_rotation, part, config)


def test_disengaged_thread_near_mating_keeps_straight_challenge():
    part = threaded_part()
    part["screw_axis_origin_local_m"] = [0, 0, 0]
    goal = np.eye(4)
    record = {
        "family": "near_mating",
        "active": 0,
        "goals": bank.matrix_pose(goal)[None],
    }
    task = {"parts": [part]}
    pose = bank.matrix_pose(goal)
    pose[2] = 0.024
    assert not bank.requires_screw_challenge(task, record, pose)
    np.testing.assert_allclose(
        bank.grasp_challenge_displacement(task, record, pose, bank.ValidationConfig()),
        [0, 0, 0.006],
    )
    pose[2] = 0.022
    assert bank.requires_screw_challenge(task, record, pose)
    del part["screw_disengaged_offset_m"]
    pose[2] = 0.06
    assert bank.requires_screw_challenge(task, record, pose)
    record["family"] = "held_transport"
    assert not bank.requires_screw_challenge(task, record, pose)


def test_unheld_lid_partial_is_offset_above_rim_around_its_center(monkeypatch):
    geometry, _ = fixture()
    vertices = np.array(
        [[x, y, z] for x in (0, 0.09) for y in (0, 0.09) for z in (0, 0.006)]
    )
    names = ("body", "lid")
    geometry.update(
        parts=[
            reset_sampling.PartSpec(name, vertices, np.eye(3), i == 0)
            for i, name in enumerate(names)
        ],
        vertices=[vertices, vertices],
    )
    task = {
        "id": "05_enclosure",
        "parts": [
            {"name": name, "assembled_transform": np.eye(4).tolist()} for name in names
        ],
    }
    monkeypatch.setattr(bank, "flange_from_joints", lambda *_: np.eye(4))
    record = bank.candidate(
        task,
        geometry,
        "partial_assembly",
        np.random.default_rng(4),
        2,
        config=bank.ValidationConfig(partial_grasp_probability=0),
    )
    goal = bank.pose_matrix(record["goals"][1])
    actual = bank.pose_matrix(record["part_poses"][1])
    delta = np.linalg.inv(goal) @ actual
    center = vertices.mean(axis=0)
    shift = delta[:3, :3] @ center + delta[:3, 3] - center
    assert 0.004 <= np.linalg.norm(shift[:2]) <= 0.008
    assert 0.004 <= shift[2] <= 0.008
    angle = abs(np.arctan2(delta[1, 0], delta[0, 0]))
    assert np.radians(3) <= angle <= np.radians(12)
    displacement = vertices @ delta[:3, :3].T + delta[:3, 3] - vertices
    assert 0.001 < np.sqrt(np.mean(np.sum(displacement**2, axis=1))) < 0.025
    assert not record["grasped_partial"]


@pytest.mark.parametrize("grasped", [False, True])
def test_thread_partial_candidates_follow_helix_for_both_support_modes(
    monkeypatch, grasped
):
    from isaac.assembly_rl.assembly_motion import withdrawal_transform

    geometry, _ = fixture()
    vertices = geometry["vertices"][0]
    geometry.update(
        parts=[
            reset_sampling.PartSpec(name, vertices, np.eye(3), i == 0)
            for i, name in enumerate(("base", "nut"))
        ],
        vertices=[vertices, vertices],
        grasps=[[], [(np.zeros(3), np.array([1.0, 0, 0]), 0.02)]],
    )
    part = threaded_part() | {"name": "nut", "assembled_transform": np.eye(4).tolist()}
    task = {
        "parts": [{"name": "base", "assembled_transform": np.eye(4).tolist()}, part]
    }
    monkeypatch.setattr(bank, "solve_ik", lambda *_: np.zeros(6))
    monkeypatch.setattr(bank, "flange_from_joints", lambda *_: np.eye(4))
    monkeypatch.setattr(bank, "gripper_spawn_clear", lambda *_: True)
    config = bank.ValidationConfig(partial_grasp_probability=float(grasped))
    record = bank.candidate(
        task, geometry, "partial_assembly", np.random.default_rng(4), 2, config=config
    )
    goal = bank.pose_matrix(record["goals"][1])
    actual = bank.pose_matrix(record["part_poses"][1])
    distance = np.dot(actual[:3, 3] - goal[:3, 3], goal[:3, 2])
    assert 0.003 <= distance <= 0.015
    np.testing.assert_allclose(
        actual, withdrawal_transform(goal, part, distance), atol=1e-10
    )
    assert record["grasped_partial"] == grasped


def test_thread_validation_commands_every_helical_waypoint_with_preserved_grasp(
    monkeypatch,
):
    from isaac.assembly_rl.assembly_motion import withdrawal_transform

    part = threaded_part() | {"mass_kg": 0.02, "screw_axis_origin_local_m": [0, 0, 0]}
    geometry, state = fixture()
    goal = np.eye(4)
    goal[2, 3] = 1.0
    initial = withdrawal_transform(goal, part, 0.01)
    flange_in_part = np.eye(4)
    flange_in_part[:3, :3] = scipy_transform.Rotation.from_euler("y", 0.4).as_matrix()
    flange_in_part[0, 3] = 0.02
    waypoints = []

    def inverse_kinematics(pose, *_):
        waypoints.append(pose.copy())
        return np.r_[
            pose[:3, 3], scipy_transform.Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        ]

    def forward_kinematics(joints):
        pose = np.eye(4)
        pose[:3, 3] = joints[:3]
        pose[:3, :3] = scipy_transform.Rotation.from_rotvec(joints[3:6]).as_matrix()
        return pose

    q = np.r_[inverse_kinematics(initial @ flange_in_part), 0.025, 0.025]
    waypoints.clear()
    record = {
        "task_id": 0,
        "active": 0,
        "family": "partial_assembly",
        "grasped_partial": True,
        "part_poses": bank.matrix_pose(initial)[None],
        "goals": bank.matrix_pose(goal)[None],
        "joint_q": q,
        "gripper_open": 1.0,
    }

    class FollowingEnvironment:
        device = "cpu"
        max_parts = 1

        def __init__(self):
            self.tasks = [{"parts": [part]}]
            self.task_names = ["thread_test"]

        def initialize(self, _ids, poses, joints, goals):
            state.update(
                part_poses=poses.numpy().copy(),
                joint_q=joints.numpy().copy(),
                goals=goals.numpy().copy(),
            )

        def hold(self, joints, _openings, *, steps):
            state["joint_q"][:, :6] = joints.numpy()
            state["part_poses"][0, 0] = bank.matrix_pose(
                forward_kinematics(state["joint_q"][0]) @ np.linalg.inv(flange_in_part)
            )

        def perturb_objects(self, _forces, *, steps):
            pass

        def snapshot(self):
            return state

    monkeypatch.setattr(bank, "solve_ik", inverse_kinematics)
    monkeypatch.setattr(bank, "flange_from_joints", forward_kinematics)
    config = bank.ValidationConfig(close_steps=1, perturb_steps=1, max_settle_steps=5)
    diagnostics = []
    _, reasons = bank.physically_validate(
        FollowingEnvironment(),
        [record],
        [geometry],
        config,
        np.random.default_rng(0),
        diagnostics=diagnostics,
    )
    assert reasons == [None]
    assert len(waypoints) == 24
    for pose, alpha in zip(waypoints, np.linspace(0, 1, 25)[1:], strict=True):
        np.testing.assert_allclose(
            pose,
            withdrawal_transform(initial, part, 0.00025 * alpha) @ flange_in_part,
            atol=1e-7,
        )
    assert diagnostics[0]["grasp_challenge"]["protocol"] == "helical_waypoints_45deg"
    assert not diagnostics[0]["grasp_challenge"]["thread_contact_coupling_claimed"]
