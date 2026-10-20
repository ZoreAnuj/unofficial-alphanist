import pathlib

import numpy as np
import pytest

from isaac.assembly_rl import model


def test_unvalidated_collision_assets_cannot_enter_training() -> None:
    root = pathlib.Path("data/training_assets")
    if not root.exists():
        pytest.skip("Local CAD cache unavailable")
    with pytest.raises(ValueError, match="precision"):
        model.load_tasks(root, ["02_motor_mount"])


def test_robot_template_retains_scene_mount_and_implicit_drives() -> None:
    root = pathlib.Path("data/training_assets")
    if not root.exists():
        pytest.skip("Local CAD cache unavailable")
    tasks = model.load_tasks(root, ["02_motor_mount"], allow_unvalidated=True)
    builder, maps, profile = model.build_template(root, tasks)
    import newton

    np.testing.assert_allclose(
        np.array(maps["base_mount"])[:3, 3], [-0.199, -0.4186, 0.8749]
    )
    assert builder.joint_target_mode[:8] == [int(newton.JointTargetMode.POSITION)] * 8
    assert len(maps["task_parts"]) == 1
    receiver = maps["task_parts"][0][0]
    assert receiver["fixed"]
    assert builder.body_flags[receiver["body"]] & int(newton.BodyFlags.KINEMATIC)
    assert profile["robot"] == "ur7e"
    assert profile["jaw"]["kd"] == 400
    assert builder.joint_target_kd[6:8] == [400.0, 400.0]
    assert builder.joint_effort_limit[6:8] == [20.0, 20.0]
    assert builder.body_mass[maps["bodies"]["wrist_camera"]] == 0.0
    assert builder.custom_attributes["mujoco:jnt_actgravcomp"].values == dict.fromkeys(
        range(8), True
    )


def test_replicated_cpu_model_keeps_receiver_mocap_and_dynamic_free_joint() -> None:
    root = pathlib.Path("data/training_assets")
    if not root.exists():
        pytest.skip("Local CAD cache unavailable")
    tasks = model.load_tasks(root, ["02_motor_mount"], allow_unvalidated=True)
    native, maps, _ = model.build_model(
        root, tasks, world_count=2, device="cpu", use_mujoco_contacts=True
    )
    assert native.body_count == 2 * maps["body_count_per_world"]
    assert native.joint_coord_count == 2 * maps["joint_coord_count_per_world"]
    fixed, moving = maps["task_parts"][0]
    assert fixed["q_start"] is None
    assert moving["q_start"] == 8
    assert moving["qd_start"] == 8
    assert moving["point_cloud_local"].shape == (128, 3)
    assert len(maps["robot_point_clouds"]) == 10
    q = native.joint_q.numpy().reshape(2, -1)
    np.testing.assert_allclose(q[0], q[1])
    assert np.isfinite(q).all()


def test_native_contact_model_preserves_filters_without_pair_enumeration(
    monkeypatch,
) -> None:
    import newton

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    for index in range(3):
        body = template.add_link(label=f"body_{index}")
        joint = template.add_joint_free(child=body)
        template.add_articulation([joint])
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    template.shape_collision_filter_pairs.append((0, 2))
    mappings = {"shape_labels": ["a", "b", "c"], "shape_roles": ["part"] * 3}
    monkeypatch.setattr(model, "build_template", lambda *_: (template, mappings, {}))
    baseline, expected_maps = model.native_model.finalize_model(
        template, mappings, world_count=2, device="cpu"
    )
    actual, actual_maps, _ = model.build_model(
        pathlib.Path("."), [], world_count=2, device="cpu", use_mujoco_contacts=True
    )
    assert baseline.shape_contact_pairs is not None
    assert actual.shape_contact_pairs is None
    assert actual_maps.pop("compact_same_body_filters_per_world") == 0
    assert actual_maps.pop("explicit_collision_filters_per_world") == 1
    assert actual_maps == expected_maps
    pairs = np.array([(a, b) for a in range(6) for b in range(a + 1, 6)])
    np.testing.assert_array_equal(
        actual.shape_collision_filter_mask(pairs),
        baseline.shape_collision_filter_mask(pairs),
    )
    for name in (
        "shape_flags",
        "shape_world",
        "shape_collision_group",
        "joint_q",
        "body_q",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name).numpy(), getattr(baseline, name).numpy()
        )
    for native in (baseline, actual):
        filtered = native.shape_collision_filter_mask(np.array([[0, 2], [3, 5]]))
        assert filtered.all()
        assert not native.shape_collision_filter_mask(np.array([[0, 1], [3, 4]])).any()


def test_native_contact_model_rejects_generic_collision_path() -> None:
    with pytest.raises(ValueError, match="MuJoCo native contacts"):
        model.build_model(
            pathlib.Path("."), [], world_count=1, use_mujoco_contacts=False
        )


def test_compact_replicated_filters_preserve_exact_queries_and_native_graph(
    monkeypatch,
) -> None:
    import newton
    import warp as wp

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    for index in range(2):
        body = template.add_link(label=f"body_{index}")
        joint = template.add_joint_free(child=body)
        template.add_articulation([joint])
        for shape in range(4):
            cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=shape != 3)
            template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    # Preserve an intentionally absent same-body filter, a self exclusion,
    # explicit disabled-shape exclusion and a cross-body exclusion exactly.
    template.shape_collision_filter_pairs.remove((0, 1))
    template.shape_collision_filter_pairs.extend([(0, 0), (0, 3), (0, 4)])
    mappings = {"shape_labels": list(map(str, range(8))), "shape_roles": ["part"] * 8}
    monkeypatch.setattr(model, "build_template", lambda *_: (template, mappings, {}))
    baseline, _ = model.native_model.finalize_model(
        template, mappings, world_count=2, device="cpu"
    )
    actual, maps, _ = model.build_model(
        pathlib.Path("."), [], world_count=2, device="cpu", use_mujoco_contacts=True
    )
    assert maps["compact_same_body_filters_per_world"] == 6
    assert maps["explicit_collision_filters_per_world"] == 2
    assert len(actual.shape_collision_filter_pairs_array()) == 4
    pairs = np.array([(a, b) for a in range(-1, 18) for b in range(-1, 18)])
    np.testing.assert_array_equal(
        actual.shape_collision_filter_mask(pairs),
        baseline.shape_collision_filter_mask(pairs),
    )
    for a, b in pairs:
        assert actual.shape_collision_filter_contains(
            a, b
        ) == baseline.shape_collision_filter_contains(a, b)
    for native in (actual, baseline):
        with pytest.raises(TypeError):
            native.shape_collision_filter_mask(np.array([[0.0, 1.0]]))
        with pytest.raises(OverflowError):
            native.shape_collision_filter_mask(np.array([[0, 2**63]], dtype=np.uint64))
    solver = newton.solvers.SolverMuJoCo
    selected = np.array([0, 1, 2, 4, 5, 6], dtype=np.int32)
    before = solver._compile_newton_collision_masks(baseline, selected)
    after = solver._compile_newton_collision_masks(actual, selected)
    assert before.exact == after.exact
    np.testing.assert_array_equal(before.collision_type, after.collision_type)
    np.testing.assert_array_equal(before.collision_affinity, after.collision_affinity)
    with wp.ScopedDevice("cpu"):
        np.testing.assert_array_equal(
            solver._color_collision_shapes(actual, selected),
            solver._color_collision_shapes(baseline, selected),
        )


def test_collision_gate_is_independent_of_training_gate(tmp_path) -> None:
    import json
    import shutil

    root = pathlib.Path("data/training_assets")
    if not root.exists():
        pytest.skip("Local CAD cache unavailable")
    task = json.loads((root / "02_motor_mount.json").read_text())
    task["collision_geometry_ready"] = True
    task["precision_training_ready"] = False
    for part in task["parts"]:
        shutil.copyfile(root / part["mesh_cache"], tmp_path / part["mesh_cache"])
    (tmp_path / "02_motor_mount.json").write_text(json.dumps(task))
    assert model.load_tasks(tmp_path, ["02_motor_mount"])[0]["collision_geometry_ready"]


def test_skinny_cad_facets_remain_closed_at_newton_precision() -> None:
    import trimesh

    class CaptureBuilder:
        def add_shape_convex_hull(self, body, **kwargs):
            self.mesh = kwargs["mesh"]
            return 0

    # Source rail facet regression: near-collinear vertices previously caused
    # default hull cleanup to remove triangles and open the collision surface.
    points = np.array(
        [
            [0.0008796622, -0.0458162099, 0.00235],
            [0.0008796622, -0.0458162099, 0.0025],
            [0.0010392305, -0.0456000015, 0.00235],
            [0.0010392305, -0.0456000015, 0.0025],
            [0.003387500072, -0.049908219000000004, 0.0025],
            [0.003387500072, -0.049897070344, 0.0025],
            [0.003387500072, -0.049897070285, 0.0025],
            [0.003389864312, -0.049912076702, 0.002496847681],
            [0.003390014196, -0.04990167084899999, 0.002496647835],
            [0.003443749794, -0.050000001, 0.00235],
            [0.003443749794, -0.050000001, 0.002425000389],
            [0.00344374981, -0.050000001, 0.002425000367],
            [0.003443749827, -0.050000001, 0.00235],
            [0.003443749827, -0.050000001, 0.002425000345],
        ]
    )
    builder = CaptureBuilder()
    mappings = {"shape_labels": [], "shape_roles": []}
    model.add_assembly_convex_shape(
        builder,
        mappings,
        {"contact": {"condim": 3}},
        body=0,
        label="rail/test",
        points=points,
        role="receiver",
    )
    mesh = trimesh.Trimesh(
        np.asarray(builder.mesh.vertices),
        np.asarray(builder.mesh.indices).reshape(-1, 3),
        process=False,
    )
    assert mesh.is_volume
    assert mappings["shape_roles"] == ["receiver"]
    assert mappings["shape_labels"] == ["rail/test"]


@pytest.mark.parametrize(("seed", "world"), [(0, 0), (1, 0), (2, 1), (3, 1)])
def test_sparse_native_masks_preserve_random_body_and_partial_filters(
    monkeypatch, seed, world
):
    import newton

    rng = np.random.default_rng(seed)
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    for i in range(5):
        body = template.add_body(label=f"body_{i}")
        for _ in range(int(rng.integers(1, 4))):
            template.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
    bodies = np.asarray(template.shape_body)
    for a in range(template.shape_count):
        for b in range(a + 1, template.shape_count):
            if bodies[a] != bodies[b] and (
                rng.random() < 0.2 or {bodies[a], bodies[b]} == {0, 1}
            ):
                template.add_shape_collision_filter_pair(a, b)
    maps = {
        "shape_labels": list(template.shape_label),
        "shape_roles": ["part"] * template.shape_count,
    }
    monkeypatch.setattr(model, "build_template", lambda *_: (template, maps, {}))
    native, _, _ = model.build_model(
        pathlib.Path("."), [], world_count=2, device="cpu", use_mujoco_contacts=True
    )
    selected = np.arange(template.shape_count) + world * template.shape_count
    bodies = native.shape_body.numpy()[selected]
    body_ids = np.unique(bodies)
    old_excludes = newton.solvers.SolverMuJoCo._find_body_collision_filter_pairs(
        native, body_ids, selected
    )
    new_excludes = model.AssemblySolverMuJoCo._find_body_collision_filter_pairs(
        native, body_ids, selected
    )
    assert new_excludes == old_excludes
    compiled = model.AssemblySolverMuJoCo._compile_newton_collision_masks(
        native, selected
    )
    assert compiled.exact
    a, b = np.triu_indices(len(selected), 1)
    allowed = ~native.shape_collision_filter_mask(
        np.column_stack((selected[a], selected[b]))
    ) & (bodies[a] != bodies[b])
    actual = ((compiled.collision_type[a] & compiled.collision_affinity[b]) != 0) | (
        (compiled.collision_type[b] & compiled.collision_affinity[a]) != 0
    )
    actual &= bodies[a] != bodies[b]
    actual &= np.array(
        [
            tuple(sorted((bodies[x], bodies[y]))) not in new_excludes
            for x, y in zip(a, b, strict=True)
        ]
    )
    np.testing.assert_array_equal(actual, allowed)


def test_native_solver_keeps_world_shape_exclusion_and_signed_universal_bit(
    monkeypatch,
):
    import newton

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(template)
    for i in range(2):
        body = template.add_body(label=f"body_{i}")
        template.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01, label=f"shape_{i}")
    table = template.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.01, label="table")
    template.add_shape_collision_filter_pair(0, table)
    maps = {
        "shape_labels": list(template.shape_label),
        "shape_roles": ["part"] * template.shape_count,
    }
    monkeypatch.setattr(model, "build_template", lambda *_: (template, maps, {}))
    native, _, _ = model.build_model(
        pathlib.Path("."), [], world_count=1, device="cpu", use_mujoco_contacts=True
    )
    compiled = model.AssemblySolverMuJoCo._compile_newton_collision_masks(
        native, np.arange(3)
    )
    allowed = (
        compiled.collision_type[:, None] & compiled.collision_affinity[None, :]
    ) != 0
    allowed |= allowed.T
    assert not allowed[0, table]
    assert allowed[0, 1] and allowed[1, table]
    solver = model.AssemblySolverMuJoCo(
        native, use_mujoco_cpu=True, use_mujoco_contacts=True, nconmax=16, njmax=32
    )
    assert np.any(solver.mj_model.geom_contype < 0)
