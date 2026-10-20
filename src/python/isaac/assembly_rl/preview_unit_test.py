import pathlib

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from isaac.assembly_rl import preview


@pytest.mark.parametrize("package", [False, True])
def test_export_preserves_actual_joint_and_part_transforms(tmp_path, package):
    scene = pathlib.Path("data/workshop.usdc")
    assets = pathlib.Path("data/training_assets")
    if not scene.exists():
        pytest.skip("Approved local workshop unavailable")
    pose = np.array([0.13, 0.09, 0.92, 0, 0, 0, 1])
    joints = np.r_[
        np.radians([59.251, -100.867, -91.991, -77.142, 90, -63.049]), 0.012, 0.019
    ]
    file = tmp_path / "state.npz"
    np.savez(
        file,
        task_names=np.array(["02_motor_mount"]),
        task_ids=np.array([0]),
        joint_q=joints[None],
        part_poses=np.tile(pose, (1, 4, 1)),
        physics_source="Newton AssemblyEnvironment.snapshot",
    )
    output = tmp_path / "state.usda"
    exported = preview.export_snapshot(file, scene, assets, output, package=package)
    assert exported.suffix == (".usdz" if package else ".usda")
    stage = Usd.Stage.Open(str(exported))
    root = "/TrainingPreview/Parts/MountBracket"
    matrix = np.array(
        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root))
    ).T
    np.testing.assert_allclose(matrix[:3, 3], pose[:3])
    right = stage.GetPrimAtPath("/vention_cell/Instances/hande_finger_right")
    left = stage.GetPrimAtPath("/vention_cell/Instances/hande_finger_left")
    body = stage.GetPrimAtPath("/vention_cell/Instances/hande_body")
    cache = UsdGeom.XformCache()
    inv = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)
    right_pose = inv @ np.array(cache.GetLocalToWorldTransform(right)).T
    left_pose = inv @ np.array(cache.GetLocalToWorldTransform(left)).T
    assert right_pose[0, 3] == pytest.approx(-0.012)
    assert left_pose[0, 3] == pytest.approx(0.019)
    assert stage.GetRootLayer().customLayerData["physics_snapshot_sha256"]
    old_parts = stage.GetPrimAtPath("/Workshop/Assembly")
    assert not old_parts or not old_parts.IsActive()


def test_export_rejects_invalid_quaternions_before_creating_scene(tmp_path):
    file = tmp_path / "state.npz"
    np.savez(
        file,
        task_names=np.array(["02_motor_mount"]),
        task_ids=np.array([0]),
        joint_q=np.zeros((1, 8)),
        part_poses=np.zeros((1, 4, 7)),
        physics_source="Newton AssemblyEnvironment.snapshot",
    )
    with pytest.raises(ValueError, match="quaternion"):
        preview.read_snapshot(file, 0)
