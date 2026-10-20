import torch

from isaac.assembly_rl import geometry


def test_pose_metric_rejects_rotation_at_matching_origins() -> None:
    points = torch.tensor([[[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]]])
    goal = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    current = goal.clone()
    current[:, 5:] = torch.tensor([1.0, 0.0])
    assert geometry.pose_error(current, goal, points).item() > 0.07
    current = goal.clone()
    current[:, 0] += 0.0005
    torch.testing.assert_close(
        geometry.pose_error(current, goal, points), torch.tensor([0.0005])
    )


def test_inverse_pose_and_quaternion_sign_are_consistent() -> None:
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.70710678, 0.70710678]])
    points = torch.tensor([[[0.2, -0.4, 0.6], [0.0, 1.0, 0.0]]])
    world = geometry.transform_points(pose, points)
    torch.testing.assert_close(
        geometry.transform_points(geometry.inverse_pose(pose), world), points
    )
    opposite = pose.clone()
    opposite[:, 3:] *= -1
    torch.testing.assert_close(geometry.transform_points(opposite, points), world)
