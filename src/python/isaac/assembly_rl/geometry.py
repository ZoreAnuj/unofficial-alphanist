"""Batched rigid transforms in metres and Newton XYZW quaternion convention."""

import torch


def rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    xyz, vector = torch.broadcast_tensors(quaternion[..., :3], vector)
    twice = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., 3:] * twice + torch.cross(xyz, twice, dim=-1)


def transform_points(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return rotate(pose[..., None, 3:], points) + pose[..., None, :3]


def inverse_pose(pose: torch.Tensor) -> torch.Tensor:
    inverse = torch.cat((-pose[..., 3:6], pose[..., 6:7]), dim=-1)
    return torch.cat((rotate(inverse, -pose[..., :3]), inverse), dim=-1)


def compose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    a, b = first[..., 3:], second[..., 3:]
    vector = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
    vector = vector + torch.cross(a[..., :3], b[..., :3], dim=-1)
    scalar = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat(
        (rotate(a, second[..., :3]) + first[..., :3], vector, scalar), dim=-1
    )


def pose_error(
    current: torch.Tensor, target: torch.Tensor, local_points: torch.Tensor
) -> torch.Tensor:
    """RMS corresponding-surface displacement; includes rotation in metre units.

    AlphaNIST leaves its pose-distance formula unpublished. Identity symmetry is
    conservative for keyed assemblies; no yaw invariance is silently assumed.
    """
    delta = transform_points(current, local_points) - transform_points(
        target, local_points
    )
    return delta.square().sum(-1).mean(-1).sqrt()
