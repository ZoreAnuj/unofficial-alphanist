"""Assembly-local withdrawal transforms in meters and radians."""

import numpy as np
from scipy.spatial.transform import Rotation


def withdrawal_transform(pose: np.ndarray, part: dict, distance_m: float) -> np.ndarray:
    """Withdraw along a local axis, following a right-handed thread when declared.

    ``pose`` maps part coordinates to world coordinates. The optional screw
    origin is a point on the axis in part coordinates, not a world-space pivot.
    """
    axis = np.asarray(part["disassembly_axis_local"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError("Invalid disassembly axis")
    axis = axis / np.linalg.norm(axis)
    delta = np.eye(4)
    delta[:3, 3] = axis * distance_m
    if "screw_pitch_m" in part:
        pitch = float(part["screw_pitch_m"])
        origin = np.asarray(
            part.get("screw_axis_origin_local_m", [0, 0, 0]), dtype=float
        )
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("Screw pitch must be finite and positive")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid screw axis origin")
        delta[:3, :3] = Rotation.from_rotvec(
            axis * (2 * np.pi * distance_m / pitch)
        ).as_matrix()
        delta[:3, 3] += origin - delta[:3, :3] @ origin
    return pose @ delta
