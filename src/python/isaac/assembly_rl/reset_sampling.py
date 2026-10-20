"""Geometry-only reset proposals and bank-global success-guided sampling.

Proposals require downstream collision, reachability, and physics validation.
Transforms map object-local coordinates into the table's world frame, in meters.
"""

import copy
import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PartSpec:
    name: str
    vertices_m: np.ndarray
    nominal_rotation: np.ndarray
    fixed: bool = False


@dataclasses.dataclass(frozen=True)
class Placement:
    transform_world_from_local: np.ndarray
    proposed_mode: str


def _rotation(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1e-7)
    ):
        raise ValueError("Expected a finite proper 3x3 rotation")
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def _sample_rotation(
    rng: np.random.Generator, mode: str, nominal: np.ndarray
) -> np.ndarray:
    if mode == "uniform":
        # Normalized isotropic Gaussian quaternions yield Haar-uniform SO(3).
        quat = rng.normal(size=4)
        w, x, y, z = quat / np.linalg.norm(quat)
        return (
            np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            @ nominal
        )
    tilt = {"upright": 0.0, "side": np.pi / 2, "inverted": np.pi}[mode]
    azimuth = rng.uniform(-np.pi, np.pi)
    tilt_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    yaw = _axis_rotation(np.array([0.0, 0.0, 1.0]), rng.uniform(-np.pi, np.pi))
    return yaw @ _axis_rotation(tilt_axis, tilt) @ nominal


def classify_orientation(
    rotation: np.ndarray, nominal_rotation: np.ndarray, tolerance_degrees: float = 20
) -> str:
    """Classify a settled attitude, ignoring yaw; this is not a stability test."""
    if not 0 < tolerance_degrees < 45:
        raise ValueError("tolerance_degrees must be between 0 and 45")
    relative = _rotation(rotation) @ _rotation(nominal_rotation).T
    angle = np.degrees(np.arccos(np.clip(relative[2, 2], -1, 1)))
    if angle <= tolerance_degrees:
        return "upright"
    if angle >= 180 - tolerance_degrees:
        return "inverted"
    if abs(angle - 90) <= tolerance_degrees:
        return "side"
    return "tilted"


def propose_placements(
    parts: list[PartSpec],
    table_xy_bounds_m: np.ndarray,
    table_height_m: float,
    seed: int,
    *,
    exclusion_aabbs_m: tuple[np.ndarray, ...] = (),
    orientation_modes: tuple[str, ...] = ("upright", "side", "inverted", "uniform"),
    max_attempts_per_part: int = 200,
    drop_clearance_m: float = 0.005,
    separation_m: float = 0.01,
) -> dict[str, Placement]:
    """Sample bounded proposals with conservative AABB rejection.

    XY bounds have rows [xmin, xmax], [ymin, ymax]. Exclusion boxes have
    rows [xmin, ymin, zmin], [xmax, ymax, zmax]. Fixed receivers are placed
    first and get yaw only, with zero drop clearance. Other parts are lifted
    above the table for settling. Sampling modes are equiprobable proposals;
    accepted and settled distributions need not be uniform.
    """
    bounds = np.asarray(table_xy_bounds_m, dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("Invalid table XY bounds")
    if (
        not np.isfinite([table_height_m, drop_clearance_m, separation_m]).all()
        or drop_clearance_m < 0
        or separation_m < 0
        or not isinstance(max_attempts_per_part, int)
        or max_attempts_per_part < 1
    ):
        raise ValueError("Invalid placement limits")
    if not orientation_modes or set(orientation_modes) - {
        "upright",
        "side",
        "inverted",
        "uniform",
    }:
        raise ValueError("Invalid orientation modes")
    if len({part.name for part in parts}) != len(parts):
        raise ValueError("Part names must be unique")
    occupied = []
    for box in exclusion_aabbs_m:
        box = np.asarray(box, dtype=float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] > box[1]):
            raise ValueError("Invalid exclusion AABB")
        occupied.append(box)
    rng = np.random.default_rng(seed)
    placements = {}
    for part in sorted(parts, key=lambda part: not part.fixed):
        vertices = np.asarray(part.vertices_m, dtype=float)
        nominal = _rotation(part.nominal_rotation)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) == 0
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(f"Invalid vertices for {part.name}")
        for _ in range(max_attempts_per_part):
            mode = "upright" if part.fixed else str(rng.choice(orientation_modes))
            rotation = _sample_rotation(rng, mode, nominal)
            rotated = vertices @ rotation.T
            lower, upper = rotated.min(axis=0), rotated.max(axis=0)
            xy_min = bounds[:, 0] - lower[:2]
            xy_max = bounds[:, 1] - upper[:2]
            if np.any(xy_min > xy_max):
                continue
            translation = np.empty(3)
            translation[:2] = rng.uniform(xy_min, xy_max)
            clearance = 0.0 if part.fixed else drop_clearance_m
            translation[2] = table_height_m + clearance - lower[2]
            box = np.array([lower + translation, upper + translation])
            if any(
                np.all(box[0] < other[1] + separation_m)
                and np.all(other[0] < box[1] + separation_m)
                for other in occupied
            ):
                continue
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            placements[part.name] = Placement(transform, mode)
            occupied.append(box)
            break
        else:
            raise RuntimeError(f"No placement for {part.name} within attempt budget")
    return placements


def move_target_with_receiver(
    target_world: np.ndarray,
    receiver_original_world: np.ndarray,
    receiver_sampled_world: np.ndarray,
) -> np.ndarray:
    """Preserve the receiver-relative mating transform after receiver relocation."""
    transforms = []
    for value in (target_world, receiver_original_world, receiver_sampled_world):
        value = np.asarray(value, dtype=float)
        if (
            value.shape != (4, 4)
            or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
        ):
            raise ValueError("Expected finite rigid 4x4 transforms")
        _rotation(value[:3, :3])
        transforms.append(value)
    target, original, sampled = transforms
    return sampled @ np.linalg.inv(original) @ target


class SuccessGuidedSampler:
    """Global-bank SGS; window and epsilon are explicit experimental settings.

    AlphaNIST Algorithm 1 averages observed outcomes, as does the cited OmniReset
    success monitor. Unseen entries have rate zero, without fictitious failures.
    Save sampler state together with the ordered bank identity.
    """

    def __init__(
        self, bank_size: int, window_size: int, epsilon: float, seed: int = 0
    ) -> None:
        if (
            not isinstance(bank_size, int)
            or not isinstance(window_size, int)
            or bank_size < 1
            or window_size < 1
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("Bank/window must be positive integers and epsilon > 0")
        self._history = np.zeros((bank_size, window_size), dtype=np.uint8)
        self._cursor = np.zeros(bank_size, dtype=np.int64)
        self._counts = np.zeros(bank_size, dtype=np.int64)
        self._epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)

    def probabilities(self) -> np.ndarray:
        rates = self._history.sum(axis=1) / self._counts.clip(min=1)
        weights = self._epsilon + np.sqrt(rates * (1 - rates))
        return weights / weights.sum()

    def sample(self, count: int) -> np.ndarray:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return self._rng.choice(len(self._history), size=count, p=self.probabilities())

    def update(self, indices: np.ndarray, successes: np.ndarray) -> None:
        indices, successes = np.asarray(indices), np.asarray(successes)
        if (
            indices.ndim != 1
            or successes.shape != indices.shape
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(self._history))
            or not np.isin(successes, [0, 1]).all()
        ):
            raise ValueError("Expected valid bank indices and matching binary outcomes")
        # Sequential writes preserve every outcome when parallel episodes share a reset.
        for index, success in zip(indices, successes, strict=True):
            cursor = self._cursor[index]
            self._history[index, cursor] = success
            self._cursor[index] = (cursor + 1) % self._history.shape[1]
            self._counts[index] = min(self._counts[index] + 1, self._history.shape[1])

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "epsilon": self._epsilon,
            "history": self._history.tolist(),
            "cursor": self._cursor.tolist(),
            "counts": self._counts.tolist(),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SuccessGuidedSampler":
        # Version 1 cannot distinguish unseen failures from recorded failures.
        if state.get("version") != 2:
            raise ValueError("Invalid SGS checkpoint version; observed counts required")
        history = np.asarray(state["history"])
        cursor = np.asarray(state["cursor"])
        counts = np.asarray(state.get("counts"))
        if (
            history.ndim != 2
            or min(history.shape) < 1
            or not np.isin(history, [0, 1]).all()
            or cursor.shape != (history.shape[0],)
            or cursor.dtype.kind not in "iu"
            or np.any(cursor < 0)
            or np.any(cursor >= history.shape[1])
            or counts.shape != (history.shape[0],)
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > history.shape[1])
            or np.any((counts < history.shape[1]) & (cursor != counts))
            or np.any(history * (np.arange(history.shape[1]) >= counts[:, None]))
        ):
            raise ValueError("Invalid SGS checkpoint")
        sampler = cls(*history.shape, epsilon=state["epsilon"])
        sampler._history[:] = history
        sampler._cursor[:] = cursor
        sampler._counts[:] = counts
        sampler._rng.bit_generator.state = copy.deepcopy(state["rng"])
        return sampler
