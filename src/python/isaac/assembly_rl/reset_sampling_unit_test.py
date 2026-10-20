import itertools
import json

import numpy as np
import pytest

from isaac.assembly_rl import reset_sampling


def _part(name: str = "part", fixed: bool = False) -> reset_sampling.PartSpec:
    vertices = np.array(list(itertools.product([-0.03, 0.03], [-0.02, 0.02], [0, 0.1])))
    return reset_sampling.PartSpec(name, vertices, np.eye(3), fixed)


@pytest.mark.parametrize("mode", ["upright", "side", "inverted", "uniform"])
def test_proposals_fit_table_and_are_reproducible(mode: str) -> None:
    parts = [_part("receiver", True), _part("a"), _part("b")]
    bounds = np.array([[-0.5, 0.5], [-0.3, 0.3]])
    exclusion = np.array([[-0.1, -0.1, 0.8], [0.1, 0.1, 1.2]])
    kwargs = {"exclusion_aabbs_m": (exclusion,), "orientation_modes": (mode,)}
    poses = reset_sampling.propose_placements(parts, bounds, 0.8, 18, **kwargs)
    again = reset_sampling.propose_placements(parts, bounds, 0.8, 18, **kwargs)
    boxes = [exclusion]
    for part in parts:
        pose = poses[part.name]
        transform = pose.transform_world_from_local
        np.testing.assert_array_equal(
            transform, again[part.name].transform_world_from_local
        )
        points = part.vertices_m @ transform[:3, :3].T + transform[:3, 3]
        assert np.all(points[:, :2] >= bounds[:, 0] - 1e-12)
        assert np.all(points[:, :2] <= bounds[:, 1] + 1e-12)
        assert points[:, 2].min() == pytest.approx(0.8 if part.fixed else 0.805)
        box = np.array([points.min(axis=0), points.max(axis=0)])
        for other in boxes:
            assert not (
                np.all(box[0] < other[1] + 0.01) and np.all(other[0] < box[1] + 0.01)
            )
        boxes.append(box)
        expected = "upright" if part.fixed else mode
        if expected != "uniform":
            assert (
                reset_sampling.classify_orientation(transform[:3, :3], np.eye(3))
                == expected
            )


def test_uniform_rotations_are_isotropic_and_proper() -> None:
    rng = np.random.default_rng(10)
    rotations = np.array(
        [
            reset_sampling._sample_rotation(rng, "uniform", np.eye(3))
            for _ in range(10000)
        ]
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-12)
    np.testing.assert_allclose(rotations.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose((rotations**2).mean(axis=0), 1 / 3, atol=0.02)


def test_classification_handles_nonidentity_nominal_rotation() -> None:
    nominal = reset_sampling._axis_rotation(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    rng = np.random.default_rng(15)
    for mode in ("upright", "side", "inverted"):
        rotation = reset_sampling._sample_rotation(rng, mode, nominal)
        assert reset_sampling.classify_orientation(rotation, nominal) == mode


def test_oversized_and_blocked_parts_fail_with_bounded_attempts() -> None:
    with pytest.raises(RuntimeError, match="attempt budget"):
        reset_sampling.propose_placements(
            [_part()], np.array([[0, 0.01], [0, 0.01]]), 0.8, 1, max_attempts_per_part=3
        )
    with pytest.raises(RuntimeError, match="attempt budget"):
        reset_sampling.propose_placements(
            [_part()],
            np.array([[0, 1], [0, 1]]),
            0.8,
            1,
            exclusion_aabbs_m=(np.array([[-1, -1, -1], [2, 2, 2]]),),
            max_attempts_per_part=3,
        )


def test_target_frame_moves_rigidly_with_receiver() -> None:
    original = np.eye(4)
    original[:3, 3] = [0.2, 0.1, 0.8]
    target = original.copy()
    target[:3, 3] += [0.05, 0, 0.1]
    sampled = np.eye(4)
    sampled[:3, :3] = reset_sampling._axis_rotation(
        np.array([0.0, 0.0, 1.0]), np.pi / 2
    )
    sampled[:3, 3] = [-0.2, 0.1, 0.8]
    result = reset_sampling.move_target_with_receiver(target, original, sampled)
    np.testing.assert_allclose(
        np.linalg.inv(sampled) @ result, np.linalg.inv(original) @ target, atol=1e-12
    )


def test_sgs_global_probabilities_and_repeated_index_updates() -> None:
    sampler = reset_sampling.SuccessGuidedSampler(3, 4, 0.05)
    np.testing.assert_allclose(sampler.probabilities(), [1 / 3] * 3)
    sampler.update(np.array([0, 0, 1, 1, 1, 1]), np.array([1, 0, 1, 1, 1, 1]))
    np.testing.assert_allclose(
        sampler.probabilities(), np.array([0.55, 0.05, 0.05]) / 0.65
    )
    sampler.update(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]))
    expected = np.array([0.05 + np.sqrt(0.25 * 0.75), 0.05, 0.05])
    np.testing.assert_allclose(sampler.probabilities(), expected / expected.sum())
    assert sampler.state_dict()["counts"] == [4, 4, 0]


def test_first_success_does_not_count_unobserved_episodes_as_failures() -> None:
    sampler = reset_sampling.SuccessGuidedSampler(2, 100, 1e-4)
    sampler.update(np.array([0]), np.array([1]))
    np.testing.assert_allclose(sampler.probabilities(), [0.5, 0.5])
    sampler.update(np.array([0]), np.array([0]))
    weights = np.array([0.5001, 0.0001])
    np.testing.assert_allclose(sampler.probabilities(), weights / weights.sum())


def test_checkpoint_roundtrip_restores_rng_and_history() -> None:
    sampler = reset_sampling.SuccessGuidedSampler(6, 3, 0.01, 89)
    sampler.update(np.array([0, 2, 2, 3]), np.array([1, 0, 1, 1]))
    sampler.sample(10)
    clone = reset_sampling.SuccessGuidedSampler.from_state_dict(
        json.loads(json.dumps(sampler.state_dict()))
    )
    np.testing.assert_array_equal(sampler.sample(50), clone.sample(50))
    sampler.update(np.array([0, 2]), np.array([0, 1]))
    clone.update(np.array([0, 2]), np.array([0, 1]))
    assert sampler.state_dict() == clone.state_dict()


@pytest.mark.parametrize(
    "counts", [None, [1], [-1, 0], [4, 0], [1.0, 0.0], [True, False], [0, 0], [2, 0]]
)
def test_checkpoint_rejects_malformed_observed_counts(counts: list | None) -> None:
    sampler = reset_sampling.SuccessGuidedSampler(2, 3, 0.01)
    sampler.update(np.array([0]), np.array([1]))
    state = sampler.state_dict()
    state["counts"] = counts
    with pytest.raises(ValueError, match="SGS checkpoint"):
        reset_sampling.SuccessGuidedSampler.from_state_dict(state)


def test_checkpoint_rejects_legacy_history_without_observed_counts() -> None:
    state = reset_sampling.SuccessGuidedSampler(2, 3, 0.01).state_dict()
    state["version"] = 1
    state.pop("counts", None)
    with pytest.raises(ValueError, match="SGS checkpoint"):
        reset_sampling.SuccessGuidedSampler.from_state_dict(state)


@pytest.mark.parametrize(
    "indices,outcomes",
    [([0, 3], [1, 0]), ([0, 1], [1, 0.5]), ([0.0, 1.0], [1, 0]), ([0, 1], [1])],
)
def test_invalid_outcomes_do_not_partially_mutate(
    indices: list, outcomes: list
) -> None:
    sampler = reset_sampling.SuccessGuidedSampler(3, 4, 0.05)
    before = sampler.state_dict()
    with pytest.raises(ValueError):
        sampler.update(np.array(indices), np.array(outcomes))
    assert sampler.state_dict() == before
