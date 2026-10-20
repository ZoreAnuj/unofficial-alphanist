import json
import pathlib

import numpy as np
import pytest

from isaac.assembly_rl import generate_bank

CATALOG = pathlib.Path(
    "data/industrial_assembly_tasks/catalog.json"
)


@pytest.fixture(scope="module")
def actual_bank() -> tuple[dict, dict]:
    if not CATALOG.exists():
        pytest.skip("Selected CAD collection is not available")
    return generate_bank.build_bank(CATALOG, 4, 987)


def test_catalog_tasks_have_orientation_coverage_and_preserved_targets(
    actual_bank: tuple[dict, dict],
) -> None:
    metadata, arrays = actual_bank
    expected_tasks = len(json.loads(CATALOG.read_text())["tasks"])
    assert metadata["sample_count"] == 4 * expected_tasks
    assert len(metadata["tasks"]) == expected_tasks
    assert metadata["training_ready"] is False
    assert metadata["physics_validated"] is False
    for task in metadata["tasks"]:
        assert {sample["orientation_family"] for sample in task["samples"]} == set(
            generate_bank.ORIENTATION_MODES
        )
        task_id = task["id"]
        receiver_index = task["receiver_index"]
        original = np.array(
            [part["original_assembled_world_transform"] for part in task["parts"]]
        )
        original_relative = np.linalg.inv(original[receiver_index]) @ original
        assert all(len(part["source_sha256"]) == 64 for part in task["parts"])
        for index, sample in enumerate(task["samples"]):
            actual = arrays[task_id + "__world_transforms"][index]
            targets = arrays[task_id + "__target_world_transforms"][index]
            np.testing.assert_allclose(
                np.linalg.inv(actual[receiver_index]) @ targets,
                original_relative,
                atol=1e-12,
            )
            assert (
                arrays[task_id + "__orientation_modes"][index, receiver_index]
                == "upright"
            )
            assert sample["physics_validated"] is False


def test_all_tasks_are_deterministic_and_filtered_task_keeps_seed(
    actual_bank: tuple[dict, dict],
) -> None:
    metadata, arrays = actual_bank
    second_metadata, second_arrays = generate_bank.build_bank(CATALOG, 4, 987)
    assert metadata == second_metadata
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], second_arrays[key])
    _, subset_arrays = generate_bank.build_bank(CATALOG, 4, 987, task="06_gear_train")
    for key in subset_arrays:
        np.testing.assert_array_equal(arrays[key], subset_arrays[key])


def test_saved_npz_contains_no_pickle_arrays(
    actual_bank: tuple[dict, dict], tmp_path: pathlib.Path
) -> None:
    _, arrays = actual_bank
    path = tmp_path / "bank.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], loaded[key])


def test_invalid_selection_is_explicit() -> None:
    if not CATALOG.exists():
        pytest.skip("Selected CAD collection is not available")
    with pytest.raises(ValueError, match="No matching task"):
        generate_bank.build_bank(CATALOG, 4, 987, task="missing")
    with pytest.raises(ValueError, match="positive"):
        generate_bank.build_bank(CATALOG, 0, 987)
