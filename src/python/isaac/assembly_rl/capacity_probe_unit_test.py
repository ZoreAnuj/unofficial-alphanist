"""Capacity bounds derive from the actual device, not a fixed GPU model."""

import argparse

import pytest

from isaac.assembly_rl.capacity_probe import (
    device_budget_gib,
    failure_category,
    positive_count,
)


def test_arbitrary_positive_environment_counts():
    for count in (1, 256, 768, 1536, 3072, 4096):
        assert positive_count(str(count)) == count
    for invalid in ("0", "-1", "3.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_count(invalid)


def test_budget_uses_reported_device_memory_and_reserve():
    assert device_budget_gib(80 * 1024**3, 2, None) == 78
    assert device_budget_gib(40 * 1024**3, 2, None) == 38
    assert device_budget_gib(80 * 1024**3, 2, 60) == 60
    for reserve in (0, -1, 80, float("nan")):
        with pytest.raises(ValueError):
            device_budget_gib(80 * 1024**3, reserve, None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("CUDA out of memory"), "gpu_oom"),
        (MemoryError("allocation failed"), "host_oom"),
        (RuntimeError("GPU memory safety bound exceeded"), "gpu_budget"),
        (RuntimeError("Spawn collision capacity exceeded"), "solver_capacity"),
        (RuntimeError("Nonfinite part poses"), "nonfinite_state"),
        (RuntimeError("shape cannot exceed signed int32 limit"), "index_limit"),
        (ValueError("SAP64 dimensions exceed signed32 capacity"), "index_limit"),
    ],
)
def test_failure_classification(error, expected):
    assert failure_category(error) == expected
