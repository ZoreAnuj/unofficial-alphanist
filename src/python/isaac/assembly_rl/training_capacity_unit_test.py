import numpy as np
import pytest
import torch

from isaac.assembly_rl.training_capacity_probe import check_dense_state, dense_entries


def test_dense_selection_requires_both_families_per_task():
    families = np.array(["near_mating", "partial_assembly"] * 2)
    tasks = np.array([0, 0, 1, 1])
    assert dense_entries(tasks, families, 2) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Missing dense reset"):
        dense_entries(tasks[:3], families[:3], 2)


def test_bounded_dense_selection_retains_each_task_family_and_active_part():
    tasks = np.repeat([0, 1], 12)
    families = np.tile(np.repeat(["near_mating", "partial_assembly"], 6), 2)
    parts = np.tile([1, 1, 2, 2, 3, 3], 4)
    assert dense_entries(tasks, families, 2) == list(range(24))
    assert dense_entries(
        tasks, families, 2, active_parts=parts, max_per_stratum=1
    ) == list(range(0, 24, 2))
    assert dense_entries(
        tasks, families, 2, active_parts=parts, max_per_stratum=2
    ) == list(range(24))
    with pytest.raises(ValueError, match="active_part"):
        dense_entries(tasks, families, 2, max_per_stratum=1)
    with pytest.raises(ValueError, match="positive"):
        dense_entries(tasks, families, 2, active_parts=parts, max_per_stratum=0)


def test_nested_metrics_validate_counters_and_allow_unobserved_success_rate():
    from isaac.assembly_rl import training_capacity_probe as probe

    metrics = {"loss": 0.2, "tasks": {"rail": {"episodes": 0, "success_rate": None}}}
    assert probe.metrics_finite(metrics)
    metrics["tasks"]["rail"]["episodes"] = float("inf")
    assert not probe.metrics_finite(metrics)
    metrics["tasks"]["rail"]["episodes"] = 1
    metrics["loss"] = float("nan")
    assert not probe.metrics_finite(metrics)


def test_dense_gate_only_allows_compliance_on_selected_held_part():
    state = {
        "failure_flags": torch.zeros(2, dtype=torch.int32),
        "safety_failure": torch.zeros(2, dtype=torch.bool),
        "max_rigid_penetration_m": torch.full((2,), 40e-6),
        "max_pad_part_penetration_m": torch.tensor([[0.0, 400e-6], [0.0, 400e-6]]),
        **{
            name: torch.zeros(2, 1)
            for name in ("part_poses", "part_velocities", "joint_q", "joint_qd")
        },
    }
    check_dense_state(state, 1)
    with pytest.raises(RuntimeError, match="penetration"):
        check_dense_state(state, -1)
    state["max_pad_part_penetration_m"][0, 0] = 60e-6
    with pytest.raises(RuntimeError, match="penetration"):
        check_dense_state(state, 1)
    state["failure_flags"][0] = 2
    with pytest.raises(RuntimeError, match="physics guard"):
        check_dense_state(state, 1)


def test_memory_boundary_trims_only_when_requested_and_in_order(monkeypatch):
    from isaac.assembly_rl import training_capacity_probe as probe

    events = []
    monkeypatch.setattr(probe.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        probe.torch.cuda, "synchronize", lambda: events.append("torch_sync")
    )
    monkeypatch.setattr(
        probe.torch.cuda, "empty_cache", lambda: events.append("torch_trim")
    )
    monkeypatch.setattr(probe.wp, "get_mempool_release_threshold", lambda device: 123)
    monkeypatch.setattr(
        probe.wp,
        "set_mempool_release_threshold",
        lambda device, value: events.append((device, value)),
    )
    monkeypatch.setattr(
        probe.wp, "synchronize_device", lambda device: events.append("warp_sync")
    )
    probe.record_memory_boundary(events.append)
    assert events == ["torch_sync", "unchanged"]
    events.clear()
    probe.record_memory_boundary(events.append, trim_unused_cache=True)
    assert events == [
        "torch_sync",
        "before_trim",
        "torch_trim",
        "torch_sync",
        "after_torch_trim",
        ("cuda:3", 0),
        "warp_sync",
        ("cuda:3", 123),
        "after_warp_trim",
    ]


def test_memory_boundary_restores_warp_threshold_on_sync_failure(monkeypatch):
    from isaac.assembly_rl import training_capacity_probe as probe

    thresholds = []
    monkeypatch.setattr(probe.torch.cuda, "current_device", lambda: 3)
    records = []
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(probe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(probe.wp, "get_mempool_release_threshold", lambda device: 456)
    monkeypatch.setattr(
        probe.wp,
        "set_mempool_release_threshold",
        lambda device, value: thresholds.append(value),
    )

    def fail_sync(device):
        raise RuntimeError("failed sync")

    monkeypatch.setattr(probe.wp, "synchronize_device", fail_sync)
    with pytest.raises(RuntimeError, match="failed sync"):
        probe.record_memory_boundary(records.append, trim_unused_cache=True)
    assert thresholds == [0, 456]
    assert records == ["before_trim", "after_torch_trim"]


def test_allocator_snapshot_distinguishes_current_from_peak(monkeypatch):
    from isaac.assembly_rl import training_capacity_probe as probe

    gib = 1024**3
    monkeypatch.setattr(probe.torch.cuda, "current_device", lambda: 3)
    readings = {
        "mem_get_info": (2 * gib, 10 * gib),
        "memory_allocated": gib,
        "memory_reserved": 2 * gib,
        "max_memory_allocated": 3 * gib,
        "max_memory_reserved": 4 * gib,
        "get_allocator_backend": "native",
        "memory_stats": {"num_alloc_retries": 7, "num_ooms": 2},
    }
    for name, value in readings.items():
        monkeypatch.setattr(probe.torch.cuda, name, lambda value=value: value)
    monkeypatch.setattr(
        probe.wp, "get_mempool_used_mem_current", lambda device: 5 * gib
    )
    monkeypatch.setattr(probe.wp, "get_mempool_used_mem_high", lambda device: 6 * gib)
    monkeypatch.setattr(
        probe.wp, "get_mempool_release_threshold", lambda device: 9 * gib
    )
    stats = probe.allocator_snapshot()
    assert stats["torch_reserved_gib"] == 2
    assert stats["torch_peak_reserved_gib"] == 4
    assert stats["torch_num_alloc_retries"] == 7
    assert stats["torch_num_ooms"] == 2
    assert stats["warp_pool_used_current_gib"] == 5
    assert stats["warp_pool_used_high_gib"] == 6
    assert stats["warp_pool_release_threshold_bytes"] == 9 * gib


def test_constructor_failure_survives_failed_allocator_diagnostic(
    monkeypatch, tmp_path
):
    import json
    import sys

    from isaac.assembly_rl import training_capacity_probe as probe

    out = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--assets",
            str(tmp_path),
            "--bank",
            str(tmp_path),
            "--out",
            str(out),
            "--num-envs",
            "1",
            "--max-device-gib",
            "77",
        ],
    )

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("original constructor OOM")

    def fail_snapshot():
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(probe.environment, "AssemblyEnvironment", fail_constructor)
    monkeypatch.setattr(probe, "allocator_snapshot", fail_snapshot)
    with pytest.raises(RuntimeError, match="original constructor OOM"):
        probe.main()
    result = json.loads((out / "result.json").read_text())
    assert result["error"] == "RuntimeError: original constructor OOM"
    assert result["failure_allocator_error"] == "diagnostic unavailable"
    assert result["trim_unused_cache"] is False


def test_evaluation_environment_stays_resident_in_production_order(
    monkeypatch, tmp_path
):
    import sys
    import types

    from isaac.assembly_rl import training_capacity_probe as probe

    bank = tmp_path / "train"
    bank.mkdir()
    np.savez(bank / "bank.npz", task_ids=[], family=[], held_part=[])
    evaluation = tmp_path / "eval"
    events = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--assets",
            str(tmp_path),
            "--bank",
            str(bank),
            "--eval-bank",
            str(evaluation),
            "--out",
            str(tmp_path / "output"),
            "--num-envs",
            "1",
            "--max-device-gib",
            "75",
        ],
    )

    def environment(_assets, path, num_envs, **kwargs):
        label = path.name
        events.append(("construct", label, num_envs))
        return types.SimpleNamespace(
            label=label,
            physics_identity="same-physics",
            task_names=["task"],
            tasks=[{}],
            bank_metadata={},
            num_envs=num_envs,
            device="cpu",
            close=lambda: events.append(("close", label)),
        )

    class Trainer:
        def __init__(self, env, **kwargs):
            events.append(("trainer", env.label))
            self.bank_identity = str(bank)
            self.model = torch.nn.Linear(1, 1)

        def train_iteration(self):
            events.append(("ppo",))
            return {"loss": 0.1, "tasks": {"task": {"success_rate": None}}}

    monkeypatch.setattr(probe.environment, "AssemblyEnvironment", environment)
    monkeypatch.setattr(probe.train, "Trainer", Trainer)
    monkeypatch.setattr(probe.train, "path_identity", str)
    monkeypatch.setattr(
        probe.train,
        "require_physics_bank",
        lambda env: events.append(("require", env.label)),
    )
    monkeypatch.setattr(
        probe.train,
        "validate_evaluation_separation",
        lambda *args: events.append(("separation",)),
    )
    monkeypatch.setattr(probe, "dense_entries", lambda *args, **kwargs: [])
    monkeypatch.setattr(probe, "allocator_snapshot", lambda: {"device_used_gib": 1.0})
    monkeypatch.setattr(
        probe, "record_memory_boundary", lambda record, **kwargs: record("unchanged")
    )
    monkeypatch.setattr(
        probe.threading,
        "Thread",
        lambda **kwargs: types.SimpleNamespace(start=lambda: None, join=lambda: None),
    )
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(probe.torch.cuda, "reset_peak_memory_stats", lambda: None)
    probe.main()
    assert events == [
        ("construct", "train", 1),
        ("require", "train"),
        ("trainer", "train"),
        ("construct", "eval", 2),
        ("require", "eval"),
        ("separation",),
        ("ppo",),
        ("close", "eval"),
        ("close", "train"),
    ]
