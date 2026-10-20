import pytest

from isaac.assembly_rl import environment


def test_contact_capacity_overrides_profile_and_changes_bank_identity(monkeypatch):
    """Allocations must reach Newton and invalidate banks from another capacity."""
    task = {"id": "test", "parts": [{"name": "fixture"}]}
    monkeypatch.setattr(environment.scene_model, "load_tasks", lambda *a, **k: [task])
    monkeypatch.setattr(
        environment.scene_model,
        "build_model",
        lambda *a, **k: (
            object(),
            {"asset_sha256": {}, "base_mount": []},
            {"solver": {"nconmax": 1024, "njmax": 8192}},
        ),
    )
    options = []

    class StopBeforeAllocationError(Exception):
        pass

    def solver(_model, **kwargs):
        options.append(kwargs)
        raise StopBeforeAllocationError

    monkeypatch.setattr(environment.scene_model, "AssemblySolverMuJoCo", solver)
    identities = []
    for config, threaded in (
        ({}, False),
        ({"solver_nconmax": 32768}, False),
        ({"solver_jacobian": "dense"}, False),
        ({}, True),
        ({"solver_nconmax": 32768}, True),
    ):
        if threaded:
            task["parts"][0]["screw_pitch_m"] = 0.002
        env = environment.AssemblyEnvironment.__new__(environment.AssemblyEnvironment)
        with pytest.raises(StopBeforeAllocationError):
            env.__init__("unused", device="cpu", config=config)
        identities.append(env.physics_identity)
        assert env.config["solver_nconmax"] == options[-1]["nconmax"]
    assert [row["nconmax"] for row in options] == [16384, 32768, 16384, 131072, 32768]
    assert [row["njmax"] for row in options] == [65536] * 5
    assert [row["jacobian"] for row in options] == [
        "sparse",
        "sparse",
        "dense",
        "sparse",
        "sparse",
    ]
    assert len(set(identities[:4])) == 4
    # The mocked assets are identical: equal effective settings have equal identity.
    assert identities[1] == identities[4]
