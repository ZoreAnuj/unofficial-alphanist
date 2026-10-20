"""Probe hashes compiled contact eligibility, independent of mask coloring."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaac.assembly_rl.single_sim_probe import capture_collision_graph


def fake_environment(colors, pair_ids):
    native = SimpleNamespace(ngeom=3)
    for name in (
        "geom_contype",
        "geom_conaffinity",
        "geom_bodyid",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "body_weldid",
        "body_parentid",
        "exclude_signature",
        "pair_geom1",
        "pair_geom2",
    ):
        setattr(native, name, np.asarray(colors))
    ids = SimpleNamespace(numpy=lambda: np.asarray(pair_ids))
    return SimpleNamespace(
        solver=SimpleNamespace(
            mj_model=native, mjw_model=SimpleNamespace(nxn_pairid=ids)
        )
    )


def test_compiled_graph_hash_ignores_color_permutation(tmp_path):
    pairs = [[-1, -1], [-2, 0], [3, -1]]
    first = capture_collision_graph(
        fake_environment([1, 2, 4], pairs), tmp_path / "a.npz"
    )
    second = capture_collision_graph(
        fake_environment([4, 1, 2], pairs), tmp_path / "b.npz"
    )
    assert (
        first["allowed_contact_graph_sha256"] == second["allowed_contact_graph_sha256"]
    )
    assert first["allowed_contact_pair_count"] == 2
    pairs[1][0] = -1
    third = capture_collision_graph(
        fake_environment([4, 1, 2], pairs), tmp_path / "c.npz"
    )
    assert (
        first["allowed_contact_graph_sha256"] != third["allowed_contact_graph_sha256"]
    )


def test_unexpected_backend_graph_layout_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        capture_collision_graph(
            fake_environment([1, 2, 4], [[-1, -1]]), tmp_path / "bad.npz"
        )
