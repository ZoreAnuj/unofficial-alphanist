import ast
import pathlib

import mujoco
import mujoco_warp
import numpy as np
import pytest
import warp as wp
from mujoco_warp._src import io

from isaac.assembly_rl import initialization


@pytest.fixture
def original_runtime(monkeypatch):
    source = pathlib.Path(io.__file__).read_bytes()
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("put_model", "_calculate_max_contact_nnz")
    ]
    namespace = dict(vars(io))
    exec(  # noqa: S102 - compile the installed host functions for equivalence checks.
        compile(ast.Module(body=functions, type_ignores=[]), io.__file__, "exec"),
        namespace,
    )
    monkeypatch.setattr(initialization, "_installed", None)
    monkeypatch.setattr(io, "put_model", namespace["put_model"])
    monkeypatch.setattr(mujoco_warp, "put_model", namespace["put_model"])
    monkeypatch.setattr(
        io, "_calculate_max_contact_nnz", namespace["_calculate_max_contact_nnz"]
    )
    return namespace


def native_model(margin=0.0, multiccd=True):
    disabled = "" if multiccd else 'multiccd="disable"'
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><option><flag {disabled}/></option><worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="a" pos="0 0 1"><freejoint/>
          <geom name="a_box" type="box" size=".1 .1 .1" margin="{margin}" contype="1" conaffinity="2"/>
          <geom type="sphere" size=".1" contype="4" conaffinity="8"/>
          <body name="weld"><geom type="sphere" size=".05"/></body>
          <body name="child"><joint type="hinge"/><geom type="capsule" size=".03 .1"/></body>
        </body>
        <body name="b" pos="0 0 2"><freejoint/>
          <geom name="b_box" type="box" size=".1 .1 .1" contype="2" conaffinity="1"/>
          <geom type="ellipsoid" size=".1 .2 .3" contype="8" conaffinity="4"/>
        </body>
        <body name="c" pos="0 0 3"><freejoint/><geom type="box" size=".1 .1 .1" contype="0" conaffinity="0"/></body>
        </worldbody><contact>
          <exclude body1="a" body2="b"/>
          <pair geom1="a_box" geom2="b_box"/>
        </contact></mujoco>"""
    )


@pytest.mark.parametrize("margin,multiccd", [(0.0, True), (0.001, False)])
def test_native_pair_tables_and_nnz_are_exact(original_runtime, margin, multiccd):
    model = native_model(margin, multiccd)
    original_nnz = original_runtime["_calculate_max_contact_nnz"](model)
    with wp.ScopedDevice("cpu"):
        before = original_runtime["put_model"](model)
        assert initialization.install() == initialization.fingerprint()
        after = mujoco_warp.put_model(model)
    for field in (
        "nxn_geom_pair",
        "nxn_geom_pair_filtered",
        "nxn_pairid",
        "nxn_pairid_filtered",
    ):
        np.testing.assert_array_equal(
            getattr(before, field).numpy(), getattr(after, field).numpy()
        )
    assert before.geom_pair_type_count == after.geom_pair_type_count
    assert before.opt.broadphase == after.opt.broadphase
    assert io._calculate_max_contact_nnz(model) == original_nnz
    assert initialization.install() == initialization.fingerprint()


def test_nonzero_margin_rejection_is_preserved(original_runtime):
    model = native_model(0.001)
    model.geom_contype[-1] = 1
    model.geom_conaffinity[-1] = 1
    with wp.ScopedDevice("cpu"):
        with pytest.raises(NotImplementedError) as before:
            original_runtime["put_model"](model)
        initialization.install()
        with pytest.raises(NotImplementedError) as after:
            mujoco_warp.put_model(model)
    assert str(after.value) == str(before.value)


def test_body_nnz_matches_native_for_mixed_masks(original_runtime):
    model = native_model()
    original = original_runtime["_calculate_max_contact_nnz"]
    initialization.install()
    rng = np.random.default_rng(61)
    for _ in range(20):
        model.geom_contype[:] = rng.integers(0, 16, model.ngeom)
        model.geom_conaffinity[:] = rng.integers(0, 16, model.ngeom)
        model.geom_contype[1] = -2147483648
        model.geom_conaffinity[4] = -2147483648
        assert io._calculate_max_contact_nnz(model) == original(model)


def test_type_histogram_crosses_chunk_boundary_and_handles_empty_pairs():
    geom_types = np.array([6, 2, 3], dtype=np.int32)
    pairs = np.tile(np.array([[0, 1], [2, 1], [0, 0]]), (350000, 1))
    expected = np.zeros(45, dtype=np.int64)
    expected[[21, 18, 39]] = 350000
    assert initialization._geom_pair_type_count(geom_types, pairs, 9) == tuple(expected)
    assert initialization._geom_pair_type_count(geom_types, pairs[:0], 9) == (0,) * 45


def test_unknown_runtime_and_conflicting_binding_fail_closed(
    original_runtime, monkeypatch
):
    with pytest.raises(RuntimeError, match="exact audited runtime"):
        initialization.transform_source(b"changed runtime")
    source = pathlib.Path(io.__file__).read_bytes()
    initialization.install()
    assert pathlib.Path(io.__file__).read_bytes() == source
    monkeypatch.setattr(mujoco_warp, "put_model", original_runtime["put_model"])
    with pytest.raises(RuntimeError, match="binding conflicts"):
        initialization.install()
