import ast
import dataclasses
import pathlib

import mujoco
import mujoco_warp
import numpy as np
import pytest
import warp as wp
from mujoco_warp._src import io, types

from isaac.assembly_rl import compact_allocation


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_sparse_allocation_is_discarded_and_final_state_unchanged(
    monkeypatch, sparse, compact
):
    original_source = pathlib.Path(io.__file__).read_bytes()
    monkeypatch.setattr(compact_allocation, "_installed", None)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="hinge"/>'
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    model.opt.jacobian = (
        mujoco.mjtJacobian.mjJAC_SPARSE if sparse else mujoco.mjtJacobian.mjJAC_DENSE
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cJ_type = next(f.type for f in dataclasses.fields(types.Data) if f.name == "cJ")
    allocations = []
    allocate = io._create_array

    def record(value, specification, sizes):
        if specification is cJ_type:
            allocations.append(
                (sizes["nworld"], sizes["njmax_pad"], sizes["nvmax_pad"])
            )
        return allocate(value, specification, sizes)

    monkeypatch.setattr(io, "_create_array", record)
    # Earlier environment tests may already have installed the process overlay.
    function = next(
        node
        for node in ast.parse(original_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "put_data"
    )
    namespace = dict(vars(io))
    exec(  # noqa: S102 - restore the installed runtime's host function for this test.
        compile(ast.Module(body=[function], type_ignores=[]), io.__file__, "exec"),
        namespace,
    )
    original = namespace["put_data"]
    monkeypatch.setattr(io, "put_data", original)
    monkeypatch.setattr(mujoco_warp, "put_data", original)
    kwargs = {"nworld": 2, "nconmax": 4, "njmax": 16, "nvmax": 1 if compact else None}
    with wp.ScopedDevice("cpu"):
        before = original(model, data, **kwargs)
        assert allocations == [(2, 16, 16)]
        allocations.clear()
        assert compact_allocation.install() == compact_allocation.fingerprint()
        after = mujoco_warp.put_data(model, data, **kwargs)
        assert allocations == ([] if sparse else [(2, 16, 16)])
        assert before.cJ.shape == after.cJ.shape
        assert after.cJ.shape == ((2, 16, 16) if compact and not sparse else (0, 0, 0))
        for field in ("qpos", "qvel", "qacc", "qLD", "nacon", "body_awake", "ncdof"):
            np.testing.assert_array_equal(
                getattr(before, field).numpy(), getattr(after, field).numpy()
            )
        assert compact_allocation.install() == compact_allocation.fingerprint()
    assert pathlib.Path(io.__file__).read_bytes() == original_source
    monkeypatch.setattr(mujoco_warp, "put_data", original)
    with pytest.raises(RuntimeError, match="binding conflicts"):
        compact_allocation.install()


def test_unknown_source_fails_closed():
    with pytest.raises(RuntimeError, match="exact audited runtime"):
        compact_allocation.transform_source(b"changed runtime")
