import sys
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from mujoco_warp._src import collision_core, collision_driver
from mujoco_warp._src.types import BroadphaseFilter, BroadphaseType

from isaac.assembly_rl import sap_wide


@wp.kernel
def lookup(prefix: wp.array[wp.int64], out: wp.array[int]):
    out[0] = collision_core.sap_binary_search(prefix, wp.int64(2147483648), 0, 3)


def test_wide_scan_overflow_boundary():
    a = wp.array(
        np.array([2147483640, 20, 30], dtype=np.int64), dtype=wp.int64, device="cpu"
    )
    b = wp.empty_like(a)
    wp.utils.array_scan(a, b)
    assert b.numpy().tolist() == [2147483640, 2147483660, 2147483690]
    out = wp.zeros(1, dtype=int, device="cpu")
    wp.launch(lookup, dim=1, inputs=[b, out], device="cpu")
    assert out.numpy()[0] == 1


def test_exact_pair_equivalence_and_ccd_binding(monkeypatch):
    original = collision_driver.sap_broadphase
    monkeypatch.setattr(collision_driver, "sap_broadphase", original)
    convex = collision_driver.convex_narrowphase
    with wp.ScopedDevice("cpu"):
        m = SimpleNamespace(
            ngeom=3,
            opt=SimpleNamespace(
                broadphase=BroadphaseType.SAP_SEGMENTED,
                enableflags=0,
                broadphase_filter=BroadphaseFilter.SPHERE,
            ),
            geom_rbound=wp.array(np.ones((1, 3)), dtype=float),
            geom_margin=wp.zeros((1, 3)),
            geom_gap=wp.zeros((1, 3)),
            geom_aabb=wp.zeros((1, 3, 2), dtype=wp.vec3),
            geom_type=wp.array([2, 2, 2], dtype=int),
            geom_bodyid=wp.array([0, 1, 2], dtype=int),
            nxn_pairid=wp.array([[-1, -1], [-1, -1], [-1, -1]], dtype=wp.vec2i),
        )
        d = SimpleNamespace(
            nworld=2,
            naconmax=16,
            geom_xpos=wp.array(np.zeros((2, 3, 3)), dtype=wp.vec3),
            geom_xmat=wp.array(
                np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy(), dtype=wp.mat33
            ),
            body_awake=wp.zeros((2, 3), dtype=int),
            ncollision=wp.zeros(1, dtype=int),
        )

        def run(fn):
            ctx = collision_core.create_collision_context(16)
            d.ncollision.zero_()
            fn(m, d, ctx)
            n = int(d.ncollision.numpy()[0])
            return sorted(
                (int(w), *map(int, p))
                for w, p in zip(
                    ctx.collision_worldid.numpy()[:n],
                    ctx.collision_pair.numpy()[:n],
                    strict=True,
                )
            )

        before = run(original)
        sap_wide.install()
        after = run(collision_driver.sap_broadphase)
        assert before == after and len(after) == 6
        assert collision_driver.convex_narrowphase is convex
        assert sap_wide.install() == sap_wide.fingerprint()
        d.nworld = 2147483647 // (m.ngeom * 5) + 1
        with pytest.raises(ValueError, match="signed32"):
            run(collision_driver.sap_broadphase)
    sys.modules.pop(sap_wide.MODULE_NAME, None)


def test_source_gate():
    with pytest.raises(RuntimeError):
        sap_wide.transform_source(b"changed", b"changed")
