import numpy as np
import shapely
from shapely.geometry import MultiPoint, box

from isaac.assembly_rl.nist_section_partition import shared_core_extrusion


def test_shared_core_preserves_holes_steps_and_gap():
    hole = box(0.003, 0.003, 0.007, 0.007)
    first = box(0, 0, 0.010, 0.010).difference(hole)
    second = box(0, 0, 0.012, 0.010).difference(hole)
    slabs = [(0.0, 0.0004, first), (0.0004, 0.0008, second), (0.001, 0.0014, first)]
    hulls, report = shared_core_extrusion(slabs)
    for low, high, expected in slabs:
        z = (low + high) / 2
        actual = shapely.union_all(
            [
                MultiPoint(np.unique(p[:, :2], axis=0)).convex_hull
                for p in hulls
                if p[:, 2].min() < z < p[:, 2].max()
            ]
        )
        assert actual.symmetric_difference(expected).area < 1e-18
        assert actual.intersection(hole).area == 0
    assert not any(p[:, 2].min() < 0.0009 < p[:, 2].max() for p in hulls)
    assert len(report["groups"]) == 2
