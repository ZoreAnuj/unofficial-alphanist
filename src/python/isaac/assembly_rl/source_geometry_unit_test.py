import numpy as np
import pytest
import scipy.spatial
import shapely
import trimesh

from isaac.assembly_rl import collision_sections as cs
from isaac.assembly_rl import prepare_source_assets as prep


def test_section_extrusion_preserves_through_bore():
    polygon = shapely.box(-0.01, -0.01, 0.01, 0.01).difference(
        shapely.Point(0, 0).buffer(0.003)
    )
    pieces, report = cs.extrusion(polygon, 2, -0.005, 0.005, error=1e-6)
    union = cs.union_mesh(pieces)
    assert report["holes"] == 1
    assert union.is_volume
    assert union.volume == pytest.approx(polygon.area * 0.01, rel=0.001)
    for piece in pieces:
        planes = scipy.spatial.ConvexHull(piece).equations
        assert not np.all(planes[:, 3] <= 0)


def test_bounded_tessellation_repair_rejects_large_opening():
    mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    mesh.update_faces(np.arange(1, len(mesh.faces)))
    with pytest.raises(ValueError, match="diameter"):
        prep.repair_tiny_triangles(mesh)


def test_bounded_tessellation_repair_closes_tiny_missing_face():
    mesh = trimesh.creation.box(extents=[1e-6, 1e-6, 1e-6])
    mesh.update_faces(np.arange(1, len(mesh.faces)))
    repaired, report = prep.repair_tiny_triangles(mesh)
    assert repaired.is_volume
    assert len(report) == 1
    assert repaired.volume == pytest.approx(1e-18)


def test_convex_clipping_preserves_raceway_halfspace():
    box = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    clipped = prep.clip_convex(box.vertices, np.array([[1.0, 0, 1.0, 0]]))
    assert np.max(clipped[:, 0] + clipped[:, 2]) < 1e-10
    assert cs.convex_mesh(clipped).volume == pytest.approx(box.volume / 2)


def test_continuous_sweep_detects_interference_between_endpoints():
    transform = np.eye(4).tolist()
    definition = {
        "id": "03_shaft_coupling",
        "parts": [
            {"name": "moving", "assembled_transform": transform, "fixed": False},
            {"name": "obstacle", "assembled_transform": transform, "fixed": True},
        ],
    }
    moving = trimesh.creation.box(extents=[0.002, 0.002, 0.002]).vertices
    obstacle = moving + [0, 0, 0.02]
    report, _ = prep.fit_task(definition, {"moving": [moving], "obstacle": [obstacle]})
    assert report["pairs"][0]["target_overlap_m3"] == 0
    assert report["pairs"][0]["swept_overlap_m3"] > 1e-10
    assert not report["passed"]


def test_convex_partition_merge_keeps_concave_corner_open():
    a = trimesh.creation.box(extents=[0.01, 0.01, 0.01]).vertices
    b = a + [0.01, 0, 0]
    c = a + [0, 0.01, 0]
    merged = cs.merge_convex_partition([a, b, c])
    assert len(merged) == 2
    assert sum(cs.convex_mesh(p).volume for p in merged) == pytest.approx(3e-6)
    outside = np.array([0.01, 0.01, 0])
    for p in merged:
        planes = scipy.spatial.ConvexHull(p).equations
        assert not np.all(planes[:, :3] @ outside + planes[:, 3] <= 0)
