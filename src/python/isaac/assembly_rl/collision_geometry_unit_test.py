import numpy as np
import pytest
import trimesh

from isaac.assembly_rl import collision_geometry


def test_exact_partition_does_not_fill_annular_bore():
    mesh = trimesh.creation.annulus(r_min=0.003, r_max=0.01, height=0.004, sections=12)
    hulls, report = collision_geometry.decompose(mesh.vertices, mesh.faces)
    assert np.isclose(
        report["source_volume_m3"], report["collision_volume_m3"], rtol=1e-8
    )
    from scipy.spatial import ConvexHull

    axis_points = np.c_[np.zeros(11), np.zeros(11), np.linspace(-0.002, 0.002, 11)]
    for h in hulls:
        equations = ConvexHull(h).equations
        assert not np.any(
            np.all(axis_points @ equations[:, :3].T + equations[:, 3] <= 1e-10, axis=1)
        )


def test_open_surface_is_rejected():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="closed"):
        collision_geometry.decompose(mesh.vertices, mesh.faces)


def test_two_tetrahedra_merge_only_when_union_is_convex():
    vertices = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        * 0.01
    )
    hulls, report = collision_geometry.merge_tetrahedra(
        vertices, np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    )
    assert len(hulls) == 1
    assert report["positive_roundoff_volume_m3"] < 1e-18


def test_enclosed_cavity_is_not_silently_filled():
    outer = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    inner = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    inner.invert()
    mesh = trimesh.util.concatenate([outer, inner])
    with pytest.raises(ValueError, match="Enclosed cavities"):
        collision_geometry.decompose(mesh.vertices, mesh.faces)
