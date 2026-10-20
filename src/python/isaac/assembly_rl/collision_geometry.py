"""Boundary-preserving convex partition of watertight triangle solids."""

import heapq

import numpy as np
import scipy.spatial
import trimesh


def merge_tetrahedra(vertices, elements, max_attempts=500000):
    """Merge adjacent solids only when their convex hull has the same volume.

    Coordinates are rescaled to millimetres for QHull conditioning. No geometric
    simplification tolerance is allowed; the relative tolerance is roundoff.
    """
    points = np.asarray(vertices, dtype=float) * 1000
    elements = np.asarray(elements, dtype=np.int64)
    q = points[elements]
    volumes = np.abs(np.linalg.det(q[:, 1:] - q[:, :1])) / 6
    if np.any(volumes <= 0):
        raise ValueError("Tetrahedralization contains degenerate cells")
    groups = {i: set(e) for i, e in enumerate(elements)}
    neighbors = {i: set() for i in groups}
    owners = {}
    for i, cell in enumerate(elements):
        for j in range(4):
            face = tuple(sorted(np.delete(cell, j)))
            if face in owners:
                other = owners.pop(face)
                neighbors[i].add(other)
                neighbors[other].add(i)
            else:
                owners[face] = i
    volume = dict(enumerate(volumes.tolist()))
    versions = {i: 0 for i in groups}
    queue = []

    def enqueue(a, b):
        a, b = sorted((a, b))
        heapq.heappush(
            queue, (-(volume[a] + volume[b]), a, b, versions[a], versions[b])
        )

    for a, ns in neighbors.items():
        for b in ns:
            if a < b:
                enqueue(a, b)
    attempts = 0
    roundoff_added = 0.0
    while queue and attempts < max_attempts:
        _, a, b, va, vb = heapq.heappop(queue)
        if a not in groups or b not in groups or va != versions[a] or vb != versions[b]:
            continue
        attempts += 1
        indices = np.array(sorted(groups[a] | groups[b]))
        try:
            hull = scipy.spatial.ConvexHull(points[indices])
        except scipy.spatial.QhullError:
            continue
        expected = volume[a] + volume[b]
        error = hull.volume - expected
        if abs(error) > max(expected * 1e-11, 1e-12):
            continue
        roundoff_added += max(0.0, error)
        groups[a] = set(indices[hull.vertices])
        groups.pop(b)
        volume[a] = hull.volume
        volume.pop(b)
        ns = (neighbors[a] | neighbors[b]) - {a, b}
        for n in ns:
            neighbors[n].discard(a)
            neighbors[n].discard(b)
            neighbors[n].add(a)
        neighbors[a] = ns
        neighbors.pop(b)
        versions[a] += 1
        for n in ns:
            enqueue(a, n)
    hulls = [points[sorted(group)] / 1000 for group in groups.values()]
    return hulls, {
        "tetrahedra": len(elements),
        "convex_pieces": len(hulls),
        "merge_attempts": attempts,
        "merge_attempt_limit": max_attempts,
        "positive_roundoff_volume_m3": roundoff_added / 1e9,
    }


def decompose(vertices, faces):
    import tetgen

    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    # Remove only zero-area isolated triangles introduced by CAD tessellation.
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Collision input must be a closed consistently wound solid")
    components = mesh.split(only_watertight=False)
    hulls, reports = [], []
    source_volume = 0.0
    for component in components:
        if not component.is_watertight or not component.is_winding_consistent:
            raise ValueError(
                "Collision input must be a closed consistently wound solid"
            )
        if component.volume <= 0:
            raise ValueError(
                "Enclosed cavities require explicit TetGen hole seeds; refusing to fill them"
            )
        generator = tetgen.TetGen(
            np.asarray(component.vertices), np.asarray(component.faces)
        )
        nodes, cells, _, _ = generator.tetrahedralize(
            quality=False, nobisect=True, quiet=True
        )
        q = nodes[cells]
        volume = np.abs(np.linalg.det(q[:, 1:] - q[:, :1])).sum() / 6
        if not np.isclose(volume, component.volume, rtol=1e-8, atol=1e-16):
            raise ValueError(
                "Constrained tetrahedralization changed solid volume; cavities may be filled"
            )
        pieces, report = merge_tetrahedra(nodes, cells)
        hulls.extend(pieces)
        source_volume += component.volume
        reports.append(report)
    if not hulls:
        raise ValueError("No solid collision pieces")
    volume = sum(scipy.spatial.ConvexHull(p).volume for p in hulls)
    if not np.isclose(volume, source_volume, rtol=1e-8, atol=1e-15):
        raise ValueError("Convex merging changed solid volume")
    return hulls, {
        "method": "TetGen constrained PLC tetrahedralization with no boundary bisection; volume-certified adjacent convex merges",
        "tetgen_version": tetgen.__version__,
        "source_volume_m3": source_volume,
        "collision_volume_m3": volume,
        "components": reports,
    }
