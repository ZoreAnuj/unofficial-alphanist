"""Source section extrusion preserving apertures, with explicit error reports."""

import collections
import itertools

import numpy as np
import scipy.spatial
import shapely
import shapely.geometry
import trimesh

from isaac.gantry_ur7 import newton_fixture_geometry as _g


def section(mesh, axis, height):
    normal = np.eye(3)[axis]
    origin = normal * height
    lines = trimesh.intersections.mesh_plane(mesh, normal, origin)
    axes = [i for i in range(3) if i != axis]
    edges = shapely.geometry.MultiLineString(np.round(lines[:, :, axes], 10).tolist())
    result = shapely.build_area(shapely.node(edges))
    if result.is_empty or not result.is_valid:
        raise ValueError("Invalid source section")
    return result


def extrusion(polygon, axis, low, high, error=1e-05):
    axes = [i for i in range(3) if i != axis]
    simplified = polygon.simplify(error, preserve_topology=True)
    err = shapely.hausdorff_distance(polygon.boundary, simplified.boundary)
    if err > error * 1.01:
        simplified = polygon
        err = 0.0
    triangles = list(
        shapely.get_parts(shapely.constrained_delaunay_triangles(simplified))
    )
    if abs(sum(x.area for x in triangles) - simplified.area) > 1e-12:
        raise ValueError("Section triangulation filled holes")
    pieces = _g._merge_convex(triangles)
    hulls = []
    discarded_volume = 0.0
    for p in pieces:
        p = p.simplify(1e-8, preserve_topology=True)
        if p.area * (high - low) < 1e-15 or 2 * p.area / p.length < 1e-8:
            discarded_volume += p.area * (high - low)
            continue
        xy = np.asarray(p.exterior.coords)[:-1]
        pts = np.zeros((2 * len(xy), 3))
        pts[: len(xy), axes] = xy
        pts[len(xy) :, axes] = xy
        pts[: len(xy), axis] = low
        pts[len(xy) :, axis] = high
        hulls.append(pts)
    return (
        hulls,
        {
            "section_boundary_error_m": float(err),
            "holes": sum(len(p.interiors) for p in shapely.get_parts(polygon)),
            "convex_pieces": len(hulls),
            "discarded_numeric_sliver_volume_m3": discarded_volume,
        },
    )


def bracket(mesh):
    """Partition the L bracket into its two leaves and side reinforcements."""
    hulls = []
    reports = []
    recipes = [
        (0, 0.0015, 0, 0.003, shapely.geometry.box(-1, 0.003, 1, 1)),
        (1, 0.025, 0, 0.05, shapely.geometry.box(0, -1, 0.003, 0.003)),
        (2, 0.0015, 0, 0.003, shapely.geometry.box(-1, -1, 0, 1)),
        (1, 0.0015, 0, 0.003, shapely.geometry.box(-1, 0.003, 0, 1)),
        (1, 0.0485, 0.047, 0.05, shapely.geometry.box(-1, 0.003, 0, 1)),
        (1, 0.025, 0.003, 0.047, shapely.geometry.box(-1, 0.003, 0, 1)),
    ]
    for axis, height, low, high, clip in recipes:
        polygon = section(mesh, axis, height)
        if clip is not None:
            polygon = shapely.union_all(
                [
                    p
                    for p in shapely.get_parts(polygon.intersection(clip))
                    if p.geom_type == "Polygon"
                ]
            )
        if polygon.is_empty:
            continue
        hs, r = extrusion(polygon, axis, low, high)
        hulls.extend(hs)
        reports.append(r)
    return (
        hulls,
        {
            "method": "Source mid-plane sections of bracket leaves, reinforcements and inner bend; constrained polygon triangulation retains holes",
            "sections": reports,
            "scope": "Needs source-to-proxy surface deviation and assembled swept-fit validation",
        },
    )


def convex_mesh(points):
    """Orient SciPy hull facets without deleting skinny source triangles."""
    points = np.asarray(points)
    hull = scipy.spatial.ConvexHull(points)
    faces = hull.simplices.copy()
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    mesh = trimesh.Trimesh(points, faces, process=False)
    # Nearly collinear facets can have a zero numeric normal; edge adjacency
    # fixes their orientation without deleting any source boundary triangle.
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    if not mesh.is_volume:
        raise ValueError("Convex partition generated an invalid solid")
    return mesh


def union_mesh(hulls):
    return trimesh.boolean.union([convex_mesh(p) for p in hulls], engine="manifold")


def layered(mesh, axis=2, error=1e-05):
    axes = [i for i in range(3) if i != axis]
    levels = np.unique(np.round(mesh.vertices[:, axis], 8))
    hulls = []
    reports = []
    for low, high in itertools.pairwise(levels):
        if high - low < 1e-07:
            continue
        polygon = section(mesh, axis, (low + high) / 2)
        hs, r = extrusion(polygon, axis, low, high, error)
        hulls.extend(hs)
        reports.append(r)
    order = axes + [axis]
    merged = _g._merge_vertical_prisms([p[:, order] for p in hulls])
    inverse = np.argsort(order)
    return (
        [p[:, inverse] for p in merged],
        {
            "method": "Source vertex-height slab midsections, holes retained; approximation of sloped surfaces requires separate validation",
            "slabs": len(reports),
            "section_error_m": error,
        },
    )


def common_sections(
    mesh, axis, levels, *, error=1e-5, sample_step=0.00025, relief=None
):
    """Extrude common solid across sampled source sections, preserving apertures.

    This is an explicitly sampled inner approximation, not a certified surface
    bound. It cannot replace mating-path validation and source-volume checks.
    """
    hulls, reports = [], []
    for low, high in itertools.pairwise(levels):
        if high - low < 1e-8:
            continue
        count = max(3, int(np.ceil((high - low) / sample_step)) + 1)
        sample_z = np.linspace(
            low + min(1e-7, (high - low) / 100),
            high - min(1e-7, (high - low) / 100),
            count,
        )
        polygons = [section(mesh, axis, float(z)) for z in sample_z]
        common = shapely.intersection_all(polygons)
        common = shapely.union_all(
            [
                p
                for p in shapely.get_parts(common)
                if p.geom_type == "Polygon" and p.area > 1e-12
            ]
        )
        if common.is_empty:
            raise ValueError("No common solid across section slab")
        if relief is not None:
            common = relief(common, low, high)
        pieces, report = extrusion(common, axis, low, high, error=error)
        hulls.extend(pieces)
        reports.append(
            {
                **report,
                "bottom_m": float(low),
                "top_m": float(high),
                "source_section_samples": count,
            }
        )
    order = [i for i in range(3) if i != axis] + [axis]
    inverse = np.argsort(order)
    merged = _g._merge_vertical_prisms([p[:, order] for p in hulls])
    return [p[:, inverse] for p in merged], {
        "method": "Convex source-section common-solid extrusion; sampled inner approximation with holes preserved",
        "section_error_m": error,
        "max_section_sample_step_m": sample_step,
        "slabs": reports,
    }


def merge_convex_partition(pieces):
    """Merge adjoining partition solids only when hull and sum volumes agree.

    Input interiors must be disjoint (as with section triangulation and clipping).
    Coordinates are welded at 0.1 nm; holes and concavities are retained by the
    convex-union volume test. This is not a decomposition of overlapping shapes.
    """
    solids = {i: np.unique(np.round(p, 10), axis=0) for i, p in enumerate(pieces)}
    volumes = {i: scipy.spatial.ConvexHull(p).volume for i, p in solids.items()}
    for _ in range(16):
        vertices = collections.defaultdict(list)
        for i, points in solids.items():
            for point in points:
                vertices[tuple(point)].append(i)
        pairs = collections.Counter(
            pair for ids in vertices.values() for pair in itertools.combinations(ids, 2)
        )
        merged = 0
        for (a, b), shared in sorted(pairs.items(), key=lambda item: -item[1]):
            if shared < 3 or a not in solids or b not in solids:
                continue
            points = np.unique(np.concatenate([solids[a], solids[b]]), axis=0)
            hull = scipy.spatial.ConvexHull(points)
            volume = volumes[a] + volumes[b]
            if abs(hull.volume - volume) > max(1e-20, 1e-8 * volume):
                continue
            solids[a] = points[hull.vertices]
            volumes[a] = volume
            del solids[b], volumes[b]
            merged += 1
        if not merged:
            break
    return list(solids.values())
