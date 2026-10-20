# mypy: disable-error-code="attr-defined"
"""Convex, hole-preserving extrusion of the fixture's layered CAD tessellation.

Slab extrusion must conservatively cover the source within a certified distance
from solid shared by every height. Each cross-section keeps its holes and
disconnected components; whole-fixture convexification would cap the receiver.
All coordinates and error limits are metres.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
from collections import deque
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import trimesh

# Numerical discretization only: 0.05µm is 1/500 of the CAD fit clearance.
SECTION_ERROR_M = 5e-8
# A coarse hull may fill holes and notches far from contact, never a feature
# this wide: a larger fill needs a finer region, not a looser bound.
COARSE_ERROR_LIMIT_M = 0.01


@dataclasses.dataclass(frozen=True)
class Decomposition:
    prisms: tuple[np.ndarray, ...]
    report: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ContactRegion:
    """Upward-unbounded vertical annulus in the decomposition frame, metres."""

    name: str
    center_xy_m: tuple[float, float]
    inner_radius_m: float
    outer_radius_m: float
    bottom_m: float


@dataclasses.dataclass(frozen=True)
class RegionLimited:
    fine_indices: tuple[int, ...]
    coarse_prisms: tuple[np.ndarray, ...]
    report: dict[str, Any]


def _topology(geometry: Any) -> tuple[int, int]:
    polygons = list(shapely.get_parts(geometry))
    if any(p.geom_type != "Polygon" for p in polygons):
        raise ValueError("fixture section must contain only polygons")
    return len(polygons), sum(len(p.interiors) for p in polygons)


def _edges(polygon: Any) -> Iterator[tuple[tuple[float, float], ...]]:
    coords = list(polygon.exterior.coords)
    for a, b in zip(coords[:-1], coords[1:], strict=True):
        yield tuple(sorted((a, b)))


def _merge_convex(
    pieces: list[Any],
    envelope: Any | None = None,
    *,
    source_boundary: Any | None = None,
    error_limit_m: float = 0,
) -> list[Any]:
    """Merge adjacent groups whose convex hull remains inside a fixed envelope.

    Groups retain their partition edges while merging, so a convex hull cannot
    hide adjacency or accumulate a fresh tolerance at each union.
    """
    if envelope is not None:
        shapely.prepare(envelope)
    if source_boundary is not None:
        shapely.prepare(source_boundary)
    polygons = dict(enumerate(pieces))
    owners: dict[tuple[tuple[float, float], ...], set[int]] = {}
    for index, polygon in polygons.items():
        for edge in _edges(polygon):
            owners.setdefault(edge, set()).add(index)
    queue = deque(edge for edge, indices in owners.items() if len(indices) == 2)
    while queue:
        edge = queue.popleft()
        indices = owners.get(edge, set())
        if len(indices) != 2:
            continue
        a, b = sorted(indices)
        merged = polygons[a].union(polygons[b])
        if merged.geom_type != "Polygon":
            continue
        hull = merged.convex_hull
        if envelope is not None:
            if not envelope.covers(hull):
                continue
            if (
                source_boundary is not None
                and error_limit_m > 0
                and source_boundary.intersects(hull.buffer(-error_limit_m))
            ):
                continue
        elif (
            hull.difference(merged).area > 1e-18
            or shapely.hausdorff_distance(hull.boundary, merged.boundary) > 1e-12
        ):
            continue
        for index in (a, b):
            for old_edge in _edges(polygons.pop(index)):
                owners[old_edge].remove(index)
        polygons[a] = merged
        for new_edge in _edges(merged):
            owners.setdefault(new_edge, set()).add(a)
            if len(owners[new_edge]) == 2:
                queue.append(new_edge)
    return [polygon.convex_hull for polygon in polygons.values()]


def _clip_height(points: np.ndarray, height: float, *, above: bool) -> np.ndarray:
    clipped = []
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
        first_inside = first[2] >= height if above else first[2] <= height
        second_inside = second[2] >= height if above else second[2] <= height
        if first_inside:
            clipped.append(first)
        if first_inside != second_inside:
            fraction = (height - first[2]) / (second[2] - first[2])
            clipped.append(first + fraction * (second - first))
    return np.asarray(clipped).reshape(-1, 3)


def _wall_sweep(mesh: trimesh.Trimesh, bottom: float, top: float) -> list[Any]:
    """Project slab-clipped side faces, bounding every change of occupied XY.

    Source tessellation can have nanometre wall taper. Between vertex heights,
    only crossing a side face can change whether an XY point belongs to solid.
    The swept boundary therefore bounds every section, without an angular
    tolerance that could silently accept a real chamfer.
    """
    triangles = mesh.triangles
    lower, upper = triangles[:, :, 2].min(axis=1), triangles[:, :, 2].max(axis=1)
    candidates = triangles[(lower < top) & (upper > bottom) & (upper > lower)]
    swept = []
    for triangle in candidates:
        points = _clip_height(triangle, bottom, above=True)
        points = _clip_height(points, top, above=False)
        if len(points) < 3:
            continue
        projection = shapely.geometry.MultiPoint(
            np.round(points[:, :2], 10)
        ).convex_hull
        # Clipping and projection preserve convexity. Rebuilding the hull also
        # removes repeated or collinear vertices from exactly vertical walls.
        if not projection.is_empty:
            swept.append(projection)
    return swept


def _slab_error_bound(
    original: Any, section: Any, swept: list[Any], limit_m: float
) -> tuple[float, float]:
    """Certify source containment and distance to solid present at every height."""
    if not section.covers(original) or (
        swept and not np.all(shapely.covers(section, swept))
    ):
        raise ValueError("fixture sloped surfaces exceed conservative section coverage")
    sweep = shapely.union_all([piece for piece in swept if piece.area > 0])
    core = original.difference(sweep)
    if core.is_empty:
        raise ValueError("fixture sloped surfaces leave no bounded slab core")
    if core.covers(section):
        return 0.0, float(sweep.area)

    # Round buffering is contained in the exact Euclidean distance offset, so
    # coverage certifies the bound even where the section has concave corners.
    source_boundaries = [original.boundary, *swept]

    def covered(distance: float) -> bool:
        if not core.buffer(distance, join_style="round").covers(section):
            return False
        # Solid distance alone would allow a long, submicron notch to disappear.
        # Every projected source wall must also remain near the output boundary.
        tube = section.boundary.buffer(distance, join_style="round")
        shapely.prepare(tube)
        return bool(np.all(shapely.covers(tube, source_boundaries)))

    if not covered(limit_m):
        raise ValueError("fixture sloped surfaces exceed the section error budget")
    return limit_m, float(sweep.area)


def _merge_vertical_prisms(prisms: list[np.ndarray]) -> list[np.ndarray]:
    """Join identical convex footprints over touching height intervals exactly."""
    groups: dict[bytes, tuple[np.ndarray, list[tuple[float, float]]]] = {}
    for points in prisms:
        xy = np.unique(points[:, :2], axis=0)
        key = xy.tobytes()
        if key not in groups:
            groups[key] = (xy, [])
        groups[key][1].append((float(points[:, 2].min()), float(points[:, 2].max())))
    result = []
    for xy, intervals in groups.values():
        combined: list[tuple[float, float]] = []
        for bottom, top in sorted(intervals):
            if combined and bottom <= combined[-1][1]:
                combined[-1] = (combined[-1][0], max(top, combined[-1][1]))
            else:
                combined.append((bottom, top))
        for bottom, top in combined:
            result.append(
                np.concatenate(
                    (
                        np.c_[xy, np.full(len(xy), bottom)],
                        np.c_[xy, np.full(len(xy), top)],
                    )
                )
            )
    return result


def decompose(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    section_error_m: float = SECTION_ERROR_M,
    boundary_error_limit_m: float | None = None,
    merge_convex: bool = True,
) -> Decomposition:
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or triangles.dtype.kind not in "iu"
        or np.any(triangles < 0)
        or np.any(triangles >= len(vertices))
    ):
        raise ValueError("fixture decomposition requires finite vertices and triangles")
    if not np.isfinite(section_error_m) or not 0 <= section_error_m <= 2e-7:
        raise ValueError("fixture section error must be within 0 to 0.2 micrometres")
    error_limit = (
        min(3 * section_error_m, 2e-7)
        if boundary_error_limit_m is None
        else boundary_error_limit_m
    )
    if not np.isfinite(error_limit) or not 0 <= error_limit <= 2e-7:
        raise ValueError("fixture boundary error must be within 0 to 0.2 micrometres")
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError(
            "fixture decomposition requires a closed consistently wound solid"
        )
    source_volume = abs(mesh.volume)
    # USD float32 vertices can differ at nominally coplanar surfaces. The
    # half-nanometre Z rounding is recorded separately from the XY error bound.
    rounded = mesh.vertices.copy()
    rounded[:, 2] = np.round(rounded[:, 2], 9)
    mesh.vertices = rounded
    levels = np.unique(rounded[:, 2])
    if len(levels) < 2:
        raise ValueError("fixture has no solid height")
    prisms = []
    rows = []
    for bottom, top in zip(levels[:-1], levels[1:], strict=True):
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, (bottom + top) / 2]
        )
        if not len(lines):
            continue
        # Endpoint quantization joins numerical copies of the same edge. Its
        # maximum XY displacement is sqrt(2)*0.05nm and is included in the report.
        edges = shapely.geometry.MultiLineString(np.round(lines[:, :, :2], 10).tolist())
        original = shapely.build_area(shapely.node(edges))
        if original.is_empty or not original.is_valid:
            raise ValueError("fixture section is open or invalid")
        simplified = shapely.simplify(original, section_error_m, preserve_topology=True)
        # Outward buffering prevents a simplified chord from cutting away solid.
        # Actual containment and the error bound are checked below.
        section = (
            simplified.buffer(section_error_m * 1.02, join_style="mitre")
            if section_error_m
            else simplified
        )
        boundary_error = float(
            shapely.hausdorff_distance(original.boundary, section.boundary)
        )
        area_error = float(original.symmetric_difference(section).area)
        if (
            not section.is_valid
            or not section.covers(original)
            or _topology(section) != _topology(original)
            or boundary_error > error_limit + 1e-14
            or area_error > original.length * 3 * section_error_m + 1e-15
        ):
            raise ValueError(
                "fixture section simplification exceeds its geometric error bound"
            )
        swept = _wall_sweep(mesh, bottom, top)
        certified_error, sweep_area = _slab_error_bound(
            original, section, swept, error_limit
        )
        core = original.difference(
            shapely.union_all([piece for piece in swept if piece.area > 0])
        )
        envelope = core.buffer(error_limit * 0.95, join_style="round")
        triangles_2d = list(
            shapely.get_parts(shapely.constrained_delaunay_triangles(section))
        )
        area = sum(piece.area for piece in triangles_2d)
        if not np.isclose(area, section.area, rtol=1e-10, atol=1e-15):
            raise ValueError("fixture triangulation lost or filled source area")
        shapely.prepare(section)
        if not np.all(shapely.covers(section, triangles_2d)) or any(
            piece.area <= 0 for piece in triangles_2d
        ):
            raise ValueError("fixture triangulation crosses a hole boundary")
        pieces = (
            _merge_convex(
                triangles_2d,
                envelope=envelope,
                source_boundary=original.boundary,
                error_limit_m=error_limit,
            )
            if merge_convex
            else triangles_2d
        )
        actual_section = shapely.union_all(pieces)
        if _topology(actual_section) != _topology(original):
            raise ValueError("fixture convex merging changed section topology")
        certified_error, sweep_area = _slab_error_bound(
            original, actual_section, swept, error_limit
        )
        section = actual_section
        boundary_error = float(
            shapely.hausdorff_distance(original.boundary, section.boundary)
        )
        area_error = float(original.symmetric_difference(section).area)
        for piece in pieces:
            points = np.asarray(piece.exterior.coords)[:-1]
            prisms.append(
                np.concatenate(
                    (
                        np.c_[points, np.full(len(points), bottom)],
                        np.c_[points, np.full(len(points), top)],
                    )
                )
            )
        rows.append(
            {
                "bottom_m": float(bottom),
                "top_m": float(top),
                "source_area_m2": float(original.area),
                "area_m2": float(section.area),
                "boundary_error_m": certified_error,
                "midsection_boundary_error_m": boundary_error,
                "side_face_sweep_area_m2": sweep_area,
                "symmetric_difference_area_m2": area_error,
                "components": _topology(section)[0],
                "holes": _topology(section)[1],
                "triangles": len(triangles_2d),
                "convex_prisms": len(pieces),
            }
        )
    volume = sum(row["area_m2"] * (row["top_m"] - row["bottom_m"]) for row in rows)
    source_section_volume = sum(
        row["source_area_m2"] * (row["top_m"] - row["bottom_m"]) for row in rows
    )
    volume_rounding_bound = sum(row["source_area_m2"] for row in rows) * 1e-9 + 1e-13
    if abs(source_section_volume - source_volume) > volume_rounding_bound:
        raise ValueError("fixture cross sections do not preserve the closed CAD volume")
    before_vertical_merge = len(prisms)
    prisms = _merge_vertical_prisms(prisms)
    return Decomposition(
        tuple(prisms),
        {
            "method": "CAD slab sections with exact projected side-face containment and common-solid distance bound; conservative simplification; constrained triangulation; adjacent convex hulls confined to the same certified section envelope",
            "source_volume_m3": float(source_volume),
            "decomposed_volume_m3": volume,
            "section_error_limit_m": section_error_m,
            "boundary_error_limit_m": error_limit,
            "section_endpoint_error_bound_m": float(np.sqrt(2) * 0.5e-10),
            "height_rounding_error_bound_m": 0.5e-9,
            "slabs": rows,
            "convex_prisms": len(prisms),
            "convex_prisms_before_vertical_merge": before_vertical_merge,
            "exact_vertical_merges": before_vertical_merge - len(prisms),
        },
    )


def _cache_key(
    vertices: np.ndarray, triangles: np.ndarray, options: dict[str, Any]
) -> str:
    digest = hashlib.sha256()
    # GEOS/shapely/trimesh numerics decide the certified output, so a library
    # or algorithm change must miss rather than replay an older certificate.
    for text in (
        pathlib.Path(__file__).read_text(),
        shapely.__version__,
        shapely.geos_version_string,
        trimesh.__version__,
        np.__version__,
        json.dumps(options, sort_keys=True),
    ):
        if text is None:
            raise RuntimeError("an unversioned library cannot key the fixture cache")
        digest.update(text.encode() + b"\0")
    for array, dtype in ((vertices, "<f8"), (triangles, "<i8")):
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        digest.update(repr(contiguous.shape).encode() + contiguous.tobytes())
    return digest.hexdigest()


def _load_cached(path: pathlib.Path, key: str) -> Decomposition | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            points, counts = data["points"], data["counts"]
            report = json.loads(str(data["report"]))
    except Exception:
        # Any unreadable entry is a miss: empty or truncated files raise
        # EOFError/BadZipFile, and flipped bytes also reach numpy's header
        # parser (TokenError) and zipfile's method switch (NotImplementedError).
        # The member CRCs stop a corrupt entry from loading as valid data.
        return None
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or counts.ndim != 1
        or int(counts.sum()) != len(points)
        or np.any(counts < 1)
    ):
        return None
    return Decomposition(tuple(np.split(points, np.cumsum(counts)[:-1])), report)


def cached_decompose(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    cache_dir: pathlib.Path | None,
    **options: Any,
) -> Decomposition:
    """Reuse a certified decomposition of identical inputs across episodes.

    Every recording rebuilds its model and the fixture CAD never changes, while
    one decomposition costs minutes of GEOS work. A miss or unreadable entry
    recomputes, so the cache can only save time, never change the result.
    """
    if cache_dir is None:
        return decompose(vertices, triangles, **options)
    key = _cache_key(vertices, triangles, options)
    path = cache_dir / f"{key}.npz"
    cached = _load_cached(path, key)
    if cached is not None:
        return cached
    result = decompose(vertices, triangles, **options)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                key=np.array(key),
                points=np.concatenate(result.prisms),
                counts=np.array([len(prism) for prism in result.prisms]),
                report=np.array(json.dumps(result.report, sort_keys=True)),
            )
            # Durable before the rename, so a crash cannot publish an empty entry.
            stream.flush()
            os.fsync(stream.fileno())
        # Concurrent recorders may race on a miss; both write identical bytes.
        os.replace(temporary, path)
    except OSError:
        pass  # A read-only or full cache disk only costs the recomputation.
    return result


def _reaches(region: ContactRegion, footprint: Any, bottom: float, top: float) -> bool:
    """Exact annulus test without polygonizing the circles.

    Radii over a connected footprint form one interval from its nearest point
    to its farthest exterior vertex, so interval overlap decides intersection.
    """
    if top <= region.bottom_m:
        return False
    center = shapely.geometry.Point(region.center_xy_m)
    return any(
        part.distance(center) <= region.outer_radius_m
        and np.max(
            np.linalg.norm(
                np.asarray(part.exterior.coords) - region.center_xy_m, axis=1
            )
        )
        >= region.inner_radius_m
        for part in shapely.get_parts(footprint)
    )


def _outward_error(section: Any, cover: Any) -> float:
    """Certified bound on how far a covering hull reaches beyond the section.

    A bound is accepted only when the buffered solid covers the filled piece.
    Round buffers lie inside the Euclidean offset, and clipping to the piece's
    bounds only removes solid, so every accepted value is an upper bound. The
    inscribed circle is a lower bound that makes a filled hole one check.
    """
    bound = 0.0
    for piece in shapely.get_parts(cover.difference(section)):
        # The nearest solid lies on the piece's closure; the micrometre margin
        # keeps solid that only touches the bounds from clipping to nothing.
        x0, y0, x1, y1 = piece.bounds
        near = shapely.clip_by_rect(section, x0 - 1e-6, y0 - 1e-6, x1 + 1e-6, y1 + 1e-6)
        low = (
            float(shapely.maximum_inscribed_circle(piece, 1e-6).length)
            if piece.area > 0
            else 0.0
        )
        high = low + 1e-6
        while not near.buffer(high).covers(piece):
            if high > COARSE_ERROR_LIMIT_M:
                raise ValueError("fixture coarse hull fills a feature beyond its limit")
            low, high = high, 2 * high
        while high - low > 1e-6:
            middle = (low + high) / 2
            if near.buffer(middle).covers(piece):
                high = middle
            else:
                low = middle
        bound = max(bound, high)
    return bound


def limit_to_regions(
    decomposition: Decomposition, regions: Sequence[ContactRegion]
) -> RegionLimited:
    """Keep certified slabs whose hull a contact region reaches; hull the rest.

    Selection is per slab, so a kept slab retains every certified prism byte
    for byte and a replaced slab leaves no partial seam. Testing the hull, not
    the section, keeps any slab whose fill would enter a region. A vertically
    merged prism kept for one slab also keeps every other slab it spans, so no
    hull overlaps kept geometry or fills a notch it leaves open. Each section
    component of a replaced slab becomes its convex hull prism, filling holes
    and notches that no modelled contact reaches.
    """
    footprints = [
        shapely.geometry.MultiPoint(points[:, :2]).convex_hull
        for points in decomposition.prisms
    ]
    spans = [
        (points[:, 2].min(), points[:, 2].max()) for points in decomposition.prisms
    ]
    slabs = []
    for slab in decomposition.report["slabs"]:
        bottom, top = slab["bottom_m"], slab["top_m"]
        active = {
            index
            for index, (low, high) in enumerate(spans)
            if low <= bottom and high >= top
        }
        section = shapely.union_all([footprints[index] for index in sorted(active)])
        pieces = [part.convex_hull for part in shapely.get_parts(section)]
        cover = shapely.union_all(pieces)
        reached = [
            region.name for region in regions if _reaches(region, cover, bottom, top)
        ]
        slabs.append((slab, active, section, pieces, cover, reached))
    # Keeping a merged prism keeps every slab it spans, which can keep more.
    fine: set[int] = set()
    keep = {i for i, (*_, reached) in enumerate(slabs) if reached}
    while True:
        fine = set().union(*(slabs[i][1] for i in keep))
        spanned = {i for i, row in enumerate(slabs) if row[1] & fine}
        if spanned <= keep:
            break
        keep |= spanned
    hulls = []
    kept = []
    replaced = []
    for i, (slab, active, section, pieces, cover, reached) in enumerate(slabs):
        bottom, top = slab["bottom_m"], slab["top_m"]
        row = {"bottom_m": bottom, "top_m": top, "certified_prisms": len(active)}
        if i in keep:
            kept.append({**row, "regions": reached})
            continue
        if not cover.covers(section):
            raise ValueError("fixture coarse hulls do not contain their section")
        error = _outward_error(section, cover) + slab["boundary_error_m"]
        if error > COARSE_ERROR_LIMIT_M:
            raise ValueError("fixture coarse hull fills a feature beyond its limit")
        for piece in pieces:
            xy = np.asarray(piece.exterior.coords)[:-1]
            hulls.append(
                np.concatenate(
                    (
                        np.c_[xy, np.full(len(xy), bottom)],
                        np.c_[xy, np.full(len(xy), top)],
                    )
                )
            )
        replaced.append(
            {**row, "hull_prisms": len(pieces), "outward_error_bound_m": error}
        )
    coarse = _merge_vertical_prisms(hulls)
    return RegionLimited(
        tuple(sorted(fine)),
        tuple(coarse),
        {
            "method": "certified slabs whose convex hull reaches a contact region kept byte-identical; every other slab replaced by the convex hull prism of each section component, verified to contain the certified section",
            "regions": [dataclasses.asdict(region) for region in regions],
            "certified_prisms": len(decomposition.prisms),
            "fine_prisms": len(fine),
            "coarse_prisms": len(coarse),
            "fine_slabs": kept,
            "coarse_slabs": replaced,
            "coarse_outward_error_bound_m": max(
                (row["outward_error_bound_m"] for row in replaced), default=0.0
            ),
            "coarse_error_limit_m": COARSE_ERROR_LIMIT_M,
        },
    )


def slab_prisms(vertices: np.ndarray, triangles: np.ndarray) -> Iterator[np.ndarray]:
    yield from decompose(vertices, triangles).prisms
