"""Factor repeated section solids without simplifying their mating boundaries."""

import numpy as np
import shapely

from isaac.gantry_ur7 import newton_fixture_geometry as geometry


def _polygons(shape):
    return [
        part
        for part in shapely.get_parts(shape)
        if part.geom_type == "Polygon" and part.area > 0
    ]


def _extrude(shape, low, high):
    hulls = []
    for polygon in _polygons(shape):
        triangles = list(
            shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))
        )
        partition = shapely.union_all(triangles)
        if partition.symmetric_difference(polygon).area > 1e-18:
            raise ValueError("Section triangulation changed the solid or its holes")
        for piece in geometry._merge_convex(triangles, envelope=polygon):
            xy = np.asarray(piece.exterior.coords)[:-1]
            hulls.append(
                np.concatenate(
                    (
                        np.c_[xy, np.full(len(xy), low)],
                        np.c_[xy, np.full(len(xy), high)],
                    )
                )
            )
    return hulls


def shared_core_extrusion(slabs, *, max_group_height_m=0.001):
    """Return Z-axis convex prisms and a report for pre-simplified slab polygons.

    Adjacent slabs share an extruded intersection; each slab retains its exact
    difference from that core. No further simplification or sliver removal occurs.
    A single input slab taller than the group limit is retained unchanged.
    """
    if not np.isfinite(max_group_height_m) or max_group_height_m <= 0:
        raise ValueError("Group height must be positive and finite")
    slabs = sorted(slabs, key=lambda slab: slab[0])
    groups = []
    previous_high = None
    for low, high, polygon in slabs:
        if (
            not np.isfinite([low, high]).all()
            or high <= low
            or not polygon.is_valid
            or polygon.is_empty
            or polygon.geom_type not in ("Polygon", "MultiPolygon")
        ):
            raise ValueError("Expected positive-height slabs with valid solid polygons")
        if previous_high is not None and low < previous_high:
            raise ValueError("Input slab interiors must not overlap")
        if (
            not groups
            or low != previous_high
            or high - groups[-1][0][0] > max_group_height_m
        ):
            groups.append([])
        groups[-1].append((low, high, polygon))
        previous_high = high

    hulls, reports = [], []
    source_volume = 0.0
    for group in groups:
        core = shapely.union_all(
            _polygons(shapely.intersection_all([p for _, _, p in group]))
        )
        low, high = group[0][0], group[-1][1]
        pieces = _extrude(core, low, high)
        core_pieces = len(pieces)
        volume = 0.0
        for bottom, top, polygon in group:
            remainder = polygon.difference(core)
            recovered = shapely.union_all([core, remainder])
            if recovered.symmetric_difference(polygon).area > 1e-18:
                raise ValueError("Common-core factoring changed a section")
            pieces.extend(_extrude(remainder, bottom, top))
            volume += polygon.area * (top - bottom)
        source_volume += volume
        hulls.extend(pieces)
        reports.append(
            {
                "bottom_m": float(low),
                "top_m": float(high),
                "slabs": len(group),
                "common_core_pieces": core_pieces,
                "convex_pieces_before_vertical_merge": len(pieces),
                "common_core_volume_m3": float(core.area * (high - low)),
            }
        )
    merged = geometry._merge_vertical_prisms(hulls)
    return merged, {
        "method": "Shared intersections and exact slab residuals; no boundary simplification or sliver removal",
        "axis": 2,
        "slabs": len(slabs),
        "groups": reports,
        "max_group_height_m": max_group_height_m,
        "input_proxy_volume_m3": float(source_volume),
        "section_symmetric_difference_tolerance_m2": 1e-18,
        "convex_pieces_before_vertical_merge": len(hulls),
        "convex_pieces": len(merged),
    }
