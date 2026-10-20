"""Reproducible collision-only mating relief for four industrial assembly tasks.

Source USD visuals are copied unchanged. Geometry readiness certifies the recorded
straight insertion paths, not dynamic contact, graspability or training throughput.
"""

import argparse
import hashlib
import itertools
import json
import pathlib

import numpy as np
import scipy.spatial
import shapely
import shapely.affinity
import trimesh

from isaac.assembly_rl import assets
from isaac.assembly_rl import collision_sections as cs

TASKS = ("03_shaft_coupling", "04_linear_guide", "05_enclosure", "06_gear_train")
MASS_KEYS = {
    "NEMA17": "motor",
    "FlexibleCoupling": "coupling",
    "MGN7Rail": "rail_mgn7",
    "MGN7Carriage": "carriage_detailed",
    "EnclosureBody": "enclosure",
    "EnclosureLid": "lid",
}


def repair_tiny_triangles(mesh, maximum_diameter=1e-5):
    """Close only isolated triangular tessellation defects under an explicit bound."""
    mesh = mesh.copy()
    boundary = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse) == 1]
    adjacent = {}
    for a, b in boundary:
        adjacent.setdefault(int(a), set()).add(int(b))
        adjacent.setdefault(int(b), set()).add(int(a))
    patches, reports = [], []
    while adjacent:
        start = next(iter(adjacent))
        group, queue = {start}, [start]
        while queue:
            for neighbor in adjacent[queue.pop()]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        if len(group) != 3 or any(len(adjacent[i]) != 2 for i in group):
            raise ValueError(
                "Only isolated tiny triangular source defects may be repaired"
            )
        face = sorted(group)
        points = mesh.vertices[face]
        diameter = float(
            np.max(np.linalg.norm(points[:, None] - points[None, :], axis=-1))
        )
        if diameter > maximum_diameter:
            raise ValueError("Source defect exceeds repair diameter bound")
        if np.any(np.all(mesh.edges == face[:2], axis=1)):
            face.reverse()
        patches.append(face)
        reports.append({"diameter_m": diameter})
        for vertex in group:
            adjacent.pop(vertex)
    if patches:
        mesh.faces = np.concatenate([mesh.faces, patches])
    if not mesh.is_volume:
        raise ValueError("Source remains invalid after bounded tessellation repair")
    return mesh, reports


def clip_convex(points, equations):
    """Intersect a convex solid with halfspaces without filling any concavity."""
    for plane in equations:
        distances = points @ plane[:3] + plane[3]
        inside = distances <= 1e-10
        if inside.all():
            continue
        if not inside.any():
            return None
        hull = scipy.spatial.ConvexHull(points)
        edges = np.unique(
            np.sort(
                np.concatenate(
                    [hull.simplices[:, pair] for pair in ([0, 1], [0, 2], [1, 2])]
                ),
                axis=1,
            ),
            axis=0,
        )
        edges = edges[inside[edges[:, 0]] != inside[edges[:, 1]]]
        a, b = edges.T
        crosses = (
            points[a]
            + (points[b] - points[a])
            * (distances[a] / (distances[a] - distances[b]))[:, None]
        )
        points = np.unique(
            np.round(np.concatenate([points[inside], crosses]), 12), axis=0
        )
        if len(points) < 4 or np.linalg.matrix_rank(points - points[0], tol=1e-11) < 3:
            return None
    return points


def rail_geometry(mesh):
    hulls = []
    for low, high, sample_z in ((0, 0.0025, 0.001), (0.0025, 0.0048, 0.0042)):
        source_section = cs.section(mesh, 2, sample_z)
        holes = [
            shapely.Polygon(ring)
            for p in shapely.get_parts(source_section)
            for ring in p.interiors
        ]
        if len(holes) != 7:
            raise ValueError("Expected seven source rail mounting holes")
        polygon = shapely.box(-0.0035, -0.1, 0.0035, 0).difference(
            shapely.union_all(holes)
        )
        pieces, _ = cs.extrusion(polygon, 2, low, high, error=1e-5)
        hulls.extend(pieces)
    report = {
        "method": "Source axial raceway extrusion intersected with two source mounting-hole slab sections; all seven through bores and counterbores retained",
        "section_error_m": 1e-5,
        "approximation": "Mounting-hole end chamfers omitted; axial raceway follows source section",
    }
    profile = cs.section(mesh, 1, -0.0575)
    envelopes, _ = cs.extrusion(profile, 1, -0.100000001, 0.000000001, error=1e-6)
    clipped = []
    for hull in hulls:
        for envelope in envelopes:
            if np.any(hull.max(0) < envelope.min(0)) or np.any(
                hull.min(0) > envelope.max(0)
            ):
                continue
            part = clip_convex(hull, scipy.spatial.ConvexHull(envelope).equations)
            if part is not None and scipy.spatial.ConvexHull(part).volume > 1e-15:
                clipped.append(part)
    report["raceway_correction"] = (
        "Slabs intersected with source axial raceway profile; retains mounting through-holes/counterbores and removes slab overhang into raceways"
    )
    return clipped, report


def recipe(name, mesh, source_meshes):
    if name == "NEMA17":
        return cs.layered(mesh, error=2.5e-5)
    if name == "FlexibleCoupling":
        sides = 96
        angles = np.arange(sides) * 2 * np.pi / sides
        radius = 0.00255 / np.cos(np.pi / sides)
        cutter = shapely.Polygon(
            np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        )

        def relief(polygon, low, high):
            return polygon.difference(cutter) if low < 0.0114 else polygon

        hulls, report = cs.common_sections(
            mesh,
            2,
            [0, 0.00025, 0.0005, 0.0114, 0.0184, 0.0224, 0.0333, 0.03355, 0.0338],
            sample_step=0.0005,
            relief=relief,
        )
        report["engineering_relief"] = {
            "description": "Lower 5 mm source shaft bore enlarged to 5.10 mm; circumscribed 96-sided aperture. Flexure slots retained as rigid geometry; clamp screws not actuated.",
            "nominal_radial_clearance_m": 0.00005,
        }
        return hulls, report
    if name == "MGN7Rail":
        return rail_geometry(mesh)
    if name == "MGN7Carriage":
        profile = cs.section(source_meshes["MGN7Rail"], 1, -0.0575)
        cutter = shapely.affinity.translate(profile, yoff=-0.0015).buffer(
            0.00005, join_style="mitre"
        )
        levels = [
            -0.01145,
            -0.01125,
            -0.01025,
            -0.00825,
            -0.00675,
            0.00675,
            0.00825,
            0.01025,
            0.01125,
            0.01145,
        ]
        hulls, report = cs.common_sections(
            mesh,
            1,
            levels,
            sample_step=0.0005,
            relief=lambda p, lo, hi: p.difference(cutter),
        )
        report["engineering_relief"] = {
            "description": "Source swept rail raceway +50 micrometres cut from carriage collision only; rolling-element behavior excluded",
            "nominal_clearance_m": 0.00005,
        }
        return hulls, report
    if name == "EnclosureBody":
        return cs.common_sections(
            mesh,
            2,
            [
                -0.00195,
                0,
                0.00135,
                *np.arange(0.003, 0.0355, 0.002),
                0.03555,
                0.0363,
                0.03655,
                0.0368,
                0.03705,
                0.0373,
                0.0375,
                0.03805,
            ],
            sample_step=0.0005,
        )
    if name == "EnclosureLid":
        body = source_meshes["EnclosureBody"]
        cutter = shapely.union_all(
            [cs.section(body, 2, z) for z in np.linspace(0.034, 0.0380499, 83)]
        ).buffer(0.00005, join_style="mitre")
        hulls, report = cs.common_sections(
            mesh,
            2,
            [0.034, 0.035, 0.037, 0.038, 0.0381, 0.039, 0.03975, 0.04],
            sample_step=0.0005,
            relief=lambda p, lo, hi: p.difference(cutter) if lo < 0.0381 else p,
        )
        report["engineering_relief"] = {
            "description": "Rigid clip interference removed along full withdrawal from top body sections +50 micrometres; elastic snap retention is excluded",
            "nominal_clearance_m": 0.00005,
        }
        return hulls, report
    if name == "GearBase":
        return cs.common_sections(
            mesh, 2, [0, 0.005, 0.0247, 0.025], sample_step=0.0005
        )
    if name in ("GearLarge", "GearMedium", "GearSmall"):
        hulls, slabs = [], []
        for low, high in itertools.pairwise([0, 0.01, 0.0247, 0.025]):
            pieces, row = cs.extrusion(
                cs.section(mesh, 1, (low + high) / 2), 1, low, high, error=2e-5
            )
            hulls.extend(pieces)
            slabs.append(row)
        return hulls, {
            "method": "Source tooth-disk and hub midsections extruded along shaft axis; bores retained; rounded tooth/end chamfers approximated",
            "section_error_m": 2e-5,
            "slabs": slabs,
        }
    raise ValueError(f"No reviewed geometry recipe for {name}")


def fit_task(definition, hulls_by_name):
    axis_local = np.array(
        [0, 1, 0]
        if definition["id"] in ("04_linear_guide", "06_gear_train")
        else [0, 0, 1]
    )
    distance = 0.065 if definition["id"] == "04_linear_guide" else 0.04
    world, proxies = {}, {}
    for part in definition["parts"]:
        matrix = np.asarray(part["assembled_transform"])
        world[part["name"]] = [
            p @ matrix[:3, :3].T + matrix[:3, 3] for p in hulls_by_name[part["name"]]
        ]
        proxies[part["name"]] = cs.union_mesh(world[part["name"]])
    checks = []
    for part in definition["parts"]:
        if part.get("fixed"):
            continue
        name = part["name"]
        axis = np.asarray(part["assembled_transform"])[:3, :3] @ axis_local
        swept = cs.union_mesh(
            [np.concatenate([p, p + axis * distance]) for p in world[name]]
        )
        for other, obstacle in proxies.items():
            if other == name:
                continue
            volumes = []
            for moving in (proxies[name], swept):
                common = trimesh.boolean.intersection(
                    [moving, obstacle], engine="manifold"
                )
                volumes.append(float(abs(common.volume)) if len(common.faces) else 0.0)
            checks.append(
                {
                    "moving": name,
                    "obstacle": other,
                    "target_overlap_m3": volumes[0],
                    "swept_overlap_m3": volumes[1],
                    "withdrawal_distance_m": distance,
                }
            )
    maximum = max(max(r["target_overlap_m3"], r["swept_overlap_m3"]) for r in checks)
    return {
        "method": "Continuous straight-path convex Minkowski extrusion followed by Manifold solid union/intersection; includes all other parts in final assembled poses",
        "pairs": checks,
        "max_overlap_m3": maximum,
        "numerical_touching_tolerance_m3": 5e-13,
        "passed": maximum <= 5e-13,
    }, axis_local.tolist()


def prepare(catalog, output, cad_mass, task_ids=TASKS):
    output.mkdir(parents=True, exist_ok=True)
    masses = json.loads(cad_mass.read_text())
    digest = hashlib.sha256(
        pathlib.Path(__file__).read_bytes() + pathlib.Path(cs.__file__).read_bytes()
    ).hexdigest()
    for task_id in task_ids:
        task_dir = catalog.parent / task_id
        definition = json.loads((task_dir / "task.json").read_text())
        source_meshes, visuals, repairs = {}, {}, {}
        for part in definition["parts"]:
            vertices, faces = assets.read_mesh(task_dir / part["asset"])
            visuals[part["name"]] = (vertices, faces)
            mesh = trimesh.Trimesh(vertices, faces, process=True)
            if part["name"] == "GearLarge":
                mesh, repairs[part["name"]] = repair_tiny_triangles(mesh)
            source_meshes[part["name"]] = mesh
        rows, hulls_by_name = [], {}
        for part in definition["parts"]:
            name = part["name"]
            print(f"Preparing {task_id}/{name}", flush=True)
            source = task_dir / part["asset"]
            mesh = source_meshes[name]
            hulls, report = recipe(name, mesh, source_meshes)
            before_merge = len(hulls)
            hulls = cs.merge_convex_partition(hulls)
            report["exact_merge_input_hulls"] = before_merge
            report["exact_merge_output_hulls"] = len(hulls)
            union = cs.union_mesh(hulls)
            if not union.is_volume:
                raise ValueError(f"Invalid collision solid: {name}")
            if name in MASS_KEYS:
                mass = masses[MASS_KEYS[name]]
                cad_source = catalog.parent / "sources" / mass["source_filename"]
                if (
                    hashlib.sha256(cad_source.read_bytes()).hexdigest()
                    != mass["source_sha256"]
                ):
                    raise ValueError(f"Stale source CAD mass properties: {cad_source}")
            else:
                if not mesh.is_volume:
                    raise ValueError(f"No valid source solid for mass: {name}")
                mesh.density = 2700.0
                properties = mesh.mass_properties
                mass = {
                    "mass_kg": properties.mass,
                    "com_m": properties.center_mass.tolist(),
                    "inertia_kg_m2": properties.inertia.tolist(),
                    "volume_m3": mesh.volume,
                    "density_kg_m3": 2700.0,
                    "provenance": "Closed source-solid volume, center of mass and inertia integrals; nominal homogeneous density, not measured material or mass",
                }
            report.update(
                source_volume_m3=mass["volume_m3"],
                collision_union_volume_m3=float(union.volume),
                volume_ratio=float(union.volume / mass["volume_m3"]),
                source_tessellation_repairs=repairs.get(name, []),
            )
            t = np.arange(128) + 0.5
            z = 1 - 2 * t / 128
            angle = np.pi * (3 - np.sqrt(5)) * t
            directions = np.concatenate(
                [
                    np.eye(3),
                    -np.eye(3),
                    np.c_[
                        np.sqrt(1 - z * z) * np.cos(angle),
                        np.sqrt(1 - z * z) * np.sin(angle),
                        z,
                    ],
                ]
            )
            source_support = (mesh.vertices @ directions.T).max(axis=0)
            proxy_support = (np.concatenate(hulls) @ directions.T).max(axis=0)
            report["support_envelope_validation"] = {
                "method": "134 sampled directions including six axial extremes; convex support only, not a certified full-surface bound",
                "directions": len(directions),
                "max_under_source_m": float(
                    np.maximum(source_support - proxy_support, 0).max()
                ),
                "max_over_source_m": float(
                    np.maximum(proxy_support - source_support, 0).max()
                ),
            }
            target = output / (task_id + "_" + name + ".npz")
            v, f = visuals[name]
            np.savez_compressed(
                target,
                vertices=v,
                faces=f,
                **{f"hull_{i}": p for i, p in enumerate(hulls)},
            )
            row = {
                **part,
                "mesh_cache": target.name,
                "hull_count": len(hulls),
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "collision_cache_sha256": hashlib.sha256(
                    target.read_bytes()
                ).hexdigest(),
                "recipe_sha256": digest,
                "mass_kg": mass["mass_kg"],
                "com_m": mass["com_m"],
                "inertia_kg_m2": mass["inertia_kg_m2"],
                "density_kg_m3": mass["density_kg_m3"],
                "mass_provenance": mass,
                "collision_report": report,
                "training_ready": False,
                "precision_training_ready": False,
            }
            rows.append(row)
            hulls_by_name[name] = hulls
        fit, axis = fit_task(definition, hulls_by_name)
        ready = fit["passed"]
        for row in rows:
            row["collision_geometry_ready"] = ready
            if not row.get("fixed"):
                row["disassembly_axis_local"] = axis
            (output / (task_id + "_" + row["name"] + ".json")).write_text(
                json.dumps(row, indent=2)
            )
        result = {
            **definition,
            "parts": rows,
            "collision_geometry_ready": ready,
            "training_ready": False,
            "precision_training_ready": False,
            "geometry_fit_validation": fit,
            "training_blockers": [
                "Newton contact, grasp/settling and inverted-pose validation pending",
                "Throughput and collision-capacity validation pending; geometry readiness does not imply large-scale training readiness",
                "Collision-only relief and explicit rigid-body approximations require task acceptance; elastic/clamp/screw fastening is not modeled",
            ],
        }
        (output / (task_id + ".json")).write_text(json.dumps(result, indent=2))
        print(
            f"{task_id}: {sum(len(h) for h in hulls_by_name.values())} hulls, fit passed={ready}, max overlap={fit['max_overlap_m3']:.3g} m3",
            flush=True,
        )
        if not ready:
            raise ValueError(f"Continuous insertion path interferes: {task_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--cad-mass", type=pathlib.Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    prepare(args.catalog, args.out, args.cad_mass, args.tasks)
