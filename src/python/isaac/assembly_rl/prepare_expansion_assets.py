"""Prepare three-object industrial tasks with dependency-aware insertion checks."""

import argparse
import hashlib
import itertools
import json
import pathlib

import numpy as np
import shapely
import trimesh

from isaac.assembly_rl import assets, expansion_geometry
from isaac.assembly_rl import collision_sections as cs

TASKS = ("07_drive_shaft", "08_shaft_bearing", "09_frame_corner", "10_dual_carriage")
REUSE = {
    "NEMA17": "03_shaft_coupling_NEMA17",
    "MGN7Rail": "04_linear_guide_MGN7Rail",
    "CarriageNegative": "04_linear_guide_MGN7Carriage",
    "CarriagePositive": "04_linear_guide_MGN7Carriage",
}


def prerequisites(parts: list[dict], name: str) -> set[str]:
    by_name = {part["name"]: part for part in parts}
    visiting, complete = set(), set()

    def visit(current):
        if current not in by_name or current in visiting:
            raise ValueError("Unknown or cyclic assembly prerequisite")
        if current in complete:
            return
        visiting.add(current)
        for parent in by_name[current].get("requires_assembled", []):
            visit(parent)
        visiting.remove(current)
        complete.add(current)

    visit(name)
    return complete - {name}


def fit_task(definition: dict, hulls_by_name: dict) -> dict:
    """Check final contacts and withdrawal with later dependent parts absent."""
    world, proxies = {}, {}
    parts = definition["parts"]
    dependencies = {p["name"]: prerequisites(parts, p["name"]) for p in parts}
    for part in parts:
        matrix = np.asarray(part["assembled_transform"])
        name = part["name"]
        world[name] = [
            p @ matrix[:3, :3].T + matrix[:3, 3] for p in hulls_by_name[name]
        ]
        proxies[name] = cs.union_mesh(world[name])

    def overlap(a, b):
        common = trimesh.boolean.intersection([a, b], engine="manifold")
        return float(abs(common.volume)) if len(common.faces) else 0.0

    final = [
        {"parts": [a, b], "overlap_m3": overlap(proxies[a], proxies[b])}
        for a, b in itertools.combinations(proxies, 2)
    ]
    paths = []
    for part in parts:
        if part["fixed"]:
            continue
        name = part["name"]
        matrix = np.asarray(part["assembled_transform"])
        axis = matrix[:3, :3] @ np.asarray(part["disassembly_axis_local"])
        if not np.isclose(np.linalg.norm(axis), 1):
            raise ValueError("Withdrawal axis must be a unit vector")
        distance = 0.07
        swept = cs.union_mesh(
            [np.concatenate([p, p + axis * distance]) for p in world[name]]
        )
        omitted = [other for other in proxies if name in dependencies[other]]
        for other, obstacle in proxies.items():
            if other == name or other in omitted:
                continue
            paths.append(
                {
                    "moving": name,
                    "obstacle": other,
                    "later_parts_absent": omitted,
                    "withdrawal_distance_m": distance,
                    "overlap_m3": overlap(swept, obstacle),
                }
            )
    maximum = max(row["overlap_m3"] for row in final + paths)
    return {
        "method": "Continuous convex Minkowski extrusion, followed by solid Boolean intersections. All final part pairs checked; descendants absent only during prerequisite insertion.",
        "final_pairs": final,
        "insertion_paths": paths,
        "max_overlap_m3": maximum,
        "numerical_touching_tolerance_m3": 5e-13,
        "passed": maximum <= 5e-13,
    }


def coupling_geometry(mesh):
    angles = np.arange(96) * 2 * np.pi / 96

    def relief(polygon, low, high):
        radius = 0.00255 if high <= 0.0114 else 0.00405 if low >= 0.0184 else None
        if radius is None:
            return polygon
        radius /= np.cos(np.pi / 96)
        cutter = shapely.Polygon(
            np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        )
        return polygon.difference(cutter)

    hulls, report = cs.common_sections(
        mesh,
        2,
        [0, 0.00025, 0.0005, 0.0114, 0.0184, 0.0224, 0.0333, 0.03355, 0.0338],
        sample_step=0.0005,
        relief=relief,
    )
    report["engineering_relief"] = {
        "description": "Both 5 mm and 8 mm source bores enlarged radially by 50 micrometres; original visual and inertial geometry retained. Rigid coupling, clamp tightening excluded.",
        "nominal_radial_clearance_m": 0.00005,
    }
    return hulls, report


def prepare(catalog: pathlib.Path, output: pathlib.Path, task_ids=TASKS) -> None:
    masses = json.loads(
        (
            catalog.parent / "sources/expansion_candidates/cad_mass_properties.json"
        ).read_text()
    )
    old_masses = json.loads((output / "cad_mass_properties.json").read_text())
    digest = hashlib.sha256(
        b"".join(
            pathlib.Path(p).read_bytes()
            for p in (__file__, cs.__file__, expansion_geometry.__file__)
        )
    ).hexdigest()
    for task_id in task_ids:
        directory = catalog.parent / task_id
        definition = json.loads((directory / "task.json").read_text())
        if (
            len(definition["parts"]) != 3
            or sum(p["fixed"] for p in definition["parts"]) != 1
        ):
            raise ValueError(
                "Expansion requires one fixed receiver and two movable bodies"
            )
        rows, hulls_by_name = [], {}
        for part in definition["parts"]:
            name = part["name"]
            print(f"Preparing {task_id}/{name}", flush=True)
            source = directory / part["asset"]
            vertices, faces = assets.read_mesh(source)
            mesh = trimesh.Trimesh(vertices, faces, process=True)
            if name in REUSE:
                prior = json.loads((output / (REUSE[name] + ".json")).read_text())
                if not prior.get("collision_geometry_ready"):
                    raise ValueError("Cannot reuse unvalidated collision geometry")
                cache = output / prior["mesh_cache"]
                if (
                    hashlib.sha256(cache.read_bytes()).hexdigest()
                    != prior["collision_cache_sha256"]
                ):
                    raise ValueError("Prior collision cache hash mismatch")
                with np.load(cache, allow_pickle=False) as data:
                    original_triangles = data["vertices"][data["faces"]]
                    new_triangles = vertices[faces]
                    # USD exporters differ in whether zero-area source faces survive.
                    nonzero = (
                        np.linalg.norm(
                            np.cross(
                                new_triangles[:, 1] - new_triangles[:, 0],
                                new_triangles[:, 2] - new_triangles[:, 0],
                            ),
                            axis=1,
                        )
                        > 0
                    )
                    new_triangles = new_triangles[nonzero]
                    if (
                        original_triangles.shape != new_triangles.shape
                        or not np.allclose(
                            original_triangles, new_triangles, atol=1e-8, rtol=0
                        )
                    ):
                        raise ValueError(f"Local mesh changed for reused part {name}")
                    hulls = [data[f"hull_{i}"] for i in range(prior["hull_count"])]
                report = prior["collision_report"]
                mass = {
                    key: prior[key]
                    for key in ("mass_kg", "com_m", "inertia_kg_m2", "density_kg_m3")
                }
                mass["provenance"] = prior.get(
                    "mass_provenance", prior.get("provenance")
                )
            else:
                hulls, report = (
                    coupling_geometry(mesh)
                    if name == "FlexibleCoupling"
                    else expansion_geometry.recipe(name, mesh)
                )
                hulls = cs.merge_convex_partition(hulls)
                mass = (
                    old_masses["coupling"]
                    if name == "FlexibleCoupling"
                    else masses[name]
                )
                cad_source = catalog.parent / "sources" / mass["source_filename"]
                if (
                    hashlib.sha256(cad_source.read_bytes()).hexdigest()
                    != mass["source_sha256"]
                ):
                    raise ValueError(f"Stale source CAD mass properties: {cad_source}")
                union = cs.union_mesh(hulls)
                if not union.is_volume:
                    raise ValueError(f"Invalid collision solid: {name}")
                report.update(
                    collision_union_volume_m3=float(union.volume),
                    source_volume_m3=mass["volume_m3"],
                    volume_ratio=float(union.volume / mass["volume_m3"]),
                )
            target = output / (task_id + "_" + name + ".npz")
            np.savez_compressed(
                target,
                vertices=vertices,
                faces=faces,
                **{f"hull_{i}": p for i, p in enumerate(hulls)},
            )
            rows.append(
                {
                    **part,
                    "mesh_cache": target.name,
                    "hull_count": len(hulls),
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "collision_cache_sha256": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest(),
                    "recipe_sha256": digest,
                    **{
                        key: mass[key]
                        for key in (
                            "mass_kg",
                            "com_m",
                            "inertia_kg_m2",
                            "density_kg_m3",
                        )
                    },
                    "mass_provenance": mass,
                    "collision_report": report,
                    "training_ready": False,
                    "precision_training_ready": False,
                }
            )
            hulls_by_name[name] = hulls
        fit = fit_task(definition, hulls_by_name)
        for row in rows:
            row["collision_geometry_ready"] = fit["passed"]
            (output / (task_id + "_" + row["name"] + ".json")).write_text(
                json.dumps(row, indent=2)
            )
        result = {
            **definition,
            "parts": rows,
            "collision_geometry_ready": fit["passed"],
            "geometry_fit_validation": fit,
            "training_ready": False,
            "training_blockers": [
                "Dynamic reset, grasp retention and settling validation pending; fastening excluded"
            ],
        }
        (output / (task_id + ".json")).write_text(json.dumps(result, indent=2))
        print(
            f"{task_id}: fit={fit['passed']}, max overlap={fit['max_overlap_m3']:.3g} m3",
            flush=True,
        )
        if not fit["passed"]:
            raise ValueError(f"Insertion geometry interferes: {task_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    prepare(args.catalog, args.out, args.tasks)
