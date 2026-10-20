"""Prepare the four NIST-derived tasks, retaining actual threads and apertures."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh

from isaac.assembly_rl import assets, collision_geometry
from isaac.assembly_rl.assembly_motion import withdrawal_transform
from isaac.assembly_rl.prepare_expansion_assets import fit_task

TASKS = (
    "07_threaded_fasteners",
    "08_rod_insertion",
    "09_keyed_pegs",
    "10_electrical_connectors",
)


def mesh_from_cache(path):
    with np.load(path, allow_pickle=False) as data:
        mesh = trimesh.Trimesh(data["vertices"], data["faces"], process=True)
        hulls = [data[f"hull_{i}"] for i in range(len(data.files) - 2)]
    return mesh, hulls


def prepare_threads(source):
    """Partition the closed source surfaces; never replace threads with cylinders."""
    meshes, hulls, reports = {}, {}, {}
    for name in ("nut_m16_low_poly", "bolt_m16_low_poly"):
        path = source / (name + "_clean.obj")
        mesh = trimesh.load_mesh(path, process=True)
        pieces, report = collision_geometry.decompose(mesh.vertices, mesh.faces)
        meshes[name], hulls[name], reports[name] = mesh, pieces, report
        report["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    fixture = trimesh.creation.box([0.12, 0.05, 0.003])
    fixture.apply_translation([0, 0, 0.0015])
    fixed_meshes, fixed_hulls = [fixture], [fixture.vertices]
    for x in (-0.04, 0.04):
        offset = np.array([x, 0, 0.003])
        bolt = meshes["bolt_m16_low_poly"].copy()
        bolt.apply_translation(offset)
        fixed_meshes.append(bolt)
        fixed_hulls.extend(p + offset for p in hulls["bolt_m16_low_poly"])
    rows = {
        "ThreadedFixture": (trimesh.util.concatenate(fixed_meshes), fixed_hulls),
        "NutLeftM16": (meshes["nut_m16_low_poly"], hulls["nut_m16_low_poly"]),
        "NutRightM16": (meshes["nut_m16_low_poly"], hulls["nut_m16_low_poly"]),
    }
    for name, (mesh, pieces) in rows.items():
        if not mesh.is_volume:
            raise ValueError("Thread source must remain a closed positive volume")
        mesh.density = 7800
        report = {
            "mass_kg": float(mesh.mass),
            "com_m": mesh.center_mass.tolist(),
            "inertia_kg_m2": mesh.moment_inertia.tolist(),
            "density_kg_m3": 7800,
            "mass_provenance": (
                "Closed source-mesh volume integral, homogeneous steel density "
                "7800 kg/m3 is an explicit engineering assumption, not a measured mass."
            ),
            "collision_report": {
                "method": "Boundary-preserving source convex partition; real helices retained",
                "sources": reports,
                "hull_count": len(pieces),
                "removed_source_artifact": (
                    "One isolated nonvolumetric triangle from the M16 nut; "
                    "the closed 1300-triangle solid and all its vertices are unchanged."
                ),
            },
        }
        np.savez_compressed(
            source / (name + ".npz"),
            vertices=mesh.vertices,
            faces=mesh.faces,
            **{f"hull_{i}": p for i, p in enumerate(pieces)},
        )
        (source / (name + ".json")).write_text(json.dumps(report, indent=2))


def threaded_fit(task, meshes):
    """Check assembled contacts and densely sampled helical withdrawal.

    This is a sampled geometry gate, not a continuous-path certificate. Native
    collision and physically exercised screw motion remain separate reset gates.
    """
    world = {}
    for part in task["parts"]:
        mesh = meshes[part["name"]].copy()
        mesh.apply_transform(np.asarray(part["assembled_transform"]))
        world[part["name"]] = mesh

    def overlap(a, b):
        common = trimesh.boolean.intersection([a, b], engine="manifold")
        return abs(float(common.volume)) if len(common.faces) else 0.0

    final = [
        {"parts": [a, b], "overlap_m3": overlap(world[a], world[b])}
        for a, b in itertools.combinations(world, 2)
    ]
    paths, controls = [], []
    for part in task["parts"]:
        if part["fixed"]:
            continue
        name = part["name"]
        pose = np.asarray(part["assembled_transform"])
        maximum = 0.0
        # 5 degrees per sample follows the authored 2 mm pitch to full release.
        distances = np.linspace(0, 0.024, 865)
        for distance in distances:
            moving = meshes[name].copy()
            moving.apply_transform(withdrawal_transform(pose, part, float(distance)))
            maximum = max(
                maximum,
                *(
                    overlap(moving, obstacle)
                    for key, obstacle in world.items()
                    if key != name
                ),
            )
        paths.append(
            {
                "part": name,
                "samples": len(distances),
                "withdrawal_m": 0.024,
                "angular_step_degrees": 5,
                "max_overlap_m3": maximum,
            }
        )
        # A straight axial shortcut must hit the actual thread flanks.
        wrong = world[name].copy()
        wrong.apply_translation(pose[:3, :3] @ np.array([0, 0, 0.0005]))
        controls.append(
            {
                "part": name,
                "straight_axial_advance_m": 0.0005,
                "max_overlap_m3": max(
                    overlap(wrong, obstacle)
                    for key, obstacle in world.items()
                    if key != name
                ),
            }
        )
    maximum = max(
        [row["overlap_m3"] for row in final] + [row["max_overlap_m3"] for row in paths]
    )
    return {
        "method": "Source-solid Boolean intersections along sampled right-handed helix",
        "continuous_path_certified": False,
        "final_pairs": final,
        "helical_paths": paths,
        "straight_shortcut_negative_controls": controls,
        "max_overlap_m3": maximum,
        "numerical_touching_tolerance_m3": 5e-13,
        "passed": maximum <= 5e-13
        and all(row["max_overlap_m3"] > 1e-9 for row in controls),
    }


def prepare(catalog, output, task_ids=TASKS):
    source_root = catalog.parent / "sources"
    if TASKS[0] in task_ids:
        prepare_threads(source_root / "nist_expansion")
    output.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        definition = json.loads((catalog.parent / task_id / "task.json").read_text())
        if (
            len(definition["parts"]) != 3
            or sum(p["fixed"] for p in definition["parts"]) != 1
        ):
            raise ValueError("Each addition must have one fixed and two movable bodies")
        source = source_root / (
            "nist_connectors" if task_id == TASKS[3] else "nist_expansion"
        )
        rows, meshes, hulls = [], {}, {}
        for part in definition["parts"]:
            name = part["name"]
            cache = source / (name + ".npz")
            metadata = json.loads((source / (name + ".json")).read_text())
            mesh, pieces = mesh_from_cache(cache)
            visual_path = catalog.parent / task_id / part["asset"]
            vertices, faces = assets.read_mesh(visual_path)
            target = output / (task_id + "_" + name + ".npz")
            np.savez_compressed(
                target,
                vertices=vertices,
                faces=faces,
                **{f"hull_{i}": p for i, p in enumerate(pieces)},
            )
            rows.append(
                {
                    **part,
                    **metadata,
                    "mesh_cache": target.name,
                    "hull_count": len(pieces),
                    "source": str(visual_path),
                    "source_sha256": hashlib.sha256(
                        visual_path.read_bytes()
                    ).hexdigest(),
                    "collision_cache_sha256": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest(),
                    "prepared_source_cache_sha256": hashlib.sha256(
                        cache.read_bytes()
                    ).hexdigest(),
                    "recipe_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "training_ready": False,
                    "precision_training_ready": False,
                }
            )
            meshes[name], hulls[name] = mesh, pieces
        fit = (
            threaded_fit(definition, meshes)
            if task_id == TASKS[0]
            else fit_task(definition, hulls)
        )
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
            "training_blockers": ["Native physical reset and grasp validation pending"],
        }
        (output / (task_id + ".json")).write_text(json.dumps(result, indent=2))
        print(
            task_id,
            "fit",
            fit["passed"],
            "hulls",
            sum(p["hull_count"] for p in rows),
            flush=True,
        )
        if not fit["passed"]:
            raise ValueError(f"Source collision mating check failed: {task_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    prepare(args.catalog, args.out, args.tasks)
