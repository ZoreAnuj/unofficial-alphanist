"""Prepare and validate the source-section motor-mount collision asset pair."""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import trimesh

from isaac.assembly_rl import assets, collision_sections


def prepare(catalog: pathlib.Path, output: pathlib.Path) -> None:
    """Write visual meshes, convex colliders, integral mass properties and fit checks.

    This certifies only the recorded geometric checks. Newton contact stability,
    grasp support and release behavior must pass before training_ready can change.
    """
    definition_path = catalog.parent / "02_motor_mount" / "task.json"
    definition = json.loads(definition_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = dict(definition, parts=[])
    proxies = []
    recipe = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
        + pathlib.Path(collision_sections.__file__).read_bytes()
    ).hexdigest()
    for part in definition["parts"]:
        source = definition_path.parent / part["asset"]
        vertices, faces = assets.read_mesh(source)
        mesh = trimesh.Trimesh(vertices, faces, process=True)
        if not mesh.is_volume:
            raise ValueError(f"Invalid source solid: {source}")
        if part["name"] == "MountBracket":
            hulls, report = collision_sections.bracket(mesh)
        else:
            hulls, report = collision_sections.layered(mesh, error=2.5e-5)
        proxy = collision_sections.union_mesh(hulls)
        if not proxy.is_volume:
            raise ValueError(f"Invalid collision union: {source}")
        proxy.apply_transform(np.asarray(part["assembled_transform"]))
        proxies.append(proxy)
        mesh.density = 2700.0
        mass = mesh.mass_properties
        target = output / ("02_motor_mount_" + part["name"] + ".npz")
        np.savez_compressed(
            target,
            vertices=vertices,
            faces=faces,
            **{f"hull_{i}": points for i, points in enumerate(hulls)},
        )
        row = {
            **part,
            "mesh_cache": target.name,
            "hull_count": len(hulls),
            "mass_kg": mass.mass,
            "com_m": mass.center_mass.tolist(),
            "inertia_kg_m2": mass.inertia.tolist(),
            "density_kg_m3": mesh.density,
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "collision_cache_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "recipe_sha256": recipe,
            "collision_report": report,
            "provenance": "Nominal homogeneous density; closed source-solid volume, COM and inertia integrals. Source section extrusion retaining apertures. Dynamic validation remains required.",
            "collision_geometry_ready": False,
            "training_ready": False,
            "precision_training_ready": False,
        }
        if not part.get("fixed"):
            row["disassembly_axis_local"] = [0, 0, -1]
        result["parts"].append(row)
    rows = []
    for withdrawal in np.linspace(0.04, 0.0, 41):
        moving = proxies[1].copy()
        moving.vertices[:, 0] -= withdrawal
        common = trimesh.boolean.intersection([proxies[0], moving], engine="manifold")
        volume = abs(common.volume) if len(common.faces) else 0.0
        rows.append({"withdrawal_m": float(withdrawal), "overlap_m3": volume})
    maximum = max(row["overlap_m3"] for row in rows)
    # 0.0001 mm^3 accepts float32 surface touching; a material CAD interference
    # (the original guide/enclosure failures) is orders of magnitude larger.
    if maximum > 1e-13:
        raise ValueError(f"Motor collision insertion path interferes: {maximum} m3")
    result.update(
        collision_geometry_ready=True,
        training_ready=False,
        precision_training_ready=False,
        geometry_fit_validation={
            "method": "41 discrete Boolean intersection checks along 40 mm straight insertion path; not a continuous proof",
            "poses": rows,
            "max_overlap_m3": maximum,
            "numerical_touching_tolerance_m3": 1e-13,
            "source_pilot_radial_clearance_m": 0.00025,
        },
        training_blockers=[
            "Newton contact and settling validation pending",
            "Motor must remain supported by the gripper during seating; screw fastening/retention is not modeled",
        ],
    )
    for row in result["parts"]:
        row["collision_geometry_ready"] = True
        (output / ("02_motor_mount_" + row["name"] + ".json")).write_text(
            json.dumps(row, indent=2)
        )
    (output / "02_motor_mount.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.out)
