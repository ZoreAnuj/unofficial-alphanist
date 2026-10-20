"""Prepare intact matched RJ45 source geometry for two fixture approach directions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

REVISION = "f9486a69b30a38355c1c2415fc15f6d6be8cb5b3"


def closed_source(path):
    mesh = trimesh.load(path, force="mesh")
    mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    solids, discarded = [], 0
    for component in mesh.split(only_watertight=False):
        if component.is_volume:
            solids.append(component)
        elif (
            abs(component.volume) < 1e-18
            and np.linalg.matrix_rank(
                component.vertices - component.vertices[0], tol=1e-10
            )
            < 3
        ):
            discarded += len(component.faces)
        else:
            raise ValueError(f"Non-solid source component: {path}")
    solid = trimesh.boolean.union(solids, engine="manifold")
    if not solid.is_volume:
        raise ValueError("Source Boolean union is not a valid closed solid")
    return solid, {
        "discarded_isolated_planar_faces": discarded,
        "added_caps": 0,
        "union_volume_m3": float(solid.volume),
        "obj_coordinate_rounding_m": 5e-9,
    }


def save_part(
    directory, name, visual, solid, hulls, collision_report, sources, density
):
    solid.density = density
    mass = solid.mass_properties
    if mass.mass <= 0 or np.any(np.linalg.eigvalsh(mass.inertia) <= 0):
        raise ValueError("Invalid source-solid mass properties")
    np.savez_compressed(
        directory / f"{name}.npz",
        vertices=visual.vertices,
        faces=visual.faces,
        **{f"hull_{i}": p for i, p in enumerate(hulls)},
    )
    report = {
        "mass_kg": float(mass.mass),
        "com_m": mass.center_mass.tolist(),
        "inertia_kg_m2": mass.inertia.tolist(),
        "density_kg_m3": density,
        "volume_m3": float(solid.volume),
        "collision_report": {**collision_report, "convex_pieces": len(hulls)},
        "source_provenance": {
            "sources": [
                {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sources
            ],
            "repository": "https://huggingface.co/datasets/UW-Lab/uwlab-assets",
            "revision": REVISION,
            "mass_method": "Whole closed Boolean-union source solid integrals at nominal homogeneous simulation density; engineering estimate, not measured mass or original multi-material CAD mass.",
            "license_status": "Asset-specific UWLab redistribution license not established; NIST provenance does not automatically establish third-party USD licensing.",
            "scope": "Rigid source insertion geometry only; electrical function and elastic latch/locking behavior are not modeled. Full fixture swept fit and dynamics require separate validation.",
        },
    }
    (directory / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, "hulls", len(hulls), "mass_kg", mass.mass, flush=True)


def prepare(directory):
    directory = Path(directory)
    solids, hulls, reports = {}, {}, {}
    for kind in ("plug", "socket"):
        key = f"rj45_{kind}"
        solid, repair = closed_source(directory / f"{key}_collision.obj")
        with np.load(directory / f"{key}_hulls.npz") as cached:
            pieces = [
                cached[f"hull_{i}"]
                for i in range(len([k for k in cached if k.startswith("hull_")]))
            ]
        detail = json.loads((directory / f"{key}_decomposition.json").read_text())
        volume = sum(trimesh.convex.convex_hull(p).volume for p in pieces)
        if not np.isclose(volume, solid.volume, rtol=1e-7, atol=1e-15):
            raise ValueError("Cached source hull volume mismatch")
        solids[kind], hulls[kind], reports[kind] = (
            solid,
            pieces,
            {"repair": repair, "decomposition": detail},
        )
    vertical = np.eye(4)
    vertical[:3, 3] = [-0.045, 0, 0.006]
    horizontal = np.eye(4)
    horizontal[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    horizontal[:3, 3] = [0.045, 0, 0.09]
    transforms = {"RJ45PlugVertical": vertical, "RJ45PlugHorizontal": horizontal}
    visual_plug = trimesh.load(directory / "rj45_plug_visual.obj", force="mesh")
    visual_socket = trimesh.load(directory / "rj45_socket_visual.obj", force="mesh")
    for name in transforms:
        save_part(
            directory,
            name,
            visual_plug,
            solids["plug"],
            hulls["plug"],
            reports["plug"],
            [directory / "rj45_plug.usd", directory / "rj45_plug_collision.obj"],
            1200.0,
        )
    plate = trimesh.creation.box(extents=[0.17, 0.06, 0.006])
    plate.apply_translation([0, 0, 0.003])
    pedestal = trimesh.creation.box(extents=[0.028, 0.024, 0.063])
    pedestal.apply_translation([0.059, 0, 0.0375])
    fixture_solids = [plate, pedestal]
    fixture_hulls = [plate.vertices, pedestal.vertices]
    fixture_visuals = [plate, pedestal]
    for transform in transforms.values():
        socket = solids["socket"].copy()
        socket.apply_transform(transform)
        fixture_solids.append(socket)
        visual = visual_socket.copy()
        visual.apply_transform(transform)
        fixture_visuals.append(visual)
        fixture_hulls.extend(
            p @ transform[:3, :3].T + transform[:3, 3] for p in hulls["socket"]
        )
    fixture = trimesh.boolean.union(fixture_solids, engine="manifold")
    if not fixture.is_volume or not np.isclose(
        fixture.volume, sum(p.volume for p in fixture_solids), rtol=1e-7, atol=1e-15
    ):
        raise ValueError("Fixture components have overlapping volume")
    visual = trimesh.util.concatenate(fixture_visuals)
    visual.export(directory / "ConnectorFixture.obj")
    save_part(
        directory,
        "ConnectorFixture",
        visual,
        fixture,
        fixture_hulls,
        {
            "method": "Two unmodified source socket partitions plus exact mounting plate and support pedestal",
            "socket": reports["socket"],
            "socket_transforms": {n: t.tolist() for n, t in transforms.items()},
            "mounting_plate_dimensions_m": [0.17, 0.06, 0.006],
            "pedestal_dimensions_m": [0.028, 0.024, 0.063],
        },
        [
            directory / "rj45_socket.usd",
            directory / "rj45_socket_collision.obj",
            directory / "ConnectorFixture.obj",
        ],
        2700.0,
    )
    original = json.loads((directory / "preview_pose_spec.json").read_text())[0]
    spec = {
        "id": "10_electrical_connectors",
        "title": "Vertical and horizontal RJ45 connector assembly",
        "units": "metres",
        "training_ready": False,
        "notes": "Two intact copies of the matched UWLab/NIST RJ45 pair approach along orthogonal directions. One fixed compound fixture and two independent movable plugs. Rigid seated insertion only; elastic latch deformation/locking and electrical function are not modeled.",
        "license_status": original["license_status"],
        "source_provenance": {
            "repository": "https://huggingface.co/datasets/UW-Lab/uwlab-assets",
            "revision": REVISION,
            "files": [
                {
                    "name": f"rj45_{k}.usd",
                    "url": f"https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/{REVISION}/Props/NIST/rj45_{k}.usd",
                    "sha256": hashlib.sha256(
                        (directory / f"rj45_{k}.usd").read_bytes()
                    ).hexdigest(),
                }
                for k in ("plug", "socket")
            ],
        },
        "pose_provenance": {
            "source_mate_relative_transform": np.eye(4).tolist(),
            "basis": "Original nist_scene.usd assigns matching plug/socket frames; each intact pair receives the same rigid fixture transform",
            "fixture_socket_transforms": {n: t.tolist() for n, t in transforms.items()},
        },
        "parts": [
            {
                "name": "ConnectorFixture",
                "fixed": True,
                "obj": str(directory / "ConnectorFixture.obj"),
                "assembled_transform": np.eye(4).tolist(),
                "dimensions_mm": (visual.extents * 1000).tolist(),
                "color": [0.24, 0.28, 0.31],
                "metallic": 0.7,
                "construction": "170 x 60 x 6 mm base and 28 x 24 x 63 mm support pedestal; unmodified authentic sockets",
            }
        ],
    }
    for name, transform in transforms.items():
        spec["parts"].append(
            {
                "name": name,
                "fixed": False,
                "obj": str(directory / "rj45_plug_visual.obj"),
                "assembled_transform": transform.tolist(),
                "dimensions_mm": (visual_plug.extents * 1000).tolist(),
                "color": [0.25, 0.32, 0.35]
                if name.endswith("Vertical")
                else [0.63, 0.37, 0.12],
                "metallic": 0.05,
                "disassembly_axis_local": [0, 0, 1],
                "disassembly_axis_assembly": transform[:3, 2].tolist(),
                "requires_assembled": [],
                "near_mating_offset_range_m": [0.02, 0.06],
            }
        )
    (directory / "preview_pose_spec.json").write_text(
        json.dumps([spec], indent=2) + "\n"
    )
    print(
        "Preview spec and all three caches ready; whole-fixture fit and dynamic validation pending",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
