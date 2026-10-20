"""Cache source CAD and coarse convex collision meshes for reset settling.

These decompositions support placement experiments, not precision insertion:
the original mating clearances still require a separate collision-fit check.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np


def read_mesh(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError(f"Expected metre-scale USD: {path}")
    vertices, triangles = [], []
    cache = UsdGeom.XformCache()
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        vertices.append(points)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[cursor : cursor + count] + offset
            triangles.extend(
                (face[0], face[i], face[i + 1]) for i in range(1, count - 1)
            )
            cursor += count
        offset += len(points)
    if not vertices:
        raise ValueError(f"No meshes: {path}")
    return np.concatenate(vertices), np.asarray(triangles, dtype=np.int32)


def prepare(catalog: pathlib.Path, output: pathlib.Path) -> None:
    import coacd
    import trimesh

    coacd.set_log_level("error")
    output.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(catalog.read_text())["tasks"]
    for task in tasks:
        task_dir = catalog.parent / task["id"]
        definition = json.loads((task_dir / "task.json").read_text())
        result = dict(definition)
        result["precision_training_ready"] = False
        result["collision_purpose"] = (
            "reset settling and throughput probe; mating fit unvalidated"
        )
        result["parts"] = []
        for part in definition["parts"]:
            source = task_dir / part["asset"]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            stem = task["id"] + "_" + part["name"]
            target = output / (stem + ".npz")
            metadata = output / (stem + ".json")
            if (
                metadata.exists()
                and target.exists()
                and json.loads(metadata.read_text())["source_sha256"] == digest
            ):
                result["parts"].append(json.loads(metadata.read_text()))
                continue
            vertices, faces = read_mesh(source)
            mesh = trimesh.Trimesh(vertices, faces, process=True)
            hulls = []
            for component in mesh.split(only_watertight=False):
                if (
                    len(component.vertices) < 4
                    or abs(component.convex_hull.volume) < 1e-12
                ):
                    continue
                if (
                    component.is_watertight
                    and abs(component.volume) / component.convex_hull.volume > 0.99
                ):
                    hulls.append(np.asarray(component.convex_hull.vertices))
                else:
                    pieces = coacd.run_coacd(
                        coacd.Mesh(
                            np.asarray(component.vertices), np.asarray(component.faces)
                        ),
                        threshold=0.02,
                        max_convex_hull=32,
                        preprocess_mode="auto",
                        resolution=1500,
                        mcts_nodes=10,
                        mcts_iterations=30,
                        mcts_max_depth=3,
                        seed=0,
                    )
                    hulls.extend(np.asarray(points) for points, _ in pieces)
            if not hulls:
                raise ValueError(f"No collision volume for {source}")
            low, high = vertices.min(0), vertices.max(0)
            density = 1200 if task["id"] in ("05_enclosure", "06_gear_train") else 2700
            volume = (
                abs(mesh.volume) if mesh.is_watertight else np.prod(high - low) * 0.4
            )
            mass = float(max(0.003, density * volume))
            lengths = high - low
            inertia = np.diag(
                mass
                / 12
                * np.array(
                    [
                        lengths[1] ** 2 + lengths[2] ** 2,
                        lengths[0] ** 2 + lengths[2] ** 2,
                        lengths[0] ** 2 + lengths[1] ** 2,
                    ]
                )
            )
            np.savez_compressed(
                target,
                vertices=vertices,
                faces=faces,
                **{f"hull_{i}": points for i, points in enumerate(hulls)},
            )
            row = {
                **part,
                "mesh_cache": target.name,
                "source_sha256": digest,
                "source": str(source),
                "hull_count": len(hulls),
                "mass_kg": mass,
                "com_m": ((low + high) / 2).tolist(),
                "inertia_kg_m2": inertia.tolist(),
                "provenance": "Nominal material density and bounding-box inertia; CoACD 0.02 normalized threshold for settling only",
            }
            metadata.write_text(json.dumps(row, indent=2))
            result["parts"].append(row)
            print(stem, len(hulls), "hulls", flush=True)
        (output / (task["id"] + ".json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.out)
