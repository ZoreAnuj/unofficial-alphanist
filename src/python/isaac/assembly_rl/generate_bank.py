"""Generate unvalidated on-table reset proposals from the selected CAD tasks.

This builds the loose-parts portion of a future reset bank. Robot joint states,
grasp resets, contact validation, and Newton settling are deliberately absent.
"""

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from isaac.assembly_rl import assets, reset_sampling

TABLE_HEIGHT_M = 0.86365
TABLE_XY_BOUNDS_M = np.array([[-0.825, 0.825], [-0.565, 0.475]])
PROPOSAL_XY_BOUNDS_M = np.array([[-0.30, 0.40], [-0.15, 0.30]])
ROBOT_BASE_EXCLUSION_M = np.array(
    [[-0.299, -0.5186, TABLE_HEIGHT_M], [-0.099, -0.3186, 1.30]]
)
ORIENTATION_MODES = ("upright", "side", "inverted", "uniform")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rigid_transform(value: list[list[float]]) -> np.ndarray:
    transform = np.asarray(value, dtype=float).copy()
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid source assembly transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6) or not np.isclose(
        np.linalg.det(rotation), 1, atol=2e-6
    ):
        raise ValueError("Source assembly transform is not rigid")
    # USD float32 rotations need projection before strict SO(3) validation.
    left, _, right = np.linalg.svd(rotation)
    transform[:3, :3] = left @ right
    return transform


def _load_task(
    catalog_path: pathlib.Path, task_id: str, scene_position: list[float]
) -> tuple[dict[str, Any], list[reset_sampling.PartSpec], np.ndarray]:
    from pxr import Usd, UsdGeom

    task_dir = catalog_path.parent / task_id
    definition_path = task_dir / "task.json"
    definition = json.loads(definition_path.read_text())
    scene = np.eye(4)
    scene[:3, 3] = scene_position
    parts, sources, targets = [], [], []
    for part in definition["parts"]:
        source = task_dir / part["asset"]
        stage = Usd.Stage.Open(str(source))
        root_transform = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetDefaultPrim())
        ).T
        if not np.allclose(root_transform, np.eye(4), atol=1e-9):
            raise ValueError(
                f"Nonidentity source root needs explicit frame handling: {source}"
            )
        vertices, _ = assets.read_mesh(source)
        assembled = _rigid_transform(part["assembled_transform"])
        parts.append(
            reset_sampling.PartSpec(
                part["name"], vertices, assembled[:3, :3], bool(part["fixed"])
            )
        )
        targets.append(scene @ assembled)
        dependencies = []
        for layer in stage.GetUsedLayers():
            if layer.anonymous:
                continue
            layer_path = pathlib.Path(layer.realPath)
            dependencies.append(
                {"path": str(layer_path), "sha256": _sha256(layer_path)}
            )
        sources.append(
            {
                "name": part["name"],
                "fixed": bool(part["fixed"]),
                "source": str(source),
                "source_sha256": _sha256(source),
                "usd_dependencies": sorted(dependencies, key=lambda row: row["path"]),
                "vertex_count": len(vertices),
                "source_root_transform": root_transform.tolist(),
                "original_assembled_world_transform": targets[-1].tolist(),
            }
        )
    if sum(part.fixed for part in parts) != 1:
        raise ValueError(f"Expected exactly one fixed receiver: {task_id}")
    metadata = {
        "id": task_id,
        "title": definition["title"],
        "definition_sha256": _sha256(definition_path),
        "parts": sources,
        "source_notes": definition.get("notes", definition.get("source", "")),
    }
    return metadata, parts, np.asarray(targets)


def _targets_fit_workspace(
    parts: list[reset_sampling.PartSpec], targets: np.ndarray
) -> bool:
    for part, target in zip(parts, targets, strict=True):
        points = part.vertices_m @ target[:3, :3].T + target[:3, 3]
        if (
            np.any(points[:, :2] < PROPOSAL_XY_BOUNDS_M[:, 0] - 1e-8)
            or np.any(points[:, :2] > PROPOSAL_XY_BOUNDS_M[:, 1] + 1e-8)
            or points[:, 2].min() < TABLE_HEIGHT_M - 1e-6
        ):
            return False
    return True


def build_bank(
    catalog: pathlib.Path, samples_per_task: int, seed: int, task: str | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if samples_per_task < 1 or seed < 0:
        raise ValueError("samples_per_task must be positive and seed nonnegative")
    catalog = catalog.resolve()
    source_catalog = json.loads(catalog.read_text())
    tasks = [
        row for row in source_catalog["tasks"] if task is None or row["id"] == task
    ]
    if not tasks:
        raise ValueError(f"No matching task: {task}")
    if not np.isclose(source_catalog["table_top_z"], TABLE_HEIGHT_M):
        raise ValueError("Catalog table height does not match this UR7 scene")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "proposal_only",
        "physics_validated": False,
        "training_ready": False,
        "bank_family": "loose_parts_on_table",
        "coordinate_frame": "UR7 workshop world; metre-scale asset-stage coordinates",
        "seed": seed,
        "samples_per_task": samples_per_task,
        "catalog": str(catalog),
        "catalog_sha256": _sha256(catalog),
        "table_height_m": TABLE_HEIGHT_M,
        "table_xy_bounds_m": TABLE_XY_BOUNDS_M.tolist(),
        "proposal_xy_bounds_m": PROPOSAL_XY_BOUNDS_M.tolist(),
        "robot_base_exclusion_aabb_m": ROBOT_BASE_EXCLUSION_M.tolist(),
        "drop_clearance_m": 0.005,
        "separation_m": 0.01,
        "orientation_schedule": "upright, side, inverted, uniform SO(3), repeated",
        "validation": {
            "mesh_bounds_and_conservative_aabbs": True,
            "target_frames_follow_receiver": True,
            "robot_full_geometry_collision": False,
            "ik_reachability": False,
            "settling": False,
            "stable_support": False,
            "precision_mating": False,
        },
        "tasks": [],
    }
    provenance = catalog.parent / source_catalog.get(
        "sources", "sources/provenance.json"
    )
    if provenance.is_file():
        metadata["provenance_sha256"] = _sha256(provenance)
    arrays = {}
    bank_index = 0
    for row in tasks:
        task_id = row["id"]
        task_metadata, parts, original_targets = _load_task(
            catalog, task_id, source_catalog["scene_position"]
        )
        receiver_index = next(index for index, part in enumerate(parts) if part.fixed)
        task_salt = int.from_bytes(
            hashlib.sha256(task_id.encode()).digest()[:4], "little"
        )
        task_metadata["receiver_index"] = receiver_index
        samples, transforms, goals, modes = [], [], [], []
        for sample_index in range(samples_per_task):
            mode = ORIENTATION_MODES[sample_index % len(ORIENTATION_MODES)]
            for attempt in range(32):
                proposal_seed = int(
                    np.random.SeedSequence(
                        [seed, task_salt, sample_index, attempt]
                    ).generate_state(1)[0]
                )
                try:
                    placements = reset_sampling.propose_placements(
                        parts,
                        PROPOSAL_XY_BOUNDS_M,
                        TABLE_HEIGHT_M,
                        proposal_seed,
                        exclusion_aabbs_m=(ROBOT_BASE_EXCLUSION_M,),
                        orientation_modes=(mode,),
                    )
                except RuntimeError:
                    continue
                sampled = np.array(
                    [placements[part.name].transform_world_from_local for part in parts]
                )
                targets = np.array(
                    [
                        reset_sampling.move_target_with_receiver(
                            target,
                            original_targets[receiver_index],
                            sampled[receiver_index],
                        )
                        for target in original_targets
                    ]
                )
                if _targets_fit_workspace(parts, targets):
                    break
            else:
                raise RuntimeError(
                    f"Cannot fit proposal and assembly target: {task_id}/{sample_index}"
                )
            part_modes = [placements[part.name].proposed_mode for part in parts]
            samples.append(
                {
                    "bank_index": bank_index,
                    "sample_index": sample_index,
                    "proposal_seed": proposal_seed,
                    "orientation_family": mode,
                    "status": "proposal_only",
                    "physics_validated": False,
                    "training_ready": False,
                    "parts": [
                        {
                            "name": part.name,
                            "proposed_mode": part_mode,
                            "world_transform": transform.tolist(),
                            "assembled_target_world_transform": target.tolist(),
                        }
                        for part, part_mode, transform, target in zip(
                            parts, part_modes, sampled, targets, strict=True
                        )
                    ],
                }
            )
            transforms.append(sampled)
            goals.append(targets)
            modes.append(part_modes)
            bank_index += 1
        arrays[task_id + "__world_transforms"] = np.asarray(transforms)
        arrays[task_id + "__target_world_transforms"] = np.asarray(goals)
        arrays[task_id + "__orientation_modes"] = np.asarray(modes)
        arrays[task_id + "__part_names"] = np.asarray([part.name for part in parts])
        arrays[task_id + "__fixed"] = np.asarray([part.fixed for part in parts])
        task_metadata["samples"] = samples
        metadata["tasks"].append(task_metadata)
    metadata["sample_count"] = bank_index
    return metadata, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--task")
    args = parser.parse_args()
    metadata, arrays = build_bank(
        args.catalog, args.samples_per_task, args.seed, args.task
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "bank.npz", **arrays)
    (args.output / "bank.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata['sample_count']} unvalidated proposals to {args.output}")


if __name__ == "__main__":
    main()
