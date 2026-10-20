"""Compare one physical world under isolated collision/storage configurations.

Old bank entries supply identical diagnostic initial states; this never certifies
or rewrites a reset bank for a changed physical implementation.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import time

import newton
import numpy as np
import torch
import warp as wp

from isaac.assembly_rl import environment, model, preview, reset_bank_builder


def restore_entry(env, arrays, index):
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    selected = lambda name: env._tensor(arrays[name][index : index + 1])
    env._initialize(
        ids,
        ids,
        selected("part_poses"),
        selected("joint_q"),
        selected("goals"),
        selected("joint_qd"),
        selected("part_velocities"),
    )
    for name in ("jaw_servo", "arm_servo", "arm_servo_velocity", "gripper_open"):
        getattr(env, name).copy_(selected(name))
    env.targets.copy_(selected("joint_targets"))


def capture_collision_graph(env, output):
    """Hash the actual compiled contact graph, including exclusions/overrides."""
    native = env.solver.mj_model
    pair_ids = env.solver.mjw_model.nxn_pairid.numpy()
    expected = native.ngeom * (native.ngeom - 1) // 2
    if pair_ids.shape != (expected, 2):
        raise ValueError("Unexpected MuJoCo Warp compiled collision graph layout")
    allowed = pair_ids[:, 0] > -2
    packed = np.packbits(allowed)
    fields = (
        "geom_contype",
        "geom_conaffinity",
        "geom_bodyid",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "body_weldid",
        "body_parentid",
        "exclude_signature",
        "pair_geom1",
        "pair_geom2",
    )
    np.savez_compressed(
        output,
        **{name: np.asarray(getattr(native, name)) for name in fields},
        allowed_pair_packed=packed,
        ngeom=native.ngeom,
    )
    return {
        "ngeom": native.ngeom,
        "allowed_contact_pair_count": int(allowed.sum()),
        "allowed_contact_graph_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "compiled_collision_file": output.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "bank", "out"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--pairs", choices=("baseline", "omitted"), required=True)
    parser.add_argument("--jacobian", choices=("dense", "sparse"), required=True)
    parser.add_argument(
        "--family", choices=("loose", "partial_assembly"), required=True
    )
    parser.add_argument("--constructor-only", action="store_true")
    parser.add_argument("--reference-model-file", type=pathlib.Path)
    args = parser.parse_args()
    probe_model = model
    if args.reference_model_file is not None:
        specification = importlib.util.spec_from_file_location(
            "assembly_reference_model", args.reference_model_file
        )
        probe_model = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(probe_model)
        environment.scene_model = probe_model
    metadata = json.loads((args.bank / "bank.json").read_text())
    with np.load(args.bank / "bank.npz", allow_pickle=False) as source:
        arrays = dict(source)
    reset_bank_builder.validate_bank_entries(arrays, metadata)
    if len(metadata["task_names"]) != 1:
        raise ValueError("This probe requires a single-task reference bank")
    index = int(np.flatnonzero(arrays["family"] == args.family)[0])
    if args.pairs == "baseline":
        # Process-local reference implementation; production keeps the optimized
        # builder and never constructs this unused pair table.
        class ReferenceBuilder(probe_model.MuJoCoNativeContactBuilder):
            _find_shape_contact_pairs = newton.ModelBuilder._find_shape_contact_pairs

        probe_model.MuJoCoNativeContactBuilder = ReferenceBuilder
    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "pairs": args.pairs,
        "jacobian": args.jacobian,
        "family": args.family,
        "task": metadata["task_names"][0],
        "entry_id": metadata["entry_ids"][index],
        "num_envs": 1,
        "diagnostic_only": True,
        "completed": False,
        "constructor_only": args.constructor_only,
        "model_source_sha256": hashlib.sha256(
            pathlib.Path(probe_model.__file__).read_bytes()
        ).hexdigest(),
        "frames": [],
    }
    env = None
    try:
        started = time.perf_counter()
        env = environment.AssemblyEnvironment(
            args.assets,
            num_envs=1,
            config={
                "task_names": metadata["task_names"],
                "solver_jacobian": args.jacobian,
            },
        )
        torch.cuda.synchronize()
        result["constructor_seconds"] = time.perf_counter() - started
        pairs = env.model.shape_contact_pairs
        result["candidate_pair_count"] = 0 if pairs is None else pairs.size
        result["physics_identity"] = env.physics_identity
        result.update(capture_collision_graph(env, args.out / "compiled_collision.npz"))
        if args.constructor_only:
            result["completed"] = True
            return
        restore_entry(env, arrays, index)
        preview.capture_snapshot(env, args.out / "initial.npz")
        targets, gripper = env.targets[:, :6].clone(), env.gripper_open.clone()
        for frame in range(10):
            started = time.perf_counter()
            env.hold(targets, gripper)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            state = env.snapshot()
            result["frames"].append(
                {
                    "seconds": elapsed,
                    "failure_flags": state["failure_flags"].tolist(),
                    "safety_failure": state["safety_failure"].tolist(),
                    "max_penetration_m": state["max_penetration_m"].tolist(),
                }
            )
            preview.capture_snapshot(env, args.out / f"frame_{frame:02d}.npz")
        result["median_interval_seconds"] = statistics.median(
            row["seconds"] for row in result["frames"][1:]
        )
        free, total = torch.cuda.mem_get_info()
        result["device_used_gib_after_step"] = (total - free) / 1024**3
        data = env.solver.mjw_data
        count = int(wp.to_torch(data.nacon)[0])
        result["contacts"] = reset_bank_builder.summarize_contacts(
            env.contacts.rigid_contact_shape0.numpy()[:count],
            env.contacts.rigid_contact_shape1.numpy()[:count],
            env.contacts.force.numpy()[:count],
            data.contact.dist.numpy()[:count],
            data.contact.worldid.numpy()[:count],
            labels=env.maps["shape_labels"],
            num_envs=1,
        )
        result["completed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
