"""Bounded real-physics capacity probe with selected task geometries resident."""

import argparse
import faulthandler
import hashlib
import json
import math
import pathlib
import resource
import statistics
import time

import numpy as np
import torch
import warp as wp

from isaac.assembly_rl import (
    ccd_workspace,
    environment,
    model,
    physics_step,
    preview,
    reset_bank_builder,
    sap_wide,
)


def positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Environment count must be a positive integer"
        ) from error
    if count < 1:
        raise argparse.ArgumentTypeError("Environment count must be a positive integer")
    return count


def device_budget_gib(
    total_bytes: int, reserve_gib: float, cap_gib: float | None
) -> float:
    total = total_bytes / 1024**3
    if not math.isfinite(reserve_gib) or not 0 < reserve_gib < total:
        raise ValueError("GPU reserve must be positive and smaller than device memory")
    if cap_gib is not None and (not math.isfinite(cap_gib) or cap_gib <= 0):
        raise ValueError("Explicit GPU cap must be finite and positive")
    return min(total - reserve_gib, cap_gib if cap_gib is not None else total)


def failure_category(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or (
        ("cuda" in message or "device" in message)
        and ("out of memory" in message or "allocate" in message)
    ):
        return "gpu_oom"
    if isinstance(error, MemoryError):
        return "host_oom"
    if "memory safety bound" in message:
        return "gpu_budget"
    if (
        "int32" in message
        or "signed32" in message
        or "32-bit" in message
        or "maximum supported dimension" in message
    ):
        return "index_limit"
    if "capacity" in message:
        return (
            "solver_capacity_or_nonfinite"
            if "nonfinite" in message
            else "solver_capacity"
        )
    if "nonfinite" in message:
        return "nonfinite_state"
    if isinstance(error, ValueError):
        return "validation_error"
    return "runtime_error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--num-envs", type=positive_count, required=True)
    parser.add_argument("--max-gpu-gib", type=float, default=None)
    parser.add_argument("--gpu-reserve-gib", type=float, default=2.0)
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "num_envs": args.num_envs,
        "rounds": [],
        "memory_samples": [],
        "completed": False,
        "physics_only_no_training_readiness_claim": True,
        "module_sha256": {
            name: hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (
                ("environment", environment),
                ("model", model),
                ("physics_step", physics_step),
                ("ccd_workspace", ccd_workspace),
                ("sap_wide", sap_wide),
            )
        },
    }
    target = args.output / "result.json"
    env = None
    started_total = time.perf_counter()
    budget = None

    def save():
        result["elapsed_seconds"] = time.perf_counter() - started_total
        result["host_peak_rss_gib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        )
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(target)

    def memory(label):
        nonlocal budget
        free, total = torch.cuda.mem_get_info()
        if budget is None:
            budget = device_budget_gib(total, args.gpu_reserve_gib, args.max_gpu_gib)
            result.update(
                device_total_gib=total / 1024**3,
                gpu_budget_gib=budget,
                gpu_reserve_gib=args.gpu_reserve_gib,
            )
        used = (total - free) / 1024**3
        result["memory_samples"].append(
            {
                "phase": label,
                "device_used_gib": used,
                "torch_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "torch_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            }
        )
        save()
        if used > budget:
            raise RuntimeError(f"GPU memory safety bound exceeded: {used:.2f} GiB")

    faulthandler.dump_traceback_later(180, repeat=True)
    try:
        memory("before_constructor")
        started = time.perf_counter()
        env = environment.AssemblyEnvironment(
            args.assets,
            num_envs=args.num_envs,
            seed=7,
            config={"task_names": args.tasks},
        )
        torch.cuda.synchronize()
        result.update(
            constructor_seconds=time.perf_counter() - started,
            timed_operation="env.hold policy interval, including physics guards and synchronization; excludes observation construction, network inference and optimization",
            physics_substeps_per_interval=env.substeps,
            tasks=[t["id"] for t in env.tasks],
            shapes_per_world=env.ns,
            compact_same_body_filters_per_world=env.maps["compact_same_body_filters_per_world"],
            explicit_collision_filters_per_world=env.maps["explicit_collision_filters_per_world"],
            hull_count_per_world=sum(
                p["hull_count"] for t in env.tasks for p in t["parts"]
            ),
            body_count_per_world=env.nb,
            joint_coords_per_world=env.nq,
            solver_njmax=int(env.config["solver_njmax"]),
            solver_nconmax=int(env.config["solver_nconmax"]),
            naconmax=int(env.solver.mjw_data.naconmax),
            physics_identity=env.physics_identity,
            jaw_controller=env.profile["jaw"],
            native_drive_ke_first_world=wp.to_torch(env.model.joint_target_ke)[
                :8
            ].tolist(),
            native_drive_kd_first_world=wp.to_torch(env.model.joint_target_kd)[
                :8
            ].tolist(),
            contact_profile=env.profile["contact"],
        )
        if args.tasks is not None and result["tasks"] != args.tasks:
            raise ValueError(
                "Capacity probe task order differs from requested task IDs"
            )
        memory("after_constructor")
        print(
            json.dumps(
                {
                    "phase": "constructor",
                    **{
                        k: result[k]
                        for k in (
                            "constructor_seconds",
                            "hull_count_per_world",
                            "shapes_per_world",
                        )
                    },
                }
            ),
            flush=True,
        )
        geometries = reset_bank_builder.load_task_geometry(args.assets, env.tasks)
        rng = np.random.default_rng(17)
        visited = [set() for _ in range(env.num_envs)]
        for round_id in range(len(env.tasks)):
            ids = (np.arange(env.num_envs) + round_id) % len(env.tasks)
            records = [
                reset_bank_builder.candidate(
                    env.tasks[index],
                    geometries[index],
                    "loose",
                    rng,
                    env.max_parts,
                    config=reset_bank_builder.ValidationConfig(),
                )
                for index in ids
            ]
            began = time.perf_counter()
            task_ids = torch.as_tensor(ids, dtype=torch.long, device=env.device)
            env.initialize(
                task_ids,
                env._tensor(np.asarray([r["part_poses"] for r in records])),
                env._tensor(np.asarray([r["joint_q"] for r in records])),
                env._tensor(np.asarray([r["goals"] for r in records])),
            )
            torch.cuda.synchronize()
            reset_seconds = time.perf_counter() - began
            if not torch.equal(env.task_ids, task_ids):
                raise RuntimeError("Global task indices changed during task switching")
            initial = env.snapshot()
            row = {
                "round": round_id,
                "task_ids": ids.tolist(),
                "reset_seconds": reset_seconds,
                "interval_seconds": [],
                "max_penetration_m": 0.0,
                "failures": [],
                "failure_flags": [],
                "max_nefc": 0,
                "max_nacon": 0,
            }
            result["rounds"].append(row)
            for _frame in range(5):
                began = time.perf_counter()
                env.hold(
                    initial["joint_q"][:, :6],
                    torch.ones(env.num_envs, device=env.device),
                )
                torch.cuda.synchronize()
                row["interval_seconds"].append(time.perf_counter() - began)
                state = env.snapshot()
                row["max_penetration_m"] = max(
                    row["max_penetration_m"], state["max_penetration_m"].max().item()
                )
                row["failures"].append(state["safety_failure"].tolist())
                row["failure_flags"].append(state["failure_flags"].tolist())
                row["max_nefc"] = max(
                    row["max_nefc"],
                    int(wp.to_torch(env.solver.mjw_data.nefc).max().item()),
                )
                row["max_nacon"] = max(
                    row["max_nacon"],
                    int(wp.to_torch(env.solver.mjw_data.nacon)[0].item()),
                )
                if not torch.isfinite(state["part_poses"]).all():
                    raise RuntimeError("Nonfinite part poses during task switching")
            row["steady_interval_median_seconds"] = statistics.median(
                row["interval_seconds"][1:]
            )
            for world, task_id in enumerate(ids):
                visited[world].add(int(task_id))
            observation = env.observe()
            if observation["points"].shape != (env.num_envs, env.num_points, 3):
                raise RuntimeError("Invalid all-task observation shape")
            row["observation_finite"] = all(
                bool(torch.isfinite(v).all()) for v in observation.values()
            )
            if not row["observation_finite"]:
                raise RuntimeError("Nonfinite all-task observation")
            snapshot_name = f"physical_snapshot_{env.tasks[int(ids[0])]['id']}.npz"
            preview.capture_snapshot(env, args.output / snapshot_name)
            row["snapshot"] = snapshot_name
            memory(f"round_{round_id}")
            print(json.dumps(row), flush=True)
        if any(v != set(range(len(env.tasks))) for v in visited):
            raise RuntimeError("Not all tasks were exercised in every world")
        result.update(
            completed=True,
            any_safety_failure=any(
                any(flags) for row in result["rounds"] for flags in row["failures"]
            ),
            peak_sampled_gpu_gib=max(
                row["device_used_gib"] for row in result["memory_samples"]
            ),
            steady_interval_median_seconds=statistics.median(
                t for row in result["rounds"] for t in row["interval_seconds"][1:]
            ),
            first_interval_seconds=result["rounds"][0]["interval_seconds"][0],
        )
        result["passed"] = not result["any_safety_failure"]
        result["failure_category"] = (
            None if result["passed"] else "physical_safety_failure"
        )
        result["world_intervals_per_second"] = (
            env.num_envs / result["steady_interval_median_seconds"]
        )
        save()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("rounds", "memory_samples")
                }
            ),
            flush=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = False
        result["failure_category"] = failure_category(exc)
        save()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
