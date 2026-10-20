"""Qualify dense accepted contacts and one unchanged PPO update at a batch size."""

import argparse
import json
import math
import pathlib
import threading
import time

import numpy as np
import torch
import warp as wp

from isaac.assembly_rl import environment, train

DENSE_HOLD_INTERVALS = 2


def metrics_finite(value):
    if isinstance(value, dict):
        return all(metrics_finite(child) for child in value.values())
    # Per-task success rate is undefined until that task completes an episode.
    return value is None or math.isfinite(value)


def dense_entries(
    task_ids, families, task_count, *, active_parts=None, max_per_stratum=None
):
    """Require both difficult families for every task; never silently skip one."""
    if max_per_stratum is not None:
        if max_per_stratum < 1:
            raise ValueError("Dense entries per stratum must be positive")
        if active_parts is None or active_parts.shape != task_ids.shape:
            raise ValueError("Bounded dense selection requires per-entry active_part")
    selected = []
    for task in range(task_count):
        for family in ("near_mating", "partial_assembly"):
            matches = np.flatnonzero((task_ids == task) & (families == family))
            if not len(matches):
                raise ValueError(f"Missing dense reset: task{task}/{family}")
            if max_per_stratum is None:
                selected.extend(matches.tolist())
            else:
                for part in np.unique(active_parts[matches]):
                    selected.extend(
                        matches[active_parts[matches] == part][
                            :max_per_stratum
                        ].tolist()
                    )
    return selected


def check_dense_state(state, held_part):
    if state["failure_flags"].any() or state["safety_failure"].any():
        raise RuntimeError("Dense-contact physics guard failed")
    pads = state["max_pad_part_penetration_m"]
    limits = torch.full_like(pads, 0.00005)
    if held_part >= 0:
        limits[:, held_part] = 0.0005
    if (state["max_rigid_penetration_m"] > 0.00005).any() or (pads > limits).any():
        raise RuntimeError("Dense-contact final penetration gate failed")
    if any(
        not torch.isfinite(state[key]).all()
        for key in ("part_poses", "part_velocities", "joint_q", "joint_qd")
    ):
        raise RuntimeError("Dense-contact state is nonfinite")


def allocator_snapshot():
    """Keep current usage separate from allocator high-water marks."""
    device = f"cuda:{torch.cuda.current_device()}"
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return {
        "device_used_gib": (total - free) / 1024**3,
        "torch_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "torch_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "torch_allocator_backend": torch.cuda.get_allocator_backend(),
        "torch_num_alloc_retries": stats.get("num_alloc_retries"),
        "torch_num_ooms": stats.get("num_ooms"),
        "warp_pool_used_current_gib": wp.get_mempool_used_mem_current(device) / 1024**3,
        "warp_pool_used_high_gib": wp.get_mempool_used_mem_high(device) / 1024**3,
        "warp_pool_release_threshold_bytes": wp.get_mempool_release_threshold(device),
    }


def record_memory_boundary(record, *, trim_unused_cache=False):
    """Trim only unused allocator caches at an existing synchronized boundary."""
    torch.cuda.synchronize()
    record("before_trim" if trim_unused_cache else "unchanged")
    if not trim_unused_cache:
        return
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    record("after_torch_trim")
    device = f"cuda:{torch.cuda.current_device()}"
    threshold = wp.get_mempool_release_threshold(device)
    try:
        wp.set_mempool_release_threshold(device, 0)
        wp.synchronize_device(device)
    finally:
        wp.set_mempool_release_threshold(device, threshold)
    record("after_warp_trim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "bank", "out"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--eval-bank", type=pathlib.Path)
    parser.add_argument("--eval-num-envs", type=int, default=2)
    parser.add_argument("--max-device-gib", type=float)
    parser.add_argument("--trim-unused-cache", action="store_true")
    parser.add_argument(
        "--dense-per-stratum",
        type=int,
        help="Maximum accepted resets per task/family/active-part stratum; default: all",
    )
    args = parser.parse_args()
    if (
        min(args.num_envs, args.eval_num_envs) < 1
        or (args.max_device_gib is not None and args.max_device_gib <= 0)
        or (args.dense_per_stratum is not None and args.dense_per_stratum < 1)
    ):
        parser.error("Positive environment count and memory budget required")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.max_device_gib is None:
        _, total = torch.cuda.mem_get_info()
        args.max_device_gib = total / 1024**3 - 2.0
    result = {
        "completed": False,
        "learning_success_demonstrated": False,
        "num_envs": args.num_envs,
        "eval_num_envs": args.eval_num_envs if args.eval_bank else 0,
        "dense_hold_intervals_per_entry": DENSE_HOLD_INTERVALS,
        "dense_per_stratum": args.dense_per_stratum,
        "device_budget_gib": args.max_device_gib,
        "trim_unused_cache": args.trim_unused_cache,
        "dense_rounds": [],
        "memory": [],
    }
    env = eval_env = None
    monitor_stop = threading.Event()
    monitor = None
    peak_device = [0.0]

    def memory(phase):
        def record(cache_stage):
            result["memory"].append(
                {"phase": phase, "cache_stage": cache_stage, **allocator_snapshot()}
            )
            (args.out / "result.json").write_text(json.dumps(result, indent=2))

        record_memory_boundary(record, trim_unused_cache=args.trim_unused_cache)
        used = result["memory"][-1]["device_used_gib"]
        if used >= args.max_device_gib:
            raise RuntimeError(f"Device memory budget exceeded: {used:.2f}GiB")

    def sample_device():
        while not monitor_stop.wait(0.1):
            free, total = torch.cuda.mem_get_info(0)
            peak_device[0] = max(peak_device[0], (total - free) / 1024**3)

    try:
        env = environment.AssemblyEnvironment(args.assets, args.bank, args.num_envs)
        train.require_physics_bank(env)
        result["physics_identity"] = env.physics_identity
        result["task_names"] = env.task_names
        memory("constructor")
        with np.load(args.bank / "bank.npz", allow_pickle=False) as bank:
            active_parts = bank["active_part"].copy() if "active_part" in bank else None
            selected = dense_entries(
                bank["task_ids"],
                bank["family"],
                len(env.tasks),
                active_parts=active_parts,
                max_per_stratum=args.dense_per_stratum,
            )
            held_parts = bank["held_part"].copy()
            family_names = bank["family"].copy()
        ids = torch.arange(env.num_envs, device=env.device)
        for entry in selected:
            indices = torch.full_like(ids, entry)
            env.reset(ids, indices)
            targets, opening = env.targets[:, :6].clone(), env.gripper_open.clone()
            row = {
                "entry_id": env.bank_metadata["entry_ids"][entry],
                "family": str(family_names[entry]),
                "seconds": [],
                "final_contact_counts": [],
            }
            if active_parts is not None:
                row["active_part"] = int(active_parts[entry])
            for _ in range(DENSE_HOLD_INTERVALS):
                started = time.perf_counter()
                env.hold(targets, opening)
                torch.cuda.synchronize()
                row["seconds"].append(time.perf_counter() - started)
                check_dense_state(env.snapshot(), int(held_parts[entry]))
                row["final_contact_counts"].append(
                    int(wp.to_torch(env.solver.mjw_data.nacon)[0])
                )
            result["dense_rounds"].append(row)
            memory(f"dense_entry_{entry}")
        torch.cuda.reset_peak_memory_stats()
        peak_device[0] = result["memory"][-1]["device_used_gib"]
        monitor = threading.Thread(target=sample_device, daemon=True)
        monitor.start()
        trainer = train.Trainer(
            env,
            bank_identity=train.path_identity(args.bank),
            asset_identity=train.path_identity(args.assets),
        )
        memory("trainer")
        if args.eval_bank:
            if train.path_identity(args.eval_bank) == trainer.bank_identity:
                raise ValueError("Evaluation bank must be held out from training")
            eval_env = environment.AssemblyEnvironment(
                args.assets, args.eval_bank, args.eval_num_envs, seed=100000
            )
            train.require_physics_bank(eval_env)
            train.validate_evaluation_separation(
                env.bank_metadata, eval_env.bank_metadata
            )
            memory("evaluation_constructor")
        started = time.perf_counter()
        metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        result["ppo_seconds"] = time.perf_counter() - started
        result["ppo_metrics"] = metrics
        if not metrics_finite(metrics):
            raise RuntimeError("PPO metrics are nonfinite")
        if not all(
            torch.isfinite(parameter).all() for parameter in trainer.model.parameters()
        ):
            raise RuntimeError("PPO parameters are nonfinite")
        memory("ppo_update")
        monitor_stop.set()
        monitor.join()
        if peak_device[0] >= args.max_device_gib:
            raise RuntimeError("Sampled PPO device memory exceeded budget")
        result["completed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        try:
            result["failure_allocator"] = allocator_snapshot()
        except Exception as snapshot_error:
            result["failure_allocator_error"] = str(snapshot_error)
        raise
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join()
        result["ppo_device_peak_sampled_100ms_gib"] = peak_device[0]
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        cleanup_errors = []
        for label, instance in (("evaluation", eval_env), ("training", env)):
            if instance is None:
                continue
            try:
                instance.close()
            except Exception as cleanup_error:
                cleanup_errors.append(f"{label}: {cleanup_error}")
        if cleanup_errors:
            result["cleanup_error"] = "; ".join(cleanup_errors)
            (args.out / "result.json").write_text(json.dumps(result, indent=2))
            if "error" not in result:
                raise RuntimeError(result["cleanup_error"])


if __name__ == "__main__":
    main()
