"""PPO rollout, per-worker SGS, atomic distributed checkpoints and frozen evaluation.

Run with python -m isaac.assembly_rl.train or torchrun. The environment owns
physics validation and rejects unaccepted reset banks by default.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import random
import subprocess
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import distributed

from isaac.assembly_rl import policy, reset_sampling


def distributed_active() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def distributed_timeout() -> datetime.timedelta:
    value = os.environ.get("ASSEMBLY_DISTRIBUTED_TIMEOUT_SECONDS", "300")
    if (
        not value.isascii()
        or not value.isdecimal()
        or str(int(value)) != value
        or not 30 <= int(value) <= 600
    ):
        raise ValueError("ASSEMBLY_DISTRIBUTED_TIMEOUT_SECONDS must be 30-600")
    return datetime.timedelta(seconds=int(value))


def rank() -> int:
    return distributed.get_rank() if distributed_active() else 0


def world_size() -> int:
    return distributed.get_world_size() if distributed_active() else 1


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(path: pathlib.Path) -> str:
    """Hash a file or every file inside a bank/asset directory."""
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(child)))
    return digest.hexdigest()


def checkpoint_snapshot_due(
    iteration: int, previous: int | None, cadence: int, final: bool
) -> bool:
    return previous is None or final or iteration // cadence > previous // cadence


def git_revision() -> str:
    """Git is optional in minimal training containers; source hashes remain recorded."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Checkpoint state must contain tensors and primitives, got {type(value)}"
    )


def random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class Trainer:
    def __init__(
        self,
        env: Any,
        config: policy.PPOConfig | None = None,
        *,
        seed: int = 0,
        device: str = "cuda:0",
        bank_identity: str,
        asset_identity: str,
        sgs_window: int = 100,
        sgs_epsilon: float = 1e-4,
        provenance: dict | None = None,
    ):
        self.env, self.config = env, config or policy.PPOConfig()
        self.device = torch.device(device)
        self.bank_identity, self.asset_identity = bank_identity, asset_identity
        self.provenance = provenance or {}
        self.iteration = 0
        self.task_names = getattr(env, "task_names", None)
        if self.task_names is None:
            self.task_names = [
                str(i) for i in range(int(torch.as_tensor(env.bank_task_ids).max()) + 1)
            ]
        if distributed_active():
            # Fail collectively before any shape-dependent parameter broadcasts.
            contract = {
                "asset_identity": asset_identity,
                "ppo_config": dataclasses.asdict(self.config),
                "environment_config": getattr(env, "config", None),
                "physics_identity": getattr(env, "physics_identity", None),
                "task_names": getattr(env, "task_names", None),
                "dimensions": (
                    env.num_envs,
                    env.num_points,
                    env.proprio_dim,
                    env.action_dim,
                ),
                "sgs": (sgs_window, sgs_epsilon),
                "source_sha256": self.provenance.get("source_sha256"),
            }
            contracts = [None] * world_size()
            distributed.all_gather_object(contracts, contract)
            if any(other != contract for other in contracts):
                raise ValueError(
                    "Distributed worker configuration differs: assets, source, "
                    "environment dimensions, PPO and SGS must match"
                )
        worker_seed = seed + rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        self.model = policy.ActorCritic(
            env.num_points, env.proprio_dim, env.action_dim
        ).to(self.device)
        if distributed_active():
            # Parameters are shared; normalizer histories remain local to each worker.
            for parameter in self.model.parameters():
                distributed.broadcast(parameter.data, src=0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.sampler = reset_sampling.SuccessGuidedSampler(
            env.bank_size, sgs_window, sgs_epsilon, worker_seed
        )
        env_ids = torch.arange(env.num_envs, device=self.device)
        env.reset(env_ids, self.sample_resets(env.num_envs))

    def sample_resets(self, count: int) -> torch.Tensor:
        return torch.as_tensor(
            self.sampler.sample(count), dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def collect_rollout(self) -> tuple[policy.PPOBatch, dict[str, Any]]:
        self.model.eval()
        storage: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "points",
                "proprio",
                "actor_proprio_normalized",
                "critic_proprio_normalized",
                "actions",
                "old_log_probs",
                "old_values",
                "old_means",
                "old_stds",
                "rewards",
                "next_values",
                "terminated",
                "truncated",
            )
        }
        episodes = successes = safety_failures = 0
        task_counters = torch.zeros(
            (len(self.task_names), 3), dtype=torch.float64, device=self.device
        )
        observation = self.env.observe()
        for _ in range(self.config.rollout_steps):
            # Cited RSL/SimBa behavior: update during collection; freeze during PPO.
            self.model.update_normalizers(observation["proprio"])
            actor_proprio = self.model.actor.normalizer(observation["proprio"])
            critic_proprio = self.model.critic.normalizer(observation["proprio"])
            distribution, values = self.model.forward_normalized(
                observation["points"], actor_proprio, critic_proprio
            )
            storage["actor_proprio_normalized"].append(actor_proprio.detach().clone())
            storage["critic_proprio_normalized"].append(critic_proprio.detach().clone())
            actions = distribution.sample()
            for key in ("points", "proprio"):
                storage[key].append(observation[key].detach().clone())
            storage["actions"].append(actions)
            storage["old_log_probs"].append(distribution.log_prob(actions))
            storage["old_values"].append(values)
            storage["old_means"].append(distribution.base_dist.loc.clone())
            storage["old_stds"].append(distribution.base_dist.scale.clone())
            next_observation, reward, terminated, truncated, info = self.env.step(
                actions.clamp(-1, 1)
            )
            # The timeout bootstrap must use its terminal observation, before resetting.
            next_values = self.model.critic(**next_observation).squeeze(-1)
            for key, value in (
                ("rewards", reward),
                ("next_values", next_values),
                ("terminated", terminated),
                ("truncated", truncated),
            ):
                storage[key].append(value.detach().clone())
            done = terminated | truncated
            observation = next_observation
            ids = done.nonzero(as_tuple=False).flatten()
            count = ids.numel()
            if count:
                self.sampler.update(
                    info["bank_index"][ids].cpu().numpy(),
                    info["success"][ids].cpu().numpy(),
                )
                episodes += count
                successes += int(info["success"][ids].sum())
                safety_failures += int(info["safety_failure"][ids].sum())
                task_counters.index_add_(
                    0,
                    info["task_id"][ids].long(),
                    torch.stack(
                        (
                            torch.ones_like(info["success"][ids]),
                            info["success"][ids],
                            info["safety_failure"][ids],
                        ),
                        dim=1,
                    ).to(torch.float64),
                )
                observation = self.env.reset(ids, self.sample_resets(count))
        stacked = {key: torch.stack(values) for key, values in storage.items()}
        # Reuse rollout storage to avoid one host synchronization per step.
        reward_sum = float(stacked["rewards"].sum(dtype=torch.float64))
        # Running moments may change before the next sampled action. Continuing
        # transitions use that next collected value, as in RSL rollout storage.
        # Time limits retain their pre-reset terminal-observation bootstrap.
        continuing = ~(stacked["terminated"][:-1] | stacked["truncated"][:-1])
        stacked["next_values"][:-1] = torch.where(
            continuing, stacked["old_values"][1:], stacked["next_values"][:-1]
        )
        advantages, returns = policy.compute_gae(
            stacked["rewards"],
            stacked["old_values"],
            stacked["next_values"],
            stacked["terminated"],
            stacked["truncated"],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        batch = policy.PPOBatch(
            **{
                key: stacked[key].flatten(0, 1)
                for key in (
                    "points",
                    "proprio",
                    "actor_proprio_normalized",
                    "critic_proprio_normalized",
                    "actions",
                    "old_log_probs",
                    "old_values",
                    "old_means",
                    "old_stds",
                )
            },
            advantages=advantages.flatten(),
            returns=returns.flatten(),
        )
        counters = torch.cat(
            (
                torch.tensor(
                    [episodes, successes, safety_failures, reward_sum],
                    dtype=torch.float64,
                    device=self.device,
                ),
                task_counters.flatten(),
            )
        )
        if distributed_active():
            distributed.all_reduce(counters)
        values = counters.tolist()
        episodes, successes, safety_failures, reward_sum = values[:4]
        task_results = {}
        for index, name in enumerate(self.task_names):
            count, wins, failures = values[4 + index * 3 : 7 + index * 3]
            task_results[name] = {
                "episodes": int(count),
                "successes": int(wins),
                "safety_failures": int(failures),
                "success_rate": wins / count if count else None,
            }
        return batch, {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / max(episodes, 1),
            "safety_failures": safety_failures,
            "mean_step_reward": reward_sum
            / (self.config.rollout_steps * self.env.num_envs * world_size()),
            "tasks": task_results,
        }

    def train_iteration(self) -> dict[str, Any]:
        batch, metrics = self.collect_rollout()
        self.model.train()
        metrics.update(
            policy.ppo_update(self.model, self.optimizer, batch, self.config)
        )
        self.iteration += 1
        metrics["iteration"] = self.iteration
        metrics["environment_steps"] = (
            self.iteration
            * self.env.num_envs
            * self.config.rollout_steps
            * world_size()
        )
        return metrics

    def save_checkpoint(
        self, output: pathlib.Path, *, capture_render_snapshot: bool = False
    ) -> pathlib.Path:
        output = pathlib.Path(output)
        generation = [uuid.uuid4().hex if rank() == 0 else None]
        if distributed_active():
            distributed.broadcast_object_list(generation, src=0)
        name = f"checkpoint-{self.iteration:08d}-{generation[0]}"
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "iteration": self.iteration,
            "world_size": world_size(),
            "rank": rank(),
            "num_envs": self.env.num_envs,
            "bank_identity": self.bank_identity,
            "asset_identity": self.asset_identity,
            "config": dataclasses.asdict(self.config),
            "provenance": self.provenance,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sampler": self.sampler.state_dict(),
            "environment": self.env.state_dict(),
            "rng": random_state(),
        }
        filename = f"rank-{rank():05d}.pt"
        target = destination / filename
        temporary = destination / f".{filename}.incomplete"
        with temporary.open("wb") as handle:
            torch.save(cpu_state(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        entry = {
            "file": filename,
            "sha256": sha256(target),
            "bank_identity": self.bank_identity,
        }
        entries = [None] * world_size()
        if distributed_active():
            distributed.all_gather_object(entries, entry)
        else:
            entries[0] = entry
        manifest = {
            "version": 2,
            "world_size": world_size(),
            "iteration": self.iteration,
            "asset_identity": self.asset_identity,
            "files": entries,
        }
        # A manifest appears only after every rank finished its file. Writing the
        # same manifest per rank supports both shared storage and separate VM disks.
        manifest_tmp = destination / f".manifest-rank-{rank()}.incomplete"
        with manifest_tmp.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_tmp.replace(destination / "manifest.json")
        if distributed_active():
            distributed.barrier()
        if capture_render_snapshot:
            run_rank_zero_evaluation(
                lambda progress: self._capture_checkpoint_snapshot(
                    destination, progress
                )
            )
        return destination

    def _capture_checkpoint_snapshot(
        self, checkpoint: pathlib.Path, progress: Callable[[], None]
    ) -> None:
        from isaac.assembly_rl import preview

        committed = checkpoint / "manifest.json"
        if json.loads(committed.read_text())["iteration"] != self.iteration:
            raise ValueError("Snapshot iteration differs from committed checkpoint")
        temporary = checkpoint / ".render_snapshot.npz.incomplete"
        preview.capture_snapshot(self.env, temporary)
        progress()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        snapshot = checkpoint / "render_snapshot.npz"
        temporary.replace(snapshot)
        with np.load(snapshot, allow_pickle=False) as archive:
            names, task_ids = archive["task_names"].tolist(), archive["task_ids"]
            frames = [
                {
                    "task_id": int(task),
                    "task_name": names[int(task)],
                    "env_index": int(np.flatnonzero(task_ids == task)[0]),
                }
                for task in np.unique(task_ids)
            ]
        for frame in frames:
            preview.read_snapshot(snapshot, frame["env_index"])
        metadata = {
            "schema_version": 1,
            "checkpoint_id": checkpoint.name,
            "checkpoint_manifest_sha256": sha256(committed),
            "iteration": self.iteration,
            "environment_steps": self.iteration
            * self.env.num_envs
            * self.config.rollout_steps
            * world_size(),
            "physics_identity": self.env.physics_identity,
            "asset_identity": self.asset_identity,
            "rank": 0,
            "world_size": world_size(),
            "task_names": names,
            "snapshot": {"file": snapshot.name, "sha256": sha256(snapshot)},
            "frames": frames,
            "unavailable_task_names": [
                name for index, name in enumerate(names) if index not in task_ids
            ],
        }
        temporary = checkpoint / ".render_snapshot.json.incomplete"
        with temporary.open("w") as handle:
            json.dump(metadata, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(checkpoint / "render_snapshot.json")

    def load_checkpoint(self, checkpoint: pathlib.Path) -> None:
        checkpoint = pathlib.Path(checkpoint)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if distributed_active():
            identity = hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest()
            identities = [None] * world_size()
            distributed.all_gather_object(identities, identity)
            if any(other != identity for other in identities):
                raise ValueError(
                    "Distributed checkpoint manifests differ across workers"
                )
        if manifest["version"] not in (1, 2) or manifest["world_size"] != world_size():
            raise ValueError(
                "Checkpoint version/world size mismatch; exact resume requires the same workers"
            )
        if manifest["asset_identity"] != self.asset_identity:
            raise ValueError("Checkpoint asset identity differs")
        expected_name = f"rank-{rank():05d}.pt"
        entries = {entry["file"]: entry for entry in manifest["files"]}
        entry = entries.get(expected_name, {})
        bank_identity = (
            manifest.get("bank_identity")
            if manifest["version"] == 1
            else entry.get("bank_identity")
        )
        if bank_identity != self.bank_identity:
            raise ValueError("Checkpoint worker bank identity differs")
        target = checkpoint / expected_name
        if entry.get("sha256") != sha256(target):
            raise ValueError("Checkpoint checksum mismatch")
        # Never unpickle arbitrary Python objects from a supplied checkpoint.
        state = torch.load(target, map_location="cpu", weights_only=True)
        if (
            state.get("version") != 1
            or state["world_size"] != world_size()
            or state["iteration"] != manifest["iteration"]
            or len(state["sampler"]["history"]) != self.env.bank_size
            or state["config"] != dataclasses.asdict(self.config)
            or state["num_envs"] != self.env.num_envs
            or state["rank"] != rank()
        ):
            raise ValueError("Checkpoint configuration or worker assignment differs")
        if (
            state["bank_identity"] != self.bank_identity
            or state["asset_identity"] != self.asset_identity
        ):
            raise ValueError("Checkpoint state identity differs")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler = reset_sampling.SuccessGuidedSampler.from_state_dict(
            state["sampler"]
        )
        self.env.load_state_dict(state["environment"])
        self.iteration = state["iteration"]
        restore_random_state(state["rng"])
        if distributed_active():
            distributed.barrier()


def run_rank_zero_evaluation(operation: Callable[[Callable[[], None]], Any]) -> Any:
    """Keep finite collective timeouts per evaluation step, not per entire sweep."""
    if not distributed_active():
        return operation(lambda: None)

    def progress() -> None:
        distributed.broadcast_object_list([None], src=0)

    if rank() != 0:
        while True:
            status = [None]
            distributed.broadcast_object_list(status, src=0)
            if status[0] is not None:
                if status[0]["error"] is not None:
                    raise RuntimeError(status[0]["error"])
                return None
    try:
        result = operation(progress)
    except Exception as error:
        distributed.broadcast_object_list(
            [
                {
                    "error": f"Rank-zero evaluation failed: {type(error).__name__}: {error}"
                }
            ],
            src=0,
        )
        raise
    distributed.broadcast_object_list([{"error": None}], src=0)
    return result


@torch.no_grad()
def evaluate(
    model: policy.ActorCritic,
    env: Any,
    *,
    trials_per_task: int = 32,
    horizon: int | None = None,
    seed: int = 0,
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Frozen bank, uniform task weight, sampled Gaussian actions; no SGS updates."""
    if horizon is None:
        horizon = getattr(env, "max_horizon_steps", 375)
    if trials_per_task < 1 or horizon < 1:
        raise ValueError("Evaluation trials and horizon must be positive")
    saved_rng, was_training = random_state(), model.training
    try:
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        task_ids = torch.as_tensor(env.bank_task_ids).cpu().numpy()
        chosen = []
        for task in np.unique(task_ids):
            candidates = np.flatnonzero(task_ids == task)
            # Cycle a fixed permutation for balanced coverage within each task.
            chosen.extend(
                np.resize(generator.permutation(candidates), trials_per_task).tolist()
            )
        device = next(model.parameters()).device
        model.eval()
        results: dict[str, dict[str, dict[str, float]]] = {"tasks": {}, "families": {}}
        for offset in range(0, len(chosen), env.num_envs):
            if progress is not None:
                progress()
            indices = chosen[offset : offset + env.num_envs]
            valid_count = len(indices)
            indices.extend([indices[-1]] * (env.num_envs - valid_count))
            bank_indices = torch.tensor(indices, device=device, dtype=torch.long)
            env_ids = torch.arange(env.num_envs, device=device)
            env.reset(env_ids, bank_indices)
            active = torch.arange(env.num_envs, device=device) < valid_count
            succeeded = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
            failed = torch.zeros_like(succeeded)
            for _ in range(horizon):
                if progress is not None:
                    progress()
                distribution, _ = model(**env.observe())
                _, _, terminated, truncated, info = env.step(
                    distribution.sample().clamp(-1, 1)
                )
                succeeded |= active & info["success"]
                failed |= active & info["safety_failure"]
                done = terminated | truncated
                active &= ~done
                if not active.any():
                    break
                # Finished rows remain masked; keep their physics in a valid reset state.
                reset_ids = done.nonzero(as_tuple=False).flatten()
                if reset_ids.numel():
                    env.reset(reset_ids, bank_indices[reset_ids])
            for index in range(valid_count):
                for group, label in (
                    ("tasks", str(int(task_ids[indices[index]]))),
                    ("families", env.bank_family_names[indices[index]]),
                ):
                    row = results[group].setdefault(
                        label, {"episodes": 0, "successes": 0, "safety_failures": 0}
                    )
                    row["episodes"] += 1
                    row["successes"] += int(succeeded[index])
                    row["safety_failures"] += int(failed[index])
        for group in results.values():
            for row in group.values():
                row["success_rate"] = row["successes"] / row["episodes"]
        return {
            "seed": seed,
            "horizon": horizon,
            "trials_per_task": trials_per_task,
            **results,
        }
    finally:
        restore_random_state(saved_rng)
        model.train(was_training)


def validate_evaluation_separation(
    training_metadata: dict, evaluation_metadata: dict | None
) -> None:
    """Reject held-out states present in any worker's independent training bank."""
    from isaac.assembly_rl import reset_bank_builder

    payload = [evaluation_metadata if rank() == 0 else None]
    if distributed_active():
        # Preserve the ID schema; large physical state arrays stay local.
        if rank() == 0 and payload[0] is not None:
            payload[0] = {
                "entry_ids": payload[0].get("entry_ids", []),
                "content_id_schema": payload[0].get("content_id_schema", 1),
                "task_names": payload[0].get("task_names"),
            }
        distributed.broadcast_object_list(payload, src=0)
    error = None
    try:
        if training_metadata.get("task_names") != (payload[0] or {}).get("task_names"):
            raise ValueError("Evaluation must cover the same ordered tasks as training")
        reset_bank_builder.assert_disjoint_bank_entries(
            training_metadata, payload[0] or {}
        )
    except ValueError as failure:
        error = f"worker {rank()}: {failure}"
    errors = [None] * world_size()
    if distributed_active():
        distributed.all_gather_object(errors, error)
    else:
        errors[0] = error
    if any(errors):
        raise ValueError("; ".join(value for value in errors if value))


def require_physics_bank(env: Any) -> None:
    """Legacy banks may support diagnostics, but cannot certify current training."""
    identity = env.bank_metadata.get("physics_identity")
    if not identity or identity != env.physics_identity:
        raise ValueError(
            "Training/evaluation bank physics identity is missing or differs; "
            "validate the bank with the current physical model and controller"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "bank", "out"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--eval-num-envs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--eval-bank", type=pathlib.Path)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--eval-trials-per-task", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--render-every",
        type=int,
        default=200,
        help="Snapshot cadence in updates; also captures the first and final checkpoints.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=375,
        help="Policy steps per assembly pair (375 = 15 seconds at25Hz)",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        min(
            args.num_envs,
            args.eval_num_envs,
            args.iterations,
            args.log_every,
            args.checkpoint_every,
            args.render_every,
            args.eval_every,
            args.eval_seeds,
            args.eval_trials_per_task,
            args.horizon,
        )
        < 1
    ):
        parser.error("Counts and intervals must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", timeout=distributed_timeout())
    from isaac.assembly_rl import environment

    env = eval_env = None
    try:
        env = environment.AssemblyEnvironment(
            args.assets,
            args.bank,
            args.num_envs,
            device=args.device,
            seed=args.seed + rank(),
            config={"horizon_steps": args.horizon},
        )
        require_physics_bank(env)
        provenance = {
            "git_revision": git_revision(),
            "torch": str(torch.__version__),
            "settings": policy.CONFIG_PROVENANCE,
            "source_sha256": {
                path.name: sha256(path)
                for path in (
                    pathlib.Path(__file__),
                    pathlib.Path(policy.__file__),
                    pathlib.Path(reset_sampling.__file__),
                    pathlib.Path(environment.__file__),
                )
            },
        }
        trainer = Trainer(
            env,
            device=args.device,
            seed=args.seed,
            bank_identity=path_identity(args.bank),
            asset_identity=path_identity(args.assets),
            provenance=provenance,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        if args.eval_bank and rank() == 0:
            if path_identity(args.eval_bank) == trainer.bank_identity:
                raise ValueError("Evaluation bank must be held out from training")
            eval_env = environment.AssemblyEnvironment(
                args.assets,
                args.eval_bank,
                args.eval_num_envs,
                device=args.device,
                seed=args.seed + 100000,
                config={"horizon_steps": args.horizon},
            )
            require_physics_bank(eval_env)
        if args.eval_bank:
            validate_evaluation_separation(
                env.bank_metadata, None if eval_env is None else eval_env.bank_metadata
            )
        args.out.mkdir(parents=True, exist_ok=True)
        last_checkpoint_iteration = last_snapshot_iteration = None
        while trainer.iteration < args.iterations:
            started = time.monotonic()
            metrics = trainer.train_iteration()
            metrics["iteration_seconds"] = time.monotonic() - started
            if rank() == 0 and trainer.iteration % args.log_every == 0:
                print(json.dumps(metrics), flush=True)
                with (args.out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
            if args.eval_bank and trainer.iteration % args.eval_every == 0:

                def evaluation_sweep(progress):
                    result = [
                        evaluate(
                            trainer.model,
                            eval_env,
                            seed=args.seed + offset,
                            trials_per_task=args.eval_trials_per_task,
                            progress=progress,
                        )
                        for offset in range(args.eval_seeds)
                    ]
                    (args.out / f"evaluation-{trainer.iteration:08d}.json").write_text(
                        json.dumps(result, indent=2)
                    )

                run_rank_zero_evaluation(evaluation_sweep)
            if trainer.iteration == 1 or trainer.iteration % args.checkpoint_every == 0:
                capture = checkpoint_snapshot_due(
                    trainer.iteration,
                    last_snapshot_iteration,
                    args.render_every,
                    trainer.iteration == args.iterations,
                )
                trainer.save_checkpoint(args.out, capture_render_snapshot=capture)
                last_checkpoint_iteration = trainer.iteration
                if capture:
                    last_snapshot_iteration = trainer.iteration
        if last_checkpoint_iteration != trainer.iteration:
            trainer.save_checkpoint(args.out, capture_render_snapshot=True)
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
        if distributed_active():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
