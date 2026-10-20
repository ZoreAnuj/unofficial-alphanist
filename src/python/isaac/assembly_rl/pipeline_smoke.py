"""Verify an actual checkpoint resume, another PPO update, and held-out evaluation.

This finite integration check does not establish policy convergence. Use train.py
for training and the full five-seed,32-trial evaluation protocol.
"""

import argparse
import json
import math
import os
import pathlib
from typing import Any

import torch

from isaac.assembly_rl import train


def assert_equal(actual: Any, expected: Any) -> None:
    """Exact nested comparison, including controller/optimizer state and RNG bytes."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            assert_equal(first, second)
    else:
        assert actual == expected


def frozen_state(trainer: train.Trainer) -> dict:
    return {
        "model": train.cpu_state(trainer.model.state_dict()),
        "sampler": trainer.sampler.state_dict(),
        "rng": train.random_state(),
    }


def verify_restored(trainer: train.Trainer, saved: dict) -> None:
    assert trainer.iteration == saved["iteration"]
    restored = {
        **frozen_state(trainer),
        "optimizer": trainer.optimizer.state_dict(),
        "environment": trainer.env.state_dict(),
    }
    for key, value in restored.items():
        assert_equal(value, saved[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "bank", "eval-bank", "checkpoint", "out"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--eval-trials-per-task", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.eval_trials_per_task) < 1:
        parser.error("Environment and trial counts must be positive")
    if train.world_size() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run this probe on one GPU; train.py supports distributed resume")
    from isaac.assembly_rl import environment, preview

    env = eval_env = None
    try:
        env = environment.AssemblyEnvironment(
            args.assets, args.bank, args.num_envs, device=args.device
        )
        train.require_physics_bank(env)
        trainer = train.Trainer(
            env,
            device=args.device,
            bank_identity=train.path_identity(args.bank),
            asset_identity=train.path_identity(args.assets),
        )
        trainer.load_checkpoint(args.checkpoint)
        saved = torch.load(
            args.checkpoint / "rank-00000.pt", map_location="cpu", weights_only=True
        )
        verify_restored(trainer, saved)
        args.out.mkdir(parents=True, exist_ok=False)
        preview.capture_snapshot(env, args.out / "restored_snapshot.npz")
        metrics = trainer.train_iteration()
        assert trainer.iteration == saved["iteration"] + 1
        assert all(math.isfinite(value) for value in metrics.values())
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        assert any(
            not torch.equal(parameter.detach().cpu(), saved["model"][name])
            for name, parameter in trainer.model.named_parameters()
        ), "Resumed PPO update did not change any policy parameters"
        resumed_checkpoint = trainer.save_checkpoint(args.out)
        eval_env = environment.AssemblyEnvironment(
            args.assets, args.eval_bank, args.num_envs, device=args.device, seed=100000
        )
        train.require_physics_bank(eval_env)
        if eval_env.task_names != env.task_names:
            raise ValueError("Evaluation must cover the same tasks as training")
        train.validate_evaluation_separation(env.bank_metadata, eval_env.bank_metadata)
        before = frozen_state(trainer)
        evaluation = train.evaluate(
            trainer.model,
            eval_env,
            trials_per_task=args.eval_trials_per_task,
            seed=100000,
        )
        assert_equal(frozen_state(trainer), before)
        preview.capture_snapshot(eval_env, args.out / "evaluation_final_snapshot.npz")
        result = {
            "integration_checks_passed": True,
            "learning_success_demonstrated": False,
            "task_names": env.task_names,
            "physics_identity": env.physics_identity,
            "checkpoint": str(args.checkpoint),
            "resumed_checkpoint": str(resumed_checkpoint),
            "restored_iteration": saved["iteration"],
            "resumed_update": metrics,
            "evaluation": evaluation,
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
