"""Finite actual-physics integration checks, never a training success claim."""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from isaac.assembly_rl import environment, preview, reset_bank_builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--task", default="02_motor_mount")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()
    start = time.monotonic()
    env = environment.AssemblyEnvironment(
        args.assets, num_envs=args.num_envs, config={"task_names": [args.task]}, seed=7
    )
    print("MODEL_READY", time.monotonic() - start, flush=True)
    geometry = reset_bank_builder.load_task_geometry(args.assets, env.tasks)[0]
    rng = np.random.default_rng(7)
    records = [
        reset_bank_builder.candidate(
            env.tasks[0],
            geometry,
            "loose",
            rng,
            env.max_parts,
            config=reset_bank_builder.ValidationConfig(),
        )
        for _ in range(env.num_envs)
    ]
    env.initialize(
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        env._tensor(np.asarray([row["part_poses"] for row in records])),
        env._tensor(np.asarray([row["joint_q"] for row in records])),
        env._tensor(np.asarray([row["goals"] for row in records])),
    )
    initial = env.snapshot()
    for step in range(args.steps):
        env.hold(initial["joint_q"][:, :6], torch.ones(env.num_envs, device=env.device))
        state = env.snapshot()
        print(
            json.dumps(
                {
                    "step": step,
                    "failure": state["safety_failure"].tolist(),
                    "penetration_m": state["max_penetration_m"].tolist(),
                    "joint_speed": state["joint_qd"].abs().amax(-1).tolist(),
                    "part_speed": state["part_velocities"]
                    .abs()
                    .amax(dim=(-1, -2))
                    .tolist(),
                }
            ),
            flush=True,
        )
    obs = env.observe()
    assert obs["points"].shape == (env.num_envs, env.num_points, 3)
    assert obs["proprio"].shape == (env.num_envs, 140)
    checkpoint = env.state_dict()
    action = torch.zeros((env.num_envs, 7), device=env.device)
    action[:, 6] = 1
    expected = env.step(action)
    expected_state = env.snapshot()
    env.load_state_dict(checkpoint)
    actual = env.step(action)
    actual_state = env.snapshot()
    delta = (
        (expected_state["part_poses"] - actual_state["part_poses"]).abs().max().item()
    )
    result = {
        "seconds": time.monotonic() - start,
        "task": args.task,
        "num_envs": env.num_envs,
        "steps": args.steps,
        "failure": actual_state["safety_failure"].tolist(),
        "resume_max_pose_delta": delta,
        "resume_reward_delta": (expected[1] - actual[1]).abs().max().item(),
        "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
        "physics_only_no_learning_claim": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    preview.capture_snapshot(env, args.output / "physical_snapshot.npz")
    env.close()


if __name__ == "__main__":
    main()
