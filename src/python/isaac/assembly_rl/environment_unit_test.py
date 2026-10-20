import types

import torch

from isaac.assembly_rl import environment


def test_episode_horizon_scales_with_each_tasks_assembly_pairs(monkeypatch):
    env = environment.AssemblyEnvironment.__new__(environment.AssemblyEnvironment)
    env.num_envs, env.nq = 2, 8
    env.config = dict(environment.DEFAULT_CONFIG)
    env.tasks = [
        {"parts": [{"fixed": True}] + [{"fixed": False}] * count} for count in (2, 3)
    ]
    env.task_ids = torch.tensor([0, 1])
    env.moving = torch.tensor([[False, True, True, False], [False, True, True, True]])
    env.last_actions = torch.zeros(2, 7)
    env.state = types.SimpleNamespace(joint_q=torch.zeros(2, 8))
    env.solver = types.SimpleNamespace(
        mjw_data=types.SimpleNamespace(qfrc_actuator=torch.zeros(2, 8))
    )
    env.vmax, env.lower, env.upper = torch.ones(8), -torch.ones(8), torch.ones(8)
    env.elapsed = torch.zeros(2, dtype=torch.long)
    env.failure = torch.zeros(2, dtype=torch.bool)
    env.bank_indices = torch.tensor([0, 1])
    poses = torch.zeros(2, 4, 7)
    poses[..., 6] = 1
    env.goals = poses.clone()
    env.goals[..., 0] = 1
    env.metric_points = torch.zeros(2, 4, 1, 3)
    env.hold = lambda *_: None
    env.observe = dict
    env.snapshot = lambda: {"part_poses": poses, "joint_qd": torch.zeros(2, 8)}
    monkeypatch.setattr(environment.wp, "to_torch", lambda value: value)

    for elapsed, expected in (
        (374, [False, False]),
        (749, [True, False]),
        (1124, [True, True]),
    ):
        env.elapsed.fill_(elapsed)
        _, _, terminated, truncated, _ = env.step(torch.zeros(2, 7))
        assert not terminated.any()
        assert truncated.tolist() == expected
    assert env.max_horizon_steps == 1125

    env.goals.copy_(poses)
    env.elapsed.fill_(1124)
    _, _, terminated, truncated, _ = env.step(torch.zeros(2, 7))
    assert terminated.all()
    assert not truncated.any()


def test_spawn_rejects_malformed_contacts_even_with_finite_near_zero_depth():
    frames = torch.eye(3).repeat(5, 1, 1)
    frames[1] = 0
    frames[2, 0] *= 0.5
    frames[3, 2] *= -1
    positions = torch.zeros(5, 3)
    positions[4, 0] = float("nan")
    valid = environment.valid_contact_geometry(
        positions, frames, torch.full((5,), -3e-9)
    )
    assert valid.tolist() == [True, False, False, False, False]


def test_reward_requires_every_movable_part_and_uses_no_progress_reward() -> None:
    errors = torch.tensor([[0.0, 0.0005, 0.002], [0.0, 0.0005, 0.0008]])
    moving = torch.tensor([[False, True, True], [False, True, True]])
    success, error = environment.assembly_success(errors, moving)
    assert success.tolist() == [False, True]
    torch.testing.assert_close(error, torch.tensor([0.002, 0.0008]))
    reward = environment.sparse_reward(
        success, torch.ones(2, 7), torch.zeros(2, 8), torch.zeros(2, 8), 1e-6
    )
    torch.testing.assert_close(reward, torch.tensor([-7e-6, 1.0 - 7e-6]))


def test_checkpoint_bank_schema_rejects_unvalidated_resets(tmp_path) -> None:
    import json

    (tmp_path / "bank.json").write_text(json.dumps({"training_ready": False}))
    import pytest

    with pytest.raises(ValueError, match="validated"):
        environment.load_bank(tmp_path, torch.device("cpu"))
