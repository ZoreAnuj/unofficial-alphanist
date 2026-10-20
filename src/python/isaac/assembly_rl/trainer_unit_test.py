import copy
import dataclasses
import json

import numpy as np
import pytest
import torch

from isaac.assembly_rl import policy, train


class ToyEnvironment:
    num_envs = 4
    num_points = 2
    proprio_dim = 3
    action_dim = 7
    bank_size = 3
    bank_task_ids = torch.tensor([0, 0, 1])
    bank_family_names = ("loose", "near", "loose")

    def __init__(self):
        self.x = torch.zeros(self.num_envs)
        self.age = torch.zeros(self.num_envs, dtype=torch.long)
        self.indices = torch.zeros(self.num_envs, dtype=torch.long)

    def reset(self, env_ids, bank_indices):
        self.indices[env_ids] = bank_indices
        self.age[env_ids] = 0
        self.x[env_ids] = bank_indices.float()
        return self.observe()

    def observe(self):
        return {
            "points": self.x[:, None, None].expand(-1, 2, 3).clone(),
            "proprio": torch.stack(
                (self.x, self.age.float(), self.indices.float()), -1
            ),
        }

    def step(self, actions):
        self.age += 1
        self.x += actions[:, 0] * 0.1
        success = (self.age >= 2) & (self.indices == 1)
        truncated = (self.age >= 3) & ~success
        return (
            self.observe(),
            success.float(),
            success,
            truncated,
            {
                "success": success,
                "safety_failure": torch.zeros_like(success),
                "bank_index": self.indices.clone(),
                "task_id": self.bank_task_ids[self.indices],
                "assembly_error_m": self.x.abs(),
            },
        )

    def state_dict(self):
        return {
            "x": self.x.clone(),
            "age": self.age.clone(),
            "indices": self.indices.clone(),
        }

    def load_state_dict(self, state):
        self.x = state["x"].clone()
        self.age = state["age"].clone()
        self.indices = state["indices"].clone()

    def close(self):
        pass


def config():
    return policy.PPOConfig(rollout_steps=4, epochs=1, minibatches=2)


def make_trainer(seed=19, bank_identity="bank-a", bank_size=3):
    environment = ToyEnvironment()
    environment.bank_size = bank_size
    if bank_size > 3:
        environment.bank_task_ids = torch.cat(
            (environment.bank_task_ids, torch.ones(bank_size - 3, dtype=torch.long))
        )
        environment.bank_family_names += ("loose",) * (bank_size - 3)
    return train.Trainer(
        environment,
        config(),
        seed=seed,
        device="cpu",
        bank_identity=bank_identity,
        asset_identity="assets-a",
    )


def test_resume_reproduces_next_update_and_environment(tmp_path):
    torch.set_num_threads(1)
    original = make_trainer()
    original.train_iteration()
    checkpoint = original.save_checkpoint(tmp_path)
    expected_metrics = original.train_iteration()
    expected = copy.deepcopy(original.model.state_dict())
    expected_env = original.env.state_dict()
    resumed = make_trainer(seed=1000)
    resumed.load_checkpoint(checkpoint)
    actual_metrics = resumed.train_iteration()
    assert actual_metrics == expected_metrics
    assert resumed.iteration == 2
    for key, value in expected.items():
        torch.testing.assert_close(
            resumed.model.state_dict()[key], value, rtol=0, atol=0
        )
    for key, value in expected_env.items():
        torch.testing.assert_close(resumed.env.state_dict()[key], value, rtol=0, atol=0)
    assert resumed.sampler.state_dict() == original.sampler.state_dict()


def test_checkpoint_rejects_changed_bank_and_corrupt_rank_file(tmp_path):
    trainer = make_trainer()
    checkpoint = trainer.save_checkpoint(tmp_path)
    other = make_trainer()
    other.bank_identity = "different"
    with pytest.raises(ValueError, match="identity"):
        other.load_checkpoint(checkpoint)
    rank_file = checkpoint / "rank-00000.pt"
    with rank_file.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        trainer.load_checkpoint(checkpoint)


def test_checkpoint_snapshot_uses_actual_state_after_commit(tmp_path, monkeypatch):
    trainer = make_trainer()
    trainer.train_iteration()
    trainer.env.task_names = ["task-a", "task-b", "not-present"]
    trainer.env.config = {"horizon_steps": 375}
    trainer.env.physics_identity = "measured-model"
    poses = torch.zeros((4, 2, 7))
    poses[..., 6] = 1
    poses[:, :, 0] = torch.arange(4)[:, None] * 0.01
    joints = torch.arange(32, dtype=torch.float32).reshape(4, 8) * 0.001
    state = {
        "task_ids": torch.tensor([1, 0, 1, 1]),
        "joint_q": joints,
        "part_poses": poses,
    }
    before = copy.deepcopy(trainer.env.state_dict())
    captured = []

    def snapshot():
        assert len(list(tmp_path.glob("checkpoint-*/manifest.json"))) == 1
        captured.append(True)
        return state

    monkeypatch.setattr(trainer.env, "snapshot", snapshot, raising=False)
    monkeypatch.setattr(
        trainer.env, "step", lambda *_: pytest.fail("Snapshot advanced physics")
    )
    checkpoint = trainer.save_checkpoint(tmp_path, capture_render_snapshot=True)
    metadata = json.loads((checkpoint / "render_snapshot.json").read_text())
    assert captured == [True]
    assert metadata["checkpoint_id"] == checkpoint.name
    assert metadata["checkpoint_manifest_sha256"] == train.sha256(
        checkpoint / "manifest.json"
    )
    assert metadata["iteration"] == 1
    assert metadata["environment_steps"] == 16
    assert metadata["physics_identity"] == "measured-model"
    assert metadata["asset_identity"] == trainer.asset_identity
    assert metadata["frames"] == [
        {"task_id": 0, "task_name": "task-a", "env_index": 1},
        {"task_id": 1, "task_name": "task-b", "env_index": 0},
    ]
    assert metadata["unavailable_task_names"] == ["not-present"]
    path = checkpoint / metadata["snapshot"]["file"]
    assert train.sha256(path) == metadata["snapshot"]["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in state.items():
            np.testing.assert_array_equal(archive[key], expected.numpy())
    for key, expected in before.items():
        torch.testing.assert_close(trainer.env.state_dict()[key], expected)
    assert not list(checkpoint.glob("*.incomplete"))


@pytest.mark.parametrize(
    ("iteration", "previous", "final", "expected"),
    [
        (50, None, False, True),
        (100, 50, False, False),
        (200, 50, False, True),
        (250, 200, False, False),
        (250, None, False, True),
        (275, 200, True, True),
    ],
)
def test_checkpoint_snapshot_cadence(iteration, previous, final, expected):
    assert train.checkpoint_snapshot_due(iteration, previous, 200, final) is expected


def test_sgs_records_all_episode_outcomes_without_resets_crossing_gae():
    trainer = make_trainer()
    ids = torch.arange(4)
    trainer.env.reset(ids, torch.tensor([1, 0, 2, 1]))
    batch, metrics = trainer.collect_rollout()
    assert metrics["episodes"] >= 4
    assert (
        sum(row["episodes"] for row in metrics["tasks"].values()) == metrics["episodes"]
    )
    assert (
        sum(row["successes"] for row in metrics["tasks"].values())
        == metrics["successes"]
    )
    assert metrics["tasks"]["1"]["successes"] == 0
    assert trainer.sampler.state_dict()["history"][1][0] == 1
    assert trainer.sampler.state_dict()["cursor"][0] >= 1
    assert trainer.sampler.state_dict()["cursor"][2] >= 1
    assert batch.actions.shape == (16, 7)
    assert torch.isfinite(batch.returns).all()


def test_timeout_bootstraps_observation_before_reset():
    trainer = make_trainer()
    trainer.config = dataclasses.replace(config(), rollout_steps=1)
    trainer.env.indices[:] = 2
    trainer.env.age[:] = 2
    trainer.env.x[:] = 10
    batch, _ = trainer.collect_rollout()
    final_x = 10 + batch.actions[:, 0].clamp(-1, 1) * 0.1
    final_observation = {
        "points": final_x[:, None, None].expand(-1, 2, 3),
        "proprio": torch.stack(
            (final_x, torch.full((4,), 3.0), torch.full((4,), 2.0)), -1
        ),
    }
    with torch.no_grad():
        _, final_value = trainer.model(**final_observation)
    torch.testing.assert_close(batch.returns, trainer.config.gamma * final_value)


def test_rollout_reuses_step_and_reset_observations(monkeypatch):
    trainer = make_trainer()
    calls = {"observe": 0, "reset": 0}
    original_observe, original_reset = trainer.env.observe, trainer.env.reset

    def observe():
        calls["observe"] += 1
        return original_observe()

    def reset(*args):
        calls["reset"] += 1
        return original_reset(*args)

    monkeypatch.setattr(trainer.env, "observe", observe)
    monkeypatch.setattr(trainer.env, "reset", reset)
    batch, _ = trainer.collect_rollout()
    assert calls["reset"] > 0
    assert calls["observe"] == 1 + trainer.config.rollout_steps + calls["reset"]
    torch.testing.assert_close(batch.points[:, 0, 0], batch.proprio[:, 0])


def test_evaluation_balances_tasks_and_does_not_update_sgs():
    trainer = make_trainer()
    before = copy.deepcopy(trainer.sampler.state_dict())
    model_before = copy.deepcopy(trainer.model.state_dict())
    rng_before = train.random_state()
    result = train.evaluate(
        trainer.model, ToyEnvironment(), trials_per_task=6, horizon=5, seed=8
    )
    assert result["tasks"]["0"]["episodes"] == 6
    assert result["tasks"]["1"]["episodes"] == 6
    assert result["tasks"]["1"]["success_rate"] == 0
    assert 0 < result["tasks"]["0"]["success_rate"] < 1
    assert trainer.sampler.state_dict() == before
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
    rng_after = train.random_state()
    for key in ("python", "numpy"):
        assert rng_after[key] == rng_before[key]
    torch.testing.assert_close(rng_after["torch"], rng_before["torch"], rtol=0, atol=0)
    for actual, expected in zip(rng_after["cuda"], rng_before["cuda"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _distributed_worker(worker_rank, rendezvous, output):
    import pathlib

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=worker_rank, world_size=2
    )
    try:
        trainer = make_trainer(
            bank_identity=f"worker-bank-{worker_rank}", bank_size=3 + worker_rank
        )
        trainer.train_iteration()
        params = torch.cat(
            [parameter.detach().flatten() for parameter in trainer.model.parameters()]
        )
        gathered = [torch.empty_like(params) for _ in range(2)]
        dist.all_gather(gathered, params)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        means = [torch.empty(3) for _ in range(2)]
        dist.all_gather(means, trainer.model.actor.normalizer.mean)
        assert not torch.equal(means[0], means[1])
        trainer.env.task_names = ["0", "1"]
        trainer.env.config = {}
        trainer.env.physics_identity = "toy-physics"

        def snapshot():
            assert worker_rank == 0
            poses = torch.zeros((4, 1, 7))
            poses[..., 6] = 1
            return {
                "task_ids": trainer.env.bank_task_ids[trainer.env.indices],
                "joint_q": torch.zeros((4, 8)),
                "part_poses": poses,
            }

        trainer.env.snapshot = snapshot
        # Worker-local banks must resume from both shared and separate VM disks.
        for directory in ("shared", f"worker-{worker_rank}"):
            checkpoint = trainer.save_checkpoint(
                pathlib.Path(output) / directory, capture_render_snapshot=True
            )
            assert (checkpoint / "render_snapshot.json").exists() == (
                directory == "shared" or worker_rank == 0
            )
            expected = trainer.train_iteration()
            resumed = make_trainer(
                seed=900,
                bank_identity=f"worker-bank-{worker_rank}",
                bank_size=3 + worker_rank,
            )
            resumed.load_checkpoint(checkpoint)
            actual = resumed.train_iteration()
            assert expected == actual
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(
                    resumed.model.state_dict()[key], value, rtol=0, atol=0
                )
        # A stale but valid manifest on one VM must not resume mixed policies.
        if worker_rank == 1:
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["iteration"] += 1
            manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="manifests differ"):
            trainer.load_checkpoint(checkpoint)
        for schema in (1, 2):
            training_metadata = {
                "content_id_schema": schema,
                "entry_ids": [f"state-{worker_rank}"],
            }
            # An overlap on worker 1 must reject evaluation on every worker.
            with pytest.raises(ValueError, match="worker 1.*overlap"):
                train.validate_evaluation_separation(
                    training_metadata,
                    {"content_id_schema": schema, "entry_ids": ["state-1"]}
                    if worker_rank == 0
                    else None,
                )
            train.validate_evaluation_separation(
                training_metadata,
                {"content_id_schema": schema, "entry_ids": ["held-out"]}
                if worker_rank == 0
                else None,
            )
        with pytest.raises(ValueError, match="same ordered tasks"):
            train.validate_evaluation_separation(
                {
                    "task_names": ["gantry", "gear"],
                    "entry_ids": [f"state-{worker_rank}"],
                },
                {"task_names": ["gantry"], "entry_ids": ["held-out"]}
                if worker_rank == 0
                else None,
            )
        # A genuinely mismatched worker schema must still fail collectively.
        with pytest.raises(ValueError, match="worker 1.*different content-ID schemas"):
            train.validate_evaluation_separation(
                {
                    "content_id_schema": 2 if worker_rank == 0 else 1,
                    "entry_ids": [f"state-{worker_rank}"],
                },
                {"content_id_schema": 2, "entry_ids": ["held-out"]}
                if worker_rank == 0
                else None,
            )
    finally:
        dist.destroy_process_group()


def test_distributed_parameters_sync_local_normalizers_and_rank_resume(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )


def test_rollout_likelihoods_stay_consistent_when_running_statistics_change():
    torch.set_num_threads(1)
    trainer = make_trainer()
    initial_mean = trainer.model.actor.normalizer.mean.clone()
    # Independent actor/critic moments must each retain their own rollout features.
    trainer.model.critic.normalizer.update(torch.full((4, 3), 10.0))
    batch, _ = trainer.collect_rollout()
    assert not torch.equal(initial_mean, trainer.model.actor.normalizer.mean)
    assert trainer.model.actor.normalizer.count > 15
    assert not torch.equal(
        batch.actor_proprio_normalized, batch.critic_proprio_normalized
    )
    with torch.no_grad():
        distribution, values = trainer.model.forward_normalized(
            batch.points,
            batch.actor_proprio_normalized,
            batch.critic_proprio_normalized,
        )
    torch.testing.assert_close(
        distribution.log_prob(batch.actions), batch.old_log_probs, rtol=1e-6, atol=2e-6
    )
    torch.testing.assert_close(values, batch.old_values, rtol=1e-6, atol=2e-6)
    # With zero optimizer learning rate, every PPO ratio must remain one despite
    # the different statistics that were used at successive collection steps.
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.0)
    metrics = policy.ppo_update(
        trainer.model, optimizer, batch, dataclasses.replace(config(), desired_kl=None)
    )
    assert abs(metrics["approx_kl"]) < 1e-7
    assert metrics["clip_fraction"] == 0


def test_gae_uses_next_collected_value_after_normalization_updates():
    trainer = make_trainer()
    trainer.config = dataclasses.replace(config(), gae_lambda=0.0)
    batch, _ = trainer.collect_rollout()
    # All first transitions continue with zero reward; lambda=0 isolates the TD
    # target, which must use the value actually collected at the next step.
    count = trainer.env.num_envs
    torch.testing.assert_close(
        batch.returns[:count],
        trainer.config.gamma * batch.old_values[count : 2 * count],
    )


def _distributed_mismatch_worker(worker_rank, rendezvous):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=worker_rank, world_size=2
    )
    try:
        for mismatch in ("assets", "dimensions", "ppo_config", "physics"):
            # Lightweight environments can omit the optional physical fingerprint.
            env = ToyEnvironment()
            if mismatch == "physics":
                env.physics_identity = f"physics-{worker_rank}"
            if mismatch == "dimensions" and worker_rank:
                env.num_points += 1
            worker_config = config()
            if mismatch == "ppo_config" and worker_rank:
                worker_config = dataclasses.replace(worker_config, epochs=2)
            asset_identity = (
                "assets-b" if mismatch == "assets" and worker_rank else "assets-a"
            )
            with pytest.raises(ValueError, match="worker configuration"):
                train.Trainer(
                    env,
                    worker_config,
                    device="cpu",
                    bank_identity=f"worker-bank-{worker_rank}",
                    asset_identity=asset_identity,
                )
    finally:
        dist.destroy_process_group()


def test_distributed_preflight_rejects_mismatched_assets_dimensions_and_config(
    tmp_path,
):
    torch.multiprocessing.spawn(
        _distributed_mismatch_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("option", ["--eval-seeds", "--eval-trials-per-task"])
def test_cli_rejects_empty_evaluation_before_constructing_environment(
    monkeypatch, option
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--assets",
            "unused",
            "--bank",
            "unused",
            "--out",
            "unused",
            option,
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2


def test_source_revision_allows_containers_without_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train.subprocess, "run", missing_git)
    assert train.git_revision() == "unknown"


@pytest.mark.parametrize("recorded", [None, "different"])
def test_training_rejects_missing_or_stale_physics_validation(recorded):
    from types import SimpleNamespace

    env = SimpleNamespace(
        bank_metadata={"physics_identity": recorded}, physics_identity="current"
    )
    with pytest.raises(ValueError, match="physics identity"):
        train.require_physics_bank(env)


def test_training_accepts_current_physics_validation():
    from types import SimpleNamespace

    train.require_physics_bank(
        SimpleNamespace(
            bank_metadata={"physics_identity": "current"}, physics_identity="current"
        )
    )


@pytest.mark.parametrize("value", ["0", "29", "601", "030", "1.5", "invalid"])
def test_distributed_timeout_rejects_unbounded_or_malformed_values(monkeypatch, value):
    monkeypatch.setenv("ASSEMBLY_DISTRIBUTED_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="30-600"):
        train.distributed_timeout()


def test_distributed_initialization_receives_finite_timeout(monkeypatch):
    import datetime

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("ASSEMBLY_DISTRIBUTED_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--assets", "unused", "--bank", "unused", "--out", "unused"],
    )
    monkeypatch.setattr(train.torch.cuda, "set_device", lambda index: None)

    def capture(**kwargs):
        assert kwargs["backend"] == "nccl"
        assert kwargs["timeout"] == datetime.timedelta(seconds=120)
        raise RuntimeError("captured initialization")

    monkeypatch.setattr(train.distributed, "init_process_group", capture)
    with pytest.raises(RuntimeError, match="captured initialization"):
        train.main()


def _evaluation_progress_worker(worker_rank, rendezvous):
    import datetime
    import time

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=worker_rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=2),
    )
    try:

        def operation(progress):
            assert worker_rank == 0
            # Total evaluation exceeds the collective timeout; individual steps do not.
            for _ in range(12):
                progress()
                time.sleep(0.25)
            return "evaluated"

        result = train.run_rank_zero_evaluation(operation)
        assert result == ("evaluated" if worker_rank == 0 else None)

        def broken(progress):
            progress()
            raise ValueError("evaluation failed")

        with pytest.raises((ValueError, RuntimeError), match="evaluation failed"):
            train.run_rank_zero_evaluation(broken)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_long_distributed_evaluation_progress_and_failure_propagation(tmp_path):
    torch.multiprocessing.spawn(
        _evaluation_progress_worker,
        args=(str(tmp_path / "evaluation-rendezvous"),),
        nprocs=2,
        join=True,
    )


def test_evaluation_emits_progress_through_resets_and_steps():
    trainer = make_trainer()
    events = []
    train.evaluate(
        trainer.model,
        ToyEnvironment(),
        trials_per_task=6,
        horizon=5,
        progress=lambda: events.append("progress"),
    )
    assert len(events) >= 6


def test_evaluation_defaults_to_environment_full_multi_pair_horizon():
    trainer = make_trainer()
    env = ToyEnvironment()
    env.max_horizon_steps = 1125
    result = train.evaluate(trainer.model, env, trials_per_task=1)
    assert result["horizon"] == 1125
