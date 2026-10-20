import pytest
import torch

from isaac.assembly_rl.policy import (
    ActorCritic,
    PPOBatch,
    PPOConfig,
    RunningNormalizer,
    compute_gae,
    ppo_update,
)


def test_terminal_and_timeout_bootstrap_without_cross_episode_leakage():
    rewards = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    values = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
    next_values = torch.tensor([[10.0, 10.0], [0.0, 0.0]])
    terminated = torch.tensor([[True, False], [True, True]])
    truncated = torch.tensor([[False, True], [False, False]])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=0.95
    )
    torch.testing.assert_close(advantages[0], torch.tensor([-1.0, 8.0]))
    torch.testing.assert_close(returns[0], torch.tensor([1.0, 10.0]))


def test_continuing_gae_propagates_and_rollout_end_bootstraps():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0.0], [4.0]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    advantages, _ = compute_gae(
        rewards, values, next_values, dones, dones, gamma=0.5, gae_lambda=0.8
    )
    torch.testing.assert_close(advantages, torch.tensor([[2.6], [4.0]]))


def test_running_normalizer_combines_batches_and_forward_is_read_only():
    norm = RunningNormalizer(2)
    data = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    norm.update(data[:1])
    norm.update(data[1:])
    count = len(data) + 1e-4
    expected_mean = data.sum(0) / count
    expected_variance = (data.square().sum(0) + 1e-4) / count - expected_mean.square()
    torch.testing.assert_close(norm.mean, expected_mean)
    torch.testing.assert_close(norm.variance, expected_variance)
    assert float(norm.count) == pytest.approx(count)
    before = {key: value.clone() for key, value in norm.state_dict().items()}
    norm(torch.full((2, 2), 100.0))
    for key, value in norm.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_architecture_and_actor_critic_independence():
    model = ActorCritic(num_points=8, proprio_dim=13)
    assert not (
        {id(p) for p in model.actor.parameters()}
        & {id(p) for p in model.critic.parameters()}
    )
    assert model.actor.point_encoder[0].in_features == 24
    assert model.actor.point_encoder[0].out_features == 256
    assert model.actor.point_encoder[2].out_features == 128
    assert model.actor.input_layer.in_features == 141
    assert len(model.actor.blocks) == 2
    assert model.actor.blocks[0].layers[1].out_features == 1024
    points, proprio = torch.randn(4, 8, 3), torch.randn(4, 13)
    distribution, values = model(points, proprio)
    assert distribution.sample().shape == (4, 7)
    assert distribution.log_prob(distribution.sample()).shape == (4,)
    assert values.shape == (4,)
    with pytest.raises(ValueError):
        model(torch.randn(4, 9, 3), proprio)


def test_ppo_updates_finitely_and_does_not_update_normalization():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = ActorCritic(num_points=8, proprio_dim=13)
    points, proprio = torch.randn(16, 8, 3), torch.randn(16, 13)
    model.update_normalizers(proprio)
    before_norm = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if "normalizer" in key
    }
    old_weight = model.actor.head.weight.detach().clone()
    with torch.no_grad():
        distribution, values = model(points, proprio)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
    batch = PPOBatch(
        points,
        proprio,
        actions,
        log_probs,
        values,
        torch.linspace(-1.0, 1.0, 16),
        values + 1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = ppo_update(model, optimizer, batch, PPOConfig())
    assert metrics["updates"] == 20
    assert not torch.equal(old_weight, model.actor.head.weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for key, value in before_norm.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    clone = ActorCritic(num_points=8, proprio_dim=13)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(points, proprio)[1], model(points, proprio)[1])


def test_cited_value_loss_has_no_half_and_analytical_kl_adapts_lr():
    torch.set_num_threads(1)
    model = ActorCritic(num_points=2, proprio_dim=3)
    points, proprio = torch.randn(4, 2, 3), torch.randn(4, 3)
    with torch.no_grad():
        distribution, values = model(points, proprio)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        old_means = distribution.base_dist.loc + 10
        old_stds = distribution.base_dist.scale.clone()
    batch = PPOBatch(
        points,
        proprio,
        actions,
        log_probs,
        values,
        torch.arange(4.0),
        values + 1,
        old_means,
        old_stds,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = ppo_update(model, optimizer, batch, PPOConfig(epochs=1, minibatches=1))
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert metrics["learning_rate"] == pytest.approx(1e-4 / 1.5)
