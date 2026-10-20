"""AlphaNIST hidden network/PPO, with explicitly attributed cited-code defaults.

Flattened ordered XYZ is the literal Fig. 3 interpretation, not confirmed author
code. No official AlphaNIST implementation is available. CONFIG_PROVENANCE
separates paper settings, cited implementation hints, and engineering choices.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, distributed, nn
from torch.distributions import Independent, Normal


class RunningNormalizer(nn.Module):
    """SimBa population moments; update during collection, freeze during PPO."""

    def __init__(self, dimension: int, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("variance", torch.ones(dimension))
        self.register_buffer("count", torch.full((), 1e-4, dtype=torch.float64))
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        if (
            values.ndim != 2
            or values.shape[1:] != self.mean.shape
            or not values.shape[0]
        ):
            raise ValueError("Expected a nonempty [batch, proprio_dim] tensor")
        batch_variance, batch_mean = torch.var_mean(values, dim=0, unbiased=False)
        batch_count = values.shape[0]
        total = self.count + batch_count
        weight = (batch_count / total).to(values.dtype)
        old_weight = (self.count / total).to(values.dtype)
        delta = batch_mean - self.mean
        self.variance.copy_(
            old_weight * self.variance
            + weight * batch_variance
            + delta.square() * old_weight * weight
        )
        self.mean.add_(delta * weight)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) * torch.rsqrt(self.variance + self.epsilon)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, 256)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Flax/JAX He-normal uses a variance-corrected truncated normal.
                standard_deviation = (
                    math.sqrt(2.0 / layer.in_features) / 0.8796256610342398
                )
                nn.init.trunc_normal_(
                    layer.weight,
                    std=standard_deviation,
                    a=-2 * standard_deviation,
                    b=2 * standard_deviation,
                )
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class GeometryNetwork(nn.Module):
    def __init__(self, num_points: int, proprio_dim: int, output_dim: int):
        super().__init__()
        self.num_points, self.proprio_dim = num_points, proprio_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3 * num_points, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU()
        )
        self.normalizer = RunningNormalizer(proprio_dim)
        self.input_layer = nn.Linear(128 + proprio_dim, 256)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.final_norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, output_dim)
        nn.init.orthogonal_(self.input_layer.weight, gain=1.0)
        nn.init.zeros_(self.input_layer.bias)
        # Point encoder and output initialization are not specified by AlphaNIST.

    def forward(
        self, points: Tensor, proprio: Tensor, *, normalize: bool = True
    ) -> Tensor:
        if points.ndim != 3 or points.shape[1:] != (self.num_points, 3):
            raise ValueError(f"Expected points [batch, {self.num_points}, 3]")
        if proprio.shape != (points.shape[0], self.proprio_dim):
            raise ValueError(f"Expected proprio [batch, {self.proprio_dim}]")
        encoded = self.point_encoder(points.flatten(start_dim=1))
        features = torch.cat(
            (encoded, self.normalizer(proprio) if normalize else proprio), dim=-1
        )
        return self.head(self.final_norm(self.blocks(self.input_layer(features))))


class ActorCritic(nn.Module):
    """Raw Gaussian actions; bound applied commands without altering logged actions.

    Update normalization during rollout collection, as in cited RSL/SimBa code;
    freeze during PPO. Residual/projection initialization follows cited SimBa.
    Point/head initialization and state-independent Gaussian std are explicit choices.
    """

    def __init__(
        self,
        num_points: int,
        proprio_dim: int,
        action_dim: int = 7,
        initial_log_std: float = 0.0,
    ):
        super().__init__()
        if min(num_points, proprio_dim, action_dim) < 1:
            raise ValueError("Network dimensions must be positive")
        self.actor = GeometryNetwork(num_points, proprio_dim, action_dim)
        self.critic = GeometryNetwork(num_points, proprio_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    @torch.no_grad()
    def update_normalizers(self, proprio: Tensor) -> None:
        self.actor.normalizer.update(proprio)
        self.critic.normalizer.update(proprio)

    def forward(self, points: Tensor, proprio: Tensor) -> tuple[Independent, Tensor]:
        mean = self.actor(points, proprio)
        distribution = Independent(
            Normal(mean, self.log_std.exp(), validate_args=False), 1
        )
        return distribution, self.critic(points, proprio).squeeze(-1)

    def forward_normalized(
        self, points: Tensor, actor_proprio: Tensor, critic_proprio: Tensor
    ) -> tuple[Independent, Tensor]:
        """Reevaluate the exact normalized features used when sampling actions."""
        mean = self.actor(points, actor_proprio, normalize=False)
        distribution = Independent(
            Normal(mean, self.log_std.exp(), validate_args=False), 1
        )
        return distribution, self.critic(
            points, critic_proprio, normalize=False
        ).squeeze(-1)

    def evaluate_actions(
        self, points: Tensor, proprio: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, values = self(points, proprio)
        return distribution.log_prob(actions), distribution.entropy(), values


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return advantages/returns [time, env]; next_values precede any autoreset.

    True terminal states have no bootstrap. Time limits bootstrap their final
    observation, but both kinds of boundary stop advantages crossing episodes.
    """
    if (
        rewards.ndim != 2
        or rewards.shape[0] == 0
        or any(
            tensor.shape != rewards.shape
            for tensor in (values, next_values, terminated, truncated)
        )
    ):
        raise ValueError(
            "All rollout tensors must have the same nonempty [time, env] shape"
        )
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("Episode boundary masks must be boolean")
    advantages = torch.empty_like(rewards)
    last = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        delta = (
            rewards[step]
            + gamma * next_values[step] * (~terminated[step])
            - values[step]
        )
        continuation = ~(terminated[step] | truncated[step])
        last = delta + gamma * gae_lambda * continuation * last
        advantages[step] = last
    return advantages, advantages + values


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 32
    epochs: int = 5
    minibatches: int = 4
    # Below: cited OmniReset/RSL defaults, not separately specified by AlphaNIST.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 1.0
    entropy_coefficient: float = 0.006
    max_grad_norm: float = 1.0
    clipped_value_loss: bool = True
    desired_kl: float | None = 0.01


@dataclass(frozen=True)
class PPOBatch:
    """Detached rollout tensors flattened across time and environments."""

    points: Tensor
    proprio: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    old_means: Tensor | None = None
    old_stds: Tensor | None = None
    # Collection-time features prevent changing running moments from changing
    # the old policy's likelihood during PPO. Raw input remains for diagnostics.
    actor_proprio_normalized: Tensor | None = None
    critic_proprio_normalized: Tensor | None = None

    def __post_init__(self):
        if (self.actor_proprio_normalized is None) != (
            self.critic_proprio_normalized is None
        ):
            raise ValueError(
                "Both actor and critic normalized rollout features are required together"
            )
        if self.actor_proprio_normalized is not None and (
            self.actor_proprio_normalized.shape != self.proprio.shape
            or self.critic_proprio_normalized.shape != self.proprio.shape
        ):
            raise ValueError("Normalized rollout features must match proprio shape")


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig | None = None,
) -> dict[str, float]:
    """PPO with cited RSL clipping/adaptive KL, and synchronous gradient averaging.

    Normalizer buffers remain local to each GPU; only trainable gradients sync.
    The caller must use equal rollout/minibatch sizes on every worker.
    """
    config = config or PPOConfig()
    size = batch.advantages.shape[0]
    if config.epochs < 1 or config.minibatches < 1 or size < config.minibatches:
        raise ValueError("Need positive epochs/minibatches and enough rollout samples")
    if size % config.minibatches:
        raise ValueError("Rollout size must divide evenly into minibatches")
    if any(
        tensor.shape[0] != size or tensor.requires_grad
        for tensor in vars(batch).values()
        if tensor is not None
    ):
        raise ValueError(
            "Rollout tensors must be detached and have matching batch dimensions"
        )
    if any(
        getattr(batch, key).shape != (size,)
        for key in ("old_log_probs", "old_values", "advantages", "returns")
    ):
        raise ValueError("Rollout statistics must have shape [batch]")
    multi_gpu = distributed.is_available() and distributed.is_initialized()
    worker_count = distributed.get_world_size() if multi_gpu else 1
    if (batch.old_means is None) != (batch.old_stds is None):
        raise ValueError(
            "Old Gaussian means and standard deviations must be provided together"
        )
    if batch.old_means is not None and (
        batch.old_means.shape != batch.actions.shape
        or batch.old_stds.shape != batch.actions.shape
        or (batch.old_stds <= 0).any()
    ):
        raise ValueError(
            "Old Gaussian statistics must match actions with positive standard deviations"
        )
    # Legacy callers without distribution statistics retain fixed-rate PPO.
    # Runner always supplies statistics, enabling the cited analytical KL rule.
    moments = torch.stack(
        (
            batch.advantages.double().sum(),
            batch.advantages.double().square().sum(),
            batch.advantages.new_tensor(size, dtype=torch.float64),
        )
    )
    if multi_gpu:
        distributed.all_reduce(moments)
    mean = moments[0] / moments[2]
    variance = (moments[1] - moments[0] * mean).clamp_min(0) / (
        moments[2] - 1
    ).clamp_min(1)
    advantages = (batch.advantages - mean.to(batch.advantages.dtype)) / (
        variance.sqrt().to(batch.advantages.dtype) + 1e-8
    )
    totals = torch.zeros(5, device=batch.points.device)
    updates = 0
    for _ in range(config.epochs):
        indices = torch.randperm(size, device=batch.points.device)
        for index in indices.chunk(config.minibatches):
            if batch.actor_proprio_normalized is None:
                # Legacy callers may supply raw inputs only while their normalizer
                # is fixed throughout collection and update.
                distribution, values = model(batch.points[index], batch.proprio[index])
            else:
                distribution, values = model.forward_normalized(
                    batch.points[index],
                    batch.actor_proprio_normalized[index],
                    batch.critic_proprio_normalized[index],
                )
            log_probs = distribution.log_prob(batch.actions[index])
            entropy = distribution.entropy()
            if config.desired_kl is not None and batch.old_means is not None:
                with torch.no_grad():
                    old_mean, old_std = batch.old_means[index], batch.old_stds[index]
                    mean, std = distribution.base_dist.loc, distribution.base_dist.scale
                    kl_mean = (
                        (
                            torch.log(std / old_std + 1e-5)
                            + (old_std.square() + (old_mean - mean).square())
                            / (2 * std.square())
                            - 0.5
                        )
                        .sum(-1)
                        .mean()
                    )
                    if multi_gpu:
                        distributed.all_reduce(kl_mean)
                        kl_mean /= worker_count
                    learning_rate = optimizer.param_groups[0]["lr"]
                    if kl_mean > 2 * config.desired_kl:
                        learning_rate = max(1e-5, learning_rate / 1.5)
                    elif 0 < kl_mean < 0.5 * config.desired_kl:
                        learning_rate = min(1e-2, learning_rate * 1.5)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
            log_ratio = log_probs - batch.old_log_probs[index]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[index]
            clipped = (
                ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                * advantages[index]
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_errors = (values - batch.returns[index]).square()
            if config.clipped_value_loss:
                clipped_values = batch.old_values[index] + (
                    values - batch.old_values[index]
                ).clamp(-config.clip_ratio, config.clip_ratio)
                value_errors = torch.maximum(
                    value_errors, (clipped_values - batch.returns[index]).square()
                )
            value_loss = value_errors.mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if multi_gpu:
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                flat = torch.cat([gradient.flatten() for gradient in gradients])
                distributed.all_reduce(flat)
                flat /= worker_count
                offset = 0
                for gradient in gradients:
                    count = gradient.numel()
                    gradient.copy_(flat[offset : offset + count].view_as(gradient))
                    offset += count
            nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                approx_kl = (ratio - 1 - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > config.clip_ratio).float().mean()
                totals.add_(
                    torch.stack(
                        (
                            policy_loss,
                            value_loss,
                            entropy_mean,
                            approx_kl,
                            clip_fraction,
                        )
                    )
                )
            updates += 1
    if multi_gpu:
        distributed.all_reduce(totals)
        totals /= worker_count
    metrics = dict(
        zip(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"),
            (totals / updates).tolist(),
            strict=True,
        )
    )
    metrics["updates"] = updates
    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    return metrics


CONFIG_PROVENANCE = {
    "alphanist": {
        "url": "https://arxiv.org/html/2610.13354v1",
        "settings": "32 rollout steps; 5 PPO epochs; 4 minibatches; separate actor/critic; 256/128 ELU point encoder; two 256/1024 Swish residual blocks",
    },
    "omnireset_legacy": {
        "url": "https://github.com/uw-lab/uwlab/tree/825cfce364f7e34b4b29aab73d37c5a1464f4375",
        "settings": "Adam 1e-4, gamma .99, lambda .95, clip .2, entropy .006, value coefficient 1, grad norm 1, adaptive KL .01, clipped values; SGS history 100 and epsilon 1e-4; five observation frames",
    },
    "rsl_ppo": {
        "url": "https://github.com/uw-lab/rsl_rl/tree/e7cd3c77bdb3c94753612f208c725e1add38a655",
        "settings": "Analytical diagonal-Gaussian KL adaptive LR, clipped squared value loss without .5, collection-time normalization updates",
    },
    "simba": {
        "url": "https://github.com/SonyResearch/simba/tree/7d0358b1f1fa3404fc5ea0ffedf9dc1424b49a45",
        "settings": "Running variance 1, count 1e-4, epsilon 1e-8; residual He truncated normal, projection orthogonal gain 1; AlphaNIST overrides ReLU with Swish",
    },
    "engineering_choices": [
        "Ordered flattened XYZ aggregation",
        "State-independent diagonal Gaussian log std initialized to zero",
        "PyTorch defaults for point encoder and output heads",
        "Global cross-worker unbiased advantage normalization",
        "Store actor/critic collection-time normalized features for PPO likelihood consistency; correctness fix beyond cited RSL raw-observation storage",
        "Apply AlphaNIST multi-pair 375-step-per-pair horizon to training and evaluation; worker environment counts are CLI settings",
        "UR7 five-frame proprioception has 140 values instead of Franka 150",
    ],
}
