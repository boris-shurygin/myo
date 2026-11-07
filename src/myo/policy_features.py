"""Actor input normalization stored inside the policy checkpoint."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NormalizedFeaturesExtractor(BaseFeaturesExtractor):
    """Keep every causal observation and normalize with training-only statistics.

    The caller supplies statistics from training episodes, excluding held-out
    episodes. Buffers travel with the actor, so runtime does not need a separate
    normalizer file or running statistics. The scale floor limits amplification
    of near-constant channels, including tendon strain and repeated history.
    """

    def __init__(self, observation_space: spaces.Box, minimum_scale: float = 0.05):
        if len(observation_space.shape) != 1:
            raise ValueError("Policy normalization requires a flat observation vector.")
        if not 0.0 < minimum_scale < float("inf"):
            raise ValueError("The normalization scale floor must be finite and positive.")
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self.minimum_scale = float(minimum_scale)
        self.register_buffer("observation_mean", torch.zeros(self.features_dim))
        self.register_buffer("observation_scale", torch.ones(self.features_dim))

    @torch.no_grad()
    def set_normalization(self, mean, scale) -> None:
        """Set per-channel training mean and standard deviation without fitting."""
        mean = torch.as_tensor(mean, dtype=self.observation_mean.dtype,
                               device=self.observation_mean.device)
        scale = torch.as_tensor(scale, dtype=self.observation_scale.dtype,
                                device=self.observation_scale.device)
        if mean.shape != self.observation_mean.shape or scale.shape != self.observation_scale.shape:
            raise ValueError("Normalization statistics must match the observation vector.")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()) or bool((scale < 0).any()):
            raise ValueError("Normalization statistics must be finite with nonnegative scales.")
        self.observation_mean.copy_(mean)
        self.observation_scale.copy_(scale.clamp_min(self.minimum_scale))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.observation_mean) / self.observation_scale
