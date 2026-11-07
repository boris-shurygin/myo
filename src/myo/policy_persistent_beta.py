"""A causal autoregressive Beta policy with exact conditional log-probability."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional

from .policy_beta import BetaExcitationDistribution, BetaExcitationPolicy


class PreviousActionFeatures(nn.Module):
    def __init__(self, base, indices):
        super().__init__()
        self.base = base
        self.latent_dim_pi = base.latent_dim_pi + len(indices)
        self.latent_dim_vf = base.latent_dim_vf
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer("previous_mean", torch.zeros(len(indices)))
        self.register_buffer("previous_scale", torch.ones(len(indices)))

    def forward_actor(self, features):
        previous = features[:, self.indices] * self.previous_scale + self.previous_mean
        return torch.cat((self.base.forward_actor(features), previous), dim=-1)

    def forward_critic(self, features):
        return self.base.forward_critic(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class PersistentBetaDistribution(BetaExcitationDistribution):
    def __init__(self, action_dim, persistence):
        super().__init__(action_dim)
        self.persistence = persistence

    def proba_distribution(self, parameters, previous):
        alpha, beta = (functional.softplus(parameters) + 0.05).chunk(2, dim=-1)
        concentration = alpha + beta
        mean = ((1 - self.persistence) * alpha / concentration
                + self.persistence * previous.clamp(0, 1))
        self.distribution = torch.distributions.Beta(mean * concentration, (1 - mean) * concentration)
        return self

    def actions_from_params(self, parameters, previous, deterministic=False):
        return self.proba_distribution(parameters, previous).get_actions(deterministic)

    def log_prob_from_params(self, parameters, previous):
        actions = self.actions_from_params(parameters, previous)
        return actions, self.log_prob(actions)


class PersistentBetaPolicy(BetaExcitationPolicy):
    def __init__(self, *args, previous_action_indices, persistence=0.9, **kwargs):
        if not 0 <= persistence < 1:
            raise ValueError("Persistence must lie in [0, 1)")
        self.previous_action_indices = previous_action_indices
        self.persistence = persistence
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        self.mlp_extractor = PreviousActionFeatures(self.mlp_extractor, self.previous_action_indices)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_dist = PersistentBetaDistribution(self.action_space.shape[0], self.persistence)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi), latent_pi[:, -self.action_space.shape[0]:])

    @torch.no_grad()
    def synchronize_previous_action_normalization(self):
        extractor = self.features_extractor
        history = self.mlp_extractor
        history.previous_mean.copy_(extractor.observation_mean[history.indices])
        history.previous_scale.copy_(extractor.observation_scale[history.indices])
