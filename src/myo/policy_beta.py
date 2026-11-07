"""Bounded muscle exploration: sampling and deterministic means share [0, 1]."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BetaExcitationDistribution(Distribution):
    """Independent Beta commands, without a clipped Gaussian latent action.

    The concentration floor keeps both tails representable during exploration;
    it does not impose a floor on individual sampled muscle commands.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, parameters):
        alpha, beta = (functional.softplus(parameters) + 0.05).chunk(2, dim=-1)
        self.distribution = torch.distributions.Beta(alpha, beta)
        return self

    def log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def mode(self):
        # SB3 calls this for deterministic inference. The mean exists even for
        # alpha/beta < 1, where the mathematical mode can be at a boundary.
        return self.distribution.mean

    def actions_from_params(self, parameters, deterministic=False):
        return self.proba_distribution(parameters).get_actions(deterministic)

    def log_prob_from_params(self, parameters):
        actions = self.actions_from_params(parameters)
        return actions, self.log_prob(actions)


class BetaExcitationPolicy(ActorCriticPolicy):
    """The usual actor/critic MLPs with two bounded-distribution heads per muscle."""

    def _build(self, lr_schedule):
        if self.use_sde or self.squash_output:
            raise ValueError("Beta exploration does not use Gaussian gSDE or tanh squashing")
        super()._build(lr_schedule)
        self.action_dist = BetaExcitationDistribution(self.action_space.shape[0])
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        del self.log_std
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        # A common initial mean 0.1 and std about 0.09; no teacher actions.
        with torch.no_grad():
            self.action_net.bias[:self.action_dist.action_dim].fill_(math.log(math.expm1(1.0 - 0.05)))
            self.action_net.bias[self.action_dist.action_dim:].fill_(math.log(math.expm1(9.0 - 0.05)))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi))
