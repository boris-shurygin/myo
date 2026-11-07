"""Shared actor/critic/dynamics features with PPO updates guarded by exact Beta KL."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, kl_divergence
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance

from .policy_beta import BetaExcitationPolicy
from .vertical_precision import VerticalPrecisionEnv


DESIGN = {"schema": "myo.shared_vertical.v1", "physical_features": 6, "command_context_features": 6, "goal_features": 1,
    "encoder": [128, 128], "actor_tail": [128, 128], "critic_tail": [128, 128],
    "predictor_tail": [128, 128], "activation": "SiLU", "prediction": "normalized residual of first 6 observation channels",
    "prediction_condition": "shared physical features plus delivered pending/0; new action only enters exact FIFO",
    "normalization": "physical delta mean/std from first valid training rollout, frozen; std floor 1e-5",
    "guard": "full current rollout mean analytic KL(old Beta || proposed Beta)<=target_kl; rollback weights AND RMSprop state",
    "transport": "exact 40ms FIFO, 20ms transition"}


def mlp(input_size, sizes):
    layers = []
    for size in sizes:
        layers.extend((nn.Linear(input_size, size), nn.SiLU()))
        input_size = size
    return nn.Sequential(*layers)


class SharedExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = mlp(6, (128, 128))
        self.policy_net = mlp(135, (128, 128))
        self.value_net = mlp(135, (128, 128))
        self.latent_dim_pi = self.latent_dim_vf = 128

    def features(self, observations):
        return torch.cat((self.encoder(observations[:, :6]), observations[:, 6:13]), dim=-1)

    def forward(self, observations):
        features = self.features(observations)
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, observations):
        return self.policy_net(self.features(observations))

    def forward_critic(self, observations):
        return self.value_net(self.features(observations))


class SharedPolicy(BetaExcitationPolicy):
    def _build_mlp_extractor(self):
        if self.features_dim != 13:
            raise ValueError("Shared vertical policy requires the variant-4 13-channel observation")
        self.mlp_extractor = SharedExtractor().to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Construct identically in both shared branches, including the unused control head.
        self.prediction_net = nn.Sequential(mlp(130, (128, 128)), nn.Linear(128, 6))
        nn.init.zeros_(self.prediction_net[-1].weight)
        nn.init.zeros_(self.prediction_net[-1].bias)
        self.register_buffer("delta_mean", torch.zeros(6))
        self.register_buffer("delta_std", torch.ones(6))
        self.register_buffer("delta_calibrated", torch.tensor(False))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def calibrate(self, observations, next_observations, valid):
        if not bool(self.delta_calibrated):
            delta = next_observations[valid, :6]-observations[valid, :6]
            if len(delta) < 2:
                raise ValueError("Not enough valid training transitions for delta normalization")
            self.delta_mean.copy_(delta.mean(0))
            self.delta_std.copy_(delta.std(0, unbiased=False).clamp_min(1e-5))
            self.delta_calibrated.fill_(True)

    def normalized_delta(self, observations):
        features = self.mlp_extractor.encoder(observations[:, :6])
        return self.prediction_net(torch.cat((features, observations[:, 6:8]), dim=-1))

    def next_observation(self, observations, actions):
        physical = observations[:, :6]+self.delta_mean+self.delta_std*self.normalized_delta(observations)
        # Observation queue order: pending/0, pending/1; previous excitation follows it.
        normalized_actions = 2*actions-1
        return torch.cat((physical, observations[:, 8:10], normalized_actions,
                          normalized_actions, observations[:, 12:13]), dim=-1)


class PredictiveSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor
    prediction_valid: torch.Tensor


class PredictiveBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.prediction_valid = np.zeros((self.buffer_size, self.n_envs, 1), dtype=bool)

    def record_next(self, observations, infos):
        # Called before the ordinary add(); vec-env already reset completed episodes.
        for index, info in enumerate(infos):
            self.next_observations[self.pos, index] = info.get("terminal_observation", observations[index])
            self.prediction_valid[self.pos, index, 0] = not info["physical_failure"]

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.next_observations = self.swap_and_flatten(self.next_observations)
            self.prediction_valid = self.swap_and_flatten(self.prediction_valid)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        base = super()._get_samples(batch_inds, env)
        return PredictiveSamples(*base, self.to_torch(self.next_observations[batch_inds]),
                                  self.to_torch(self.prediction_valid[batch_inds].flatten()))


class GuardedPPO(PPO):
    def __init__(self, *args, prediction_coef=0., **kwargs):
        self.prediction_coef = prediction_coef
        self.accepted_steps = self.rejected_steps = 0
        self.last_audit = {}
        super().__init__(*args, **kwargs)

    def guarded_step(self, loss, observations, old_distribution):
        parameters = {key: value.detach().clone() for key, value in self.policy.state_dict().items()}
        optimizer = copy.deepcopy(self.policy.optimizer.state_dict())
        self.policy.optimizer.zero_grad()
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm))
        try:
            self.policy.optimizer.step()
            with torch.no_grad():
                current = self.policy.get_distribution(observations).distribution
                divergence = float(kl_divergence(old_distribution, current).sum(-1).mean().clamp_min(0))
        except ValueError:
            divergence = float("inf")
        accepted = np.isfinite(divergence) and np.isfinite(gradient_norm) and divergence <= self.target_kl
        if not accepted:
            self.policy.load_state_dict(parameters)
            self.policy.optimizer.load_state_dict(optimizer)
            self.rejected_steps += 1
        else:
            self.accepted_steps += 1
        return bool(accepted), divergence, gradient_norm

    def train(self):
        if self.target_kl is None or self.clip_range_vf is not None or self.use_sde:
            raise ValueError("This narrow experiment requires a KL limit, no value clip and no gSDE")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        buffer = self.rollout_buffer
        observations = torch.as_tensor(buffer.observations.reshape(-1, 13), device=self.device)
        next_observations = torch.as_tensor(buffer.next_observations.reshape(-1, 13), device=self.device)
        valid = torch.as_tensor(buffer.prediction_valid.reshape(-1), device=self.device)
        with torch.no_grad():
            previous = self.policy.get_distribution(observations).distribution
            old_distribution = Beta(previous.concentration1.clone(), previous.concentration0.clone())
            if isinstance(self.policy, SharedPolicy):
                self.policy.calibrate(observations, next_observations, valid)
        clip = self.clip_range(self._current_progress_remaining)
        metrics = {key: [] for key in ("policy_loss", "value_loss", "prediction_loss", "entropy_loss", "approx_kl", "clip_fraction")}
        accepted_kls, rejected_kls = [], []
        before = self.accepted_steps
        proceed = True
        for _ in range(self.n_epochs):
            for batch in buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, batch.actions)
                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio = torch.exp(log_prob-batch.old_log_prob)
                policy_loss = -torch.minimum(advantages*ratio, advantages*ratio.clamp(1-clip, 1+clip)).mean()
                value_loss = (values.flatten()-batch.returns).square().mean()
                entropy_loss = -entropy.mean()
                prediction_loss = torch.zeros((), device=self.device)
                if self.prediction_coef and batch.prediction_valid.any():
                    predicted = self.policy.normalized_delta(batch.observations[batch.prediction_valid])
                    target = (batch.next_observations[batch.prediction_valid, :6]-batch.observations[batch.prediction_valid, :6]
                              -self.policy.delta_mean)/self.policy.delta_std
                    prediction_loss = (predicted-target).square().mean()
                loss = policy_loss+self.vf_coef*value_loss+self.ent_coef*entropy_loss+self.prediction_coef*prediction_loss
                with torch.no_grad():
                    log_ratio = log_prob-batch.old_log_prob
                    approx = float(((torch.exp(log_ratio)-1)-log_ratio).mean())
                accepted, divergence, _ = self.guarded_step(loss, observations, old_distribution)
                if not accepted:
                    rejected_kls.append(divergence if np.isfinite(divergence) else None)
                    proceed = False
                    break
                accepted_kls.append(divergence)
                for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("entropy_loss", entropy_loss), ("prediction_loss", prediction_loss),
                    ("approx_kl", approx), ("clip_fraction", (torch.abs(ratio-1)>clip).float().mean())):
                    metrics[key].append(float(value.detach()) if isinstance(value, torch.Tensor) else value)
            self._n_updates += 1
            if not proceed:
                break
        self.last_audit = {"accepted_steps": self.accepted_steps-before,
            "max_accepted_kl": max(accepted_kls, default=0.), "rejected_kl": rejected_kls,
            "prediction_valid": int(valid.sum()), "physical_delta_scale":
                self.policy.delta_std.cpu().tolist() if isinstance(self.policy, SharedPolicy) else None}
        for key, numbers in metrics.items():
            self.logger.record("train/"+key, float(np.mean(numbers)) if numbers else 0.)
        self.logger.record("train/explained_variance", explained_variance(buffer.values.flatten(), buffer.returns.flatten()))
        self.logger.record("train/accepted_steps", self.accepted_steps)
        self.logger.record("train/rejected_steps", self.rejected_steps)
        self.logger.record("train/max_exact_kl", max(accepted_kls, default=0.))


class SharedVerticalEnv(VerticalPrecisionEnv):
    def __init__(self, config=None, episode_seconds=4., trial="shared_prediction"):
        from .vertical_muscles import VerticalConfig
        super().__init__(config or VerticalConfig(variant=4), episode_seconds)
        self.trial = trial

    def contract(self):
        return {**super().contract(), "shared_design": DESIGN,
                "trial": self.trial,
                "predictor_identity": (None if self.trial == "independent" else
                    "shared residual head; auxiliary training" if self.trial == "shared_prediction" else
                    "shared residual head; untrained control"),
                "shared_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def make_model(env, trial, *, n_steps=1024, batch_size=128, n_epochs=10):
    if trial not in ("independent", "shared", "shared_prediction"):
        raise ValueError("Unknown shared-encoder trial")
    return GuardedPPO(BetaExcitationPolicy if trial == "independent" else SharedPolicy, env,
        prediction_coef=.05 if trial == "shared_prediction" else 0.,
        seed=0, device="cpu", learning_rate=3e-4, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5,
        target_kl=.03, rollout_buffer_class=PredictiveBuffer,
        policy_kwargs={"net_arch": {"pi": [128]*4, "vf": [128]*4}, "activation_fn": nn.SiLU,
            "optimizer_class": torch.optim.RMSprop, "optimizer_kwargs": {"eps": 1e-5, "weight_decay": 0.}})


def load_shared(path, episode_seconds=4.):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    config = metadata["contract"]["physical"]["config"]
    from .vertical_muscles import VerticalConfig
    env = SharedVerticalEnv(VerticalConfig(**config), episode_seconds, metadata["training"]["trial"])
    if metadata["contract"] != env.contract():
        raise ValueError("Shared policy contract mismatch")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Shared policy checkpoint hash mismatch")
    model = GuardedPPO.load(path, device="cpu")
    if model.num_timesteps != metadata["trained_steps"]:
        raise ValueError("Shared policy step counter mismatch")
    expected = metadata["training"]["trial"]
    if isinstance(model.policy, SharedPolicy) != (expected != "independent"):
        raise ValueError("Shared policy architecture mismatch")
    if model.prediction_coef != (.05 if expected == "shared_prediction" else 0.):
        raise ValueError("Shared policy prediction coefficient mismatch")
    return model, env, metadata
