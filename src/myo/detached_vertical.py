"""Predictor-owned features consumed by independent PPO branches without gradients."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, kl_divergence

from .policy_beta import BetaExcitationPolicy
from .shared_vertical import GuardedPPO, PredictiveBuffer, SharedPolicy, mlp
from .vertical_muscles import VerticalConfig
from .vertical_precision import VerticalPrecisionEnv


DESIGN = {
    "schema": "myo.detached_vertical.v1",
    "predictor": "physical6 + delivered2 -> 4x128 SiLU -> linear delta6",
    "actor": "concat(detached predictor128, private 13->2x128 SiLU) -> 2x128 SiLU -> Beta4",
    "critic": "concat(detached predictor128, private 13->2x128 SiLU) -> 2x128 SiLU -> linear1",
    "gradients": "prediction only owns predictor; PPO owns private branches and heads",
    "schedule": "collect 1024 transitions; guarded PPO with fixed predictor; then supervised predictor phase",
    "prediction_optimizer": "RMSprop lr=3e-4 alpha=.99 eps=1e-5; 10 epochs batch128; grad norm .5",
    "prediction_kl": "measure predictor-only and total policy KL; predictor phase is not KL guarded",
    "normalization": "first valid training rollout only, frozen; delta STD floor1e-5",
    "target": "next physical observation at20ms; no BPTT, no teacher commands",
}
TRIALS = ("frozen", "online")


class DetachedExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = mlp(8, (128,)*4)
        self.actor_private = mlp(13, (128, 128))
        self.critic_private = mlp(13, (128, 128))
        self.policy_net = mlp(256, (128, 128))
        self.value_net = mlp(256, (128, 128))
        self.latent_dim_pi = self.latent_dim_vf = 128

    def predictor_features(self, observations):
        return self.predictor(observations[:, :8])

    def forward_actor(self, observations):
        features = self.predictor_features(observations).detach()
        return self.policy_net(torch.cat((features, self.actor_private(observations)), dim=-1))

    def forward_critic(self, observations):
        features = self.predictor_features(observations).detach()
        return self.value_net(torch.cat((features, self.critic_private(observations)), dim=-1))

    def forward(self, observations):
        return self.forward_actor(observations), self.forward_critic(observations)


class DetachedPolicy(SharedPolicy):
    def _build_mlp_extractor(self):
        if self.features_dim != 13:
            raise ValueError("Detached vertical policy requires 13 observations")
        self.mlp_extractor = DetachedExtractor().to(self.device)

    def _build(self, lr_schedule):
        BetaExcitationPolicy._build(self, lr_schedule)
        self.prediction_net = nn.Linear(128, 6).to(self.device)
        nn.init.zeros_(self.prediction_net.weight)
        nn.init.zeros_(self.prediction_net.bias)
        self.register_buffer("delta_mean", torch.zeros(6, device=self.device))
        self.register_buffer("delta_std", torch.ones(6, device=self.device))
        self.register_buffer("delta_calibrated", torch.tensor(False, device=self.device))
        predictor = self.prediction_parameters()
        predictor_ids = {id(p) for p in predictor}
        private = [p for p in self.parameters() if id(p) not in predictor_ids]
        self.optimizer = self.optimizer_class(private, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.predictor_optimizer = torch.optim.RMSprop(predictor, lr=3e-4, eps=1e-5)

    def prediction_parameters(self):
        return list(self.mlp_extractor.predictor.parameters())+list(self.prediction_net.parameters())

    def normalized_delta(self, observations):
        return self.prediction_net(self.mlp_extractor.predictor_features(observations))


def distribution_snapshot(policy, observations):
    with torch.no_grad():
        distribution = policy.get_distribution(observations).distribution
        return Beta(distribution.concentration1.clone(), distribution.concentration0.clone())


def distribution_kl(before, after):
    return float(kl_divergence(before, after).sum(-1).mean().clamp_min(0))


class DetachedPPO(GuardedPPO):
    def __init__(self, *args, train_predictor=True, **kwargs):
        self.train_predictor = train_predictor
        self.predictor_steps = 0
        self.predictor_rng = np.random.default_rng(1701)
        super().__init__(*args, **kwargs)

    def train(self):
        observations = torch.as_tensor(self.rollout_buffer.observations.reshape(-1, 13), device=self.device)
        old = distribution_snapshot(self.policy, observations)
        self.policy.zero_grad(set_to_none=True)
        super().train()  # prediction_coef=0; detached features remain fixed throughout PPO.
        self.last_audit.update(self.update_predictor(observations))
        self.last_audit["total_phase_kl"] = distribution_kl(old, distribution_snapshot(self.policy, observations))

    def update_predictor(self, observations):
        buffer = self.rollout_buffer
        valid = torch.as_tensor(buffer.prediction_valid.reshape(-1), device=self.device)
        labels = torch.as_tensor(buffer.next_observations.reshape(-1, 13), device=self.device)
        x = observations[valid]
        target = (labels[valid, :6]-x[:, :6]-self.policy.delta_mean)/self.policy.delta_std
        before = distribution_snapshot(self.policy, observations)
        if not len(x):
            return {"predictor_loss_before": None, "predictor_loss_after": None,
                    "predictor_steps": 0, "predictor_phase_kl": 0.}
        with torch.no_grad():
            first_loss = float((self.policy.normalized_delta(x)-target).square().mean())
        start = self.predictor_steps
        if self.train_predictor and len(x):
            for _ in range(10):
                for indices in np.array_split(self.predictor_rng.permutation(len(x)), max(1, int(np.ceil(len(x)/128)))):
                    self.policy.zero_grad(set_to_none=True)
                    loss = (self.policy.normalized_delta(x[indices])-target[indices]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite predictor loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(self.policy.prediction_parameters(), .5)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite predictor gradient")
                    self.policy.predictor_optimizer.step()
                    self.predictor_steps += 1
        with torch.no_grad():
            last_loss = float((self.policy.normalized_delta(x)-target).square().mean())
        self.policy.zero_grad(set_to_none=True)
        return {"predictor_loss_before": first_loss, "predictor_loss_after": last_loss,
                "predictor_steps": self.predictor_steps-start,
                "predictor_phase_kl": distribution_kl(before, distribution_snapshot(self.policy, observations))}

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts+["policy.predictor_optimizer"], variables


class DetachedVerticalEnv(VerticalPrecisionEnv):
    def __init__(self, config=None, episode_seconds=4., trial="online"):
        if trial not in TRIALS:
            raise ValueError("Unknown detached-predictor trial")
        super().__init__(config or VerticalConfig(variant=4), episode_seconds)
        self.trial = trial

    def contract(self):
        shared_source = Path(__file__).with_name("shared_vertical.py")
        return {**super().contract(), "detached_design": DESIGN, "trial": self.trial,
                "predictor_identity": f"detached {self.trial} physical8-to-delta6",
                "detached_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "guard_source_sha256": hashlib.sha256(shared_source.read_bytes()).hexdigest()}


def make_model(env, trial, *, n_steps=1024, batch_size=128, n_epochs=10):
    if trial not in TRIALS:
        raise ValueError("Unknown detached-predictor trial")
    return DetachedPPO(DetachedPolicy, env, train_predictor=trial == "online", prediction_coef=0.,
        seed=0, device="cpu", learning_rate=3e-4, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5,
        target_kl=.03, rollout_buffer_class=PredictiveBuffer,
        policy_kwargs={"activation_fn": nn.SiLU, "optimizer_class": torch.optim.RMSprop,
                       "optimizer_kwargs": {"eps": 1e-5, "weight_decay": 0.}})


def load_detached(path, episode_seconds=4.):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    env = DetachedVerticalEnv(VerticalConfig(**metadata["contract"]["physical"]["config"]),
                              episode_seconds, metadata["training"]["trial"])
    if metadata["contract"] != env.contract():
        raise ValueError("Detached policy contract mismatch")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Detached policy checkpoint hash mismatch")
    model = DetachedPPO.load(path, device="cpu")
    if model.num_timesteps != metadata["trained_steps"]:
        raise ValueError("Detached policy step counter mismatch")
    if not isinstance(model.policy, DetachedPolicy) or model.prediction_coef != 0.:
        raise ValueError("Detached policy architecture mismatch")
    if model.train_predictor != (env.trial == "online"):
        raise ValueError("Detached predictor mode mismatch")
    return model, env, metadata
