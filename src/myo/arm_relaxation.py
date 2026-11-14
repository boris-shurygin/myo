"""Likelihood-consistent, coordinated antagonist relaxation for the 10-ms arm.

Guidance is an explicit, causal mechanical approximation, not a learned synergy
or an instantaneous intervention in physics. Native delayed commands are intact.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import torch
from torch.distributions import Beta, Categorical, Independent, MixtureSameFamily
import torch.nn.functional as functional

from .arm_efficiency import ArmEfficiencyEnv, EfficiencyReward
from .arm_exploration import ExplorationPolicy, ExplorationSpec, PressureExtractor
from .arm_learning import ArmPPO
from .arm_timing import ArmTiming
from .arm_workload import spec_from_dict
from .policy_beta import BetaExcitationDistribution


@dataclass(frozen=True)
class RelaxationSpec:
    mode: str = 'joint'
    probability: float = .5
    fraction: float = .25
    concentration_multiplier: float = 16.

    def __post_init__(self):
        if (self.mode not in ('joint', 'independent')
                or not 0 < self.probability < 1 or not 0 < self.fraction < 1
                or not np.isfinite(self.concentration_multiplier)
                or self.concentration_multiplier < 1):
            raise ValueError('Invalid relaxation distribution')


def torque_gains(base):
    """Nm per unit activation at fixed current fiber length, isometric proxy.

    This is NOT d(current tendon torque)/d(new excitation), which is initially
    zero under transport delay. Slack, passive force and dynamic compliance are
    tested by native counterfactual rollouts, not removed by this approximation.
    """
    return np.asarray([r*m._fiber_forces(m.fiber_length, 1., m.capacity)[0]
        for r, m in zip(base.moment_arms, base.muscles)], dtype=np.float32)


def relaxed_mean(mean, gains, fraction):
    """Remove equal opposing ACTIVE torque proxies; preserve gains @ mean."""
    positive = gains.clamp_min(0.)*mean
    negative = (-gains).clamp_min(0.)*mean
    p, n = positive.sum(-1, keepdim=True), negative.sum(-1, keepdim=True)
    removed = fraction*torch.minimum(p, n)
    positive_scale = 1.-removed/p.clamp_min(1e-12)
    negative_scale = 1.-removed/n.clamp_min(1e-12)
    scale = torch.where(gains > 0, positive_scale,
                        torch.where(gains < 0, negative_scale, torch.ones_like(gains)))
    return mean*scale


class RelaxationDistribution(BetaExcitationDistribution):
    """Joint and factorized mixtures have identical one-muscle marginals."""

    def configure(self, alpha, beta, gains, spec):
        self.nominal_mean = alpha/(alpha+beta)
        self.relaxed_mean = relaxed_mean(self.nominal_mean, gains, spec.fraction)
        concentration = (alpha+beta)*spec.concentration_multiplier
        a = torch.stack((self.nominal_mean*concentration, self.relaxed_mean*concentration), -2)
        b = torch.stack(((1-self.nominal_mean)*concentration,
                         (1-self.relaxed_mean)*concentration), -2)
        weights = alpha.new_tensor([1-spec.probability, spec.probability])
        if spec.mode == 'joint':
            self.distribution = MixtureSameFamily(Categorical(probs=weights.expand(*alpha.shape[:-1], 2)),
                                                  Independent(Beta(a, b), 1))
        else:
            self.distribution = Independent(MixtureSameFamily(
                Categorical(probs=weights.expand(*alpha.shape, 2)), Beta(a.transpose(-1, -2),
                                                                         b.transpose(-1, -2))), 1)
        return self

    def log_prob(self, actions):
        # The mixture already treats all six commands as one event.
        return self.distribution.log_prob(actions)

    def mode(self):
        # Assessment disables the extra relaxation mechanism. It must show
        # improvement in the learned nominal commands, not cheaper probes alone.
        return self.nominal_mean

    def entropy(self):
        # Mixture entropy has no closed form. Fresh on-policy samples give an
        # unbiased score-function gradient, including dependence of sampling on
        # the weights. Reusing rollout actions here would not estimate H(pi_new).
        action = self.distribution.sample((4,))
        logp = self.log_prob(action)
        value = -logp.mean(0)
        surrogate = -((logp.detach()+1.)*logp).mean(0)
        return value.detach()+surrogate-surrogate.detach()


class ArmRelaxationEnv(ArmEfficiencyEnv):
    def __init__(self, spec=None, exploration=None, timing=None, reward=None, relaxation=None,
                 *, worker=0, workers=1):
        self.relaxation = relaxation or RelaxationSpec()
        super().__init__(spec, exploration, timing, reward, worker=worker, workers=workers)
        self.base_width = len(self.observation_names)
        self.observation_names = self.observation_names+[
            f'relaxation/active_torque_gain_Nm/{name}' for name in self.unwrapped.muscle_names]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.base_width+6,), dtype=np.float32)

    def observation(self):
        # Add guidance after inherited reset/step have updated pressure. Inserting
        # it in encode() would make their observation[-1] overwrite a muscle gain.
        return np.concatenate((super().observation(), torque_gains(self.unwrapped))).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return self.observation(), info

    def step(self, action):
        observation, reward, failed, done, info = super().step(action)
        augmented = np.concatenate((observation, torque_gains(self.unwrapped))).astype(np.float32)
        return augmented, reward, failed, done, info

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_relaxation.v1', relaxation=asdict(self.relaxation),
            relaxation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_observation_width=self.base_width,
            observation_scale=contract['observation_scale']+[1.]*6,
            observation_offset=contract['observation_offset']+[0.]*6,
            relaxation_rule='causal fixed-fiber isometric active torque gains; remove fraction*min('
                'positive,negative) from both groups; shared joint or independent per-muscle component; '
                'exact mixture likelihood, fresh 4-sample score-function entropy; '
                'deterministic execution uses nominal mean with relaxation disabled')
        return contract


class RelaxationExtractor(PressureExtractor):
    def __init__(self, width, indices, mode):
        super().__init__(width-6, indices, mode)
        self.latent_dim_pi = 135  # Original 128, pressure, six mechanical gains.

    def forward_actor(self, observations):
        return torch.cat((super().forward_actor(observations[:, :-6]), observations[:, -6:]), -1)

    def forward_critic(self, observations):
        return super().forward_critic(observations[:, :-6])


class RelaxationPolicy(ExplorationPolicy):
    def __init__(self, *args, relaxation, **kwargs):
        self.relaxation = RelaxationSpec(**relaxation)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = RelaxationExtractor(self.features_dim, self.predictor_indices,
                                                 self.feature_mode).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Gains guide only the sampling distribution; no new trainable path.
        self.action_net = torch.nn.Linear(129, 12).to(self.device)
        self.action_dist = RelaxationDistribution(6)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        alpha, beta = (functional.softplus(self.action_net(latent_pi[:, :129]))+.05).chunk(2, -1)
        pressure = latent_pi[:, 128:129].clamp(0., 1.)
        requested = 1.+(self.exploration.maximum_temperature-1.)*pressure
        temperature = (torch.minimum(requested, torch.minimum(alpha, beta)/.05)
                       if self.exploration.mode == 'stall' else torch.ones_like(alpha))
        return self.action_dist.configure(alpha/temperature, beta/temperature,
                                           latent_pi[:, -6:], self.relaxation)


def make_relaxation_model(vector, parent, contract, *, learning_rate=1e-5,
                          n_steps=256, batch_size=128, n_epochs=10):
    kwargs = copy.deepcopy(parent.policy_kwargs)
    kwargs['relaxation'] = contract['relaxation']
    model = ArmPPO(RelaxationPolicy, vector, seed=0, device='cpu', learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=parent.gamma,
        gae_lambda=parent.gae_lambda, clip_range=.2, ent_coef=parent.ent_coef,
        vf_coef=parent.vf_coef, max_grad_norm=parent.max_grad_norm, target_kl=parent.target_kl,
        policy_kwargs=kwargs)
    model.policy.load_state_dict(parent.policy.state_dict(), strict=True)
    model.policy.optimizer.load_state_dict(copy.deepcopy(parent.policy.optimizer.state_dict()))
    for group in model.policy.optimizer.param_groups:
        group['lr'] = learning_rate
    model.num_timesteps, model.optimizer_steps = parent.num_timesteps, parent.optimizer_steps
    model._n_updates = parent._n_updates
    return model


def load_relaxation(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    c = meta['contract']
    if c['schema'] != 'myo.arm_relaxation.v1':
        raise ValueError('Not an arm relaxation checkpoint')
    env = ArmRelaxationEnv(spec_from_dict(c['spec']), ExplorationSpec(**c['exploration']),
        ArmTiming(**c['timing']), EfficiencyReward(**c['efficiency_reward']), RelaxationSpec(**c['relaxation']))
    try:
        if env.contract() != c or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Arm relaxation physics/input/distribution/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != c or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, RelaxationPolicy)
                or asdict(model.policy.relaxation) != c['relaxation']
                or asdict(model.policy.exploration) != c['exploration']
                or model.observation_space != env.observation_space or model.policy.feature_mode != 'none'):
            raise ValueError('Arm relaxation embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta
