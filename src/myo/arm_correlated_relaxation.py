"""Full-rank spatial exploration with an exact bounded joint likelihood."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import (Categorical, MixtureSameFamily, MultivariateNormal,
                                 SigmoidTransform, TransformedDistribution)
import torch.nn.functional as functional

from .arm_efficiency import EfficiencyReward
from .arm_exploration import ExplorationSpec
from .arm_learning import ArmPPO
from .arm_relaxation import (ArmRelaxationEnv, RelaxationDistribution, RelaxationPolicy,
                             RelaxationSpec, relaxed_mean)
from .arm_timing import ArmTiming
from .arm_workload import spec_from_dict


@dataclass(frozen=True)
class CorrelatedSpec:
    mode: str = 'correlated'
    probability: float = .5
    fraction: float = .25
    common_std: float = .15
    residual_std: float = .003

    def __post_init__(self):
        if (self.mode not in ('correlated', 'diagonal')
                or not 0 < self.probability < 1 or not 0 < self.fraction < 1
                or not np.isfinite([self.common_std, self.residual_std]).all()
                or self.common_std <= 0 or self.residual_std <= 0):
            raise ValueError('Invalid correlated exploration specification')


class CorrelatedDistribution(RelaxationDistribution):
    def configure(self, alpha, beta, gains, spec):
        self.nominal_mean = alpha/(alpha+beta)
        relaxed = relaxed_mean(self.nominal_mean, gains, spec.fraction)
        centers = torch.stack((self.nominal_mean, relaxed), -2)
        locations = torch.logit(centers)
        # At each component's center, remove the first-order active-torque
        # component from the common logit direction. Nonlinear muscle dynamics
        # are deliberately not claimed to preserve actual tendon torque.
        direction = (torch.logit(self.nominal_mean)-torch.logit(relaxed)).unsqueeze(-2)
        h = gains.unsqueeze(-2)*centers*(1-centers)
        h = functional.normalize(h, dim=-1, eps=1e-12)
        direction = direction-(direction*h).sum(-1, keepdim=True)*h
        direction = functional.normalize(direction, dim=-1, eps=1e-12)
        covariance = spec.common_std**2*direction.unsqueeze(-1)*direction.unsqueeze(-2)
        covariance = covariance+spec.residual_std**2*torch.eye(6, device=alpha.device, dtype=alpha.dtype)
        if spec.mode == 'diagonal':
            covariance = torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
        self.locations, self.covariance = locations, covariance
        weights = alpha.new_tensor([1-spec.probability, spec.probability])
        self.latent_distribution = MixtureSameFamily(
            Categorical(probs=weights.expand(*alpha.shape[:-1], 2)),
            MultivariateNormal(locations, covariance_matrix=covariance))
        self.distribution = TransformedDistribution(self.latent_distribution, [SigmoidTransform()])
        return self


class ArmCorrelatedEnv(ArmRelaxationEnv):
    def __init__(self, spec=None, exploration=None, timing=None, reward=None, correlated=None,
                 *, worker=0, workers=1):
        self.correlated = correlated or CorrelatedSpec()
        # Reuse the causal gain observation, not the old sampling law.
        super().__init__(spec, exploration, timing, reward, RelaxationSpec(),
                         worker=worker, workers=workers)

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_correlated_relaxation.v1', correlated=asdict(self.correlated),
            correlated_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            correlated_rule='two-component full-rank logit-normal; causal torque-proxy-null '
                'common covariance plus positive isotropic residual; exact transformed mixture density; '
                'diagonal control has identical component marginals; deterministic nominal mean; '
                'four-sample score-function entropy; original Beta head concentration is unused')
        return contract


class CorrelatedPolicy(RelaxationPolicy):
    def __init__(self, *args, correlated, **kwargs):
        self.correlated = CorrelatedSpec(**correlated)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_dist = CorrelatedDistribution(6)

    def _get_action_dist_from_latent(self, latent_pi):
        alpha, beta = (functional.softplus(self.action_net(latent_pi[:, :129]))+.05).chunk(2, -1)
        # A fixed, recorded spatial exploration law replaces the old temperature
        # controller in this adapter. Pressure remains an original actor input.
        return self.action_dist.configure(alpha, beta, latent_pi[:, -6:], self.correlated)


def make_correlated_model(vector, parent, contract, *, learning_rate=1e-5,
                          n_steps=256, batch_size=128, n_epochs=10):
    kwargs = copy.deepcopy(parent.policy_kwargs)
    kwargs.update(relaxation=contract['relaxation'], correlated=contract['correlated'])
    model = ArmPPO(CorrelatedPolicy, vector, seed=0, device='cpu', learning_rate=learning_rate,
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


def load_correlated(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    c = meta['contract']
    if c['schema'] != 'myo.arm_correlated_relaxation.v1':
        raise ValueError('Not a correlated arm checkpoint')
    env = ArmCorrelatedEnv(spec_from_dict(c['spec']), ExplorationSpec(**c['exploration']),
        ArmTiming(**c['timing']), EfficiencyReward(**c['efficiency_reward']), CorrelatedSpec(**c['correlated']))
    try:
        if env.contract() != c or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Correlated arm physics/input/distribution/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != c or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, CorrelatedPolicy)
                or asdict(model.policy.correlated) != c['correlated']
                or asdict(model.policy.exploration) != c['exploration']
                or model.observation_space != env.observation_space or model.policy.feature_mode != 'none'):
            raise ValueError('Correlated arm embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta
