"""Small self-labelled command weakening followed by task-only PPO correction.

This is an engineered auxiliary learning rule, not a reproduction of a neural
plasticity mechanism. No command is modified after sampling or at inference.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence
import torch.nn.functional as functional

from .arm_efficiency import EfficiencyReward
from .arm_exploration import ExplorationSpec
from .arm_learning import ArmPPO
from .arm_relaxation import (ArmRelaxationEnv, RelaxationPolicy, RelaxationSpec,
                             relaxed_mean)
from .arm_timing import ArmTiming
from .arm_workload import spec_from_dict
from .policy_beta import BetaExcitationDistribution


@dataclass(frozen=True)
class WeakeningSpec:
    fraction: float = .05
    error_gate_deg: float = 3.
    speed_gate_deg_s: float = 5.
    concentration_multiplier: float = 16.
    loss_scale: float = .05
    mean_kl_limit: float = .1
    maximum_kl_limit: float = .5

    def __post_init__(self):
        if (not all(np.isfinite(x) and x > 0 for x in asdict(self).values())
                or self.fraction >= 1 or self.concentration_multiplier < 1):
            raise ValueError('Invalid weakening specification')


class ArmWeakeningEnv(ArmRelaxationEnv):
    def __init__(self, spec=None, exploration=None, timing=None, reward=None, weakening=None,
                 *, worker=0, workers=1):
        self.weakening = weakening or WeakeningSpec()
        super().__init__(spec, exploration, timing, reward, RelaxationSpec(), worker=worker, workers=workers)

    def contract(self):
        c = super().contract()
        c.update(schema='myo.arm_weakening.v1', weakening=asdict(self.weakening),
            weakening_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weakening_rule='single independent Beta at16x concentration (configured multiplier); '
                'stagnation temperature retained before scaling; deterministic nominal mean; '
                'auxiliary self-label lowers opposing active torque proxy only in settled states; '
                'nonsettled states anchor original means; PPO task-only physical reward; '
                'no post-sampling modification; mechanical gains excluded from learned features')
        return c


class NominalBetaDistribution(BetaExcitationDistribution):
    def mode(self):
        return self.nominal_mean


class WeakeningPolicy(RelaxationPolicy):
    def __init__(self, *args, weakening, **kwargs):
        self.weakening = WeakeningSpec(**weakening)
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        alpha, beta = (functional.softplus(self.action_net(latent_pi[:, :129]))+.05).chunk(2, -1)
        pressure = latent_pi[:, 128:129].clamp(0., 1.)
        requested = 1+(self.exploration.maximum_temperature-1)*pressure
        temperature = (torch.minimum(requested, torch.minimum(alpha,beta)/.05)
                       if self.exploration.mode == 'stall' else torch.ones_like(alpha))
        a, b = alpha/temperature, beta/temperature
        self.action_dist.distribution = Beta(a*self.weakening.concentration_multiplier,
                                             b*self.weakening.concentration_multiplier)
        self.action_dist.nominal_mean = a/(a+b)
        return self.action_dist

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Use the independent distribution's summed log density/analytic entropy,
        # not the inherited joint-mixture methods.
        self.action_dist = NominalBetaDistribution(6)


def make_weakening_model(vector, parent, contract, *, learning_rate=1e-5,
                          n_steps=256, batch_size=128, n_epochs=10):
    kwargs = copy.deepcopy(parent.policy_kwargs)
    kwargs.update(relaxation=contract['relaxation'], weakening=contract['weakening'])
    model = ArmPPO(WeakeningPolicy, vector, seed=0, device='cpu', learning_rate=learning_rate,
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


def weakening_labels(model, observations, contract):
    """Frozen targets from this policy, current sensations and known muscle gains."""
    obs = torch.as_tensor(observations, device=model.device)
    names = contract['observation_names']
    scale, offset = contract['observation_scale'], contract['observation_offset']

    def decode(name):
        i = names.index(name)
        return (obs[:, i]-offset[i])/scale[i]

    # Native angular_velocity_scaled is velocity/4; centered angles share the
    # same reference center, which cancels in the position error.
    error = decode('angle_centered_rad')-decode('requested_target_centered_rad')
    speed = 4*decode('angular_velocity_scaled')
    spec = model.policy.weakening
    gate = (error.abs() <= np.deg2rad(spec.error_gate_deg)) & (speed.abs() <= np.deg2rad(spec.speed_gate_deg_s))
    with torch.no_grad():
        d = model.policy.get_distribution(obs).distribution
        old_alpha, old_beta = d.concentration1.clone(), d.concentration0.clone()
        mean = d.mean.clone()
        lower = relaxed_mean(mean, obs[:, -6:], spec.fraction)
        target = torch.where(gate[:, None], lower, mean)
    return obs, target, gate, old_alpha, old_beta


def weaken_actor(model, observations, contract, *, updates=64, batch_size=256):
    """Explicit supervised steps, independently counted from PPO transitions.

    No backprop through native dynamics. Task error is corrected by subsequent
    real PPO rollouts; a separate native quality gate accepts/rejects candidates.
    """
    obs, target, gate, a, b = weakening_labels(model, observations, contract)
    if not gate.any():
        return {'attempted_updates': 0, 'accepted_updates': 0, 'gate_fraction': 0., 'stop': 'no_settled_states'}
    old_distribution = Beta(a, b)
    anchor = torch.arange(0, len(obs), max(1, len(obs)//2048), device=model.device)
    count, rejected, mean_kl, maximum_kl = 0, False, 0., 0.
    losses = []
    model.policy.set_training_mode(True)
    for index in range(updates):
        selected = (torch.arange(batch_size, device=model.device)+index*batch_size) % len(obs)
        before = copy.deepcopy(model.policy.state_dict())
        optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
        mean = model.policy.get_distribution(obs[selected]).distribution.mean
        loss = .5*((mean-target[selected])/model.policy.weakening.loss_scale).square().mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
        model.policy.optimizer.step()
        model.optimizer_steps += 1
        losses.append(float(loss.detach()))
        with torch.no_grad():
            new = model.policy.get_distribution(obs[anchor]).distribution
            # The analytic Beta expression subtracts large terms. Float32 can
            # report zero/negative KL for a small real update at concentration16x.
            kl = kl_divergence(Beta(a[anchor].double(), b[anchor].double()),
                Beta(new.concentration1.double(), new.concentration0.double())).sum(-1).clamp_min(0.)
            trial_mean, trial_maximum = float(kl.mean()), float(kl.max())
        if trial_mean > model.policy.weakening.mean_kl_limit or trial_maximum > model.policy.weakening.maximum_kl_limit:
            model.policy.load_state_dict(before)
            model.policy.optimizer.load_state_dict(optimizer)
            rejected = True
            break
        count += 1
        mean_kl, maximum_kl = trial_mean, trial_maximum
    with torch.no_grad():
        new_mean = model.policy.get_distribution(obs).distribution.mean
        old_mean = old_distribution.mean
    return {'attempted_updates': len(losses), 'accepted_updates': count, 'gate_fraction': float(gate.float().mean()),
        'mean_loss': float(np.mean(losses)), 'mean_kl': mean_kl, 'maximum_kl': maximum_kl,
        'mean_command_reduction': float((old_mean-new_mean).mean()),
        'settled_command_reduction': float((old_mean[gate]-new_mean[gate]).mean()),
        'nonsettled_command_change_rms': float((new_mean[~gate]-old_mean[~gate]).square().mean().sqrt()) if (~gate).any() else 0.,
        'stop': 'kl_budget' if rejected else 'update_budget'}


def load_weakening(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    c = meta['contract']
    if c['schema'] != 'myo.arm_weakening.v1':
        raise ValueError('Not a weakening arm checkpoint')
    env = ArmWeakeningEnv(spec_from_dict(c['spec']), ExplorationSpec(**c['exploration']),
        ArmTiming(**c['timing']), EfficiencyReward(**c['efficiency_reward']), WeakeningSpec(**c['weakening']))
    try:
        if env.contract() != c or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Weakening arm physics/input/distribution/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != c or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, WeakeningPolicy)
                or asdict(model.policy.weakening) != c['weakening']
                or asdict(model.policy.exploration) != c['exploration']
                or model.observation_space != env.observation_space or model.policy.feature_mode != 'none'):
            raise ValueError('Weakening arm embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta
