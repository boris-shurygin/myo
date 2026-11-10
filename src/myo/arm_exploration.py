"""Causal stagnation context and likelihood-consistent Beta exploration."""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from .arm_learning import ArmExtractor, ArmPolicy, ArmPPO
from .arm_reward_curriculum import ArmRewardCurriculumEnv, RewardStage
from .arm_workload import spec_from_dict


@dataclass(frozen=True)
class ExplorationSpec:
    mode: str = 'stall'
    window_s: float = .4
    error_threshold_deg: float = 5.
    progress_threshold_deg: float = .5
    rise_s: float = .2
    fall_s: float = .1
    maximum_temperature: float = 16.

    def __post_init__(self):
        if self.mode not in ('off', 'stall'):
            raise ValueError('Unknown exploration mode')
        if not all(np.isfinite(v) and v > 0 for v in (
                self.window_s, self.error_threshold_deg, self.progress_threshold_deg,
                self.rise_s, self.fall_s, self.maximum_temperature)) or self.maximum_temperature < 1:
            raise ValueError('Exploration scales must be finite and positive, temperature >= 1')


class StagnationDetector:
    def __init__(self, spec, dt):
        self.spec, self.dt = spec, dt
        self.reset()

    def reset(self):
        self.time, self.pressure, self.progress_deg = 0., 0., 0.
        self.target, self.stagnant = None, False
        self.history = deque()

    def update(self, q, target):
        if self.target is None or abs(target - self.target) > 1e-8:
            self.reset()
            self.target = target
        error = float(abs(np.rad2deg(q - target)))
        self.history.append((self.time, error))
        while self.time - self.history[0][0] > self.spec.window_s + 1e-9:
            self.history.popleft()
        full = self.time - self.history[0][0] >= self.spec.window_s - 1e-9
        self.progress_deg = self.history[0][1] - error
        self.stagnant = bool(full and error > self.spec.error_threshold_deg
                             and self.progress_deg < self.spec.progress_threshold_deg)
        delta = self.dt / (self.spec.rise_s if self.stagnant else self.spec.fall_s)
        self.pressure = float(np.clip(self.pressure + (delta if self.stagnant else -delta), 0., 1.))
        self.time += self.dt
        return self.pressure


class ArmExplorationEnv(ArmRewardCurriculumEnv):
    def __init__(self, spec=None, reward_stage=None, exploration=None, *, worker=0, workers=1):
        self.exploration = exploration or ExplorationSpec()
        self.detector = StagnationDetector(self.exploration, .02)
        super().__init__(spec, reward_stage, worker=worker, workers=workers)
        self.observation_names = self.observation_names + ['exploration/stagnation_pressure']
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(156,), dtype=np.float32)

    def encode(self, observation):
        return np.concatenate((super().encode(observation), [self.detector.pressure])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.detector.reset()
        observation, info = super().reset(seed=seed, options=options)
        self.detector.update(info['q'], info['requested_target_q'])
        observation[-1] = self.detector.pressure
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation[-1] = self.detector.update(info['q'], info['requested_target_q'])
        info['exploration'] = {'pressure': self.detector.pressure,
            'stagnant': self.detector.stagnant, 'progress_deg': self.detector.progress_deg}
        return observation, reward, terminated, truncated, info

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_exploration.v1', exploration=asdict(self.exploration),
            exploration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            observation_scale=contract['observation_scale'] + [1.],
            observation_offset=contract['observation_offset'] + [0.],
            exploration_rule='T=min(1+(Tmax-1)*pressure,min(alpha,beta)/0.05); '
                'independent Beta(alpha/T,beta/T), mean unchanged; off uses T=1; '
                'detector runs in both arms; no reward or physical-state intervention')
        return contract


class PressureExtractor(ArmExtractor):
    def __init__(self, width, indices, mode):
        super().__init__(width, indices, mode)
        self.latent_dim_pi = 129

    def forward_actor(self, observations):
        return torch.cat((super().forward_actor(observations), observations[:, -1:]), dim=-1)


class ExplorationPolicy(ArmPolicy):
    def __init__(self, *args, exploration, **kwargs):
        self.exploration = ExplorationSpec(**exploration)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = PressureExtractor(self.features_dim, self.predictor_indices,
                                             self.feature_mode).to(self.device)

    def _get_action_dist_from_latent(self, latent_pi):
        alpha, beta = (functional.softplus(self.action_net(latent_pi)) + .05).chunk(2, dim=-1)
        pressure = latent_pi[:, -1:].clamp(0., 1.)
        requested = 1. + (self.exploration.maximum_temperature - 1.) * pressure
        # Keep the existing numerical concentration floor while preserving the
        # mean exactly. Near-silent muscles therefore receive little broadening.
        temperature = (torch.minimum(requested, torch.minimum(alpha, beta) / .05)
                       if self.exploration.mode == 'stall' else torch.ones_like(alpha))
        self.action_dist.distribution = torch.distributions.Beta(alpha / temperature, beta / temperature)
        return self.action_dist


def make_exploration_model(env, parent, contract, *, learning_rate=1e-5,
                           n_steps=256, batch_size=128, n_epochs=10):
    native = contract['native']
    model = ArmPPO(ExplorationPolicy, env, seed=0, device='cpu', learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': native['predictor_indices'],
            'physical_indices': native['physical_indices'], 'feature_mode': 'none',
            'exploration': contract['exploration'], 'activation_fn': nn.SiLU,
            'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})
    previous = dict(parent.policy.named_parameters())
    with torch.no_grad():
        for name, parameter in model.policy.named_parameters():
            source = previous[name]
            if parameter.shape == source.shape:
                parameter.copy_(source)
            else:
                if (name not in ('mlp_extractor.actor_private.0.weight',
                                 'mlp_extractor.critic_private.0.weight', 'action_net.weight')
                        or parameter.shape != (source.shape[0], source.shape[1]+1)):
                    raise ValueError('Unexpected exploration transfer shape')
                parameter.zero_()
                parameter[:, :source.shape[1]].copy_(source)
            if source in parent.policy.optimizer.state and parameter.requires_grad:
                state = copy.deepcopy(parent.policy.optimizer.state[source])
                for field, value in state.items():
                    if isinstance(value, torch.Tensor) and value.shape == source.shape and value.shape != parameter.shape:
                        state[field] = torch.zeros_like(parameter)
                        state[field][:, :source.shape[1]].copy_(value)
                model.policy.optimizer.state[parameter] = state
        for name, buffer in model.policy.named_buffers():
            buffer.copy_(dict(parent.policy.named_buffers())[name])
    model.num_timesteps, model.optimizer_steps = parent.num_timesteps, parent.optimizer_steps
    model._n_updates = parent._n_updates
    return model


def load_exploration(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    contract = meta['contract']
    if contract['schema'] != 'myo.arm_exploration.v1':
        raise ValueError('Not an arm exploration checkpoint')
    env = ArmExplorationEnv(spec_from_dict(contract['spec']), RewardStage(**contract['reward_stage']),
                            ExplorationSpec(**contract['exploration']))
    try:
        if env.contract() != contract:
            raise ValueError('Arm exploration physics/input/reward identity mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Arm exploration checkpoint hash mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != contract or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, ExplorationPolicy)
                or asdict(model.policy.exploration) != contract['exploration']
                or model.observation_space.shape != (156,) or model.policy.feature_mode != 'none'):
            raise ValueError('Arm exploration embedded identity mismatch')
    except Exception:
        env.close()
        raise
    return model, env, meta


def evaluation_job(path, conditions):
    """Deterministic means only: no additional exploration during assessment."""
    torch.set_num_threads(1)
    model, env, _ = load_exploration(path)
    rows = []
    try:
        for condition in conditions:
            observation, _ = env.reset(seed=0, options=condition)
            trace = []
            while True:
                action = model.predict(observation, deterministic=True)[0]
                observation, _, failed, truncated, info = env.step(action)
                trace.append((env.unwrapped.time, np.rad2deg(info['q']), np.rad2deg(info['qvel']),
                    info['workload']['activation_squared_mean'], info['workload']['active_fraction_mean'],
                    info['activation'], info['exploration']['pressure']))
                if failed or truncated:
                    break
            times = np.array([s[0] for s in trace])
            error = np.array([s[1] for s in trace]) - condition['case'][1]
            velocity = np.array([s[2] for s in trace])
            tail = times > condition['seconds'] - 1. - 1e-8
            if not tail.any():
                tail = np.ones(len(trace), dtype=bool)
            full = not failed and len(trace) == round(condition['seconds']/.02)
            settled = (abs(error) <= 3.) & (abs(velocity) <= 5.)
            fraction = float(settled[tail].mean()) if full else 0.
            stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
            direction = np.sign(condition['case'][1]-condition['case'][0])
            rows.append({**condition, 'failure': bool(failed), 'passed': bool(full and fraction >= .9),
                'settled_fraction': fraction, 'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
                'full_rmse_deg': float(np.sqrt(np.mean(error**2))),
                'overshoot_deg_20ms': float(max(0., np.max(direction*error))) if direction else None,
                'settling_time_s_20ms': float(times[stable[0]]) if len(stable) and full else None,
                'activation_squared_mean': float(np.mean([s[3] for s in trace])),
                'active_fraction_mean': float(np.mean([s[4] for s in trace])),
                'tail_activation_by_muscle': np.mean([s[5] for s, used in zip(trace, tail) if used], axis=0).tolist(),
                'pressure_fraction': float(np.mean([s[6] > .5 for s in trace]))})
    finally:
        env.close()
    return {'cases': rows}
