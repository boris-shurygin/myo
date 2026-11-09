"""Variable-load and periodic-mount arm tasks with explicit effort/fatigue proxies."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from .arm_learning import ArmPPO, ArmPolicy
from .arm_precision import ArmPrecisionEnv, TRAIN_CASES, VALIDATION_CASES, TEST_CASES
from .compact_protocol import initialize_command_at_reset


@dataclass(frozen=True)
class WorkloadSpec:
    masses: tuple = (.5, 1., 2.)
    tilts: tuple = (30.,)
    capacities: tuple = (1.,)
    cases: tuple = TRAIN_CASES
    seconds: float = 4.
    activation_squared_weight: float = 0.
    active_fraction_weight: float = 0.


class ArmWorkloadEnv(ArmPrecisionEnv):
    def __init__(self, spec=None, *, worker=0, workers=1):
        super().__init__()
        self.workload = spec or WorkloadSpec()
        self.worker, self.workers, self.episode_index = worker, workers, 0
        self.native_contract = super().contract()
        self.native_config = self.unwrapped.config
        self.observation_names = self.observation_names+['mount/sin_minus_sin30', 'mount/cos_minus_cos30']
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(155,), dtype=np.float32)
        self.tilt_deg = 30.
        self.mount_id = self.unwrapped.model.body('arm_mount').id
        # One reproducible permutation, not seed sweeps. All Cartesian cells
        # appear once per cycle, and workers take disjoint schedule positions.
        self.schedule = [(q, target, mass, tilt, capacity) for q, target in self.workload.cases
                         for mass in self.workload.masses for tilt in self.workload.tilts
                         for capacity in self.workload.capacities]
        permutation = np.random.default_rng(0).permutation(len(self.schedule))
        self.schedule = [self.schedule[i] for i in permutation]

    def encode(self, observation):
        native = super().encode(observation)
        angle = np.deg2rad(self.tilt_deg)
        features = [np.sin(angle)-np.sin(np.deg2rad(30.)), np.cos(angle)-np.cos(np.deg2rad(30.))]
        return np.concatenate((native, features)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        index = self.worker+self.workers*self.episode_index
        q, target, mass, tilt, capacity = self.schedule[index % len(self.schedule)]
        self.episode_index += 1
        self.case = tuple(options.pop('case', (q, target)))
        mass = float(options.pop('payload_mass', mass))
        self.tilt_deg = float(options.pop('tilt_deg', tilt)) % 360.
        capacity = float(options.pop('capacity', capacity))
        seconds = float(options.pop('seconds', self.workload.seconds))
        base = self.unwrapped
        base.config = replace(self.native_config, arm_mount_tilt=float(np.deg2rad(self.tilt_deg)),
                              episode_seconds=seconds)
        half = np.deg2rad(self.tilt_deg)/2
        base.model.body_quat[self.mount_id] = np.asarray([
            np.cos(half), np.cos(half), -np.sin(half), np.sin(half)])/np.sqrt(2.)
        # Native reset recomputes constants and kinematics after mount/mass changes,
        # then initializes the real delayed and series-elastic muscle states.
        _, info = base.reset(seed=seed, options={'q': float(np.deg2rad(self.case[0])),
            'target_angle': float(np.deg2rad(self.case[1])), 'trigger': 1.,
            'payload_mass': mass, 'capacity': capacity, **options})
        initialize_command_at_reset(base, self.case[1], 1.)
        return self.observation(), info

    def step(self, action):
        before = np.array([m.capacity for m in self.unwrapped.muscles])
        observation, reward, terminated, truncated, info = super().step(action)
        activation = np.array(info['activation'])
        capacity = np.array(info['capacity'])
        metrics = {'activation_mean': float(activation.mean()),
            'activation_squared_mean': float(np.mean(activation**2)),
            'active_fraction_mean': float(np.mean(activation*capacity)),
            'flexor_activation': float(activation[:3].mean()),
            'extensor_activation': float(activation[3:].mean()),
            'coactivation_proxy': float(min(activation[:3].mean(), activation[3:].mean())),
            'capacity_net_loss_per_s': float(np.mean(before-capacity)/self.unwrapped.control_dt)}
        costs = {'activation_squared_proxy_cost': -self.workload.activation_squared_weight*metrics['activation_squared_mean'],
                 'active_fraction_fatigue_proxy_cost': -self.workload.active_fraction_weight*metrics['active_fraction_mean']}
        info['reward_components'].update(costs)
        info['workload'] = {'tilt_deg': self.tilt_deg, **metrics}
        # Match the existing 0.1 reward multiplier. Costs use actual activation,
        # after transport/filtering, not an instant simulated command effect.
        return observation, reward+.1*sum(costs.values()), terminated, truncated, info

    def contract(self):
        return {'schema': 'myo.arm_workload.v1', 'native': self.native_contract,
            'spec': json.loads(json.dumps(asdict(self.workload))),
            'observation_names': self.observation_names,
            'observation_scale': self.scale.tolist()+[1., 1.],
            'observation_offset': self.offset.tolist()+[0., 0.],
            'action_mapping': self.native_contract['action_mapping'],
            'mount': 'whole fixed arm Ry(-tilt)*Rx(pi/2), radians internally; modulo360 at reset',
            'variable_conditions': 'payload_mass>0, tilt any finite degrees, capacity(0,1], seconds>0 at reset only',
            'initial_muscles': 'native activation0.03 and native fiber equilibrium; no teacher preload',
            'energy': 'end-of-control-interval activation proxies; not metabolic joules; reward multiplier0.1',
            'sampling': 'seed0 permutation of Cartesian cells; worker+workers*episode_index',
            'predictor': None, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def spec_from_dict(value):
    value = dict(value)
    for name in ('masses', 'tilts', 'capacities'):
        value[name] = tuple(value[name])
    value['cases'] = tuple(tuple(c) for c in value['cases'])
    return WorkloadSpec(**value)


def make_workload_model(env, parent, contract, *, learning_rate=3e-5, n_steps=256,
                        batch_size=128, n_epochs=10):
    native = contract['native']
    model = ArmPPO(ArmPolicy, env, seed=0, device='cpu', learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=.99, gae_lambda=.95,
        clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs={'predictor_indices': native['predictor_indices'],
            'physical_indices': native['physical_indices'], 'feature_mode': 'none',
            'activation_fn': nn.SiLU, 'optimizer_class': torch.optim.RMSprop,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 0.}})
    previous = dict(parent.policy.named_parameters())
    with torch.no_grad():
        for name, parameter in model.policy.named_parameters():
            source = previous[name]
            if source.shape == parameter.shape:
                parameter.copy_(source)
            else:
                if name not in ('mlp_extractor.actor_private.0.weight', 'mlp_extractor.critic_private.0.weight'):
                    raise ValueError('Unexpected workload transfer shape')
                parameter.zero_()
                parameter[:, :source.shape[1]].copy_(source)
            if source in parent.policy.optimizer.state and parameter.requires_grad:
                state = copy.deepcopy(parent.policy.optimizer.state[source])
                for field, value in state.items():
                    if isinstance(value, torch.Tensor) and value.shape == source.shape and source.shape != parameter.shape:
                        state[field] = torch.zeros_like(parameter)
                        state[field][:, :source.shape[1]].copy_(value)
                model.policy.optimizer.state[parameter] = state
        # Carry nonparameter calibration buffers too, even for the unused head.
        for name, buffer in model.policy.named_buffers():
            buffer.copy_(dict(parent.policy.named_buffers())[name])
    model.num_timesteps, model.optimizer_steps = parent.num_timesteps, parent.optimizer_steps
    model._n_updates = parent._n_updates
    return model


def save_workload(model, env, path, training):
    path = Path(path)
    model.workload_contract = env.contract()
    model.workload_training = json.loads(json.dumps(training, allow_nan=False))
    model.save(path)
    meta = {'contract': model.workload_contract, 'training': model.workload_training,
            'steps': model.num_timesteps, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')


def load_workload(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = ArmWorkloadEnv(spec_from_dict(meta['contract']['spec']))
    try:
        if env.contract() != meta['contract']:
            raise ValueError('Arm workload physics/input/reward identity mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Arm workload checkpoint hash mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != meta['contract'] or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or model.policy.feature_mode != 'none'
                or model.observation_space.shape != (155,)):
            raise ValueError('Arm workload embedded identity mismatch')
    except Exception:
        env.close()
        raise
    return model, env, meta


def scenarios(cases, masses=(1.,), tilts=(30.,), capacities=(1.,), seconds=4.):
    return [{'case': tuple(case), 'payload_mass': mass, 'tilt_deg': tilt, 'capacity': capacity, 'seconds': seconds}
            for case in cases for mass in masses for tilt in tilts for capacity in capacities]


def evaluate_workload(model, conditions, *, samples=False, mask_extensors=False):
    env = ArmWorkloadEnv()
    rows = []
    with torch.no_grad():
        for condition in conditions:
            observation, _ = env.reset(seed=0, options=condition)
            trace, metrics, failed = [], [], False
            initial_capacity = np.array([m.capacity for m in env.unwrapped.muscles])
            while True:
                if isinstance(model, str):
                    action = env.unwrapped.baseline_action() if model == 'heuristic' else np.zeros(6, np.float32)
                else:
                    width = model.observation_space.shape[0]
                    action = model.predict(observation[:width], deterministic=True)[0]
                if mask_extensors:
                    action = np.asarray(action).copy(); action[3:] = 0.
                observation, _, failed, truncated, info = env.step(action)
                trace.append([env.unwrapped.time, float(np.rad2deg(info['q'])),
                    float(np.rad2deg(info['qvel'])), float(np.rad2deg(info['requested_target_q'])),
                    *np.asarray(action).tolist(), *info['activation'], *info['capacity']])
                metrics.append([info['workload'][key] for key in ('activation_mean', 'activation_squared_mean',
                    'active_fraction_mean', 'flexor_activation', 'extensor_activation', 'coactivation_proxy')])
                if failed or truncated:
                    break
            a, m = np.asarray(trace), np.asarray(metrics)
            error = a[:, 1]-a[:, 3]
            tail = a[:, 0] > condition['seconds']-1.-1e-8
            if not tail.any():
                tail = np.ones(len(a), dtype=bool)
            full = not failed and len(a) == round(condition['seconds']/.02)
            settled = (abs(error) <= 3.) & (abs(a[:, 2]) <= 5.)
            fraction = float(settled[tail].mean()) if full else 0.
            direction = np.sign(condition['case'][1]-condition['case'][0])
            stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
            row = {**condition, 'failure': bool(failed), 'passed': bool(full and fraction >= .9),
                'settled_fraction': fraction, 'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
                'full_rmse_deg': float(np.sqrt(np.mean(error**2))),
                'overshoot_deg_20ms': float(max(0., np.max(direction*error))) if direction else None,
                'settling_time_s_20ms': float(a[stable[0], 0]) if len(stable) and full else None,
                'duration_s': float(a[-1, 0]),
                'capacity_loss_by_muscle': (initial_capacity-np.asarray(info['capacity'])).tolist(),
                'tail_activation_by_muscle': a[tail, 10:16].mean(0).tolist()}
            for i, key in enumerate(('activation_mean', 'activation_squared_mean', 'active_fraction_mean',
                                      'flexor_activation', 'extensor_activation', 'coactivation_proxy')):
                row[key] = float(m[:, i].mean())
                row['tail_'+key] = float(m[tail, i].mean())
                row[key+'_integral_s'] = float(m[:, i].sum()*.02)
            if samples:
                row['samples'] = trace
            rows.append(row)
    env.close()
    groups = {}
    for field in ('payload_mass', 'tilt_deg', 'capacity'):
        groups[field] = {str(value): {'score_deg': float(np.mean([r['tail_rmse_deg'] for r in rows if r[field] == value])),
            'passed': sum(r['passed'] for r in rows if r[field] == value),
            'total': sum(r[field] == value for r in rows)} for value in sorted({r[field] for r in rows})}
    return {'cases': rows, 'score_deg': float(np.mean([r['tail_rmse_deg'] for r in rows])),
        'passed': sum(r['passed'] for r in rows), 'total': len(rows),
        'failures': sum(r['failure'] for r in rows), 'groups': groups,
        'activation_squared_mean': float(np.mean([r['activation_squared_mean'] for r in rows])),
        'active_fraction_mean': float(np.mean([r['active_fraction_mean'] for r in rows]))}


def validation_rank(result):
    return result['failures'], -result['passed']/result['total'], result['score_deg']


def stage_passed(result):
    return (result['failures'] == 0 and result['passed']/result['total'] >= .9
        and result['score_deg'] <= 3.
        and all(group['passed']/group['total'] >= .75
                for values in result['groups'].values() for group in values.values()))
