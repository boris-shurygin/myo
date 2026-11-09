"""Short, fixed MyoArm positioning tasks with causal compressed motor queues."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .compact_protocol import initialize_command_at_reset
from .env import PreacherCurlEnv
from .runtime import read_config, dynamics_digest


TRAIN_CASES = tuple(dict.fromkeys((q, t) for t in (15, 35, 55, 75, 95, 115, 135)
                                  for q in (t, 20, 75, 125)))
VALIDATION_CASES = ((35, 95), (95, 35), (45, 75), (105, 75), (25, 25),
                    (65, 65), (105, 105), (125, 125), (75, 75))
TEST_CASES = ((20, 75), (75, 20), (30, 30), (60, 60), (90, 90), (120, 120),
              (130, 130), (15, 15), (25, 105), (105, 25), (60, 125))
REWARD = {"position": "anchored_log1p_abs", "position_weight": .85,
    "position_epsilon_deg": 3., "position_anchor_deg": 90., "velocity_weight": .15,
    "velocity_scale_rad_s": 1.5, "other_costs": "unchanged native costs",
    "contact_proxy_cost": ".1*(peak contact N/100)^2", "multiplier": .1,
    "experimental_domain_failure": "q>160deg or source path reversal; no mechanical upper stop"}


def arm_config():
    path = Path(__file__).resolve().parents[2]/"configs/d1_myoarm.json"
    return replace(read_config(path), history_steps=2, episode_seconds=4.)


class ArmPrecisionEnv(gym.Wrapper):
    def __init__(self):
        super().__init__(PreacherCurlEnv(arm_config()))
        base = self.unwrapped
        self.cases, self.next_case = TRAIN_CASES, 0
        self.case = self.cases[0]
        self.queue_steps = len(base.muscles[0].delay.pending)
        if self.queue_steps != 2*base.config.frame_skip or base.control_dt != .02:
            raise ValueError("Arm precision requires exact 40ms motor delay and 20ms commands")
        self.prefix = len(base.observation_names)-6*self.queue_steps
        self.observation_names = base.observation_names[:self.prefix]+[
            f"pending/{age}/{name}" for age in range(2) for name in base.muscle_names]
        self.scale = np.ones(len(self.observation_names))
        self.offset = np.zeros(len(self.observation_names))
        for i, name in enumerate(self.observation_names):
            field = name.split('/')[-1]
            if 'centered_rad' in field:
                self.scale[i] = 2/np.pi
            elif field in ('activation', 'remaining_capacity', 'trigger') or 'excitation/' in name or name.startswith('pending/'):
                self.scale[i], self.offset[i] = 2., -1.
            elif field == 'fiber_length_normalized':
                self.scale[i], self.offset[i] = 5., -5.
            elif field == 'tendon_length_normalized':
                self.scale[i], self.offset[i] = 25., -25.
            elif field == 'tendon_strain':
                self.scale[i] = 25.
        physical_names = ['angle_centered_rad', 'angular_velocity_scaled']
        physical_names += [f'{name}/{field}' for field in ('fiber_length_normalized', 'activation', 'remaining_capacity')
                           for name in base.muscle_names]
        self.physical_indices = [self.observation_names.index(n) for n in physical_names]
        self.pending_indices = [self.observation_names.index(f'pending/{age}/{name}')
                                for age in range(2) for name in base.muscle_names]
        self.predictor_indices = self.physical_indices+self.pending_indices[:6]+[2]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32)

    def encode(self, observation):
        pending = observation[self.prefix:].reshape(6, 2, self.unwrapped.config.frame_skip)
        if not np.all(pending == pending[:, :, :1]):
            raise ValueError("Motor queue cannot be compressed without losing command changes")
        compact = np.concatenate((observation[:self.prefix], pending[:, :, 0].T.reshape(-1)))
        return (compact*self.scale+self.offset).astype(np.float32)

    def observation(self):
        return self.encode(self.unwrapped.observe())

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        self.case = tuple(options.pop('case', self.cases[self.next_case]))
        self.next_case = (self.next_case+1) % len(self.cases)
        initial, target = self.case
        _, info = self.env.reset(seed=seed, options={
            'q': float(np.deg2rad(initial)), 'target_angle': float(np.deg2rad(target)),
            'trigger': 1., 'payload_mass': 1., 'capacity': options.pop('capacity', 1.), **options})
        initialize_command_at_reset(self.env, target, 1.)
        return self.observation(), info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        outside = bool(info['q'] > np.deg2rad(160) or info['path_direction_reversal'])
        if outside:
            terminated, truncated = True, False
            self.unwrapped._episode_done = True
            info['termination_reason'] = 'outside_experimental_geometry_domain'
        terms = {k: float(v) for k, v in info['reward_components'].items()
                 if k not in ('position_tracking', 'velocity_tracking')}
        error = abs(info['q']-info['requested_target_q'])
        c = (1-np.exp(-1))/np.log1p(90/3)
        tracking = float(info['trigger'] > 0)
        terms['position_log'] = tracking*.85*(1-c*np.log1p(error/np.deg2rad(3)))
        terms['velocity'] = tracking*.15*float(np.exp(-(info['qvel']/1.5)**2))
        terms['contact_force_proxy_cost'] = -.1*(info['substep_peaks']['tissue_contact_force_N']/100.)**2
        if terminated:
            terms['physical_failure'] = -1.
        info['reward_components'] = terms
        info['arm_precision_failure'] = bool(terminated)
        return self.encode(observation), .1*sum(terms.values()), terminated, truncated, info

    def contract(self):
        root = Path(__file__).parent
        return {'schema': 'myo.arm_precision.v1', 'config': asdict(self.unwrapped.config),
            'dynamics_sha256': dynamics_digest(), 'geometry_sha256': self.unwrapped.geometry.fingerprint,
            'action_mapping': list(self.unwrapped.muscle_names), 'action_semantics': 'six direct excitations before delay',
            'observation_names': self.observation_names, 'scale': self.scale.tolist(), 'offset': self.offset.tolist(),
            'physical_indices': self.physical_indices, 'predictor_indices': self.predictor_indices,
            'pending_indices': self.pending_indices, 'reward': REWARD,
            'source_hashes': {n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                              for n in ('arm_precision.py', 'arm_learning.py', 'policy_beta.py')}}


def evaluate_arm(model, cases=VALIDATION_CASES, *, samples=False):
    env = ArmPrecisionEnv()
    results = []
    for case in cases:
        obs, _ = env.reset(seed=0, options={'case': case})
        trace, failed, reward = [], False, 0.
        while True:
            action = env.unwrapped.baseline_action() if model == 'heuristic' else model.predict(obs, deterministic=True)[0]
            obs, r, failed, truncated, info = env.step(action)
            trace.append([env.unwrapped.time, float(np.rad2deg(info['q'])),
                          float(np.rad2deg(info['qvel'])), float(np.rad2deg(info['requested_target_q'])),
                          *np.asarray(action).tolist()])
            reward += r
            if failed or truncated:
                break
        a = np.asarray(trace)
        error = a[:, 1]-a[:, 3]
        tail = a[:, 0] > 3.-1e-8
        settled = (abs(error) <= 3.) & (abs(a[:, 2]) <= 5.)
        full = len(a) == 200 and not failed
        fraction = float(np.mean(settled[tail])) if tail.any() and full else 0.
        direction = np.sign(case[1]-case[0])
        stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
        row = {'initial_deg': case[0], 'target_deg': case[1], 'failure': bool(failed),
            'passed': bool(full and fraction >= .9), 'settled_fraction': fraction,
            'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))) if tail.any() else float(np.sqrt(np.mean(error**2))),
            'full_rmse_deg': float(np.sqrt(np.mean(error**2))), 'return': reward,
            'overshoot_deg_20ms': float(max(0., np.max(direction*error))) if direction else None,
            'settling_time_s_20ms': float(a[stable[0], 0]) if len(stable) and full else None,
            'duration_s': float(a[-1, 0])}
        if samples:
            row['samples'] = trace
        results.append(row)
    env.close()
    return {'cases': results, 'score_deg': float(np.mean([r['tail_rmse_deg'] for r in results])),
        'passed': sum(r['passed'] for r in results), 'total': len(results),
        'failures': sum(r['failure'] for r in results)}


def rank(result):
    return result['failures'], -result['passed'], result['score_deg']
