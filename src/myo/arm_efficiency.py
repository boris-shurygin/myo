"""Stable-hold/effort reward adapter; the pinned 10/10-ms physics stays unchanged."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .arm_exploration import ExplorationSpec, ExplorationPolicy
from .arm_learning import ArmPPO
from .arm_tension_diagnostics import antagonist_metrics
from .arm_timing import ArmTiming, ArmTimingEnv, load_timing
from .arm_workload import WorkloadSpec, spec_from_dict
from .arm_workload_evaluation import merge_results


@dataclass(frozen=True)
class EfficiencyReward:
    velocity_weight: float = .15
    velocity_scale_deg_s: float = 5.
    near_goal_scale_deg: float = 10.
    activation_squared_weight: float = 0.

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in asdict(self).values())
                or min(self.velocity_weight, self.activation_squared_weight) < 0
                or min(self.velocity_scale_deg_s, self.near_goal_scale_deg) <= 0):
            raise ValueError('Invalid stability/effort reward scales')


def velocity_cost(error_rad, velocity_rad_s, reward, reference_velocity_rad_s=0.):
    """Bounded tracking cost; future moving references must supply their velocity."""
    gate = np.exp(-np.square(np.rad2deg(error_rad)/reward.near_goal_scale_deg))
    v2 = np.square(np.rad2deg(velocity_rad_s-reference_velocity_rad_s)/reward.velocity_scale_deg_s)
    return float(-reward.velocity_weight*gate*v2/(1.+v2))


class ArmEfficiencyEnv(ArmTimingEnv):
    def __init__(self, spec=None, exploration=None, timing=None, reward=None, *, worker=0, workers=1):
        self.efficiency_reward = reward or EfficiencyReward()
        super().__init__(spec, exploration, timing, worker=worker, workers=workers)
        # Do not apply the inherited workload costs a second time.
        if self.workload.activation_squared_weight or self.workload.active_fraction_weight:
            raise ValueError('Efficiency adapter owns its activation cost')

    def step(self, action):
        observation, _, failed, done, info = super().step(action)
        terms = dict(info['reward_components'])
        terms['near_goal_velocity_cost'] = velocity_cost(
            info['q']-info['requested_target_q'], info['qvel'], self.efficiency_reward)
        terms['activation_squared_proxy_cost'] = (
            -self.efficiency_reward.activation_squared_weight*info['workload']['activation_squared_mean'])
        info['reward_components'] = terms
        return observation, .1*sum(terms.values()), failed, done, info

    def contract(self):
        contract = super().contract()
        contract.update(schema='myo.arm_efficiency.v1', efficiency_reward=asdict(self.efficiency_reward),
            efficiency_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            efficiency_rule='0.1*(position_log+physical_failure-velocity_weight*'
                'exp(-(error_deg/near_goal_scale_deg)^2)*v2/(1+v2)-activation_weight*mean(actual_activation^2)); '
                'v2=(velocity_error_deg_s/velocity_scale_deg_s)^2; held reference velocity=0')
        return contract


def load_efficiency(path):
    path = Path(path)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    contract = meta['contract']
    if contract['schema'] != 'myo.arm_efficiency.v1':
        raise ValueError('Not an arm efficiency checkpoint')
    env = ArmEfficiencyEnv(spec_from_dict(contract['spec']), ExplorationSpec(**contract['exploration']),
        ArmTiming(**contract['timing']), EfficiencyReward(**contract['efficiency_reward']))
    try:
        if env.contract() != contract or hashlib.sha256(path.read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Arm efficiency physics/input/reward/checkpoint identity mismatch')
        model = ArmPPO.load(path, device='cpu')
        if (model.workload_contract != contract or model.workload_training != meta['training']
                or model.num_timesteps != meta['steps'] or not isinstance(model.policy, ExplorationPolicy)
                or asdict(model.policy.exploration) != contract['exploration']
                or model.observation_space != env.observation_space or model.policy.feature_mode != 'none'):
            raise ValueError('Arm efficiency embedded identity mismatch')
    except BaseException:
        env.close()
        raise
    return model, env, meta


def load_parent(path):
    schema = json.loads(Path(str(path)+'.metadata.json').read_text())['contract']['schema']
    return load_efficiency(path) if schema == 'myo.arm_efficiency.v1' else load_timing(path)


def evaluate_job(path, conditions, trace_path=None):
    """Separate-process evaluation: deterministic means, engineering tension metrics."""
    torch.set_num_threads(1)
    model, env, _ = load_parent(path)
    rows, traces = [], {}
    try:
        for index, condition in enumerate(conditions):
            observation, initial = env.reset(seed=0, options=condition)
            samples = []
            while True:
                action = model.predict(observation, deterministic=True)[0]
                observation, _, failed, done, info = env.step(action)
                samples.append([info['t'], info['q'], info['qvel'], *info['activation'],
                    *info['tendon_force_N'], *info['moment_arm_m'], *info['capacity']])
                if failed or done:
                    break
            a = np.asarray(samples)
            traces[f'case_{index:03d}'] = a
            error = np.rad2deg(a[:, 1])-condition['case'][1]
            velocity = np.rad2deg(a[:, 2])
            tail = a[:, 0] > condition['seconds']-1.-1e-8
            if not tail.any():
                tail = np.ones(len(a), dtype=bool)
            full = not failed and len(a) == round(condition['seconds']/env.unwrapped.control_dt)
            settled = (abs(error) <= 3.) & (abs(velocity) <= 5.)
            stable = np.flatnonzero(np.logical_and.accumulate(settled[::-1])[::-1])
            direction = np.sign(condition['case'][1]-condition['case'][0])
            metrics = antagonist_metrics(a[:, 3:9], a[:, 9:15], a[:, 15:21], a[:, 2])
            row = {**condition, 'failure': bool(failed), 'passed': bool(full and settled[tail].mean() >= .9),
                'steps': len(a), 'settled_fraction': float(settled[tail].mean()) if full else 0.,
                'tail_rmse_deg': float(np.sqrt(np.mean(error[tail]**2))),
                'tail_velocity_rms_deg_s': float(np.sqrt(np.mean(velocity[tail]**2))),
                'full_rmse_deg': float(np.sqrt(np.mean(error**2))),
                'overshoot_deg': float(max(0., np.max(direction*error))) if direction else None,
                'settling_time_s': float(a[stable[0], 0]) if len(stable) and full else None,
                'activation_squared_mean': float(metrics['activation_squared_mean'].mean()),
                'active_fraction_mean': float(np.mean(a[:, 3:9]*a[:, 21:27])),
                'capacity_net_loss_by_muscle': (np.array(initial['capacity'])-a[-1, 21:27]).tolist(),
                'tail_activation_by_muscle': a[tail, 3:9].mean(0).tolist()}
            for key, value in metrics.items():
                row['full_'+key] = float(value.mean())
                row['tail_'+key] = float(value[tail].mean())
            rows.append(row)
        if trace_path:
            np.savez_compressed(trace_path, **traces)
    finally:
        env.close()
    result = merge_results([{'cases': rows}])
    for key in ('tail_velocity_rms_deg_s', 'tail_activation_squared_mean', 'tail_activation_coactivation',
                'tail_opposing_muscle_torque_Nm', 'tail_summed_tendon_force_proxy_N'):
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def quality_gate(result, baseline):
    """Per-condition retention; a good average cannot hide a failed hold or movement."""
    if len(result['cases']) != len(baseline['cases']):
        raise ValueError('Quality gate requires the same ordered conditions')
    reasons = []
    for i, (row, old) in enumerate(zip(result['cases'], baseline['cases'])):
        for key in ('case', 'payload_mass', 'tilt_deg', 'capacity', 'seconds'):
            if np.any(np.asarray(row[key]) != np.asarray(old[key])):
                raise ValueError('Quality gate condition mismatch')
        if row['failure'] or not row['passed']:
            reasons.append([i, 'pass'])
        if row['tail_rmse_deg'] > old['tail_rmse_deg']+.25:
            reasons.append([i, 'position'])
        if row['tail_velocity_rms_deg_s'] > max(5., 1.1*old['tail_velocity_rms_deg_s']):
            reasons.append([i, 'speed'])
        if old['settling_time_s'] is not None and (row['settling_time_s'] is None
                or row['settling_time_s'] > old['settling_time_s']+.2+1e-8):
            reasons.append([i, 'settling'])
        if old['overshoot_deg'] is not None and (row['overshoot_deg'] is None
                or row['overshoot_deg'] > old['overshoot_deg']+1.):
            reasons.append([i, 'overshoot'])
    return not reasons, reasons
