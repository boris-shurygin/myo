"""Calibrate spatial exploration using full native-state counterfactuals."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_correlated_relaxation import CorrelatedDistribution, CorrelatedSpec
from myo.arm_efficiency import EfficiencyReward, load_parent, velocity_cost
from myo.arm_relaxation import torque_gains
from myo.delayed_acceleration import NativeResponse, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path,
        default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--common-std', type=float)
    parser.add_argument('--residual-std', type=float, default=.001)
    parser.add_argument('--validation-goals', type=float, nargs='+', default=[55., 95., 125.])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    parent, env, meta = load_parent(args.parent)
    reward = EfficiencyReward(activation_squared_weight=.1)
    summary = {'parent': str(args.parent), 'parent_sha256': meta['checkpoint_sha256'],
        'validation_goals': args.validation_goals, 'fixed_common_std': args.common_std,
        'fixed_residual_std': args.residual_std,
        'seed': 0, 'samples': args.samples, 'reward': asdict(reward), 'calibration': [], 'validation': [],
        'protocol': 'first10ms action varied, next90ms fixed nominal; restored full native state; '
            'discounted100ms return gamma.99 without critic bootstrap; not PPO advantages; '
            'calibrate75deg only, validate55/95/125deg after selecting covariance; '
            'selection maximizes cheap+positive-return fraction, then mean return; '
            'proceed if >=35% beneficial and positive mean return and mean activation reduction'}
    arrays = {}
    try:
        def probe(goal, specs):
            obs, _ = env.reset(seed=0, options={'case': (goal, goal), 'payload_mass': .5,
                'tilt_deg': 30., 'capacity': 1., 'seconds': 8.})
            for _ in range(250):
                obs, _, failed, done, _ = env.step(parent.predict(obs, deterministic=True)[0])
                assert not failed and not done
            state = capture(env.unwrapped)
            nominal = parent.predict(obs, deterministic=True)[0]
            gains = torque_gains(env.unwrapped)
            kernel = NativeResponse(replace(env.unwrapped.config, payload_mass=.5), horizon_steps=10)
            try:
                def response(action):
                    kernel.restore(state)
                    rows = []
                    for i in range(10):
                        kernel.step(action if i == 0 else nominal)
                        info = kernel.env.telemetry()
                        a2 = float(np.mean(np.square(info['activation'])))
                        error = info['q']-np.deg2rad(goal)
                        c = (1-np.exp(-1))/np.log1p(90/3)
                        position = .1*.85*(1-c*np.log1p(abs(error)/np.deg2rad(3.)))
                        velocity = .1*velocity_cost(error, info['qvel'], reward)
                        effort = -.1*reward.activation_squared_weight*a2
                        rows.append([info['q'], info['qvel'], a2, position+velocity+effort,
                                     position, velocity, effort])
                    return np.asarray(rows)

                baseline = response(nominal)
                for i in range(10):
                    _, old_reward, failed, done, info = env.step(nominal)
                    assert not failed and not done
                    expected = old_reward+.1*(velocity_cost(info['q']-np.deg2rad(goal),
                        info['qvel'], reward)-reward.activation_squared_weight*np.mean(np.square(info['activation'])))
                    np.testing.assert_allclose(baseline[i, 3], expected, rtol=0., atol=1e-12)
                rows = []
                for spec in specs:
                    torch.manual_seed(0)
                    mu = torch.as_tensor(nominal[None])
                    with torch.no_grad():
                        d = CorrelatedDistribution(6).configure(mu, 1-mu, torch.as_tensor(gains[None]), spec)
                        actions = d.distribution.sample((args.samples,))[:, 0].numpy()
                    traces = np.asarray([response(action) for action in actions])
                    reduction = 1-traces[:, :, 2].mean(-1)/baseline[:, 2].mean()
                    dq, dv = np.rad2deg(traces[:, :, 0]-baseline[:, 0]), np.rad2deg(traces[:, :, 1]-baseline[:, 1])
                    delta = ((traces[:, :, 3:]-baseline[:, 3:])*np.power(.99, np.arange(10))[None, :, None]).sum(1)
                    quiet = (abs(dq).max(-1) <= .25) & (abs(dv).max(-1) <= 5.)
                    useful = (reduction > 0) & (delta[:, 0] > 0) & quiet
                    row = {'goal': goal, 'distribution': asdict(spec), 'beneficial_count': int(useful.sum()),
                        'quiet_count': int(quiet.sum()), 'positive_return_count': int((delta[:, 0] > 0).sum()),
                        'mean_activation_reduction': float(reduction.mean()),
                        'mean_return_delta': float(delta[:, 0].mean()),
                        'position_delta': float(delta[:, 1].mean()), 'velocity_delta': float(delta[:, 2].mean()),
                        'effort_delta': float(delta[:, 3].mean()),
                        'proxy_torque_std_Nm': float(np.std((actions-nominal)@gains)),
                        'mean_max_speed_delta_deg_s': float(abs(dv).max(-1).mean())}
                    key = f'{goal}_{spec.mode}_{spec.common_std}_{spec.residual_std}'
                    arrays[key+'_actions'], arrays[key+'_response'], arrays[key+'_baseline'] = actions, traces, baseline
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                return rows
            finally:
                kernel.close()

        specs = ([CorrelatedSpec(common_std=s, residual_std=r)
                  for s in (.05, .15, .3) for r in (.001, .003, .01)] if args.common_std is None
                 else [CorrelatedSpec(common_std=args.common_std, residual_std=args.residual_std)])
        summary['calibration'] = probe(75., specs)
        chosen = max(summary['calibration'], key=lambda row: (row['beneficial_count'], row['mean_return_delta']))
        selected = CorrelatedSpec(**chosen['distribution'])
        summary['selected'] = asdict(selected)
        for goal in args.validation_goals:
            summary['validation'].extend(probe(goal, [selected, replace(selected, mode='diagonal')]))
        summary['calibration_control'] = probe(75., [replace(selected, mode='diagonal')])
        checked = [chosen]+[r for r in summary['validation'] if r['distribution']['mode'] == 'correlated']
        summary['proceed'] = all(r['beneficial_count'] >= .35*args.samples
            and r['mean_return_delta'] > 0 and r['mean_activation_reduction'] > 0 for r in checked)
        summary['status'] = 'complete'
    finally:
        env.close()
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        np.savez_compressed(args.out/'traces.npz', **arrays)


if __name__ == '__main__':
    main()
