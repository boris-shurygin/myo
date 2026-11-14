"""Measure actual 10-ms exploration probes, not just held 200-ms commands."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_efficiency import EfficiencyReward, load_parent, velocity_cost
from myo.arm_relaxation import load_relaxation, relaxed_mean, torque_gains
from myo.delayed_acceleration import NativeResponse, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('Positive sample count required')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    protocol = json.loads((args.root/'protocol.json').read_text())
    reward_spec = EfficiencyReward(**protocol['reward'])
    parent, env, _ = load_parent(protocol['parent'])
    result, arrays = {}, {}
    try:
        obs, _ = env.reset(seed=0, options={'case': (75., 75.), 'payload_mass': .5,
            'tilt_deg': 30., 'capacity': 1., 'seconds': 8.})
        for _ in range(250):
            obs, _, failed, done, _ = env.step(parent.predict(obs, deterministic=True)[0])
            assert not failed and not done
        state = capture(env.unwrapped)
        nominal = parent.predict(obs, deterministic=True)[0]
        gains = torque_gains(env.unwrapped)
        augmented = np.r_[obs, gains].astype(np.float32)
        kernel = NativeResponse(replace(env.unwrapped.config, payload_mass=.5), horizon_steps=10)
        try:
            def response(action):
                kernel.restore(state)
                rows = []
                for i in range(10):
                    kernel.step(action if i == 0 else nominal)
                    info = kernel.env.telemetry()
                    a2 = float(np.mean(np.square(info['activation'])))
                    error = info['q']-np.deg2rad(75.)
                    c = (1-np.exp(-1))/np.log1p(90/3)
                    position = .1*.85*(1-c*np.log1p(abs(error)/np.deg2rad(3.)))
                    velocity = .1*velocity_cost(error, info['qvel'], reward_spec)
                    effort = -.1*reward_spec.activation_squared_weight*a2
                    rows.append([info['q'], info['qvel'], a2, info['muscle_torque_Nm'],
                                 position+velocity+effort, position, velocity, effort])
                return np.asarray(rows)

            baseline = response(nominal)
            # Verify the duplicated reward expression against the original
            # wrapper's position reward plus the exact efficiency adapter costs.
            for i in range(10):
                _, old_reward, failed, done, info = env.step(nominal)
                assert not failed and not done
                expected = old_reward+.1*(velocity_cost(info['q']-np.deg2rad(75.),
                    info['qvel'], reward_spec)-reward_spec.activation_squared_weight
                    *np.mean(np.square(info['activation'])))
                np.testing.assert_allclose(baseline[i, 4], expected, rtol=0., atol=1e-12)
            arrays['nominal_response'] = baseline
            # Separate the proposed direction itself from residual Beta noise.
            action = relaxed_mean(torch.as_tensor(nominal), torch.as_tensor(gains),
                                  protocol['relaxation']['fraction']).numpy()
            pure = response(action)
            arrays['mean_relaxation_response'] = pure
            delta = ((pure[:, 4:]-baseline[:, 4:])*np.power(.99, np.arange(10))[:, None]).sum(0)
            result['mean_relaxation'] = {
                'integrated_activation_reduction': float(1-pure[:, 2].mean()/baseline[:, 2].mean()),
                'max_absolute_angle_delta_deg': float(np.max(abs(np.rad2deg(pure[:, 0]-baseline[:, 0])))),
                'max_absolute_speed_delta_deg_s': float(np.max(abs(np.rad2deg(pure[:, 1]-baseline[:, 1])))),
                'return_delta': float(delta[0]), 'position_reward_delta': float(delta[1]),
                'velocity_reward_delta': float(delta[2]), 'effort_reward_delta': float(delta[3])}
            print(json.dumps({'mean_relaxation': result['mean_relaxation']}), flush=True)
            for mode in ('independent', 'joint'):
                model, check, _ = load_relaxation(args.root/mode/'initial.zip')
                try:
                    torch.manual_seed(0)
                    with torch.no_grad():
                        dist = model.policy.get_distribution(torch.as_tensor(augmented[None]))
                        actions = dist.distribution.sample((args.samples,))[:, 0].numpy()
                    traces = np.asarray([response(action) for action in actions])
                    reduction = 1-traces[:, :, 2].mean(-1)/baseline[:, 2].mean()
                    angle_delta = np.rad2deg(traces[:, :, 0]-baseline[:, 0])
                    speed_delta = np.rad2deg(traces[:, :, 1]-baseline[:, 1])
                    # A diagnostic threshold, NOT PPO reward/GAE or acceptance.
                    cheap = reduction >= .02
                    quiet = ((abs(angle_delta).max(-1) <= .25)
                             & (abs(speed_delta).max(-1) <= 5.))
                    discount = np.power(.99, np.arange(10))
                    reward_delta = ((traces[:, :, 4:]-baseline[:, 4:])*discount[None, :, None]).sum(1)
                    row = {'samples': args.samples, 'cheap_count': int(cheap.sum()),
                        'quiet_count': int(quiet.sum()), 'cheap_and_quiet_count': int((cheap & quiet).sum()),
                        'positive_100ms_return_delta_count': int((reward_delta[:, 0] > 0).sum()),
                        'cheap_quiet_positive_return_delta_count': int((cheap & quiet & (reward_delta[:, 0] > 0)).sum()),
                        'mean_100ms_return_delta': float(reward_delta[:, 0].mean()),
                        'mean_100ms_position_reward_delta': float(reward_delta[:, 1].mean()),
                        'mean_100ms_velocity_reward_delta': float(reward_delta[:, 2].mean()),
                        'mean_100ms_effort_reward_delta': float(reward_delta[:, 3].mean()),
                        'mean_integrated_activation_reduction': float(reduction.mean()),
                        'proxy_torque_delta_std_Nm': float(np.std((actions-nominal)@gains)),
                        'max_absolute_angle_delta_rms_deg': float(np.sqrt(np.mean(abs(angle_delta).max(-1)**2))),
                        'max_absolute_speed_delta_mean_deg_s': float(abs(speed_delta).max(-1).mean()),
                        'cheap_and_quiet_definition': '100ms mean actual a² reduced>=2%; '
                            'all sampled |q-q_nominal|<=.25deg and |v-v_nominal|<=5deg/s'}
                    result[mode] = row
                    arrays[mode+'_actions'] = actions
                    arrays[mode+'_responses'] = traces
                    print(json.dumps({mode: row}), flush=True)
                finally:
                    check.close()
        finally:
            kernel.close()
    finally:
        env.close()
    summary = {'protocol': 'one shared 75deg state after2.5s nominal control; '
        f'{args.samples} probes per initial distribution, seed0; first10ms command varied; '
        'next9 commands identical nominal, native state restored before every probe; '
        '100ms return delta uses actual reward and gamma.99 without critic bootstrap; '
        'diagnostic criteria and return deltas are not PPO advantages; no learning', 'results': result}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(args.out/'traces.npz', **arrays)


if __name__ == '__main__':
    main()
