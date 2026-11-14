"""Counterfactual coordinated relaxation, restoring full native delayed states."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_efficiency import load_parent
from myo.arm_relaxation import relaxed_mean, torque_gains
from myo.delayed_acceleration import NativeResponse, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path,
        default=Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    model, env, meta = load_parent(args.parent)
    rows, traces = [], {}
    try:
        for goal in (55., 75., 95., 125.):
            obs, _ = env.reset(seed=0, options={'case': (goal, goal), 'payload_mass': .5,
                'tilt_deg': 30., 'capacity': 1., 'seconds': 8.})
            for _ in range(250):
                obs, _, failed, done, _ = env.step(model.predict(obs, deterministic=True)[0])
                assert not failed and not done
            state = capture(env.unwrapped)
            command = model.predict(obs, deterministic=True)[0]
            gain = torque_gains(env.unwrapped)
            kernel = NativeResponse(replace(env.unwrapped.config, payload_mass=.5), horizon_steps=20)
            try:
                reference = None
                for fraction in (0., .1, .25, .5):
                    action = relaxed_mean(torch.as_tensor(command), torch.as_tensor(gain), fraction).numpy()
                    kernel.restore(state)
                    trace = []
                    for step in range(20):
                        kernel.step(action)
                        info = kernel.env.telemetry()
                        trace.append([info['q'], info['qvel'], *info['activation'],
                            *info['tendon_force_N'], info['muscle_torque_Nm']])
                        if fraction == 0.:
                            # Independent wrapper continuation checks complete
                            # state restoration, geometry, queue and histories.
                            _, _, failed, done, actual = env.step(command)
                            assert not failed and not done
                            for key in ('q', 'qvel', 'activation', 'tendon_force_N', 'capacity'):
                                np.testing.assert_allclose(info[key], actual[key], rtol=0., atol=1e-10)
                    trace = np.asarray(trace)
                    if fraction == 0.:
                        reference = trace
                    traces[f'{goal:g}_{fraction:g}'] = trace
                    for end in (6, 10, 20):
                        a2, old = np.mean(trace[end-1, 2:8]**2), np.mean(reference[end-1, 2:8]**2)
                        rows.append({'goal_deg': goal, 'fraction': fraction, 'horizon_ms': end*10,
                            'command': action.tolist(), 'torque_gain_Nm': gain.tolist(),
                            'proxy_torque_delta_Nm': float(gain@(action-command)),
                            'q_deg': float(np.rad2deg(trace[end-1, 0])),
                            'q_delta_deg': float(np.rad2deg(trace[end-1, 0]-reference[end-1, 0])),
                            'speed_deg_s': float(np.rad2deg(trace[end-1, 1])),
                            'speed_delta_deg_s': float(np.rad2deg(trace[end-1, 1]-reference[end-1, 1])),
                            'activation_squared': float(a2), 'activation_reduction': float(1-a2/old),
                            'actual_torque_delta_Nm': float(trace[end-1, -1]-reference[end-1, -1])})
                print(json.dumps({'goal': goal, 'rows': rows[-9:]}), flush=True)
            finally:
                kernel.close()
    finally:
        env.close()
    result = {'parent': str(args.parent), 'parent_sha256': meta['checkpoint_sha256'],
        'state_restoration': 'all 20 reference steps match the independently continued original wrapper',
        'protocol': 'settle 2.5s; freeze nominal command for all counterfactual branches, 200ms; '
                    'full muscle/delay/command/history and MuJoCo integration state restored', 'rows': rows}
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.out/'traces.npz', **traces)


if __name__ == '__main__':
    main()
