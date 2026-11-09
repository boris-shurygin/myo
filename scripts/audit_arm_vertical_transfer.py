"""Compare execution noise and native reset preparation across the two stands."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_precision import ArmPrecisionEnv
from myo.pretrained_vertical import load_pretrained
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import load_policy
from myo.vertical_precision import VerticalPrecisionEnv
from myo.vertical_muscles import VerticalConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/arm_bottlenecks/vertical_transfer.json'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    torch.set_num_threads(1)
    result = {'schema': 'myo.arm_vertical_transfer.v1', 'reset_arm': [], 'reset_vertical': [],
              'vertical_execution': [], 'models': {}}
    arm = ArmPrecisionEnv()
    for initial in (20, 75, 125):
        arm.reset(seed=0, options={'case': (initial, 75)})
        base = arm.unwrapped
        t = base.telemetry()
        row = {'initial_deg': initial, 'activation': t['activation'],
               'muscle_torque_Nm': t['muscle_torque_Nm'],
               'passive_torque_Nm': t['passive_limit_torque_Nm'],
               'gravity_bias_Nm': float(base.data.qfrc_bias[base.dof_index])}
        row['net_nonconstraint_torque_Nm'] = row['muscle_torque_Nm']+row['passive_torque_Nm']-row['gravity_bias_Nm']
        for _ in range(2):
            arm.step(np.full(6, .1))
        row['angle_after_transport_only_40ms_deg'] = float(np.rad2deg(base.data.qpos[base.qpos_index]))
        result['reset_arm'].append(row)
    arm.close()
    path = Path('runs/vertical_precision/combined/best.zip')
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    env = VerticalPrecisionEnv(VerticalConfig(**meta['contract']['physical']['config']))
    baseline, meta = load_policy(path, env)
    frozen, frozen_env, frozen_meta = load_pretrained(Path('runs/pretrained_vertical/pretrained_frozen/best.zip'))
    for label, model, stand, identity in (('precision', baseline, env, meta),
                                         ('pretrained_frozen', frozen, frozen_env, frozen_meta)):
        before = parameter_hash(model)
        result['models'][label] = identity['checkpoint_sha256']
        for case in ((-.03, .06), (.025, -.055), (.07, .07)):
            for stochastic, seed in ((False, 0), *((True, s) for s in range(4))):
                torch.manual_seed(seed)
                obs, _ = stand.reset(seed=0, options={'initial_position_m': case[0], 'target_m': case[1]})
                if label == 'precision' and not stochastic:
                    physical = stand.physical
                    forces = physical.forces()
                    result['reset_vertical'].append({'initial_m': case[0],
                        'activation': physical.activation.tolist(), 'forces_N': forces.tolist(),
                        'net_force_N': float(forces[0]-forces[1]-9.81)})
                trace = []
                while True:
                    action = model.predict(obs, deterministic=not stochastic)[0]
                    obs, _, failed, truncated, info = stand.step(action)
                    trace.append([info['position_m'], info['velocity_m_s'], *action,
                                  *stand.physical.activation])
                    if failed or truncated:
                        break
                a = np.asarray(trace); tail = a[-50:]
                result['vertical_execution'].append({'model': label, 'case_m': list(case),
                    'stochastic': stochastic, 'seed': seed, 'failure': bool(failed),
                    'tail_rmse_mm': float(1000*np.sqrt(np.mean((tail[:, 0]-case[1])**2))),
                    'tail_mean_command': tail[:, 2:4].mean(0).tolist(),
                    'tail_mean_activation': tail[:, 4:6].mean(0).tolist()})
        assert parameter_hash(model) == before
        stand.close()
        print(json.dumps({'completed': label}), flush=True)
    result.update(status='complete', unchanged_weights=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
