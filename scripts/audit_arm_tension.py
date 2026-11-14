"""Audit old effort continuations and current demo without training or physics edits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

import myo.arm_tension_diagnostics as metric_module
from myo.arm_exploration import load_exploration
from myo.arm_tension_diagnostics import antagonist_metrics
from myo.arm_timing import load_timing
from myo.arm_workload import scenarios, spec_from_dict


FIELDS = ('t', 'q', 'qvel', 'requested_target_q', 'activation', 'excitation',
    'delayed_excitation', 'tendon_force_N', 'moment_arm_m', 'capacity', 'tendon_strain',
    'fiber_length_m', 'fiber_velocity_m_s', 'active_fiber_force_N', 'passive_fiber_force_N',
    'damping_force_N', 'passive_limit_torque_Nm')


def aggregate(rows):
    """Every condition has equal weight, including the deliberately longer probe."""
    result = {'total': len(rows), 'passed': sum(row['passed'] for row in rows),
              'failures': sum(row['failure'] for row in rows)}
    for section in ('full', 'tail'):
        keys = rows[0][section]
        result[section] = {key: np.mean([row[section][key] for row in rows], axis=0).tolist()
                           for key in keys}
    result['position_return_mean'] = float(np.mean([r['position_return'] for r in rows]))
    result['counterfactual_effort_return_mean'] = float(np.mean([r['counterfactual_effort_return'] for r in rows]))
    return result


def job(label, path, groups, out, heuristic=False):
    torch.set_num_threads(1)
    path, out = Path(path), Path(out)
    meta = json.loads(Path(str(path)+'.metadata.json').read_text())
    loader = load_timing if meta['contract']['schema'] == 'myo.arm_timing.v1' else load_exploration
    model, env, meta = loader(path)
    native = env.unwrapped
    parameters = [muscle.params for muscle in native.muscles]
    f0 = np.array([p.max_isometric_force for p in parameters])
    tendon_k = np.array([p.tendon_stiffness for p in parameters])
    optimal_l = np.array([p.optimal_fiber_length for p in parameters])
    passive_k = np.array([p.passive_stiffness for p in parameters])
    result = {'checkpoint': str(path), 'checkpoint_sha256': meta['checkpoint_sha256'],
              'controller': 'native heuristic' if heuristic else 'deterministic Beta means',
              'contract': meta['contract'], 'groups': {}}
    traces = {}
    try:
        index = 0
        for group, conditions in groups.items():
            rows = []
            for condition in conditions:
                observation, initial = env.reset(seed=0, options=condition)
                records, position_rewards, rewards, contact, peaks, bias, constraints = [], [], [], [], [], [], []
                while True:
                    action = native.baseline_action() if heuristic else model.predict(observation, deterministic=True)[0]
                    observation, reward, failed, done, info = env.step(action)
                    records.append({field: info[field] for field in FIELDS})
                    rewards.append(reward)
                    position_rewards.append(.1*info['reward_components']['position_log'])
                    contact.append(info['tissue_contact']['normal_force_N'])
                    peaks.append(info['substep_peaks']['summed_tendon_force_N'])
                    bias.append(float(native.data.qfrc_bias[native.dof_index]))
                    constraints.append(float(native.data.qfrc_constraint[native.dof_index]))
                    if failed or done:
                        break
                data = {field: np.asarray([r[field] for r in records]) for field in FIELDS}
                metrics = antagonist_metrics(data['activation'], data['tendon_force_N'],
                                             data['moment_arm_m'], data['qvel'])
                error = np.rad2deg(data['q']-data['requested_target_q'])
                speed = np.rad2deg(data['qvel'])
                stretch = np.maximum(0., data['fiber_length_m']/optimal_l-1.)
                metrics.update(
                    active_fraction_mean=(data['activation']*data['capacity']).mean(axis=1),
                    tendon_elastic_energy_J=(np.square(data['tendon_force_N'])/(2*tendon_k)).sum(axis=1),
                    passive_fiber_elastic_energy_J=(f0*passive_k*optimal_l*stretch**3/3).sum(axis=1),
                    active_fiber_force_sum_N=data['active_fiber_force_N'].sum(axis=1),
                    passive_fiber_force_sum_N=data['passive_fiber_force_N'].sum(axis=1),
                    activation_over_80pct_muscle_time_fraction=(data['activation'] > .8).mean(axis=1),
                    activation_max_across_muscles=data['activation'].max(axis=1),
                    tendon_strain_max_across_muscles=data['tendon_strain'].max(axis=1),
                    force_F0_max_across_muscles=(data['tendon_force_N']/f0).max(axis=1),
                    capacity_mean=data['capacity'].mean(axis=1),
                    contact_force_proxy_N=np.asarray(contact),
                    control_interval_peak_summed_tendon_force_N=np.asarray(peaks),
                    generalized_bias_torque_Nm=np.asarray(bias),
                    generalized_constraint_torque_Nm=np.asarray(constraints),
                    passive_limit_torque_Nm=data['passive_limit_torque_Nm'],
                )
                tail = data['t'] > condition['seconds']-1.-1e-8
                tail_reached = bool(tail.any())
                if not tail_reached:
                    tail = np.ones(len(records), dtype=bool)
                full = not failed and len(records) == round(condition['seconds']/native.control_dt)
                settled = (np.abs(error) <= 3.) & (np.abs(speed) <= 5.)

                def summarize(mask):
                    return {**{key: float(value[mask].mean()) for key, value in metrics.items()},
                        'angle_rmse_deg': float(np.sqrt(np.mean(error[mask]**2))),
                        'velocity_rms_deg_s': float(np.sqrt(np.mean(speed[mask]**2))),
                        'position_fraction': float((np.abs(error[mask]) <= 3.).mean()),
                        'speed_fraction': float((np.abs(speed[mask]) <= 5.).mean()),
                        'settled_fraction': float(settled[mask].mean()),
                        'activation_by_muscle': data['activation'][mask].mean(axis=0).tolist(),
                        'force_by_muscle_N': data['tendon_force_N'][mask].mean(axis=0).tolist(),
                        'signed_torque_by_muscle_Nm': (data['tendon_force_N']*data['moment_arm_m'])[mask].mean(axis=0).tolist(),
                        'capacity_by_muscle': data['capacity'][mask].mean(axis=0).tolist()}

                row = {'condition': condition, 'failure': bool(failed),
                    'tail_reached': tail_reached,
                    'passed': bool(full and settled[tail].mean() >= .9), 'steps': len(records),
                    'position_return': float(sum(position_rewards)), 'actual_return': float(sum(rewards)),
                    'counterfactual_effort_return': float(-.01*metrics['activation_squared_mean'].sum()),
                    'capacity_net_loss_by_muscle': (np.array(initial['capacity'])-data['capacity'][-1]).tolist(),
                    'full': summarize(np.ones(len(records), dtype=bool)), 'tail': summarize(tail)}
                rows.append(row)
                traces[f'case_{index:03d}'] = np.column_stack([
                    data[field] if data[field].ndim == 2 else data[field][:, None] for field in FIELDS])
                index += 1
            result['groups'][group] = {'aggregate': aggregate(rows), 'cases': rows}
        result['trace_fields'] = [{'field': field, 'width': int(data[field].shape[1]) if data[field].ndim == 2 else 1}
                                  for field in FIELDS]
        np.savez_compressed(out/f'{label}_traces.npz', **traces)
        (out/f'{label}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return label, {name: result['aggregate'] for name, result in result['groups'].items()}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'metrics_source.py').write_bytes(Path(metric_module.__file__).read_bytes())
    root = Path('runs/arm_effort_curriculum')
    parent = Path('runs/arm_exploration/stall_audited_continuation/best.zip')
    meta = json.loads(Path(str(parent)+'.metadata.json').read_text())
    spec = spec_from_dict(meta['contract']['spec'])
    groups = {'train': scenarios(spec.cases, spec.masses, spec.tilts, spec.capacities, spec.seconds),
              'validation': meta['training']['validation']}
    jobs = [('old_parent', parent, groups, False)]
    jobs += [(name, root/name/'last.zip', groups, False) for name in (
        'control', 'activation_01', 'control_lr3e6', 'activation_01_lr3e6')]
    jobs.append(('old_heuristic', parent, groups, True))
    current = Path('runs/arm_timing/curriculum_10ms/angle_range/best.zip')
    current_cases = [{'case': case, 'payload_mass': .5, 'tilt_deg': 30., 'capacity': 1., 'seconds': seconds}
        for case, seconds in (((15.,15.),4.), ((15.,15.),30.), ((75.,75.),4.),
                              ((125.,125.),4.), ((20.,75.),4.), ((75.,20.),4.))]
    jobs += [('current_10ms', current, {'diagnostic': current_cases}, False),
             ('current_heuristic', current, {'diagnostic': current_cases}, True)]
    manifest = {'schema': 'myo.arm_tension_audit.v1', 'scope':
        'read-only replay; old 18 train/27 validation, five learned snapshots and native heuristic; '
        'current 10/10-ms policy has separate conditions, not a matched old/new comparison; no test or training',
        'sampling': 'common end-of-control-step pose and tendon forces; overlap before time averaging; tail last1s; equal case weighting',
        'counterfactual_effort_return': '-0.1*0.1*sum_t(mean_i(activation_i^2)); diagnostic only, not applied to physics or policy',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metrics_sha256': hashlib.sha256(Path(metric_module.__file__).read_bytes()).hexdigest(),
        'workers': args.workers, 'jobs': [{'label': label, 'path': str(path), 'groups': groups, 'heuristic': heuristic}
                                         for label, path, groups, heuristic in jobs]}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.monotonic()
    summary = {'status': 'running', 'manifest': manifest, 'results': {}}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(job, label, path, groups, args.out, heuristic) for label, path, groups, heuristic in jobs]
        for future in futures:
            label, result = future.result()
            summary['results'][label] = result
            summary['elapsed_s'] = time.monotonic()-start
            (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'policy': label, 'groups': {name: {
                'passed': row['passed'], 'total': row['total'],
                'activation_mean': row['tail']['activation_mean'],
                'coactivation': row['tail']['activation_coactivation'],
                'opposing_torque_Nm': row['tail']['opposing_muscle_torque_Nm'],
                'angle_rmse_deg': row['tail']['angle_rmse_deg']} for name, row in result.items()},
                'elapsed_s': summary['elapsed_s']}), flush=True)
    summary['status'] = 'complete'
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
