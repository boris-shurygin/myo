"""Compare fixed motor-probe data with fresh stochastic arm-policy diagnostics."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from myo.arm_learning import load_arm, predictor_identity
from myo.arm_precision import TRAIN_CASES
from myo.train_arm_precision import load_data, starts, forecast


ROOT = Path('runs/arm_preacher')
OUT = ROOT/'movement_coverage'
SNAPSHOTS = {'initial': 'none/initial.zip', 'none': 'none/best.zip', 'frozen': 'frozen/best.zip'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def collect_policy(name, path):
    model, env, identity = load_arm(path)
    torch.manual_seed(0)
    arrays = {key: [] for key in ('observations', 'actions', 'next_observations', 'valid')}
    episodes = []
    for case in TRAIN_CASES:
        obs, _ = env.reset(seed=0, options={'case': case})
        start = len(arrays['valid'])
        for _ in range(200):
            action = model.predict(obs, deterministic=False)[0]
            following, _, failed, truncated, _ = env.step(action)
            for key, value in zip(arrays, (obs, action, following, not failed), strict=True):
                arrays[key].append(value)
            obs = following
            if failed or truncated:
                break
        episodes.append({'split': name, 'case': case, 'start': start,
                         'stop': len(arrays['valid']), 'failure': bool(failed)})
    metadata = {'contract': env.contract(), 'episodes': episodes, 'checkpoint': identity,
                'sampling': 'fresh stochastic policy, torch seed0, full27 TRAIN_CASES; not historical PPO rollout'}
    env.close()
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(OUT/f'{name}.npz', **arrays)
    write(OUT/f'{name}.json', metadata)
    print(json.dumps({'collected': name, 'transitions': len(arrays['valid'])}), flush=True)
    return arrays, metadata


def quantiles(value):
    a = np.asarray(value)
    return dict(zip(('min', 'p01', 'p50', 'p95', 'p99', 'max'),
                    map(float, np.quantile(a, (0, .01, .5, .95, .99, 1)))))


def statistics(data, meta, split):
    ids = starts(data, meta, split)
    names = meta['contract']['observation_names']
    raw = (data['observations'][ids]-np.array(meta['contract']['offset']))/np.array(meta['contract']['scale'])
    q = np.rad2deg(raw[:, 0])+75
    speed = np.rad2deg(raw[:, 1]*4)
    indices = lambda field: [i for i, n in enumerate(names[:39]) if n.endswith('/'+field)]
    du, da, acceleration, low_speed_runs = [], [], [], []
    for episode in meta['episodes']:
        if episode['split'] != split:
            continue
        a, b = episode['start'], episode['stop']
        du.extend(np.abs(np.diff(data['actions'][a:b], axis=0)).ravel())
        activation = (data['observations'][a:b][:, indices('activation')]+1)/2
        da.extend(np.abs(np.diff(activation, axis=0)).ravel())
        # Acceleration is diagnostic finite difference of real interval velocities.
        v = data['observations'][a:b, 1]*4*180/np.pi
        vn = data['next_observations'][a:b, 1]*4*180/np.pi
        acceleration.extend(np.abs((vn-v)/.02))
        qe = data['observations'][a:b, 0]*90+75
        slow_middle = (qe >= 30) & (qe < 120) & (abs(v) <= 5)
        edges = np.diff(np.r_[False, slow_middle, False].astype(int))
        low_speed_runs.extend((np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)).tolist())
    fractions = {f'{lo}-{hi}': float(np.mean((q >= lo) & (q < hi)))
                 for lo, hi in zip((0, 15, 30, 60, 90, 120, 135, 150), (15, 30, 60, 90, 120, 135, 150, 180))}
    episodes = [e for e in meta['episodes'] if e['split'] == split]
    return {'transitions': len(ids), 'episodes': len(episodes), 'failures': sum(e['failure'] for e in episodes),
        'q_deg': quantiles(q), 'abs_velocity_deg_s': quantiles(abs(speed)),
        'slow_fraction_abs_velocity_le5_deg_s': float(np.mean(abs(speed) <= 5)),
        'abs_acceleration_deg_s2': quantiles(acceleration), 'angle_bin_fraction': fractions,
        'excitation': quantiles(data['actions'][ids]), 'activation': quantiles(raw[:, indices('activation')]),
        'capacity': quantiles(raw[:, indices('remaining_capacity')]),
        'fiber_ratio': quantiles(raw[:, indices('fiber_length_normalized')]),
        'tendon_force_over_F0': quantiles(raw[:, indices('force_normalized')]),
        'absolute_command_change_per20ms': quantiles(du),
        'absolute_activation_change_per20ms': quantiles(da),
        'middle_low_speed_runs': {'frames': sum(low_speed_runs),
            'maximum_contiguous_duration_s': max(low_speed_runs, default=0)*.02,
            'runs_at_least_500ms': sum(k >= 25 for k in low_speed_runs)}}


def coverage(reference, query):
    std = reference.std(0)
    keep = std > 1e-6
    mean = reference.mean(0)[keep]; std = std[keep]
    a = torch.as_tensor((reference[:, keep]-mean)/std)
    b = torch.as_tensor((query[:, keep]-mean)/std)
    distances = []
    for block in b.split(128):
        nearest = torch.cdist(block, a).min(1).values/np.sqrt(int(keep.sum()))
        distances.extend(nearest.tolist())
    outside = (query < reference.min(0)-1e-6) | (query > reference.max(0)+1e-6)
    return {'distance_definition': 'nearest train state; RMS distance after train-STD input scaling, nonconstant channels only',
        'query_samples': len(query), 'varying_channels': int(keep.sum()),
        'nearest_distance': quantiles(distances), 'any_input_outside_train_range_fraction': float(outside.any(1).mean()),
        'per_input_outside_train_range_fraction': outside.mean(0).tolist()}


@torch.no_grad()
def conditional_errors(body, data, meta, split):
    ids = starts(data, meta, split)
    physical = meta['contract']['physical_indices']
    obs = torch.as_tensor(data['observations'][ids])
    actual = torch.as_tensor(data['next_observations'][ids][:, physical])
    predicted = body.next_physical(obs[:, meta['contract']['predictor_indices']])
    errors = predicted-actual
    persistence = obs[:, physical]-actual
    q = obs[:, 0]*90+75
    slow = obs[:, 1].abs()*4*180/np.pi <= 5
    regions = {'lower_lt30': q < 30, 'middle30_120': (q >= 30) & (q < 120),
               'upper_ge120': q >= 120, 'slow_le5_deg_s': slow, 'moving_gt5_deg_s': ~slow,
               'middle_and_slow': (q >= 30) & (q < 120) & slow}
    result = {}
    for name, mask in regions.items():
        if not mask.any():
            result[name] = {'samples': 0}
            continue
        rmse = errors[mask].square().mean(0).sqrt()
        baseline = persistence[mask].square().mean(0).sqrt()
        result[name] = {'samples': int(mask.sum()), 'angle_rmse_deg': float(rmse[0]*90),
            'velocity_rmse_deg_s': float(rmse[1]*4*180/np.pi),
            'persistence_angle_rmse_deg': float(baseline[0]*90),
            'persistence_velocity_rmse_deg_s': float(baseline[1]*4*180/np.pi)}
    return result


def plots(datasets, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'motor_train': 'Predictor motor train', 'initial': 'Initial PPO',
              'none': 'Trained PPO without h1', 'frozen': 'Trained PPO with frozen h1'}
    velocity_limit = np.ceil(max(s['abs_velocity_deg_s']['max'] for s in stats.values())/25)*25
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, (data, meta, split) in datasets.items():
        ids = starts(data, meta, split)
        axes[0].hist(data['observations'][ids, 0]*90+75, bins=np.arange(0, 161, 5),
                     histtype='step', density=True, label=labels[name])
        velocity = np.abs(data['observations'][ids, 1]*4*180/np.pi)
        axes[1].hist(velocity, bins=np.linspace(0, velocity_limit, 61), histtype='step', density=True)
        change = []
        for e in meta['episodes']:
            if e['split'] == split:
                change.extend(np.abs(np.diff(data['actions'][e['start']:e['stop']], axis=0)).ravel())
        axes[2].hist(change, bins=np.linspace(0, 1, 101), histtype='step', density=True)
    axes[0].set(xlabel='Angle, deg', ylabel='Density')
    axes[1].set(xlabel='Absolute velocity, deg/s', xlim=(0, velocity_limit), yscale='log')
    axes[2].set(xlabel='Absolute excitation change / 20 ms', yscale='log')
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT/'distributions.png', dpi=150); fig.savefig(OUT/'distributions.svg'); plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    motor, meta, _ = datasets['motor_train']
    for row, family in enumerate(('independent', 'reciprocal', 'common', 'multisine')):
        e = next(e for e in meta['episodes'] if e['split'] == 'train' and e['family'] == family and e['limit'] == .4)
        a, b = e['start'], e['stop']; t = np.arange(b-a)*.02
        axes[row, 0].plot(t, motor['observations'][a:b, 0]*90+75, color='black')
        axes[row, 0].set(title=f"{family}, start {e['position_deg']} deg, max command .4", ylabel='Angle, deg', xlabel='Time, s')
        for name in ('initial', 'none', 'frozen'):
            data, pm, _ = datasets[name]
            case = ((20, 75), (75, 15), (125, 75), (75, 135))[row]
            ep = next(e for e in pm['episodes'] if tuple(e['case']) == case)
            a, b = ep['start'], ep['stop']
            axes[row, 1].plot(np.arange(b-a)*.02, data['observations'][a:b, 0]*90+75, label=labels[name])
        axes[row, 1].axhline(case[1], ls='--', color='black', lw=.8)
        axes[row, 1].set(title=f'PPO train task {case[0]} -> {case[1]} deg', ylabel='Angle, deg', xlabel='Time, s')
    axes[0, 1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT/'trajectories.png', dpi=150); fig.savefig(OUT/'trajectories.svg'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', action='store_true', help='Analyze existing diagnostic trajectories without new simulations')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=args.reuse)
    body_data, body_meta = load_data(ROOT)
    datasets = {'motor_train': (body_data, body_meta, 'train')}
    for name, relative in SNAPSHOTS.items():
        if args.reuse:
            data = dict(np.load(OUT/f'{name}.npz'))
            meta = json.loads((OUT/f'{name}.json').read_text())
            _, checked, identity = load_arm(ROOT/relative)
            checked.close()
            if identity != meta['checkpoint']:
                raise ValueError('Diagnostic checkpoint changed')
        else:
            data, meta = collect_policy(name, ROOT/relative)
        datasets[name] = (data, meta, name)
    body, identity = predictor_identity(ROOT/'body/best.pt')
    result = {'scope': 'fixed motor train versus new stochastic diagnostics at saved checkpoints; no PPO training or reconstruction of historical rollouts',
        'predictor_identity': identity, 'statistics': {}, 'forecast': {}, 'coverage': {}, 'conditional_errors20': {}}
    reference_ids = starts(body_data, body_meta, 'train')
    pindices = body_meta['contract']['predictor_indices']
    reference = body_data['observations'][reference_ids][:, pindices]
    for name, (data, meta, split) in datasets.items():
        result['statistics'][name] = statistics(data, meta, split)
        result['forecast'][name] = {str(20*h): forecast(body, data, meta, split, h) for h in (1, 3)}
        result['conditional_errors20'][name] = conditional_errors(body, data, meta, split)
        if name != 'motor_train':
            query = data['observations'][starts(data, meta, split)[::4]][:, pindices]
            result['coverage'][name] = coverage(reference, query)
    result['forecast']['motor_test'] = {str(20*h): forecast(body, body_data, body_meta, 'test', h) for h in (1, 3)}
    result['conditional_errors20']['motor_test'] = conditional_errors(body, body_data, body_meta, 'test')
    test_query = body_data['observations'][starts(body_data, body_meta, 'test')[::4]][:, pindices]
    result['coverage']['motor_test'] = coverage(reference, test_query)
    write(OUT/'comparison.json', result)
    plots(datasets, result['statistics'])
    print(json.dumps({'status': 'complete', 'path': str(OUT/'comparison.json')}), flush=True)


if __name__ == '__main__':
    main()
