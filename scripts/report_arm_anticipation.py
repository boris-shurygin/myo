"""Evaluate direct forecasts, command contrasts and current-policy distribution shift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_arm_predictor_lr import load_checkpoint
from myo.arm_anticipation import load_predictor, predictions, physical_errors
from myo.arm_learning import predictor_identity
from myo.arm_precision import ArmPrecisionEnv, TEST_CASES
from myo.train_arm_anticipation import ROOT, SOURCE
from myo.train_arm_precision import load_data
from myo.train_vertical_ppo import write_json


def replay_start(env, episode):
    position = episode.get('position_deg', episode.get('case', [None])[0])
    target = episode.get('target_deg', episode.get('case', [position, position])[1])
    return env.reset(seed=episode.get('seed', 0), options={
        'case': (position, target), 'capacity': episode.get('capacity', 1.)})[0]


@torch.no_grad()
def contrasts(root, body, old, data, meta):
    selected = [e for e in meta['episodes'] if e['split'] == 'test' and e['source'] == 'new_probe'
        and e.get('capacity') == 1. and (
            (e.get('family') == 'hold' and e.get('coactivation') == 0.)
            or (e.get('family') == 'ramp' and e.get('speed_deg_s') == 60.))]
    assert len(selected) == 8
    env = ArmPrecisionEnv()
    physical = env.physical_indices
    input_indices = physical+env.pending_indices+[2]
    report = {'scope': 'Post-selection native counterfactuals; none of these labels train the model',
        'perturbation': 'recorded u plus/minus [0.15,0.15,0.15,-0.15,-0.15,-0.15], clipped0..1',
        'cases': []}
    actual_pairs, new_pairs, old_pairs = [], [], []
    for episode in selected:
        for step in (5, 50):
            index = episode['start']+step
            reference = data['observations'][index]
            command = data['actions'][index]
            perturbation = np.array([.15]*3+[-.15]*3, dtype=np.float32)
            candidates = np.stack((np.clip(command-perturbation, 0, 1), np.clip(command+perturbation, 0, 1)))
            actual = []
            for candidate in candidates:
                obs = replay_start(env, episode)
                for previous in data['actions'][episode['start']:index]:
                    obs, _, failed, _, _ = env.step(previous)
                    assert not failed
                np.testing.assert_array_equal(obs[input_indices], reference[input_indices])
                for k in range(3):
                    # Later commands are causally irrelevant for this endpoint.
                    obs, _, failed, _, _ = env.step(candidate if k == 0 else np.zeros(6, np.float32))
                    assert not failed
                actual.append(obs[physical])
            x = np.repeat(reference[input_indices][None], 2, axis=0)
            new = predictions(body, x, candidates)
            baseline = predictions(old, x, candidates, iterative=True)
            actual_pairs.append(actual[1]-actual[0]); new_pairs.append(new[1]-new[0]); old_pairs.append(baseline[1]-baseline[0])
            report['cases'].append({'family': episode['family'], 'position_deg': episode['position_deg'],
                'step': step, 'actions': candidates.tolist(), 'initial_physical': reference[physical].tolist(),
                'actual_delta_between_commands': actual_pairs[-1].tolist(),
                'direct_delta_between_commands': new_pairs[-1].tolist(),
                'iterative_delta_between_commands': old_pairs[-1].tolist()})
    env.close()
    actual, new, baseline = map(np.asarray, (actual_pairs, new_pairs, old_pairs))
    for label, predicted in (('direct', new), ('iterative', baseline)):
        # Same utility: here delta60 is the actual effect of changing the action.
        metric = physical_errors(predicted, actual, np.zeros_like(actual), actual,
                                 body_metadata(root)['identity']['output_names'])
        velocity = actual[:, 1]*4*180/np.pi
        appreciable = abs(velocity) > .1
        metric['velocity_effect_sign_accuracy_gt_01_deg_s'] = float(np.mean(
            np.sign(predicted[appreciable, 1]) == np.sign(actual[appreciable, 1])))
        report[label] = metric
    write_json(root/'command_contrasts.json', report)
    return report


def body_metadata(root):
    return json.loads((root/'best.pt.metadata.json').read_text())


@torch.no_grad()
def policy_probes(root, body, old, contract):
    from myo.arm_anticipation import examples
    report = {}
    for label, checkpoint in (
        ('none_successful', Path('runs/arm_continuation/best.zip')),
        ('frozen_lower_lr', Path('runs/arm_predictor_lr/frozen_3e5/best.zip')),
    ):
        model, env, model_meta = load_checkpoint(checkpoint)
        assert env.contract() == contract
        data = {k: [] for k in ('observations', 'actions', 'next_observations', 'valid')}
        episodes = []
        for case in TEST_CASES:
            start = len(data['valid'])
            obs, _ = env.reset(seed=0, options={'case': case})
            while True:
                action = model.predict(obs, deterministic=True)[0]
                following, _, failed, done, _ = env.step(action)
                for key, value in zip(data, (obs, action, following, not failed), strict=True):
                    data[key].append(value)
                obs = following
                if failed or done:
                    break
            episodes.append({'case': case, 'split': 'test', 'start': start, 'stop': len(data['valid'])})
        env.close()
        data = {k: np.asarray(v) for k, v in data.items()}
        meta = {'contract': contract, 'episodes': episodes}
        ids, x, u, actual = examples(data, meta, 'test')
        one_step = data['next_observations'][ids][:, contract['physical_indices']]
        result = {'checkpoint': model_meta, 'cases': TEST_CASES, 'samples': len(ids)}
        for name, predicted in (('direct', predictions(body, x, u)),
                                ('iterative', predictions(old, x, u, iterative=True)),
                                ('persistence', x[:, :20])):
            result[name] = physical_errors(predicted, actual, x[:, :20], one_step,
                                           body_metadata(root)['identity']['output_names'])
        report[label] = result
        write_json(root/'policy_probes.json', report)
    return report


def report(root, plot):
    summary = json.loads((root/'training.json').read_text())
    test = json.loads((root/'test.json').read_text())
    lines = ['# Прямой прогноз60 ms', '',
        f"Updates{summary['updates']}, best{summary['best_update']}, stop `{summary['stop_reason']}`.", '',
        '| Источник | Direct q, ° | Iterative q, ° | Persistence q, ° | Direct v, °/s | Iterative v, °/s | Persistence v, °/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group, values in test['groups'].items():
        a, b, c = [values[k] for k in ('direct', 'iterative', 'persistence')]
        lines.append(f"| {group} | {a['angle_rmse_deg']:.4f} | {b['angle_rmse_deg']:.4f} | {c['angle_rmse_deg']:.4f} | "
            f"{a['velocity_rmse_deg_s']:.3f} | {b['velocity_rmse_deg_s']:.3f} | {c['velocity_rmse_deg_s']:.3f} |")
    lines += ['', '| Канал | Единицы | Direct RMSE | Iterative RMSE | RMSE/изменение60 | RMSE/изменение20 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for a, b in zip(test['groups']['all']['direct']['channels'], test['groups']['all']['iterative']['channels'], strict=True):
        lines.append(f"| {a['name']} | {a['unit']} | {a['rmse']:.6g} | {b['rmse']:.6g} | "
                     f"{a['error_over_delta60']:.4f} | {a['error_over_delta20']:.4f} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if plot:
        import matplotlib.pyplot as plt
        h = summary['history']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
        for split in ('train', 'validation'):
            axes[0].plot([r['update'] for r in h], [r[split]['balanced_mse'] for r in h], label=split)
        axes[0].set(xlabel='Optimizer updates', ylabel='Normalized delta60 MSE', yscale='log', title='Direct predictor learning')
        groups = list(test['groups'])
        x = np.arange(len(groups))
        for offset, label in ((-.25, 'direct'), (0., 'iterative'), (.25, 'persistence')):
            axes[1].bar(x+offset, [test['groups'][g][label]['angle_rmse_deg'] for g in groups], .25, label=label)
            axes[2].bar(x+offset, [test['groups'][g][label]['velocity_rmse_deg_s'] for g in groups], .25, label=label)
        axes[1].set(ylabel='Angle RMSE, degrees', title='Held-out episodes, horizon60 ms')
        axes[2].set(ylabel='Velocity RMSE, deg/s', title='Held-out episodes, horizon60 ms')
        for ax in axes[1:]:
            ax.set_xticks(x, groups, rotation=30)
        for ax in axes:
            ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
        for suffix in ('.png', '.svg'):
            fig.savefig(root/('comparison'+suffix), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.root/'training.json').read_text())
    if summary['status'] != 'complete' or not (args.root/'frozen_selection.json').exists():
        raise ValueError('Finish training and freeze selection first')
    if args.diagnostics:
        body, _ = load_predictor(args.root/'best.pt')
        old, _ = predictor_identity(args.source/'body/best.pt')
        data, meta = load_data(args.source)
        contrasts(args.root, body, old, data, meta)
        policy_probes(args.root, body, old, meta['contract'])
    report(args.root, args.plot)
    print(str(args.root/'report.md'))


if __name__ == '__main__':
    main()
