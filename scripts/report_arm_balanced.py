"""Report actual coverage, held-out prediction, and PPO for the broader arm dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_balanced import ROOT, OLD, sampling_weights
from myo.arm_learning import predictor_identity
from myo.train_arm_precision import load_data, starts, forecast
from compare_arm_movement_coverage import statistics, conditional_errors


def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1)
    data, meta = load_data(ROOT); old_data, old_meta = load_data(OLD)
    audit = {'statistics': {s: statistics(data, meta, s) for s in ('train', 'validation', 'test')},
             'old_train': statistics(old_data, old_meta, 'train')}
    ids = starts(data, meta, 'train'); obs = data['observations'][ids]
    probability, bins = sampling_weights(obs, meta)
    q = obs[:, 0]*90+75; velocity = obs[:, 1]*4*180/np.pi
    audit['sampling'] = {**bins, 'effective_examples': float(1/(probability**2).sum()),
        'expected_middle30_120_fraction': float(probability[(q >= 30)&(q < 120)].sum()),
        'expected_lower_lt30_fraction': float(probability[q < 30].sum()),
        'expected_moving_gt5_fraction': float(probability[abs(velocity) > 5].sum())}
    # Ensure every generated episode stays within its declared window.
    checked = meta['contract']['physical_indices']+meta['contract']['pending_indices']
    for e in meta['episodes']:
        a, b = e['start'], e['stop']
        np.testing.assert_array_equal(data['next_observations'][a:b-1][:, checked],
                                      data['observations'][a+1:b][:, checked])
    assert all(np.isfinite(data[k]).all() for k in ('observations', 'next_observations', 'actions'))
    audit['physical_and_fifo_continuity'] = 'passed; commands may update causally between frames'
    write(ROOT/'data_audit.json', audit)
    print(json.dumps({'coverage': {s: {'n': a['transitions'], 'middle': sum(a['angle_bin_fraction'][k] for k in ('30-60','60-90','90-120')),
        'lower': a['angle_bin_fraction']['0-15']+a['angle_bin_fraction']['15-30'], 'middle_slow': a['middle_low_speed_runs']}
        for s,a in audit['statistics'].items()}, 'sampling': audit['sampling']}), flush=True)
    if args.data_only: return
    training = read(ROOT/'body/training.json')
    assert training['status'] == 'complete'
    old, _ = predictor_identity(OLD/'body/best.pt'); new, _ = predictor_identity(ROOT/'body/best.pt')
    assert torch.equal(old.delta_mean, new.delta_mean) and torch.equal(old.delta_std, new.delta_std)
    comparisons = {'all_test': training['test']}
    groups = {
        'old_motor': lambda e: e['source'] == 'old_motor',
        'hold': lambda e: e.get('family') == 'hold', 'ramp': lambda e: e.get('family') == 'ramp',
        'jitter': lambda e: e.get('family') == 'jitter', 'policy': lambda e: e.get('family') == 'policy',
        'open_loop': lambda e: e['source'] == 'new_probe' and e.get('limit') is not None}
    for group, match in groups.items():
        subset = {**meta, 'episodes': [e for e in meta['episodes'] if match(e)]}
        comparisons[group] = {name: {str(20*h): forecast(model, data, subset, 'test', h) for h in (1, 3)}
                              for name, model in [('old', old), ('new', new)]}
    comparisons['conditional_test20'] = {name: conditional_errors(model, data, meta, 'test')
                                         for name, model in [('old', old), ('new', new)]}
    write(ROOT/'prediction_comparison.json', comparisons)
    lines = ['## Результаты', '', '### Покрытие данных', '',
        '| Набор | Переходов | Ниже30°, % | 30–120°, % | Кадров middle/slow | Максимальная последовательность middle/slow, s |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, stats in [('Старый train', audit['old_train']), *audit['statistics'].items()]:
        b = stats['angle_bin_fraction']; slow = stats['middle_low_speed_runs']
        lines.append(f"| {label} | {stats['transitions']} | {100*(b['0-15']+b['15-30']):.2f} | {100*sum(b[k] for k in ('30-60','60-90','90-120')):.2f} | {slow['frames']} | {slow['maximum_contiguous_duration_s']:.2f} |")
    lines += ['', f"Ожидаемая доля30–120° после смешанного sampling: {100*audit['sampling']['expected_middle30_120_fraction']:.2f}%.",
        f"Предиктор: дополнительных updates{training['updates']}, best{training['best_update']}, stop `{training['stop_reason']}`.",
        'Mean/STD исходного train сохранились побитово.', '',
        '### Прогноз на общем новом test', '',
        '| Источник | Старый: q20, ° | Новый: q20, ° | Старый: v20, °/s | Новый: v20, °/s | Старый: q60, ° | Новый: q60, ° |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, values in comparisons.items():
        if name == 'conditional_test20': continue
        a,b = values['old'],values['new']
        lines.append(f"| {name} | {a['20']['angle_rmse_deg']:.4f} | {b['20']['angle_rmse_deg']:.4f} | {a['20']['velocity_rmse_deg_s']:.3f} | {b['20']['velocity_rmse_deg_s']:.3f} | {a['60']['angle_rmse_deg']:.4f} | {b['60']['angle_rmse_deg']:.4f} |")
    if (ROOT/'frozen/summary.json').exists():
        controller = read(ROOT/'frozen/summary.json')
        if controller['status'] == 'complete':
            assert read(ROOT/'evaluation_audit.json')['status'] == 'passed'
            lines += ['', '### PPO', '',
                '| Признаки | Переходов | Best | Validation: °; pass | Test best: °; pass | Test last: °; pass |',
                '| --- | ---: | ---: | --- | --- | --- |']
            trials = [('Без h1', read(OLD/'none/summary.json')), ('Старый h1', read(OLD/'frozen/summary.json')),
                      ('Расширенный train h1', controller)]
            def score(r): return f"{r['score_deg']:.3f};{r['passed']}/{r['total']}"
            for label, s in trials:
                lines.append(f"| {label} | {s['steps']} | {s['best_steps']} | {score(s['best_validation'])} | {score(s['best_test'])} | {score(s['last_test'])} |")
            lines += ['', f"Новый PPO stop `{controller['stop_reason']}`, actual optimizer steps{controller['optimizer_steps']}.",
                'Test содержит две знакомые опоры15→15° и20→75°; остальные9 пар новые относительно TRAIN_CASES.', '',
                '| Старт→цель, ° | Новый PPO tail RMSE, ° | Pass | Перелёт20ms, ° | Установление20ms, s |',
                '| --- | ---: | --- | ---: | ---: |']
            for case in controller['best_test']['cases']:
                overshoot = '—' if case['overshoot_deg_20ms'] is None else f"{case['overshoot_deg_20ms']:.3f}"
                settling = 'нет' if case['settling_time_s_20ms'] is None else f"{case['settling_time_s_20ms']:.3f}"
                lines.append(f"| {case['initial_deg']}→{case['target_deg']} | {case['tail_rmse_deg']:.3f} | {'да' if case['passed'] else 'нет'} | {overshoot} | {settling} |")
    report = '\n'.join(lines)+'\n'
    path = Path('docs/arm-balanced-predictor.md'); before = path.read_text()
    suffix = before.partition('## Интерпретация')[2]
    path.write_text(before.split('## Результаты')[0]+report+('\n## Интерпретация'+suffix if suffix else ''))
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, d, m in [('old train', old_data, old_meta), ('new train', data, meta)]:
        selection = starts(d,m,'train'); values = d['observations'][selection]
        axes[0].hist(values[:,0]*90+75, bins=np.arange(0,161,5), histtype='step', density=True, label=name)
    hist = training['history']
    for name in ('train','validation'):
        axes[1].plot([h['update'] for h in hist], [h[name]['balanced_mse'] for h in hist], label=name)
    for name, path in [('without h1', OLD/'none/summary.json'),('old h1', OLD/'frozen/summary.json'),('new h1', ROOT/'frozen/summary.json')]:
        if not path.exists(): continue
        h = read(path)['history']
        axes[2].plot([x['steps'] for x in h], [x['validation']['score_deg'] for x in h], label=name)
    axes[0].set(xlabel='Angle, deg', ylabel='Density', title='Raw train states')
    axes[1].set(xlabel='Additional predictor updates', ylabel='Balanced MSE, original units', yscale='log')
    axes[2].set(xlabel='PPO transitions', ylabel='Validation tail RMSE, deg', yscale='log')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(ROOT/f'comparison.{ext}', dpi=150)
    plt.close(fig)
    if (ROOT/'evaluation_audit.json').exists():
        fig,axes = plt.subplots(2,3,figsize=(14,7))
        for label,path in [('without h1',OLD/'none/best_test.json'),('old h1',OLD/'frozen/best_test.json'),
                           ('new h1',ROOT/'frozen/best_test.json')]:
            cases = [c for c in read(path)['cases'] if c['initial_deg'] != c['target_deg']]
            for ax,case in zip(axes.flat,cases):
                a = np.asarray(case['samples'])
                ax.plot(a[:,0],a[:,1],label=label)
                if label == 'without h1':
                    ax.axhline(case['target_deg'],color='black',ls='--',lw=.8)
                    ax.set(title=f"{case['initial_deg']} -> {case['target_deg']} deg",xlabel='Time, s',ylabel='Angle, deg')
        axes.flat[-1].axis('off'); axes[0,0].legend(fontsize=8);fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(ROOT/f'movement.{ext}',dpi=150)
        plt.close(fig)


if __name__ == '__main__': main()
