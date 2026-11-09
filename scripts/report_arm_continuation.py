"""Report validation, regressions and matched mean/sample probes of continuation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from myo.arm_learning import load_arm
from myo.arm_precision import evaluate_arm
from myo.train_vertical_ppo import write_json
from continue_arm_precision import PROBE_CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/arm_continuation'))
    p.add_argument('--measure-probes', action='store_true')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    s = json.loads((args.root/'summary.json').read_text())
    cache_path = args.root/'deterministic_probes.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    steps = sorted({s['source_steps'], s['best_steps'], *[b['steps'] for b in s['blocks']]})
    if args.measure_probes:
        for step in steps:
            if str(step) in cache:
                continue
            path = args.root/'checkpoints'/f'step_{step}.zip'
            model, env, meta = load_arm(path)
            env.close()
            cache[str(step)] = {'steps': step, 'checkpoint_sha256': meta['checkpoint_sha256'],
                                'evaluation': evaluate_arm(model, PROBE_CASES)}
            write_json(cache_path, cache)
    lines = ['# Продолжение low-LR PPO', '', f"Статус: `{s['status']}`; остановка: `{s.get('stop_reason', 'ещё не остановлено')}`.", '',
        '| Шаг | Validation RMSE, ° | С равным весом целей, ° | Pass | SGD от родителя |',
        '| --- | ---: | ---: | --- | ---: |']
    for h in s['history']:
        lines.append(f"| {h['steps']} | {h['validation']['score_deg']:.3f} | "
            f"{h['goal_balanced_validation']['mean_tail_rmse_deg']:.3f} | {h['validation']['passed']}/9 | "
            f"{h['optimizer_steps']-s['source_optimizer_steps']} |")
    lines += ['', '| Блок | Завершён на шаге | Значимое улучшение | Блоков без улучшения |', '| --- | ---: | --- | ---: |']
    for b in s['blocks']:
        lines.append(f"| {b['block']} | {b['steps']} | {b['significant_progress']} | {b['stagnant_blocks']} |")
    lines += ['', '## Изменение отдельных validation-сценариев', '',
        '| Старт→цель | Parent RMSE, ° | Best RMSE, ° | Parent pass | Best pass |',
        '| --- | ---: | ---: | --- | --- |']
    for old, new in zip(s['history'][0]['validation']['cases'], s['best_validation']['cases'], strict=True):
        assert (old['initial_deg'], old['target_deg']) == (new['initial_deg'], new['target_deg'])
        lines.append(f"| {old['initial_deg']}→{old['target_deg']} | {old['tail_rmse_deg']:.3f} | "
                     f"{new['tail_rmse_deg']:.3f} | {old['passed']} | {new['passed']} |")
    points = [(s['source_steps'], s['stochastic_initial'])]
    points += [(b['steps'], b['stochastic']) for b in s['blocks']]
    if 'best_stochastic' in s:
        points.append((s['best_steps'], s['best_stochastic']))
    points = list(dict(points).items())
    lines += ['', '## Средние команды и samples на одинаковых заданиях', '',
        '| Шаг | Старт→цель | Mean RMSE, ° | Samples0/1 RMSE, ° |', '| --- | --- | ---: | --- |']
    for step, sampled in points:
        if str(step) not in cache:
            continue
        for case, det in zip(PROBE_CASES, cache[str(step)]['evaluation']['cases'], strict=True):
            rows = [r for r in sampled if (r['initial_deg'], r['target_deg']) == case]
            errors = ' / '.join(f"{r['tail_rmse_deg']:.3f}" for r in rows)
            lines.append(f"| {step} | {case[0]}→{case[1]} | {det['tail_rmse_deg']:.3f} | {errors} |")
    for label in ('best', 'balanced_best', 'last'):
        if label+'_test' in s:
            r = s[label+'_test']
            lines += ['', f"Test {label}: {r['score_deg']:.3f}°; {r['passed']}/{r['total']}, failures={r['failures']}."]
    (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        x = [h['steps']/1000 for h in s['history']]
        axes[0, 0].plot(x, [h['validation']['score_deg'] for h in s['history']], 'o-', label='Original mean')
        axes[0, 0].plot(x, [h['goal_balanced_validation']['mean_tail_rmse_deg'] for h in s['history']], 'o-', label='Equal weight per goal')
        axes[0, 0].set(title='Validation error', xlabel='Lifetime training transitions, thousands', ylabel='Tail RMSE, degrees')
        axes[0, 0].legend()
        axes[0, 1].step(x, [h['validation']['passed'] for h in s['history']], where='mid')
        axes[0, 1].set(title='Native validation passes', xlabel='Lifetime training transitions, thousands', ylabel='Passed / 9', ylim=(-.2, 9.2))
        old = s['history'][0]['validation']['cases']; new = s['best_validation']['cases']; indices = np.arange(len(old))
        axes[1, 0].bar(indices-.2, [r['tail_rmse_deg'] for r in old], .4, label='Parent')
        axes[1, 0].bar(indices+.2, [r['tail_rmse_deg'] for r in new], .4, label='Best')
        axes[1, 0].set(xticks=indices, xticklabels=[f"{r['initial_deg']}→{r['target_deg']}" for r in old],
            title='Per-case validation error', ylabel='Tail RMSE, degrees')
        axes[1, 0].tick_params(axis='x', labelrotation=40); axes[1, 0].legend()
        step, sampled = points[-1]
        if str(step) in cache:
            det = cache[str(step)]['evaluation']['cases']; av, lo, hi = [], [], []
            for case in PROBE_CASES:
                a = [r['tail_rmse_deg'] for r in sampled if (r['initial_deg'], r['target_deg']) == case]
                av.append(np.mean(a)); lo.append(min(a)); hi.append(max(a))
            k = np.arange(len(PROBE_CASES)); av = np.asarray(av)
            axes[1, 1].bar(k-.2, [r['tail_rmse_deg'] for r in det], .4, label='Mean commands')
            axes[1, 1].bar(k+.2, av, .4, yerr=[av-lo, hi-av], capsize=4, label='Samples (range of 2 streams)')
            axes[1, 1].set(xticks=k, xticklabels=[f'{a}→{b}' for a, b in PROBE_CASES],
                title=f'Execution comparison, step {step}', ylabel='Tail RMSE, degrees')
            axes[1, 1].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        for suffix in ('.png', '.svg'):
            fig.savefig(args.root/('continuation'+suffix), dpi=150)
        plt.close(fig)
    print(json.dumps({'status': s['status'], 'steps': s['history'][-1]['steps'],
                      'blocks': len(s['blocks']), 'best_steps': s['best_steps']}))


if __name__ == '__main__':
    main()
