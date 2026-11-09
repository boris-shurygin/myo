"""Plot the measured arm hypotheses, including the bounded budget continuation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path('runs/arm_hypotheses')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for path, label in (('runs/arm_preacher/none/summary.json', 'No predictor'),
                        ('runs/arm_conditioning/gain_025/summary.json', 'Frozen predictor, gain 0.25'),
                        (root/'budget_none/summary.json', 'Unchanged continuation'),
                        (root/'lr_1e4_none/summary.json', 'Continuation: LR 1e-4'),
                        (root/'order_none/summary.json', 'Continuation: interleaved goals')):
        s = read(path)
        ax.plot([r['steps']/1000 for r in s['history']],
                [r['validation']['score_deg'] for r in s['history']], 'o-', label=label,
                markersize=3)
    ax.axvline(131.072, color='grey', linestyle='--')
    ax.set(title='H0: original runs ended at the budget limit', xlabel='Training transitions, thousands',
           ylabel='Validation tail RMSE, degrees')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    old = read('runs/arm_bottlenecks/audit.json')['execution']
    new = read(root/'concentration.json')['results']
    for case, label in (([20, 75], '20 → 75'), ([75, 20], '75 → 20'), ([25, 105], '25 → 105')):
        means, lows, highs = [], [], []
        for k in (1, 4, 16, 64):
            rows = ([r for r in old if r['model'] == 'gain025_best' and r['case'] == case
                     and r['stochastic'] and r['seed'] in (0, 1)] if k == 1 else
                    [r for r in new if r['model'] == 'gain025_best' and r['case'] == case
                     and r['concentration_multiplier'] == k])
            values = [r['tail_rmse_deg'] for r in rows]
            means.append(np.mean(values)); lows.append(min(values)); highs.append(max(values))
        deterministic = next(r['tail_rmse_deg'] for r in old if r['model'] == 'gain025_best'
                             and r['case'] == case and not r['stochastic'])
        means.append(deterministic); lows.append(deterministic); highs.append(deterministic)
        means = np.asarray(means)
        ax.errorbar(np.arange(5), means, yerr=[means-np.asarray(lows), np.asarray(highs)-means],
                    marker='o', capsize=3, label=label)
    ax.set(xticks=np.arange(5), xticklabels=['1', '4', '16', '64', 'Mean'],
           title='H1: narrowing the frozen policy (ranges of 2 streams)',
           xlabel='Concentration multiplier; last point deterministic', ylabel='Tail RMSE, degrees')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    gradient = read(root/'gradient.json')['results']
    labels = ['Initial', 'None best', 'Gain .25', 'Raw h', 'Vertical']
    keys = ['none_initial', 'none_best', 'gain025_best', 'raw_frozen_best', 'vertical_best']
    ax.bar(np.arange(5), [100*gradient[k]['summary']['critic_fraction_squared']['median'] for k in keys])
    ax.axhline(90, color='black', linestyle='--', label='Predeclared dominance criterion')
    ax.set(xticks=np.arange(5), xticklabels=labels, ylim=(0, 100),
           title='H2: critic does not dominate the measured updates',
           ylabel='Critic share of squared gradient norm, median %')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    critic = read(root/'critic.json')['results']
    x = np.arange(4)
    ax.plot(x, [r['value'] for r in critic], 'o', label='Critic V')
    ax.errorbar(x, [r['mc_12s_mean'] for r in critic], yerr=[r['mc_12s_se'] for r in critic],
                marker='s', linestyle='none', capsize=4, label='12-s MC mean ± SE (8 streams)')
    ax.set(xticks=x, xticklabels=['None reset', 'None +1s', 'Gain reset', 'Gain +1s'],
           title='H3: independent stochastic returns reveal value errors', ylabel='Discounted return')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for suffix in ('.png', '.svg'):
        fig.savefig(root/('hypotheses'+suffix), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
