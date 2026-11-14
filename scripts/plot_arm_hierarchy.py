"""Plot real held-out-trajectory rollouts; does not run or alter the simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    runs = root/'runs/arm_hierarchy'
    heuristic = json.loads((runs/'feasibility/heuristic.json').read_text())['cases']
    first = json.loads((runs/'inverse_relabelled/evaluation.json').read_text())['test']['cases']
    grid = json.loads((runs/'inverse_cold_grid/evaluation.json').read_text())['test']['cases']
    fig, axes = plt.subplots(1,3, figsize=(12,3.7), sharey=True, layout='constrained')
    for ax, original, refined in zip(axes, first, grid):
        baseline = next(c for c in heuristic if c['motion'] == original['motion'])
        trace = original['trace']
        ax.plot([s['t'] for s in trace], np.rad2deg([s['reference_q'] for s in trace]),
                'k--', linewidth=1.8, label='Эталон')
        for case, label, color in ((baseline,'Эвристика','#277da1'),
                                   (original,'Первая inverse','#f9844a'),
                                   (refined,'Сетка стартов','#43aa8b')):
            ax.plot([s['t'] for s in case['trace']], np.rad2deg([s['q'] for s in case['trace']]),
                    color=color, label=label, linewidth=1.4)
        ax.axvspan(0,.5,color='gray',alpha=.1)
        ax.set_title(original['motion'])
        ax.set_xlabel('Время, с')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Угол локтя, °')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Исходная механика, 1 кг, наклон 120°. Известные test-траектории')
    out = root/'docs/images/arm-hierarchy-trajectories.svg'
    fig.savefig(out, metadata={'Date':None})
    print(out)


if __name__ == '__main__':
    main()
