"""Plot completed LR branches and diagnose frozen-feature use without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_arm_predictor_features import audit_checkpoint, digest
from compare_arm_predictor_lr import ROOT, TRIALS, load_checkpoint
from continue_arm_precision import stochastic_probe
from myo.train_vertical_ppo import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--diagnostics', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = {t: json.loads((args.root/t/'summary.json').read_text()) for t in TRIALS}
    if (not all(s['status'] in ('training_complete', 'complete') for s in rows.values())
            or not (args.root/'frozen_selection.json').exists()):
        raise ValueError('Finish all four branches and freeze their selection first')
    details = {}
    for t, s in rows.items():
        updates = s['updates']
        details[t] = {'added_transitions': s['added_transitions'],
            'sgd': s['added_optimizer_steps'], 'rollouts': len(updates),
            'rollouts_below_80_sgd': sum(u['optimizer_steps'] < 80 for u in updates),
            'initial_validation_deg': s['history'][0]['validation']['score_deg'],
            'best_balanced_validation_deg': min(h['goal_balanced_validation']['mean_tail_rmse_deg'] for h in s['history']),
            'last_explained_variance': updates[-1]['explained_variance'],
            'elapsed_s': s['elapsed_s']}
    write_json(args.root/'optimization.json', details)
    if args.diagnostics:
        source = Path('runs/arm_preacher_balanced/selected_policy_probe.npz')
        with np.load(source) as data:
            observations = data['observations'].copy()
        metadata = json.loads(source.with_suffix('.json').read_text())
        diagnostics = {'scope': 'Post-selection only; fixed historical training-state batch, no fitting',
            'observations': {'path': str(source), 'sha256': digest(source), 'count': len(observations)},
            'features': {}, 'stochastic': {}}
        for name, path in (
            ('parent_frozen', Path(TRIALS['frozen_1e4'][0])),
            *((t, args.root/t/'best.zip') for t in ('frozen_1e4', 'frozen_3e5')),
        ):
            model, env, _ = load_checkpoint(path)
            env.close()
            del model
            diagnostics['features'][name] = audit_checkpoint(path, observations, metadata, True)
            write_json(args.root/'diagnostics.json', diagnostics)
        for t in TRIALS:
            model, env, _ = load_checkpoint(args.root/t/'best.zip')
            env.close()
            diagnostics['stochastic'][t] = stochastic_probe(model)
            write_json(args.root/'diagnostics.json', diagnostics)
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for t, s in rows.items():
            x = np.array([h['added_transitions'] for h in s['history']])/1000
            dashed = t.endswith('3e5')
            color = '#b34d19' if t.startswith('frozen') else '#2563a6'
            style = '--' if dashed else '-'
            for ax, values in (
                (axes[0, 0], [h['validation']['score_deg'] for h in s['history']]),
                (axes[0, 1], [h['validation']['passed'] for h in s['history']]),
                (axes[1, 0], [h['goal_balanced_validation']['mean_tail_rmse_deg'] for h in s['history']]),
                (axes[1, 1], [h['optimizer_steps']-s['source_optimizer_steps'] for h in s['history']]),
            ):
                ax.plot(x, values, marker='o', linestyle=style, color=color, label=t)
        for ax, title, label in (
            (axes[0, 0], 'Native validation', 'Tail RMSE, degrees'),
            (axes[0, 1], 'Validation passes', 'Passed / 9'),
            (axes[1, 0], 'Equal weight per target', 'Tail RMSE, degrees'),
            (axes[1, 1], 'Actual optimizer updates', 'Added SGD steps'),
        ):
            ax.set(title=title, xlabel='Additional transitions, thousands', ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0, 1].set_ylim(-.2, 9.3)
        fig.suptitle('Warm continuations: different parents for frozen and none branches')
        for suffix in ('.png', '.svg'):
            fig.savefig(args.root/('comparison'+suffix), dpi=150)
        plt.close(fig)
    print(json.dumps(details))


if __name__ == '__main__':
    main()
