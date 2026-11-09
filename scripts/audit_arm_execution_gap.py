"""Probe deterministic basins and activation rectification without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_arm_learning_bottlenecks import arm_episode
from myo.arm_conditioning import load_conditioned_arm
from myo.train_vertical_architecture import parameter_hash


def activation_filter(commands, tau_off):
    # Exact block solution of the existing piecewise exponential activation law.
    # Transport starts filled with .03, as in the native arm reset.
    activation = np.full(6, .03)
    delivered = np.vstack((np.full((2, 6), .03), commands[:-2]))
    result = []
    for command in delivered:
        tau = np.where(command > activation, .015, tau_off)
        activation = command+(activation-command)*np.exp(-.02/tau)
        result.append(activation.copy())
    return np.asarray(result)


def plot_report(result, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for row in result['execution']:
        a = np.asarray(row['trace'])
        deterministic = not row['stochastic']
        ax.plot(.02*np.arange(1, len(a)+1), a[:, 0],
                label='Mean commands' if deterministic else f"Samples, seed {row['seed']}",
                linewidth=2.5 if deterministic else 1., alpha=1. if deterministic else .7)
    ax.axhline(75, color='black', linestyle='--', label='Goal')
    ax.set(title='Same policy: 20 → 75 degrees', xlabel='Time, s', ylabel='Angle, degrees')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot([r['case'][0] for r in result['basins']],
            [r['tail_mean_deg'] for r in result['basins']], 'o-')
    ax.axhline(75, color='black', linestyle='--')
    ax.set(title='Deterministic execution from different starts', xlabel='Initial angle, degrees',
           ylabel='Last-second mean angle, degrees')
    ax = axes[1, 0]
    audit = result['execution'][1]['activation_audit']
    x = np.arange(6)
    for offset, key, label in ((-.25, 'tail_mean_command', 'Command'),
                               (0., 'tail_mean_activation', 'Activation, 15/50 ms'),
                               (.25, 'tail_mean_symmetric_filter', 'Offline filter, 15/15 ms')):
        ax.bar(x+offset, audit[key], .25, label=label)
    ax.set(xticks=x, xticklabels=['BIC long', 'BIC short', 'BRA', 'TRI long', 'TRI lat', 'TRI med'],
           title='Same recorded commands: activation asymmetry (seed 0)', ylabel='Last-second mean')
    ax.tick_params(axis='x', labelrotation=25)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for row in result['execution']:
        a = np.asarray(row['trace'])
        ax.plot(.02*np.arange(1, len(a)+1), a[:, 6],
                label='Mean commands' if not row['stochastic'] else f"Samples, seed {row['seed']}",
                alpha=.8)
    ax.set(title='BICshort excitation: mean and sampled execution', xlabel='Time, s', ylabel='Excitation')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for suffix in ('.png', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/arm_bottlenecks/execution_gap.json'))
    p.add_argument('--plot-only', action='store_true', help='Plot an existing report without running physics')
    args = p.parse_args()
    if args.plot_only:
        plot_report(json.loads(args.output.read_text()), args.output)
        return
    if args.output.exists():
        p.error('Output already exists')
    torch.set_num_threads(1)
    model, env, meta = load_conditioned_arm(Path('runs/arm_conditioning/gain_025/best.zip'))
    before = parameter_hash(model)
    result = {'schema': 'myo.arm_execution_gap.v1', 'checkpoint_sha256': meta['checkpoint_sha256'],
              'basins': [], 'execution': []}
    for initial in (20, 22, 24, 25, 26, 30, 35, 45):
        row = arm_episode(model, env, (initial, 75))
        result['basins'].append(row)
        print(json.dumps({'start_deg': initial, 'tail_deg': row['tail_mean_deg']}), flush=True)
    for stochastic, seed in ((False, 0), *((True, s) for s in range(4))):
        row = arm_episode(model, env, (20, 75), stochastic=stochastic, seed=seed, return_trace=True)
        commands = np.asarray(row['trace'])[:, 5:]
        actual = np.asarray(row['activation_trace'])
        exact = activation_filter(commands, .05)
        error = float(np.max(np.abs(actual-exact)))
        assert error < 1e-12, error
        symmetric = activation_filter(commands, .015)
        row['activation_audit'] = {'max_abs_native_reconstruction_error': error,
            'tail_mean_command': commands[-50:].mean(0).tolist(),
            'tail_mean_activation': actual[-50:].mean(0).tolist(),
            'tail_mean_symmetric_filter': symmetric[-50:].mean(0).tolist(),
            'tail_command_std': commands[-50:].std(0).tolist()}
        result['execution'].append(row)
    assert parameter_hash(model) == before
    env.close()
    result.update(status='complete', unchanged_weights=True,
                  trigger='fixed 1; no excitation cap below 1; excess penalty identically zero')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
