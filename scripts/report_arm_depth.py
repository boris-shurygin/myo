"""Report the fixed-gain two-versus-four-layer arm comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from myo.train_arm_conditioning import balanced_goals
from myo.train_arm_depth import ROOT, REFERENCE


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--audit-features', action='store_true')
    args = parser.parse_args()
    deep, reference = read(ROOT/'summary.json'), read(REFERENCE/'summary.json')
    assert deep['status'] == reference['status'] == 'complete'
    assert deep['contract'] == reference['contract']
    assert deep['training']['ppo'] == reference['training']['ppo']
    assert deep['training']['conditioning'] == reference['training']['conditioning']
    assert deep['training']['predictor'] == reference['training']['predictor']
    assert deep['initial_torch_rng_sha256'] == reference['initial_torch_rng_sha256']
    lines = ['## Результаты глубины', '',
             '| После concat | Переходы | SGD | KL-stop / rollouts | Best шаг | Validation: °;pass | Test best: °;pass | Test last: °;pass |',
             '| --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    comparison = {}
    for name, summary in [('2×128', reference), ('4×128', deep)]:
        updates = [u['optimizer_steps'] for u in summary['updates']]
        assert sum(updates) == summary['optimizer_steps'] and len(updates)*1024 == summary['steps']
        def score(r):
            return f"{r['score_deg']:.3f};{r['passed']}/{r['total']}"
        stopped = sum(u < 80 for u in updates)
        lines.append(f"| {name} | {summary['steps']} | {summary['optimizer_steps']} | {stopped}/{len(updates)} | "
            f"{summary['best_steps']} | {score(summary['best_validation'])} | {score(summary['best_test'])} | {score(summary['last_test'])} |")
        comparison[name] = {'steps': summary['steps'], 'optimizer_steps': summary['optimizer_steps'],
            'kl_stopped_rollouts': stopped, 'rollouts': len(updates), 'best_steps': summary['best_steps'],
            'stop_reason': summary['stop_reason'], 'elapsed_s': summary['elapsed_s'],
            'best_validation': summary['best_validation'], 'best_test': summary['best_test'],
            'last_test': summary['last_test'], 'goal_balanced_validation': balanced_goals(summary['best_validation'])}
    lines += ['', f"Новый stop:`{deep['stop_reason']}`, elapsed{deep['elapsed_s']:.1f} s, "
              f"PPO-параметров{deep['ppo_parameter_count']}. Elapsed не чистый CPU-бенчмарк:",
              'старый контроль выполнялся одновременно с двумя другими PPO, новый — отдельно.',
              '', '### Test по сценариям', '', '| Старт→цель, ° | 2 слоя: °;pass | 4 слоя: °;pass |',
              '| --- | --- | --- |']
    for a, b in zip(reference['best_test']['cases'], deep['best_test']['cases'], strict=True):
        assert (a['initial_deg'], a['target_deg']) == (b['initial_deg'], b['target_deg'])
        lines.append(f"| {a['initial_deg']}→{a['target_deg']} | {a['tail_rmse_deg']:.3f};{'да' if a['passed'] else 'нет'} | "
                     f"{b['tail_rmse_deg']:.3f};{'да' if b['passed'] else 'нет'} |")
    lines += ['', 'Общие исходные tensors и Torch RNG совпали, predictor frozen проверен побитово.',
              'Best hash зафиксирован до test, native validation воспроизведён; last не переотобран по test.',
              'Дополнительные слои изменяют исходную функцию и условия оптимизации;',
              'это не чистая проверка предельной представительной способности при найденном оптимуме.', '']
    demo = None
    if (ROOT/'demo_best.json').exists():
        demo = read(ROOT/'demo_best.json')
        frames = [json.loads(line) for line in (ROOT/'demo_best.jsonl').read_text().splitlines()]
        assert len(frames) == round(demo['seconds']/.02)
        assert len({f['episode'] for f in frames}) == len({f['run_id'] for f in frames}) == 1
        assert [f['sequence'] for f in frames] == list(range(len(frames)))
        np.testing.assert_allclose([f['simulation_time_s'] for f in frames], np.arange(1, len(frames)+1)*.02, rtol=0, atol=1e-9)
        demo['end_angles_deg'] = [float(np.rad2deg(frames[i]['joints'][0]['angle_rad'])) if i < len(frames) else None
                                  for i in (199, 399, 599, 799)]
        demo['final_min_capacity_proxy'] = min(m['remaining_capacity'] for m in frames[-1]['muscles'])
        demo['continuity'] = 'passed; sequential v2 frames/times, no reset'
        comparison['demo'] = demo
        lines += ['### Непрерывное демо', '',
                  f"75→20→105→45°, по4 s без reset: duration{demo['seconds']:.2f} s, "
                  f"failure={demo['failure']}, full RMSE{demo['full_rmse_deg']:.3f}°.",
                  f"Конечные углы:{demo['end_angles_deg']}; final min capacity proxy{demo['final_min_capacity_proxy']:.4f}.",
                  f"Проверено{len(frames)} v2-кадров. GUI не открывался; full RMSE не равен tail RMSE test.", '']
    (ROOT/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')
    results = '\n'.join(lines)
    (ROOT/'report.md').write_text(results)
    doc = Path('docs/arm-predictor-readout-and-depth.md')
    document = doc.read_text()
    before = document.split('## Результаты глубины\n', 1)[0]
    interpretation = ('\n## Выводы глубины\n'+document.split('\n## Выводы глубины\n', 1)[1]
                      if '\n## Выводы глубины\n' in document else '')
    doc.write_text(before+results+interpretation)
    if args.audit_features:
        import torch
        from audit_arm_predictor_features import audit_checkpoint
        torch.set_num_threads(1)
        path = Path('runs/arm_preacher_balanced/selected_policy_probe.npz')
        with np.load(path) as data:
            observations = data['observations'].copy()
        audit = audit_checkpoint(ROOT/'best.zip', observations, read(path.with_suffix('.json')), False)
        (ROOT/'feature_use_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for name, summary in [('2 layers', reference), ('4 layers', deep)]:
            history = summary['history']
            steps = [h['steps'] for h in history]
            for ax, values in zip(axes, ([h['validation']['score_deg'] for h in history],
                [h['validation']['passed'] for h in history], [h['optimizer_steps'] for h in history]), strict=True):
                ax.plot(steps, values, marker='.', label=name)
        for ax, title, label in zip(axes, ('Validation error', 'Validation passes', 'Actual optimization'),
            ('Mean tail RMSE, deg', 'Passed cases / 9', 'Optimizer steps'), strict=True):
            ax.set(title=title, xlabel='Environment transitions', ylabel=label)
            ax.grid(alpha=.25)
            ax.legend()
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axes[1].set_ylim(-.2, 9.2)
        fig.suptitle('MyoArm post-concatenation depth; fixed width 128, gain 0.25, seed 0')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(ROOT/f'depth.{suffix}', dpi=160)
        plt.close(fig)
    print(results)


if __name__ == '__main__':
    main()
