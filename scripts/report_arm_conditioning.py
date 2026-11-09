"""Summarize the declared one-factor arm PPO runs after frozen selection/test."""
import argparse
import json
from pathlib import Path

import numpy as np

from myo.train_arm_conditioning import ROOT, REFERENCE, TRIALS, balanced_goals


def read(path):
    return json.loads(path.read_text())


def summarize(summary):
    updates = np.asarray([u['optimizer_steps'] for u in summary['updates']])
    assert updates.sum() == summary['optimizer_steps']
    assert len(updates)*1024 == summary['steps']
    return {
        'steps': summary['steps'], 'optimizer_steps': summary['optimizer_steps'],
        'rollouts': len(updates), 'kl_stopped_rollouts': int((updates < 80).sum()),
        'median_optimizer_steps_per_rollout': float(np.median(updates)),
        'best_steps': summary['best_steps'], 'best_validation': summary['best_validation'],
        'goal_balanced_best_validation': balanced_goals(summary['best_validation']),
        'best_test': summary['best_test'], 'last_test': summary['last_test'],
        'goal_balanced_best_test': balanced_goals(summary['best_test']),
        'elapsed_s': summary['elapsed_s'], 'stop_reason': summary['stop_reason'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--audit-features', action='store_true')
    args = parser.parse_args()
    reference = read(REFERENCE/'frozen/summary.json')
    summaries = {name: read(ROOT/name/'summary.json') for name in TRIALS}
    assert all(s['status'] == 'complete' for s in summaries.values())
    for summary in summaries.values():
        assert summary['contract'] == reference['contract']
        assert summary['initial_private_sha256'] == reference['initial_private_sha256']
        assert summary['initial_torch_rng_sha256'] == reference['initial_torch_rng_sha256']
        assert summary['training']['predictor'] == reference['training']['predictor']
        for key, value in reference['training']['ppo'].items():
            if key != 'learning_rate':
                assert summary['training']['ppo'][key] == value
    values = {'reference': summarize(reference), **{k: summarize(s) for k, s in summaries.items()}}
    (ROOT/'comparison.json').write_text(json.dumps(values, indent=2, allow_nan=False)+'\n')
    def score(s):
        return f"{s['score_deg']:.3f};{s['passed']}/{s['total']}"
    header = ['| Настройка | Переходы | SGD steps | KL-stop / rollouts | Best шаг | Validation: °;pass | Test best: °;pass | Test last: °;pass |',
              '| --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    def row(name, key):
        v = values[key]
        return (f"| {name} | {v['steps']} | {v['optimizer_steps']} | {v['kl_stopped_rollouts']}/{v['rollouts']} | "
            f"{v['best_steps']} | {score(v['best_validation'])} | {score(v['best_test'])} | {score(v['last_test'])} |")
    lines = ['## Результаты', '', '### Learning rate при gain1', '', *header,
             row('3e−4, контроль', 'reference'), row('1e−4', 'lr_1e4'), row('3e−5', 'lr_3e5'),
             '', '### Масштаб h1 при learning rate3e−4', '', *header,
             row('gain1, контроль', 'reference'), row('gain0,25', 'gain_025'),
             '', '### Остановка и фактическая стоимость', '',
             '| Настройка | Stop | Elapsed, s | Median SGD/rollout | Goal-balanced validation RMSE, ° |',
             '| --- | --- | ---: | ---: | ---: |']
    for name, v in values.items():
        lines.append(f"| {name} | {v['stop_reason']} | {v['elapsed_s']:.1f} | {v['median_optimizer_steps_per_rollout']:.0f} | {v['goal_balanced_best_validation']['mean_tail_rmse_deg']:.3f} |")
    lines += ['', 'Elapsed разных запусков не является чистым CPU-бенчмарком: новые ветки',
              'выполнялись одновременно на отдельных CPU. Фактические переходы и SGD указаны.',
              '', '### Test выбранных best по сценариям', '',
              '| Старт→цель, ° | Контроль: °;pass | lr1e−4: °;pass | lr3e−5: °;pass | gain0,25: °;pass |',
              '| --- | --- | --- | --- | --- |']
    rows = [v['best_test']['cases'] for v in values.values()]
    for cases in zip(*rows, strict=True):
        assert len({(r['initial_deg'], r['target_deg']) for r in cases}) == 1
        a = cases[0]
        cells = [f"{r['tail_rmse_deg']:.3f};{'да' if r['passed'] else 'нет'}" for r in cases]
        lines.append(f"| {a['initial_deg']}→{a['target_deg']} | "+' | '.join(cells)+' |')
    lines += ['', 'Все source/physical contracts, predictor identity, исходные обучаемые веса и',
              'Torch RNG совпали. Encoder/head/normalization frozen проверены побитово;',
              'gain сохранён и проверен строгим loader. Best validation воспроизведён,',
              'выбранные hashes зафиксированы до test. Last показан без переотбора best.',
              'Гиперпараметры подбираются на уже известном наборе условий; test не новый',
              'независимый benchmark и один seed не устанавливает устойчивость результата.', '']
    demo_audit = {}
    for trial in TRIALS:
        folder = ROOT/trial
        if not (folder/'demo_best.json').exists():
            continue
        demo = read(folder/'demo_best.json')
        frames = [json.loads(line) for line in (folder/'demo_best.jsonl').read_text().splitlines()]
        assert len(frames) == 800 and demo['seconds'] == 16. and not demo['failure']
        assert len({f['run_id'] for f in frames}) == len({f['episode'] for f in frames}) == 1
        assert [f['sequence'] for f in frames] == list(range(800))
        assert all(f['schema'] == 'myo.character_state.v2' for f in frames)
        np.testing.assert_allclose([f['simulation_time_s'] for f in frames], np.arange(1, 801)*.02, atol=1e-9, rtol=0)
        errors = np.rad2deg([f['joints'][0]['angle_rad']-f['command']['requested_target_angle_rad'] for f in frames])
        np.testing.assert_allclose(np.sqrt(np.mean(errors**2)), demo['full_rmse_deg'], atol=1e-8, rtol=0)
        demo_audit[trial] = {**demo, 'frames': len(frames),
            'end_angles_deg': [float(np.rad2deg(frames[i]['joints'][0]['angle_rad'])) for i in (199, 399, 599, 799)],
            'final_min_capacity_proxy': min(m['remaining_capacity'] for m in frames[-1]['muscles']),
            'continuity': 'passed: one episode, sequential frames/times, no reset'}
    (ROOT/'demo_audit.json').write_text(json.dumps(demo_audit, indent=2, allow_nan=False)+'\n')
    if demo_audit:
        lines += ['### Непрерывное демо', '',
                  '75→20→105→45°, по 4 s без reset. Full RMSE включает переходы и не равен',
                  'tail RMSE отдельных test-сценариев. GUI не открывался.', '',
                  '| Ветка | Full RMSE, ° | Конечные углы четырёх участков, ° | Final min capacity proxy |',
                  '| --- | ---: | --- | ---: |']
        for trial, d in demo_audit.items():
            angles = ', '.join(f'{v:.2f}' for v in d['end_angles_deg'])
            lines.append(f"| {trial} | {d['full_rmse_deg']:.3f} | {angles} | {d['final_min_capacity_proxy']:.4f} |")
        lines += ['', 'Проверены 800 v2-кадров на ветку, последовательность времени и отсутствие reset.', '']
    results = '\n'.join(lines)
    (ROOT/'report.md').write_text(results)
    doc = Path('docs/arm-ppo-conditioning.md')
    document = doc.read_text()
    before = document.split('## Результаты\n', 1)[0]
    interpretation = ('\n## Интерпретация\n'+document.split('\n## Интерпретация\n', 1)[1]
                      if '\n## Интерпретация\n' in document else '')
    doc.write_text(before+results+interpretation)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for name, summary in {'reference': reference, **summaries}.items():
            history = summary['history']
            steps = [r['steps'] for r in history]
            axes[0].plot(steps, [r['validation']['score_deg'] for r in history], marker='.', label=name)
            axes[1].plot(steps, [r['validation']['passed'] for r in history], marker='.', label=name)
            axes[2].plot(steps, [r['optimizer_steps'] for r in history], marker='.', label=name)
        for ax, title, ylabel in zip(axes, ('Validation error', 'Validation passes', 'Actual optimization'),
                                    ('Mean tail RMSE, deg', 'Passed cases / 9', 'Optimizer steps'), strict=True):
            ax.set_title(title)
            ax.set_xlabel('Environment transitions')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axes[0].legend(fontsize=8)
        axes[1].set_ylim(-.2, 9.2)
        fig.suptitle('MyoArm: one-factor PPO comparison; seed 0, frozen predictor')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(ROOT/f'conditioning.{suffix}', dpi=160)
        plt.close(fig)
    if args.audit_features:
        import torch
        from audit_arm_predictor_features import audit_checkpoint
        from myo.arm_conditioning import load_conditioned_arm
        from myo.arm_learning import load_arm
        torch.set_num_threads(1)
        path = REFERENCE/'selected_policy_probe.npz'
        with np.load(path) as data:
            observations = data['observations'].copy()
        meta = read(path.with_suffix('.json'))
        audits = {trial: audit_checkpoint(ROOT/trial/'best.zip', observations, meta, False) for trial in TRIALS}
        (ROOT/'feature_use_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False)+'\n')
        branches = {}
        x = torch.as_tensor(observations)
        for name, folder in [('reference', REFERENCE/'frozen'), *[(k, ROOT/k) for k in TRIALS]]:
            branches[name] = {}
            for label in ('initial', 'best'):
                loader = load_arm if name == 'reference' else load_conditioned_arm
                model, env, _ = loader(folder/f'{label}.zip')
                try:
                    with torch.no_grad():
                        extractor = model.policy.mlp_extractor
                        h = extractor.features(x)
                        private = extractor.actor_private(x)
                        first = extractor.policy_net[0].weight
                        rms = lambda v: float(v.square().mean().sqrt())
                        beta = model.policy.get_distribution(x).distribution
                        branches[name][label] = {
                            'h1_rms': rms(h), 'actor_private_rms': rms(private),
                            'h1_first_layer_contribution_rms_without_bias': rms(h@first[:, :128].T),
                            'private_first_layer_contribution_rms_without_bias': rms(private@first[:, 128:].T),
                            'mean_excitation': float(beta.mean.mean()), 'mean_conditional_std': float(beta.stddev.mean()),
                            'scope': 'same fixed 5400 probe states; not the current training occupancy distribution'}
                finally:
                    env.close()
        (ROOT/'branch_scale_audit.json').write_text(json.dumps(branches, indent=2, allow_nan=False)+'\n')
    print(results)


if __name__ == '__main__':
    main()
