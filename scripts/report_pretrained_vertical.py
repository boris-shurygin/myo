"""Generate the controlled pretrained-feature PPO comparison from audited results."""
import json
from pathlib import Path

from myo.pretrained_vertical import TRIALS

ROOT = Path('runs/pretrained_vertical')
LABELS = ('Случайный фиксированный', 'Предобученный фиксированный', 'Предобученный дообучаемый')


def main():
    summaries = {name: json.loads((ROOT/name/'summary.json').read_text()) for name in TRIALS}
    assert all(s['status'] == 'complete' for s in summaries.values())
    assert json.loads((ROOT/'evaluation_audit.json').read_text())['status'] == 'passed'
    fmt = lambda r: f"{r['score_m']*1000:.3f}; {r['passed']}/{r['total']}"
    rows = ['## Результаты', '',
        'Все три обучения завершены: 3×131072 PPO-перехода, seed0. Selections',
        'зафиксированы по validation до test. Ниже RMSE последней секунды, не всего движения.', '',
        '| Признаки | Best шаг | Validation best: mm; pass | Test best: mm; pass | Test last: mm; pass |',
        '| --- | ---: | --- | --- | --- |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        s = summaries[name]
        rows.append(f"| {label} | {s['best_steps']} | {fmt(s['best_validation'])} | {fmt(s['best_test'])} | {fmt(s['last_test'])} |")
    rows += ['', '### Прогноз на независимых моторных test-эпизодах', '',
        '| Признаки / checkpoint | q20, mm | q60, mm | v20, mm/s | v60, mm/s | Normalized MSE20 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        for checkpoint in ('initial', 'best', 'last'):
            p = summaries[name]['body_forecasts'][checkpoint]
            rows.append(f"| {label} / {checkpoint} | {p['20']['q_rmse_mm']:.4f} | {p['60']['q_rmse_mm']:.4f} | {p['20']['v_rmse_mm_s']:.3f} | {p['60']['v_rmse_mm_s']:.3f} | {p['20']['normalized_mse']:.6f} |")
    rows += ['', 'Это один и тот же фиксированный test предобучения, не траектории трёх разных policy.',
        'Нормировка общая и заморожена. Controller selection не использует forecast-метрики.', '',
        '### Прогноз на траекториях контроллеров', '',
        '| Признаки / validation-данные | q20, mm | q60, mm | v20, mm/s | v60, mm/s |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        for key, source in (('prediction_evaluation', 'одинаковая прежняя policy'),
                            ('policy_prediction_evaluation', 'собственная policy')):
            p = summaries[name][key]
            rows.append(f"| {label} / {source} | {p['20']['position_rmse_mm']:.4f} | {p['60']['position_rmse_mm']:.4f} | {p['20']['velocity_rmse_mm_s']:.3f} | {p['60']['velocity_rmse_mm_s']:.3f} |")
    rows += ['', 'Первый набор одинаков у всех ветвей, второй различается из-за поведения policy.',
        'Улучшение на узких управляющих траекториях может сопровождаться забыванием',
        'более широких моторных проб. Меньшая ошибка прогноза не равна лучшему управлению.', '',
        '### Перелёты и торможение', '',
        '| Признаки | Движение, mm | RMSE последней секунды, mm | Перелёт, mm | Установление, s |',
        '| --- | --- | ---: | ---: | ---: |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        for m in summaries[name]['motion']:
            settling = 'нет' if m['settling_time_s'] is None else f"{m['settling_time_s']:.3f}"
            rows.append(f"| {label} | {m['initial_position_m']*1000:+g}→{m['target_m']*1000:+g} | {m['last_second_position_rmse_m']*1000:.3f} | {m['overshoot_m']*1000:.3f} | {settling} |")
    rows += ['', 'Установление требует |ошибка|≤5 mm и |v|≤.03 m/s на ВСЕХ оставшихся',
        'физических samples до конца4 s. Проверены все33 native test-replay с шагом1 ms.', '',
        '### Обновления и сдвиг политики', '',
        '| Признаки | PPO accepted/rejected | Predictor updates | Predictor KL mean/max | Total KL max | Rollout без PPO update |',
        '| --- | --- | ---: | --- | ---: | ---: |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        s = summaries[name]; d = s['drift']; k = d['predictor_phase_kl']
        rows.append(f"| {label} | {s['accepted_optimizer_steps']}/{s['rejected_optimizer_steps']} | {s['predictor_steps']} | {k['mean']:.6g}/{k['max']:.6g} | {d['total_phase_kl']['max']:.6g} | {d['rollouts_no_ppo_update']} |")
    rows += ['', 'KL — точный средний KL(old Beta || new Beta) на полном train-rollout,',
        'с суммой по двум мышцам. PPO-guard не ограничивает predictor-фазу.', '',
        '### Использование h1', '',
        '| Признаки | KL при замене h1 его средним | Среднее изменение возбуждения | RMSE изменения V |',
        '| --- | ---: | ---: | ---: |']
    for name, label in zip(TRIALS, LABELS, strict=True):
        f = summaries[name]['feature_use']
        rows.append(f"| {label} | {f['policy_kl']:.6g} | {f['mean_action_abs_change']:.6g} | {f['value_rmse_change']:.6g} |")
    rows += ['', 'Read-only проверка на одинаковых1800 validation-входах. Собственные ветви',
        'сохраняют все входы; ablation не переобучает сеть и не доказывает пользу h1.', '']
    if all((ROOT/name/'critic_audit.json').exists() for name in TRIALS):
        critics = {name: json.loads((ROOT/name/'critic_audit.json').read_text()) for name in TRIALS}
        assert len({c['state_observations_sha256'] for c in critics.values()}) == 1
        rows += ['### Критик: Monte Carlo', '',
            '| Признаки | RMSE V к MC mean | Bias | Explained variance | Failures |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name, label in zip(TRIALS, LABELS, strict=True):
            c = critics[name]
            rows.append(f"| {label} | {c['rmse_against_mc_means']:.5f} | {c['mean_bias']:+.5f} | {c['explained_variance_against_mc_means']:.4f} | {c['failures']} |")
        rows += ['', 'Одинаковые12 native состояний, 6 stochastic продолжений по16 s на состояние.',
            'Каждый critic сравнивается с return своей policy, reward multiplier .1, gamma .99,',
            'без bootstrap. Верхняя граница положительного остатка return .003222;',
            'это небольшая диагностическая выборка, не общая оценка точности критика.', '']
    report = '\n'.join(rows)
    path = Path('docs/pretrained-predictor-ppo.md')
    previous = path.read_text()
    interpretation = previous.partition('## Интерпретация')[2]
    path.write_text(previous.split('## Результаты')[0]+report+('\n## Интерпретация'+interpretation if interpretation else ''))
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, label in zip(TRIALS, LABELS, strict=True):
        history = summaries[name]['history']
        x = [h['steps'] for h in history]
        axes[0].plot(x, [h['validation']['score_m']*1000 for h in history], label=label)
        axes[1].plot(x, [h['validation']['passed'] for h in history], label=label)
        axes[2].plot(x, [h['body_validation']['v_rmse_mm_s'] for h in history], label=label)
    axes[0].set(xlabel='PPO transitions', ylabel='Steady RMSE, mm', yscale='log', title='Control validation')
    axes[1].set(xlabel='PPO transitions', ylabel='Passed / 9', title='Control success')
    axes[2].set(xlabel='PPO transitions', ylabel='20-ms velocity RMSE, mm/s', yscale='log', title='Fixed body validation')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT/f'comparison.{extension}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name, label in zip(TRIALS, LABELS, strict=True):
        cases = json.loads((ROOT/name/'best_test.json').read_text())['cases']
        moving = [c for c in cases if c['initial_position_m'] != c['target_m']]
        for ax, case in zip(axes.flat, moving, strict=True):
            ax.plot([row[0] for row in case['samples']], [row[1]*1000 for row in case['samples']], label=label)
            if name == TRIALS[0]:
                ax.axhline(case['target_m']*1000, color='black', ls='--', lw=.8)
                ax.set(xlabel='Time, s', ylabel='Position, mm', title=f"{case['initial_position_m']*1000:+g} -> {case['target_m']*1000:+g} mm")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT/f'movement.{extension}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
