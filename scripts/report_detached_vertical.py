"""Generate the detached predictor-feature report and a standalone comparison plot."""
import json
from pathlib import Path

import numpy as np


ROOT = Path('runs/detached_vertical')
NAMES = ('frozen', 'online')
LABELS = {'frozen': 'Фиксированные признаки', 'online': 'Обучаемые признаки'}


def main():
    summaries = {n: json.loads((ROOT/n/'summary.json').read_text()) for n in NAMES}
    assert all(s['status'] == 'complete' for s in summaries.values())
    critics = {n: json.loads((ROOT/n/'critic_audit.json').read_text()) for n in NAMES}
    assert len({c['state_observations_sha256'] for c in critics.values()}) == 1
    fmt = lambda r: f"{r['score_m']*1000:.3f}; {r['passed']}/{r['total']}"
    text = ['## Результаты', '',
        'Оба обучения завершены по фиксированному бюджету: 2×131072 перехода. Один seed.',
        'Test выполнен после фиксации двух selections по tracking validation.', '',
        '| Ветка | Params | Best шаг | Val best, mm; pass | Val last, mm; pass | Test best, mm; pass | Test last, mm; pass |',
        '| --- | ---: | ---: | --- | --- | --- | --- |']
    for name, s in summaries.items():
        text.append(f"| {LABELS[name]} | {s['parameter_count']} | {s['best_steps']} | {fmt(s['best_validation'])} | {fmt(s['history'][-1]['validation'])} | {fmt(s['best_test'])} | {fmt(s['last_test'])} |")
    text.extend(['', 'RMSE — среднее по случаям RMSE последней секунды; pass: в ≥90% этой секунды',
        'ошибка≤5 mm и скорость≤0,03 m/s, без physical failure. Params включают predictor.',
        f"У predictor {summaries['online']['predictor_parameter_count']} параметров. Архитектура собственных PPO-ветвей одинакова.", '',
        '### Прогноз физических состояний', '',
        '| Ветка / данные | q20, mm | q60, mm | v20, mm/s | v60, mm/s |',
        '| --- | ---: | ---: | ---: | ---: |'] )
    for name in NAMES:
        s = summaries[name]
        for key, label in (('prediction_evaluation', 'fixed validation'), ('policy_prediction_evaluation', 'собственная validation')):
            p = s[key]
            text.append(f"| {LABELS[name]} / {label} | {p['20']['position_rmse_mm']:.4f} | {p['60']['position_rmse_mm']:.4f} | {p['20']['velocity_rmse_mm_s']:.3f} | {p['60']['velocity_rmse_mm_s']:.3f} |")
    p = summaries['online']['persistence']['fixed']
    text.extend(['', f"Контроль постоянного q/v на fixed validation: q20={p['20']['position_rmse_mm']:.4f} mm, q60={p['60']['position_rmse_mm']:.4f} mm;",
        f"v20={p['20']['velocity_rmse_mm_s']:.3f} mm/s, v60={p['60']['velocity_rmse_mm_s']:.3f} mm/s.",
        'Fixed validation одинакова у ветвей; собственные validation-траектории различаются по policy.',
        'Frozen-head не обучалась: это исходный прогноз текущего состояния плюс среднее train-приращение.',
        'Предиктор не учился на этих validation-эпизодах; predictor loss использует только свои train-rollout.',
        f"Jacobian скорости на 60 ms: relative RMSE к нулевому прогнозу {summaries['online']['jacobian']['relative_rmse_to_zero']:.4f}, cosine {summaries['online']['jacobian']['cosine']:.4f}.",
        'Это те же 6 почти установившихся native-проб из прошлого отчёта, не проверка всех движений.', '',
        '### Сдвиг политики от обновления предиктора', '',
        '| Ветка | PPO accepted/rejected | Predictor updates | Predictor KL mean/max | Total KL mean/max | Total>0,03 | Rollout без PPO update |',
        '| --- | --- | ---: | --- | --- | ---: | ---: |'])
    for name, s in summaries.items():
        d = s['drift']; p = d['predictor_phase_kl']; total = d['total_phase_kl']
        text.append(f"| {LABELS[name]} | {s['accepted_optimizer_steps']}/{s['rejected_optimizer_steps']} | {s['predictor_steps']} | {p['mean']:.6g}/{p['max']:.6g} | {total['mean']:.6g}/{total['max']:.6g} | {d['total_phase_above_ppo_limit']} | {d['rollouts_no_ppo_update']} |")
    text.extend(['', 'KL — средний аналитический KL(old Beta || new Beta) на всём соответствующем rollout.',
        'Total KL сравнивает policy до PPO и после обеих фаз; это не сумма двух KL.',
        'Точный guard≤0,03 ограничивает только PPO-фазу; stop_gradient не фиксирует policy при смене признаков.', '',
        '### Используют ли heads переменную часть h1', '',
        '| Ветка | KL при замене h1 средним | Среднее изменение возбуждений | RMSE изменения V | Средний STD признака |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for name, s in summaries.items():
        f = s['feature_use']
        text.append(f"| {LABELS[name]} | {f['policy_kl']:.6g} | {f['mean_action_abs_change']:.6g} | {f['value_rmse_change']:.6g} | {f['mean_feature_std']:.6g} |")
    text.extend(['', 'Read-only диагностика на fixed validation: h1 заменён его средним вектором по этим 1800 входам.',
        'Исходные 13 входов собственных ветвей сохранены. Это чувствительность голов к вариациям h1,',
        'не отдельное переобучение и не доказательство вклада в качество. В обычной оценке h1 не заменяется.', '',
        '### Движения на test: native replay с шагом 1 ms', '',
        '| Движение mm | Фиксированные: steady mm / перелёт mm / установление s | Обучаемые: steady mm / перелёт mm / установление s |',
        '| --- | --- | --- |'])
    for left, right in zip(summaries['frozen']['motion'], summaries['online']['motion'], strict=True):
        cells = []
        for r in (left, right):
            settling = 'нет' if r['settling_time_s'] is None else f"{r['settling_time_s']:.3f}"
            cells.append(f"{r['last_second_position_rmse_m']*1000:.3f} / {r['overshoot_m']*1000:.3f} / {settling}")
        text.append(f"| {left['initial_position_m']*1000:+g}→{left['target_m']*1000:+g} | {' | '.join(cells)} |")
    text.extend(['', 'Установление: после этого момента ВСЕ оставшиеся физические samples входят в допуск',
        '5 mm/0,03 m/s до 4 s. Отсутствие перелёта само по себе не означает достижения цели.', '',
        '### Независимая проверка критика', '',
        '| Ветка | MC RMSE | Bias | Explained variance |',
        '| --- | ---: | ---: | ---: |'])
    for name, c in critics.items():
        text.append(f"| {LABELS[name]} | {c['rmse_against_mc_means']:.4f} | {c['mean_bias']:.4f} | {c['explained_variance_against_mc_means']:.4f} |")
    text.extend(['', 'Те же 12 физических snapshots, что в предыдущем опыте; по 6 stochastic-продолжений 16 s.',
        'Returns соответствуют собственной stochastic policy каждой ветви; это разные целевые значения.',
        'Малая MC-выборка не доказывает точность на всём пространстве состояний.', ''])
    report = '\n'.join(text)
    path = Path('docs/detached-predictor-ppo.md')
    previous = path.read_text()
    conclusion = previous.partition('## Интерпретация')[2]
    path.write_text(previous.split('## Результаты')[0]+report+('\n## Интерпретация'+conclusion if conclusion else ''))
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, s in summaries.items():
        steps = [r['steps'] for r in s['history']]
        axes[0].plot(steps, [r['validation']['score_m']*1000 for r in s['history']], label=LABELS[name])
        axes[1].plot(steps, [r['validation']['passed'] for r in s['history']], label=LABELS[name])
        a = np.asarray(json.loads((ROOT/name/'best_test.json').read_text())['cases'][0]['samples'])
        axes[2].plot(a[:, 0], a[:, 1]*1000, label=LABELS[name])
    axes[0].set(xlabel='Transitions', ylabel='Steady RMSE, mm', yscale='log', title='Validation (9 cases)')
    axes[1].set(xlabel='Transitions', ylabel='Passed / 9', title='Validation success')
    axes[2].axhline(60, color='black', lw=.8)
    axes[2].set(xlabel='Time, s', ylabel='Position, mm', title='Test reach -30 to +60 mm')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(ROOT/f'comparison.{ext}', dpi=160)
    plt.close(fig)
    print('Updated docs/detached-predictor-ppo.md; exported PNG/SVG')


if __name__ == '__main__':
    main()
