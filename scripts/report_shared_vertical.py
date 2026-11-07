"""Write the shared-encoder comparison from completed, frozen evaluations."""
import json
from pathlib import Path

import numpy as np


ROOT = Path('runs/shared_vertical')
NAMES = ('independent', 'shared', 'shared_prediction')
LABELS = {'independent': 'Раздельные', 'shared': 'Общий', 'shared_prediction': 'Общий + прогноз'}


def main():
    summaries = {n: json.loads((ROOT/n/'summary.json').read_text()) for n in NAMES}
    critics = {n: json.loads((ROOT/n/'critic_audit.json').read_text()) for n in NAMES}
    assert len({c['state_observations_sha256'] for c in critics.values()}) == 1
    text = ['## Результаты', '',
        'Завершены три обучения по 131072 transitions, всего 393216; остановка по бюджету.',
        'Выбор best завершён до test. Ни optimizer, ни коэффициент auxiliary-loss по test не подбирались.', '',
        '| Ветка | Params | Best шаг | Val best, mm; pass | Val last, mm; pass | Test best, mm; pass | Test last, mm; pass |',
        '| --- | ---: | ---: | --- | --- | --- | --- |']
    fmt = lambda r: f"{r['score_m']*1000:.3f}; {r['passed']}/{r['total']}"
    for name, s in summaries.items():
        text.append(f"| {LABELS[name]} | {s['parameter_count']} | {s['best_steps']} | {fmt(s['best_validation'])} | {fmt(s['history'][-1]['validation'])} | {fmt(s['best_test'])} | {fmt(s['last_test'])} |")
    text.extend(['', 'Test здесь — расширенные 11 случаев. RMSE — средняя RMSE последней секунды.',
        'Необучаемая predictor-head включена в Params обоих shared-вариантов, но у `shared` не обновляется.', '',
        '| Ветка | Принято optimizer steps | Отменено | Максимальный принятый KL | Wall time train+validation, s |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for name, s in summaries.items():
        text.append(f"| {LABELS[name]} | {s['accepted_optimizer_steps']} | {s['rejected_optimizer_steps']} | {max(u['max_accepted_kl'] for u in s['updates']):.6f} | {s['elapsed_s']:.1f} |")
    text.extend(['', 'Параллельные CPU-прогоны и разное число принятых updates не являются чистым сравнением скорости.', '',
        '### Движение', '',
        '| Старт→цель mm | Перелёт раздельные / общий / +прогноз, mm | Установление, s |',
        '| --- | --- | --- |'])
    for i, reference in enumerate(summaries['independent']['motion']):
        rows = [summaries[n]['motion'][i] for n in NAMES]
        overshoot = ' / '.join(f"{r['overshoot_m']*1000:.2f}" for r in rows)
        settle = ' / '.join(f"{r['settling_time_s']:.3f}" if r['settling_time_s'] is not None else 'нет' for r in rows)
        text.append(f"| {reference['initial_position_m']*1000:+.0f}→{reference['target_m']*1000:+.0f} | {overshoot} | {settle} |")
    text.extend(['', 'Установление: ≤5 mm и ≤0,03 m/s на всех оставшихся физических отсчётах до конца 4 s.', '',
        '### Критик: независимый Monte Carlo', '',
        'У всех трёх одинаковые 12 физических начальных состояний: они сформированы старым',
        'precision best, совпадение SHA наблюдений проверено. Продолжения выполняет уже',
        'собственная stochastic policy каждой модели: её V и оценивается. Шесть повторов по',
        '16 s, gamma=0,99, без critic-bootstrap. Returns включают training multiplier=0,1.', '',
        '| Ветка | RMSE против средних MC | Среднее смещение | EV | Диапазон средних returns |',
        '| --- | ---: | ---: | ---: | --- |'])
    for name, c in critics.items():
        text.append(f"| {LABELS[name]} | {c['rmse_against_mc_means']:.3f} | {c['mean_bias']:+.3f} | {c['explained_variance_against_mc_means']:.3f} | {c['mc_mean_range'][0]:.3f}–{c['mc_mean_range'][1]:.3f} |")
    text.extend(['', 'Это малая выборка с шумом MC, не оценка всего пространства; качество critic разных',
        'policy измеряется относительно разных собственных будущих returns.', '',
        '### Предиктор', '',
        '| Ветка | q RMSE 20 ms, mm | q RMSE 60 ms, mm | v RMSE 60 ms, mm/s |',
        '| --- | ---: | ---: | ---: |'])
    for name in ('shared', 'shared_prediction'):
        result = summaries[name]['prediction_evaluation']
        text.append(f"| {LABELS[name]} | {result['20']['position_rmse_mm']:.4f} | {result['60']['position_rmse_mm']:.4f} | {result['60']['velocity_rmse_mm_s']:.3f} |")
    jac = summaries['shared_prediction']['jacobian']
    text.extend(['', 'У `shared` predictor-head не обучалась: это контроль начального перехода, а не',
        'сопоставимый отдельно обученный predictor. Все forecast-метрики на одном фиксированном',
        'наборе validation, не на test и не на идентичных по распределению новых actor-траекториях.',
        f"У обучаемой головы ошибка Jacobian скорости на 60 ms относительно нулевого: {jac['relative_rmse_to_zero']:.3f}; cosine {jac['cosine']:.3f}, шесть native-проб.", '',
        '### Перенос по массе без обучения', '',
        '| Масса kg | Раздельные: mm; pass | Общий: mm; pass | Общий + прогноз: mm; pass |',
        '| ---: | --- | --- | --- |'])
    masses = {n: json.loads((ROOT/n/'mass_audit.json').read_text()) for n in NAMES}
    for i, row in enumerate(masses['independent']['rows']):
        values = [fmt(masses[n]['rows'][i]) for n in NAMES]
        text.append(f"| {row['mass_kg']:g} | {' | '.join(values)} |")
    text.extend(['', 'Масса меняется только между эпизодами, начальный преднатяг сбалансирован под неё.',
        'Это перенос frozen policy с 1 kg, а не обучение универсальности по массе.', ''])
    report = '\n'.join(text)
    doc = Path('docs/shared-encoder-ppo.md')
    previous = doc.read_text()
    conclusion = previous.partition('## Интерпретация результатов')[2]
    doc.write_text(previous.split('## Результаты')[0]+report+('\n## Интерпретация результатов'+conclusion if conclusion else ''))
    (ROOT/'report.md').write_text(report)
    print(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, s in summaries.items():
        steps = [h['steps'] for h in s['history']]
        axes[0].plot(steps, [h['validation']['score_m']*1000 for h in s['history']], label=LABELS[name])
        axes[1].plot(steps, [h['validation']['passed'] for h in s['history']], label=LABELS[name])
        c = s['best_test']['cases'][0]
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


if __name__ == '__main__':
    main()
