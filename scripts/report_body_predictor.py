"""Summarize standalone motor-exploration training from saved native measurements."""
import json
from pathlib import Path

import numpy as np


ROOT = Path('runs/body_predictor')
LABELS = {'separate_best': 'Отдельный best', 'separate_last': 'Отдельный last',
          'parallel_actor_best': 'Прежний actor-best', 'parallel_last': 'Прежний parallel-last'}


def main():
    data = dict(np.load(ROOT/'dataset.npz'))
    metadata = json.loads((ROOT/'dataset.json').read_text())
    training = json.loads((ROOT/'training.json').read_text())
    evaluation = json.loads((ROOT/'evaluation.json').read_text())
    assert training['status'] == 'complete' and evaluation['status'] == 'passed'
    x = data['observations']
    text = ['## Результаты', '',
        '| Split | Эпизодов | Полных переходов | Physical failures |', '| --- | ---: | ---: | ---: |']
    for split in ('train', 'validation', 'test'):
        e = [e for e in metadata['episodes'] if e['split'] == split]
        text.append(f"| {split} | {len(e)} | {sum(v['complete_transitions'] for v in e)} | {sum(v['physical_failure'] for v in e)} |")
    text.extend(['', 'Наблюдаемое покрытие на control-кадрах (20 ms), не экстремумы всех физических подшагов:', '',
        f"- Положение груза: {x[:, 0].min()*.1:.4f}…{x[:, 0].max()*.1:.4f} m.",
        f"- Скорость груза: {x[:, 1].min()*.5:.4f}…{x[:, 1].max()*.5:.4f} m/s.",
        f"- Возбуждения: {data['actions'].min():.4f}…{data['actions'].max():.4f}.",
        f"- Активации: {(x[:, 4:6].min()+1)*.5:.4f}…{(x[:, 4:6].max()+1)*.5:.4f}.",
        f"- Приложенные tendon forces: {data['forces_N'].min():.3f}…{data['forces_N'].max():.3f} N.",
        f"- Положительное растяжение сухожилия: 0…{max(0., float(data['tendon_strain'].max())):.5f}.",
        f"- Минимальный signed геометрический strain: {data['tendon_strain'].min():.5f}; отрицательная часть означает слабину, не сжимающую силу.", '',
        '| Семейство | Median корреляции команд внутри train-эпизода |', '| --- | ---: |'])
    for family in ('independent', 'reciprocal', 'common', 'multisine'):
        correlations = [np.corrcoef(data['actions'][e['start']:e['stop']].T)[0, 1]
                        for e in metadata['episodes'] if e['split'] == 'train' and e['family'] == family]
        text.append(f"| {family} | {np.median(correlations):.4f} |")
    best, last = training['best_update'], training['updates']
    text.extend(['', '### Обучение и плато', '',
        f"Остановка: `{training['stop_reason']}`, update **{last}**, best **{best}**.",
        f"Исходных train-переходов {training['unique_train_transitions']}; предъявлений с повторами {training['sampled_transitions']}.",
        'Ни архитектура, ни optimizer, ни learning rate не менялись по ходу обучения.', '',
        '| Update | Train one-step MSE | Validation one-step MSE | Val q20, mm | Val v20, mm/s |',
        '| ---: | ---: | ---: | ---: | ---: |'])
    chosen = {0, best, last, 1000, 5000, 10000}
    for row in training['history']:
        if row['update'] in chosen:
            v = row['validation']
            text.append(f"| {row['update']} | {row['train']['normalized_mse']:.6g} | {v['normalized_mse']:.6g} | {v['q_rmse_mm']:.4f} | {v['v_rmse_mm_s']:.3f} |")
    text.extend(['', '### Общий новый test после выбора', '',
        '| Модель | q20, mm | v20, mm/s | q60, mm | v60, mm/s | q120, mm | v120, mm/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for label, name in LABELS.items():
        result = evaluation['metrics'][label]['test']
        values = [value for horizon in ('20', '60', '120') for value in (result[horizon]['q_rmse_mm'], result[horizon]['v_rmse_mm_s'])]
        text.append(f"| {name} | {' | '.join(f'{v:.4f}' for v in values)} |")
    persistence = evaluation['metrics']['separate_best']['test']
    values = [value*1000 for horizon in ('20', '60', '120') for value in persistence[horizon]['persistence_physical_channel_rmse'][:2]]
    text.append(f"| Постоянный q/v | {' | '.join(f'{v:.4f}' for v in values)} |")
    linear = {}
    for h in (1, 2, 3, 6):
        starts = np.concatenate([np.arange(e['start'], e['stop']-h+1) for e in metadata['episodes'] if e['split'] == 'test'])
        initial = data['observations'][starts].astype(np.float64)
        actual = data['next_observations'][starts+h-1].astype(np.float64)
        error = (initial[:, 0]-actual[:, 0])*.1+initial[:, 1]*.5*(.02*h)
        linear[str(h*20)] = {'q_rmse_mm': float(np.sqrt(np.mean(error**2))*1000),
            'v_rmse_mm_s': persistence[str(h*20)]['persistence_physical_channel_rmse'][1]*1000}
    values = [value for horizon in ('20', '60', '120') for value in linear[horizon].values()]
    text.append(f"| Постоянная скорость: q+v·t | {' | '.join(f'{v:.4f}' for v in values)} |")
    (ROOT/'simple_baselines.json').write_text(json.dumps({'constant_velocity_test': linear}, indent=2)+'\n')
    text.extend(['', 'Все модели проверены на одних и тех же началах forecast, известных командах и истинных labels.',
        'Прежние parallel-веса не менялись и не участвовали в выборе нового best.', '',
        '### Ошибка относительно изменения за один шаг', '',
        '| Канал | Новый test RMSE, физические единицы | Постоянный прогноз RMSE | Отношение |',
        '| --- | ---: | ---: | ---: |'])
    m = persistence['20']
    names = ('q, m', 'v, m/s', 'верхнее волокно, m', 'нижнее волокно, m', 'верхняя активация', 'нижняя активация')
    for i, name in enumerate(names):
        text.append(f"| {name} | {m['physical_channel_rmse'][i]:.6g} | {m['persistence_physical_channel_rmse'][i]:.6g} | {m['relative_channel_rmse_to_persistence'][i]:.4f} |")
    text.extend(['', 'Знаменатель — RMS фактического изменения канала за20 ms, а не полный диапазон ощущений.',
        'Меньше1 означает точнее постоянного прогноза. Для активаций единицы безразмерные.', '',
        '### Test по семействам: отдельный best', '',
        '| Семейство | q20, mm | v20, mm/s | q60, mm | v60, mm/s |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for family, row in evaluation['metrics']['separate_best']['test_families'].items():
        values = [value for h in ('20', '60') for value in (row[h]['q_rmse_mm'], row[h]['v_rmse_mm_s'])]
        text.append(f"| {family} | {' | '.join(f'{v:.4f}' for v in values)} |")
    text.extend(['', '### Перенос на прежние fixed-policy validation-движения', '',
        '| Модель | q20, mm | v20, mm/s | q60, mm | v60, mm/s |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for label, row in evaluation['old_fixed_policy_transfer'].items():
        values = [value for h in ('20', '60') for value in (row[h]['q_rmse_mm'], row[h]['v_rmse_mm_s'])]
        text.append(f"| {LABELS[label]} | {' | '.join(f'{v:.4f}' for v in values)} |")
    j = evaluation['native_sensitivities']
    text.extend(['', '### Чувствительность будущей скорости к новой команде', '',
        f"Native replay точный; **{j['probes']}** состояний на60 ms, позиции {min(j['positions_m']):.4f}…{max(j['positions_m']):.4f} m,",
        f"скорости {min(j['velocities_m_s']):.4f}…{max(j['velocities_m_s']):.4f} m/s.", '',
        '| Модель | RMS ошибки Jacobian / RMS native Jacobian | Cosine | Проекция на native |', '| --- | ---: | ---: | ---: |'])
    native = np.asarray(j['native'])
    for label, result in j['models'].items():
        gain = np.sum(np.asarray(result['predicted'])*native)/np.sum(native*native)
        text.append(f"| {LABELS[label]} | {result['relative_rmse_to_zero']:.4f} | {result['cosine']:.4f} | {gain:.4f} |")
    text.extend(['', 'Это независимые от предыдущих шестипробных аудитов состояния: прямые численные сравнения',
        'процентов между разными наборами проб некорректны. Здесь все четыре модели имеют общие native-эталоны.',
        'Проекция = dot(J_pred,J_native)/dot(J_native,J_native); у точного масштаба1.',
        'Проб мало: это первые test-эпизоды с амплитудой0,25, а не весь диапазон данных.',
        'Качество h1 для будущего PPO этим опытом не измерено; новый actor не обучался.', ''])
    report = '\n'.join(text)
    doc = Path('docs/body-predictor-pretraining.md')
    previous = doc.read_text()
    suffix = previous.partition('## Интерпретация')[2]
    doc.write_text(previous.split('## Результаты')[0]+report+('\n## Интерпретация'+suffix if suffix else ''))
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    h = training['history']; updates = [r['update'] for r in h]
    axes[0, 0].plot(updates, [r['train']['normalized_mse'] for r in h], label='Train')
    axes[0, 0].plot(updates, [r['validation']['normalized_mse'] for r in h], label='Validation')
    axes[0, 0].set(yscale='log', xlabel='Updates', ylabel='One-step normalized MSE')
    axes[0, 0].axvline(best, color='gray', linestyle='--'); axes[0, 0].legend()
    axes[0, 1].plot(updates, [r['validation']['q_rmse_mm'] for r in h])
    axes[0, 1].set(yscale='log', xlabel='Updates', ylabel='Validation q20 RMSE, mm')
    axes[1, 0].plot(updates, [r['validation']['v_rmse_mm_s'] for r in h])
    axes[1, 0].set(yscale='log', xlabel='Updates', ylabel='Validation v20 RMSE, mm/s')
    horizons = [20, 40, 60, 120]
    for label in LABELS:
        m = evaluation['metrics'][label]['test']
        axes[1, 1].plot(horizons, [m[str(t)]['v_rmse_mm_s'] for t in horizons], marker='o', label=LABELS[label])
    axes[1, 1].set(yscale='log', xlabel='Forecast horizon, ms', ylabel='Test velocity RMSE, mm/s')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT/f'learning_and_forecast.{extension}', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for column, family in enumerate(('independent', 'reciprocal', 'common', 'multisine')):
        episode = next(e for e in metadata['episodes'] if e['split'] == 'test' and e['family'] == family)
        segment = slice(episode['start'], episode['stop'])
        t = np.arange(episode['complete_transitions'])*.02
        axes[0, column].plot(t, data['actions'][segment, 0], label='Upper')
        axes[0, column].plot(t, data['actions'][segment, 1], label='Lower')
        axes[0, column].set(title=family, ylabel='Excitation')
        axes[1, column].plot(t, data['observations'][segment, 0]*100)
        axes[1, column].set(xlabel='Time, s', ylabel='Position, mm')
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(ROOT/f'motor_examples.{extension}', dpi=150)
    plt.close(fig)
    print('Updated docs/body-predictor-pretraining.md; exported plots')


if __name__ == '__main__':
    main()
