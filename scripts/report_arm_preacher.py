"""Generate the arm-control transfer comparison from completed local experiments."""
import json
from pathlib import Path

import numpy as np

from myo.arm_precision import TRAIN_CASES

ROOT = Path('runs/arm_preacher')
LABELS = {'none': 'Без признаков предиктора', 'frozen': 'С предобученным фиксированным h1'}


def main():
    results = {k: json.loads((ROOT/k/'summary.json').read_text()) for k in LABELS}
    assert all(s['status'] == 'complete' for s in results.values())
    assert json.loads((ROOT/'evaluation_audit.json').read_text())['status'] == 'passed'
    body = json.loads((ROOT/'body/training.json').read_text())
    data = json.loads((ROOT/'body/dataset.json').read_text())
    fmt = lambda r: f"{r['score_deg']:.3f}; {r['passed']}/{r['total']}"
    text = ['## Результаты первого блока', '',
        'Один seed, две собственные actor/critic-ветви с одинаковой инициализацией.',
        'Контроллеры не получали эталонные возбуждения эвристики. Best выбран по validation.', '',
        '| Ветка | Фактические переходы | Best шаг | Validation: °; pass | Test best: °; pass | Test last: °; pass |',
        '| --- | ---: | ---: | --- | --- | --- |']
    for name, s in results.items():
        text.append(f"| {LABELS[name]} | {s['steps']} | {s['best_steps']} | {fmt(s['best_validation'])} | {fmt(s['best_test'])} | {fmt(s['last_test'])} |")
    text += ['', 'RMSE относится к последней секунде каждого случая, затем усредняется по случаям.',
        'В общих списках есть знакомые опоры: validation 75→75°, test 15→15° и 20→75°.',
        'Для новых сочетаний старта и цели, исключая эти опоры:', '',
        '| Ветка | Новые validation: °; pass | Новые test best: °; pass |',
        '| --- | --- | --- |']
    def novel(result):
        rows = [r for r in result['cases'] if (r['initial_deg'], r['target_deg']) not in TRAIN_CASES]
        return f"{np.mean([r['tail_rmse_deg'] for r in rows]):.3f}; {sum(r['passed'] for r in rows)}/{len(rows)}"
    for name, s in results.items():
        text.append(f"| {LABELS[name]} | {novel(s['best_validation'])} | {novel(s['best_test'])} |")
    text += ['',
        f"Параметров каждой policy всего {results['none']['parameter_count']}; PPO-параметров {results['none']['ppo_parameter_count']}.",
        'В none encoder не вычисляется и h1 заменён нулями. Стоимость motor-проб и предобучения дополнительная.', '',
        '### Motor-пробы и предиктор', '',
        f"Собрано {len(data['episodes'])} эпизодов, failures {sum(e['failure'] for e in data['episodes'])}.",
        f"Train-переходов {body['train_transitions']}; predictor-updates {body['updates']}, выбран {body['best_update']}.",
        f"Остановка: `{body['stop_reason']}`. Split целыми эпизодами; failed-конечные пары маскированы.", '',
        '| Горизонт | RMSE угла, ° | RMSE скорости, °/s | Persistence угла, ° | Persistence скорости, °/s |',
        '| --- | ---: | ---: | ---: | ---: |']
    for h, m in body['test'].items():
        text.append(f"| {h} ms | {m['angle_rmse_deg']:.4f} | {m['velocity_rmse_deg_s']:.3f} | {m['persistence_angle_rmse_deg']:.4f} | {m['persistence_velocity_rmse_deg_s']:.3f} |")
    text += ['', 'Ошибки выше относятся к плавным motor-пробам. Последующая',
        '[проверка покрытия движений](arm-predictor-movement-coverage.md) выявила',
        'сдвиг распределения и рост ошибки угла20 до 0,31–0,33° на стохастических PPO-прогонах.', '',
        '### Сценарии test', '',
        '| Старт→цель, ° | Без h1: RMSE °; pass | Frozen h1: RMSE °; pass |',
        '| --- | --- | --- |']
    for left, right in zip(results['none']['best_test']['cases'], results['frozen']['best_test']['cases'], strict=True):
        text.append(f"| {left['initial_deg']}→{left['target_deg']} | {left['tail_rmse_deg']:.3f}; {'да' if left['passed'] else 'нет'} | {right['tail_rmse_deg']:.3f}; {'да' if right['passed'] else 'нет'} |")
    text += ['', '### Движения и обновления', '',
        '| Ветка | Движение, ° | Перелёт 20ms, ° | Установление 20ms, s |',
        '| --- | --- | ---: | ---: |']
    for name, s in results.items():
        for r in s['best_test']['cases']:
            if r['initial_deg'] == r['target_deg']: continue
            settling = 'нет' if r['settling_time_s_20ms'] is None else f"{r['settling_time_s_20ms']:.3f}"
            text.append(f"| {LABELS[name]} | {r['initial_deg']}→{r['target_deg']} | {r['overshoot_deg_20ms']:.3f} | {settling} |")
    for name, s in results.items():
        updates = s['updates']
        text += ['', f"{LABELS[name]}: actual optimizer steps {s['optimizer_steps']}, stop `{s['stop_reason']}`;",
            f"последний rollout critic EV {updates[-1]['explained_variance']:.4f}, approximate KL {updates[-1]['approx_kl']:.5f}."]
    text += ['', 'Explained variance к bootstrap-target не заменяет независимую MC-проверку critic.',
        'Перелёты здесь измерены на 20-ms кадрах, не на каждом физическом подшаге.', '']
    report = '\n'.join(text)
    path = Path('docs/arm-preacher-training.md')
    previous = path.read_text(); suffix = previous.partition('## Интерпретация')[2]
    prefix = previous.split('## Результаты первого блока')[0]
    path.write_text(prefix+report+('\n## Интерпретация'+suffix if suffix else ''))
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, s in results.items():
        x = [h['steps'] for h in s['history']]
        axes[0].plot(x, [h['validation']['score_deg'] for h in s['history']], label=LABELS[name])
        axes[1].plot(x, [h['validation']['passed'] for h in s['history']], label=LABELS[name])
    history = body['history']
    for key in ('train', 'validation'):
        axes[2].plot([h['update'] for h in history], [h[key]['normalized_mse'] for h in history], label=key)
    axes[0].set(xlabel='PPO transitions', ylabel='Steady RMSE, deg', yscale='log', title='Control validation')
    axes[1].set(xlabel='PPO transitions', ylabel='Passed / 9', title='Control success')
    axes[2].set(xlabel='Predictor updates', ylabel='Normalized delta MSE', yscale='log', title='Body predictor')
    axes[0].legend(fontsize=7); axes[2].legend(); fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(ROOT/f'comparison.{ext}', dpi=160)
    plt.close(fig)
    moving = {}
    for name in results:
        cases = json.loads((ROOT/name/'best_test.json').read_text())['cases']
        moving[name] = [c for c in cases if c['initial_deg'] != c['target_deg']]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for name in results:
        for ax, case in zip(axes.flat, moving[name]):
            ax.plot([r[0] for r in case['samples']], [r[1] for r in case['samples']], label=LABELS[name])
            if name == 'none':
                ax.axhline(case['target_deg'], color='black', ls='--', lw=.8)
                ax.set(title=f"{case['initial_deg']} -> {case['target_deg']} deg", xlabel='Time, s', ylabel='Angle, deg')
    axes.flat[-1].axis('off'); axes[0, 0].legend(fontsize=7); fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(ROOT/f'movement.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
