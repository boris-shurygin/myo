"""Report the fixed-budget Adam/Muon experiment without selecting on test."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/vertical_mimic_adam")
    folders = {"Muon": Path("runs/vertical_mimic/variant_4"), "Adam": root/"variant_4"}
    summaries = {name: json.loads((folder/"summary.json").read_text()) for name, folder in folders.items()}
    benchmark = json.loads((root/"update_benchmark.json").read_text())
    assert json.loads((root/"independent_evaluation.json").read_text())["status"] == "passed"
    for key in ("contract", "parameter_count"):
        assert summaries["Adam"][key] == summaries["Muon"][key]
    # Optional logging fields and optimizer selection are the only differences.
    shared = set(summaries["Muon"]["training"])-{"optimizer"}
    for key in shared:
        assert summaries["Adam"]["training"][key] == summaries["Muon"]["training"][key], key
    assert summaries["Adam"]["initial_validation"] == summaries["Muon"]["initial_validation"]
    for summary in summaries.values():
        assert summary["status"] == "complete" and len(summary["history"]) == 128

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, summary in summaries.items():
        validations = [r for r in summary["history"] if "validation" in r]
        axes[0, 0].plot([0]+[r["update"] for r in validations],
            [1000*summary["initial_validation"]["q_rmse_m"]]+
            [1000*r["validation"]["q_rmse_m"] for r in validations], "o-", label=name)
        axes[0, 1].plot([r["update"] for r in summary["history"]],
                        [r["raw_reward_mean"] for r in summary["history"]], label=name)
        data = json.loads((folders[name]/"test.json").read_text())
        for ax, index in ((axes[1, 0], 0), (axes[1, 1], 4)):
            samples = np.asarray(data["cases"][index]["samples"])
            if name == "Muon":
                ax.plot(samples[:, 0], samples[:, 3]*1000, "k--", label="Reference")
            ax.plot(samples[:, 0], samples[:, 1]*1000, label=name)
    axes[0, 0].set(title="Validation: full clips", xlabel="PPO update", ylabel="Position RMSE, mm")
    axes[0, 1].set(title="Training reward before normalization", xlabel="PPO update", ylabel="Mean reward")
    axes[1, 0].set(title="Held-out sine: best checkpoint", xlabel="Time, s", ylabel="Position, mm")
    axes[1, 1].set(title="Held-out hold +20 mm: best checkpoint", xlabel="Time, s", ylabel="Position, mm")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(root/f"comparison.{extension}", dpi=150)
    plt.close(fig)

    text = """# Adam и Muon на контроллере MuscleMimic

Опыт 2026-11-06: меняем только optimizer на физическом варианте 4 — упругие
мышцы/сухожилия, motor delay 40 ms и динамика активации 15/50 ms.
Задача — imitation известных кинематических траекторий, как в
[основном опыте](vertical-musclemimic.md), а не команды ускорения.

Сохраняются actor/critic 4×256, SiLU/LayerNorm, seed=0, начальные веса и running
statistics, 128 сред, rollout 20, одна эпоха/32 minibatch, LR=0,0004,
нормализация, reward, GAE, observation/action и физика. На каждый optimizer:
128 PPO-итераций, 327 680 transitions, 4096 minibatch-обновлений.
Начальные параметры/statistics побитово совпали; initial validation также
совпадает целиком. Конфигурации и версии пакетов проверены отчётным скриптом.

Adam — исходный `build_optimizer("adamw", ..., weight_decay=0)`, эквивалентный
Adam: beta1=0,9, beta2=0,999, eps=1e−5, global grad norm ≤1.
Muon сохраняет исходные настройки: beta=0,95, Nesterov, пять Newton–Schulz
итераций, adaptive=false; weight decay=0, Adam fallback для векторов.
Для Adam отдельно LR не подбирался. Один seed не устанавливает превосходство
одного метода в общем случае.

## Точность

Best выбирается только по восьми полным validation-клипам (включая старты
с 35% клипа). Пять отложенных test-клипов оцениваются после окончания обучения.
Strict pass: полный клип, q RMSE ≤10 mm и v RMSE ≤0,05 m/s одновременно.
Средняя RMSE ниже — среднее по клипам, а не ошибка одной общей склейки.

| Optimizer | Best iteration | Best validation q, mm | Last validation q, mm | Test q, mm | Test v, m/s | Test strict | Completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for name, s in summaries.items():
        validations = [r["validation"]["selection_score_m"] for r in s["history"] if "validation" in r]
        best = min([s["initial_validation"]["selection_score_m"]]+validations)
        t = s["test"]
        text += (f"| {name} | {s['best_step']//2560} | {best*1000:.2f} | {validations[-1]*1000:.2f} "
                 f"| {t['q_rmse_m']*1000:.2f} | {t['v_rmse_m_s']:.4f} | {t['passed']}/5 | {t['completed']}/5 |\n")
    text += "\n| PPO iteration | Muon validation q, mm | Adam validation q, mm |\n| ---: | ---: | ---: |\n"
    mh = [r for r in summaries["Muon"]["history"] if "validation" in r]
    ah = [r for r in summaries["Adam"]["history"] if "validation" in r]
    for m, a in zip(mh, ah):
        assert m["update"] == a["update"]
        text += f"| {m['update']} | {1000*m['validation']['q_rmse_m']:.2f} | {1000*a['validation']['q_rmse_m']:.2f} |\n"
    text += "\n| Test clip | Muon q RMSE, mm | Adam q RMSE, mm | Muon strict | Adam strict |\n| --- | ---: | ---: | --- | --- |\n"
    for m, a in zip(summaries["Muon"]["test"]["cases"], summaries["Adam"]["test"]["cases"]):
        assert m["motion"] == a["motion"]
        text += (f"| {m['motion']['name']} | {1000*m['q_rmse_m']:.2f} | {1000*a['q_rmse_m']:.2f} "
                 f"| {'да' if m['passed'] else 'нет'} | {'да' if a['passed'] else 'нет'} |\n")
    progress = json.loads((folders["Adam"]/"test_progress.json").read_text())
    text += "\nПосле завершения обучения дополнительно сопоставлены initial/best/last Adam\nна тех же test-клипах. Эти оценки не влияли на остановку или выбор best.\n\n| Adam checkpoint | PPO iteration | Test q RMSE, mm | Test v RMSE, m/s |\n| --- | ---: | ---: | ---: |\n"
    for name in ("initial", "best", "last"):
        entry = progress[name]
        text += f"| {name} | {entry['steps']//2560} | {1000*entry['q_rmse_m']:.2f} | {entry['v_rmse_m_s']:.4f} |\n"
    text += "\nНа test есть улучшение относительно инициализации, но после validation-best\nухудшение подтверждается и на test. Это три контрольные точки, не полная\nкривая test по итерациям. Полезное управление частично выучено; продолжение\nдо 128 итераций его не улучшило. Обе policy не прошли строгую приёмку ни\nодного test-клипа. У Adam точнее test-hold, у Muon — четыре движущихся test-клипа.\n"
    text += "\n![Validation, reward и отложенные движения](../runs/vertical_mimic_adam/comparison.png)\n\n## Время\n\n"
    text += "Изолированный замер: один и тот же native rollout 20×128, одинаковые начальные\nвеса и statistics, полный PPO loss/backprop и 32 шага optimizer. Физика, сбор\nопыта и validation исключены; JAX-компиляция прогрета отдельно, ожидание результата\nявное. Шесть замеров каждого optimizer чередуются на тех же двух CPU threads.\n\n| Optimizer | Медиана полного PPO-update после прогрева, s |\n| --- | ---: |\n"
    for name in summaries:
        text += f"| {name} | {benchmark['optimizers'][name.lower()]['median_seconds']:.4f} |\n"
    text += f"\nВ этом замере PPO-update с Muon дороже в **{benchmark['muon_over_adam']:.2f} раза**.\n"
    text += "Это стоимость всей оптимизации с backprop, не отдельно Muon/Adam kernel.\n\n"
    text += "Во время замера native-аудит работал на других CPU cores; частоты и\nтепловой режим не фиксировались. Все шесть исходных времён каждого optimizer\nсохранены в `update_benchmark.json`; отношение — ориентир для этого CPU.\n\n"
    for name, summary in summaries.items():
        median = np.median([r["seconds"] for r in summary["history"][1:]])
        text += f"- {name}: весь опыт {summary['wall_seconds']:.1f} s; медиана rollout+update без первой итерации {median:.3f} s.\n"
    s = summaries["Adam"]
    collection = np.median([r["collection_seconds"] for r in s["history"][1:]])
    update = np.median([r["update_seconds"] for r in s["history"][1:]])
    text += (f"\nУ Adam медиана сбора опыта (forward, native физика, обмен NumPy/JAX) — {collection:.3f} s, "
             f"PPO-update — {update:.3f} s.\n")
    text += "Muon — прежний опыт с четырьмя одновременно обучаемыми вариантами; Adam\nзапускался отдельно (часть времени параллельно шли тесты на других CPU cores).\nПоэтому отношение полного wall time не является изолированной оценкой optimizer.\n\n## Воспроизведение\n\n```bash\n.venv/bin/python3 -m myo.train_vertical_mimic --variant 4 --optimizer adam --output-dir runs/vertical_mimic_adam/variant_4 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --seed 0 --cpu-threads 2\n.venv/bin/python3 scripts/evaluate_vertical_mimic.py --root runs/vertical_mimic_adam --variants 4 --checkpoint-progress\n.venv/bin/python3 scripts/benchmark_vertical_mimic_optimizers.py\nPYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_mimic_optimizers.py\n```\n\nCheckpoint сохраняет выбор и полное состояние optimizer. Совместимость исходных\nMuon-checkpoints сохранена по явно закреплённому прежнему hash trainer; остальные\nпроверки исходников/физики/наблюдений/весов остаются обязательными.\nВеса, benchmark и траектории остаются в ignored `runs/`. Новое GUI не проверялось.\n"
    Path("docs/vertical-mimic-optimizers.md").write_text(text)


if __name__ == "__main__":
    main()
