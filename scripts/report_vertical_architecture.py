"""Tables by varied factor and standalone plots for the native PPO trials."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    from myo.train_vertical_architecture import TRIALS
    root = Path("runs/vertical_architecture")
    audit = json.loads((root/"independent_evaluation.json").read_text())
    assert audit["status"] == "passed"
    summaries = {name: json.loads((root/name/"summary.json").read_text()) for name in audit["trials"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name, summary in summaries.items():
        h = summary["history"]
        axes[0, 0].plot([r["steps"] for r in h], [r["validation"]["score_m"]*1000 for r in h], "o-", label=name)
        axes[0, 1].plot([r["steps"] for r in h], [r["validation"]["passed"] for r in h], "o-", label=name)
        cases = json.loads((root/name/"test.json").read_text())["cases"]
        for ax, case_index in ((axes[1, 0], 0), (axes[1, 1], 2)):
            samples = np.asarray(cases[case_index]["samples"])
            if name == "control":
                ax.plot(samples[:, 0], samples[:, 3]*1000, "k--", label="Target")
            ax.plot(samples[:, 0], samples[:, 1]*1000, label=name)
    axes[0, 0].set(title="Validation: last-second error / failure penalty", xlabel="Native transitions", ylabel="mm", yscale="log")
    axes[0, 1].set(title="Validation: settled cases", xlabel="Native transitions", ylabel="Passed / 5", yticks=range(6))
    axes[1, 0].set(title="Held-out reach -30 to +60 mm: best checkpoint", xlabel="Time, s", ylabel="Position, mm")
    axes[1, 1].set(title="Held-out hold -30 mm: best checkpoint", xlabel="Time, s", ylabel="Position, mm")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root/f"comparison.{ext}", dpi=150)
    plt.close(fig)

    text = """# Активация, глубина и optimizer собственного PPO

Опыт начат 2026-11-06, завершён 2026-11-07. Вертикальный стенд, механика 4:
упругие мышца/сухожилие, transport delay 40 ms, activation dynamics 15/50 ms.
Это [наше позиционирование](vertical-ppo.md), а не imitation из MuscleMimic.

## Протокол

Все шесть моделей обучаются с нуля, seed=0, на одном цикле девяти train-заданий.
Контроль — отдельные actor/critic 2×128 Tanh и Adam. SiLU меняет только скрытую
активацию, deep — число скрытых слоёв до 4×128 с Tanh; LayerNorm не добавляется.
Пять неглубоких сетей начинают с побитово одинаковых весов. У deep другая форма
и больше параметров: этот опыт не отделяет эффект глубины от увеличения ёмкости.

На каждый опыт 131 072 transitions = 128 rollout по 1024 шага; batch=128,
до 10 эпох на rollout. Target KL=0,03 может остановить эпохи раньше, поэтому
реальное число optimizer steps измеряется отдельно. Controller 50 Hz, физика
1 kHz, эпизод 4 s. Бюджет фиксирован, остановки по плато/готовности нет.

Общая награда и наблюдения прежние: position scale=80 mm, velocity scale=0,4 m/s,
веса 0,85/0,15, effort/change penalties, training multiplier=0,1.
Beta-распределение и прямые возбуждения [0,1] сохранены; teacher/predictor нет.
LR=0,0003; gamma=0,99; GAE lambda=0,95; clip=0,2; entropy_coef=0,001;
vf_coef=0,5; global grad norm≤0,5. Оптимизаторам отдельно LR не подбирается.

Adam/RAdam: betas=(0,9;0,999), eps=1e−5; RMSprop: alpha=0,99, eps=1e−5,
momentum=0, centered=false. Weight decay везде 0. Muon делегируется PyTorch:
матрицы — momentum=0,95, Nesterov, пять Newton–Schulz итераций и scaling
`original`; векторы — Adam с eps=1e−5. Это не утверждение численного равенства
с Optax Muon из предыдущего MuscleMimic-опыта. PyTorch CPU Muon использует
собственную реализацию и точность внутренних матричных вычислений.

Пять validation-сценариев проверяются в начале и каждые 8192 transitions.
Выбор best лексикографический: меньше физических срывов, больше успешно
установившихся сценариев, меньше средняя RMSE последней секунды. Для успеха
нужны |q−target|≤5 mm и |v|≤0,03 m/s хотя бы 90% последней секунды,
без физического срыва. Срыв получает selection score=300 mm.
Best всех шести моделей фиксируются до первых test-прогонов; test — пять
других пар начального/целевого положений, плюс отдельная смена команд без reset.

## Функция активации

"""

    def table(names):
        result = "| Опыт | Params actor+critic | Best step | Validation best, mm; pass | Validation last, mm; pass | Test best steady, mm; pass | Test best full, mm | Test last steady, mm; pass |\n| --- | ---: | ---: | --- | --- | --- | ---: | --- |\n"
        for name in names:
            s = summaries[name]
            v, last, test, last_test = s["best_validation"], s["history"][-1]["validation"], s["test"], s["last_test"]
            full = np.mean([c["position_rmse_m"] for c in test["cases"]])*1000
            result += (f"| {name} | {s['parameter_count']} | {s['best_steps']} | {v['score_m']*1000:.2f}; {v['passed']}/5 "
                f"| {last['score_m']*1000:.2f}; {last['passed']}/5 | {test['score_m']*1000:.2f}; {test['passed']}/5 "
                f"| {full:.2f} | {last_test['score_m']*1000:.2f}; {last_test['passed']}/5 |\n")
        return result

    text += table(("control", "silu"))
    text += "\n## Глубина\n\n"+table(("control", "deep"))
    text += "\n## Optimizer\n\n"+table(("control", "radam", "rmsprop", "muon"))
    if "combined" in summaries:
        text += "\n## Комбинация полезных факторов\n\nПосле завершения control/SiLU/deep/RMSprop каждый изменённый фактор выиграл\nу контроля по зафиксированному validation-критерию. До обращения к test\nдобавлен седьмой опыт: **4×128 + SiLU + RMSprop**, с нуля, seed=0, тот же\nбюджет и PPO. Это адаптивно выбранная комбинация, а не однофакторная строка.\nОснования выбора сохранены до обучения в `combination_protocol.json`.\n\n"
        text += table(("control", "silu", "deep", "rmsprop", "combined"))
    text += "\nSteady — средняя RMSE последней секунды по пяти случаям; при срыве здесь\nстоит штраф selection score, поэтому отдельно проверяются failures/pass. Full —\nсредняя RMSE всего четырёхсекундного достижения и удержания. Это разные метрики.\n\n"
    text += "| Test-сценарий: start → target, mm | "+" | ".join(summaries)+" |\n| --- |"+" ---: |"*len(summaries)+"\n"
    for i, case in enumerate(summaries["control"]["test"]["cases"]):
        text += f"| {case['initial_position_m']*1000:.0f} → {case['target_m']*1000:.0f} | "
        cells = []
        for s in summaries.values():
            c = s["test"]["cases"][i]
            value = c["last_second_rmse_m"]
            cells.append(f"{value*1000:.2f}; {'да' if c['passed'] else 'нет'}" if value is not None else "срыв")
        text += " | ".join(cells)+" |\n"
    text += "\nЯчейки — steady RMSE, mm; pass. Test не используется для выбора настроек.\n\n"
    text += """## Наблюдения по этому запуску

На test Muon уменьшил среднюю ошибку удержания с 3,99 до 2,44 mm, глубокая
сеть — до 2,92 mm. Для полной траектории достижения и удержания наименьшая
средняя ошибка у deep: 7,54 mm, затем Muon: 7,76 mm, контроль: 8,50 mm.
Комбинация не превзошла Muon по steady-ошибке. Одна замена на SiLU и одна
замена на RMSprop улучшили best validation, но не улучшили test в целом.

Ни одна модель не прошла все пять test-кейсов. У Muon, deep и комбинации
единственный провал — удержание −30 mm: устойчивое смещение составляет
соответственно 5,21 / 5,39 / 5,32 mm при пороге 5 mm. Это небольшой, но
систематический остаточный сдвиг, а не потеря способности двигаться.
У SiLU и RMSprop удержания точнее, зато хуже достижение новых целей.

SiLU/RMSprop/combined заметно теряют качество после лучших checkpoints;
последняя сеть не равнозначна лучшей. У Muon лучший checkpoint последний:
фиксированные 128 rollout не доказывают плато или завершённую сходимость.
Результат не устанавливает превосходство optimizer при отдельно подобранном LR.

"""
    diagnostic_path = root/"equilibrium_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        text += "## Проверка смещения равновесия — 2026-11-07\n\nВниз смещаются не все цели: Muon при +30 mm стабилизируется выше цели.\nОтдельные четырёхсекундные удержания с началом в целевой точке:\n\n| Цель, mm | Равновесие Muon, mm | Ошибка со знаком, mm |\n| ---: | ---: | ---: |\n"
        for row in diagnostic["policy_holds"]:
            text += f"| {row['target_m']*1000:.0f} | {row['mean_q_m']*1000:.2f} | {row['mean_error_mm']:+.2f} |\n"
        text += "\nПри цели −30 mm старты −40 и −20 mm приводят к тому же равновесию\n−35,211 mm; скорость в конце практически нулевая. Разность сил верхней\nи нижней мышцы в этой точке 9,81 N, равна весу груза 1 kg. Это устойчивая\nошибка положения. Постоянные сбалансированные возбуждения из native reset\nудерживают −60/−30/0/+30/+60 mm без значимого дрейфа за те же 4 s.\nФизика допускает точное удержание, положение не ограничено мышечной силой.\n\nTrain-hold заданы только на −60/0/+60 mm, validation-hold только на 0;\nпромежуточные положения — проверка интерполяции. Наблюдаемый рисунок ошибок\nсогласуется с неточной интерполяцией выученного статического управления,\nно причинная роль покрытия данных отдельно не проверялась. При ошибке 5 mm\nпотеря сырой position reward всего 0,003314 при максимуме общей reward около 1.\nСлабый стимул точности — ещё одна гипотеза; выгодность смещения из-за effort\nэтим не установлена. При постоянном возбуждении delay/activation уже не\nсоздают продолжающегося запаздывания, хотя влияют на обучение и переходы.\nНовых обучений, изменения reward или механики в этой диагностике не было.\n\n"
    text += "![Validation и отложенные reach/hold](../runs/vertical_architecture/comparison.png)\n\n## Время и проверки\n\n"
    text += "| Опыт | Training+validation до test, s | Learn без validation, s | Реальных optimizer steps | Последний rollout critic EV |\n| --- | ---: | ---: | ---: | ---: |\n"
    for name, s in summaries.items():
        ev = s["rollouts"][-1]["critic_ev"]
        ev_text = "—" if ev is None else f"{ev:.4f}"
        text += f"| {name} | {s['elapsed_s']:.1f} | {s['training_seconds']:.1f} | {s['optimizer_steps']} | {ev_text} |\n"
    benchmark = json.loads((root/"optimizer_benchmark.json").read_text())
    text += "\nОтдельный benchmark `optimizer.step`: одинаковые исходные 37 253 параметра,\nфиксированные синтетические градиенты, 10 шагов прогрева и 30 чередующихся\nзамеров каждого optimizer на одном CPU thread. Исключены forward/backprop,\nclipping, физика и подготовка градиентов.\n\n| Optimizer | Медиана step, ms | p95 step, ms |\n| --- | ---: | ---: |\n"
    for name, entry in benchmark["optimizers"].items():
        text += f"| {name} | {entry['median_seconds']*1000:.3f} | {entry['p95_seconds']*1000:.3f} |\n"
    precision = json.loads((root/"matmul_precision_benchmark.json").read_text())
    fp32 = precision["results"]["torch.float32"]["median_seconds"]
    bf16 = precision["results"]["torch.bfloat16"]["median_seconds"]
    text += (f"\nPyTorch сообщает CPU capability `{precision['cpu_capability']}`. Для одного "
             f"`128×128 @ transpose` float32 занимает {fp32*1000:.3f} ms, "
             f"bfloat16 — {bf16*1000:.3f} ms ({bf16/fp32:.2f}× медленнее). "
             "Реализация PyTorch Muon явно переводит матрицы NS в bfloat16.\n"
             "Следовательно, стоимость этого опыта отражает также выбор dtype/backend,\n"
             "а не только математическую сложность Muon. Отношение для одного GEMM\n"
             "не равно ожидаемому ускорению полного optimizer; NS в float32 здесь\n"
             "не обучался. Все сырые времена сохранены в двух benchmark JSON.\n")
    text += f"\nНезависимо проверены **{audit['strict_checkpoints']}** checkpoints initial/best/last "
    text += f"и **{audit['native_replays']}** точных native replay test-траекторий. "
    text += "Проверены source/physics/observation/action contracts, сохранение разных\noptimizer states, выбор best по validation и равенство исходных весов.\n\nИсходные шесть опытов работали параллельно на разных CPU cores, комбинация\nзапущена после завершения выбранных одиночных факторов; wall time отражает\nэтот запуск с общей нагрузкой/тепловым режимом, а не изолированный kernel benchmark.\nИзменение числа эпох из-за target KL тоже влияет на время. Один фиксированный\nseed и один LR дают предварительное сравнение; проигрыш здесь не устанавливает\nпредел метода. GUI, анатомическая точность и полный D1 этим опытом не проверяются.\n\n## Воспроизведение\n\n```bash\n.venv/bin/python3 -m myo.train_vertical_architecture --trial control --steps 131072 --eval-every 8192 --defer-test\n# Повторить с --trial silu, deep, radam, rmsprop, muon.\n# Комбинация — до первых test-прогонов:\n.venv/bin/python3 scripts/train_vertical_combined.py\n.venv/bin/python3 scripts/evaluate_vertical_architecture.py --include-combined\ntaskset -c 0 .venv/bin/python3 scripts/benchmark_vertical_native_optimizers.py\nPYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_architecture.py\n```\n\nВерсии пакетов, физический контракт, порядок входов, mapping действий, конфигурация\nсетей и полное состояние optimizer сохранены с checkpoint. Веса и rollout\nостаются в ignored `runs/vertical_architecture`.\n"
    Path("docs/vertical-architecture-ppo.md").write_text(text)


if __name__ == "__main__":
    main()
