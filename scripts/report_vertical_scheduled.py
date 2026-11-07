"""Report the actual scheduled-acceleration PPO experiment in Russian."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metric(result):
    return f"{result['passed']}/{result['total']}; {result['score_m_s2']:.3f}; {result['failures']}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_scheduled_ppo"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    assert summary["status"] == "complete" and audit["status"] == "passed"
    lines = ["# Команда ускорения со сроком исполнения через 40 ms", "",
             "Опыт 2026-11-06 на вертикальном стенде: низкоуровневый PPO получает непосредственно",
             "заданное ускорение будущего интервала. Четыре отдельные policy с нуля.", "",
             "## Время задания и награды", "",
             "При шаге контроллера 20 ms команда `x`, выданная в момент t, относится к среднему",
             "ускорению на интервале **t+40…t+60 ms**. Награда появляется после окончания этого",
             "интервала, в t+60 ms. Это измерение среднего за интервал, не мгновенного ускорения",
             "в точке t+40 ms и не обещание полной активации мышцы за 40 ms.", "",
             "| Момент выбора мышечной команды | Три известных задания во входе | Что сейчас оценивается |",
             "| --- | --- | --- |",
             "| t | x[t−40], x[t−20], x[t] | Среднее a за t…t+20 против x[t−40] |",
             "| t+20 | x[t−20], x[t], x[t+20] | Среднее a за t+20…t+40 против x[t−20] |",
             "| t+40 | x[t], x[t+20], x[t+40] | Среднее a за t+40…t+60 против x[t] |", "",
             "Во входе только уже выданные задания. Полный сценарий хранится в генераторе среды,",
             "не передаётся policy; невыпущенная часть не влияет на текущее observation.",
             "Очередь заданий отдельна от моторной очереди. Первые два интервала после reset",
             "имеют начальное требование a=0; последние два интервала дополняют эпизод, чтобы",
             "все выданные команды успели получить оценку. Первые два интервала не входят в",
             "метрики исполнения выданных команд, но их награда за начальное a=0 сохраняется.", "",
             "Физика интегрируется MuJoCo с шагом 1 ms. В механиках 1/2 нет моторной задержки;",
             "policy может выбирать возбуждение под действующее сейчас задание. В механиках 3/4",
             "сигнал проходит транспорт 40 ms, поэтому команда выбирается заранее. В 2/3/4",
             "сохранены упругие мышца и отдельное сухожилие, в 4 — активация 15/50 ms.",
             "Груз 1 kg, утомления нет; скрытых сил и коррекции q/v нет.", "",
             "## Сеть и обучение", "",
             "Два отдельных MLP actor/critic 128×128 Tanh. Прямые Beta-excitations двух мышц,",
             "без predictor, teacher actions и градиента через MuJoCo. Наблюдения:", "",
             "- прежние физические ощущения, очередь моторных команд и предыдущие возбуждения;",
             "- три задания ускорения на текущий интервал и через 20/40 ms, в m/s².", "",
             "Итого **7 / 9 / 13 / 15** входов у механик 1/2/3/4. Цели положения и PD-пересчёта",
             "нет. Нулевое ускорение допускает ненулевую скорость, поэтому бонус за остановку убран.", "",
             "```text", "a_actual = (v_after - v_before) / actual_elapsed_time",
             "error = a_actual - task_due_now",
             "r_raw = exp(-(error / 0.5m/s²)^2)",
             "      - 0.01 * mean(u²) - 0.002 * mean((u-u_previous)²)",
             "      - 1.0 * physical_failure", "r_training = 0.1 * r_raw", "```", "",
             "Награда остаётся на физическом шаге результата. PPO/GAE связывает будущую награду",
             "с ранней командой; ручного переноса reward к другому action и прямого backprop через",
             "моторную очередь нет. Critic учится по bootstrapped returns этой награды.", "",
             "Seed 0, CPU, один torch-поток на процесс; lr 0,0003, rollout 1024, batch 128,",
             "10 epochs, gamma 0,99, GAE lambda 0,95, clip 0,2, entropy 0,001, vf_coef 0,5,",
             "max_grad_norm 0,5, target KL 0,03. Старые checkpoint несовместимы по входам/контракту.", "",
             "## Программа команд", "",
             "Один цикл состоит из [A, −A, −A, A, 0], каждый элемент держится H шагов; два цикла",
             "на эпизод плюс два шага для опустошения очереди. Средние две одинаковые лобы дают",
             "непрерывное −A длительностью 2H. При идеальном исполнении цикл возвращает скорость",
             "и смещение к исходным. Это свойство задания; физика не подгоняет под него состояние.", "",
             "Train: A=±0,5 при H=5 (100 ms); A=±1 при H=8 (160 ms), начальные q=∓30 mm;",
             "A=±0,75 при H=10 (200 ms); один нулевой случай при H=8. Семь фиксированных случаев.",
             "Validation: A=±0,6/H=6 и ±0,9/H=9, плюс нулевой H=7.",
             "Test: A=±0,7/H=7 и ±0,8/H=10, плюс нулевой H=9; начальные q также отличаются.",
             "Все ненулевые задания содержат и положительные, и отрицательные ускорения.", "",
             "Остановка: минимум 32768, максимум 131072 перехода среды; проверки каждые 8192.",
             "Два подряд полных validation-успеха либо шесть проверок без существенного улучшения",
             "max(0,01 m/s², 2% anchor). Best выбирается по меньшему числу падений, полному",
             "прохождению программы и меньшему среднему acceleration RMSE; число отдельных",
             "успехов разрешает только ничью. Один нулевой успех не перекрывает прогресс движений. Test обученных policy вычислен после",
             "фиксации всех четырёх best. Best на конце бюджета не устанавливает плато.", "",
             "Профиль успешен при отсутствии падения, RMSE ≤0,25 m/s² и не менее 80% интервалов",
             "с |ошибкой|≤0,35 m/s². Все переходы команд входят в оценку. Physical failure — прежний",
             "выход за ±0,28 m; ему назначается score 3 m/s² вместо неполного RMSE.", "",
             "## Результаты", "",
             "Ячейка: **успешных случаев; mean RMSE/штрафной score в m/s²; физических падений**.", "",
             "| Механика | Train best | Validation best | Validation last | Test best | Best / всего шагов | Остановка |",
             "| ---: | --- | --- | --- | --- | --- | --- |"]
    for v in ("1", "2", "3", "4"):
        r, a = summary["variants"][v], audit["variants"][v]
        lines.append(f"| {v} | {metric(a['train'])} | {metric(r['selected_validation'])} | {metric(r['history'][-1]['validation'])} | "
                     f"{metric(r['test'])} | {r['best_steps']} / {r['trained_steps']} | {r['stop_reason']} |")
    active = [case for r in summary["variants"].values() for case in r["test"]["cases"]
              if case["case"]["amplitude_m_s2"] != 0.]
    lines += ["", f"Успешных ненулевых test-профилей: **{sum(c['passed'] for c in active)}/{len(active)}**.",
              "Рабочие очереди и верное вычисление награды не означают освоение управления.", ""]
    lines += ["", "## Диагностические контроллеры на тех же test-командах", "",
              "Constant support выдаёт постоянное исходное возбуждение для удержания веса; задания",
              "игнорирует. Force inverse приближённо переводит требуемую силу m·(g+a) в возбуждения",
              "по текущим длинам и скорости. Для задержанных механик использует задание, только что",
              "выданное на срок через 40 ms; для остальных — действующее задание. Будущую физику",
              "он не симулирует, активацию и compliance не инвертирует точно. Это ориентир, не оптимум.",
              "Ни один из них не размечает действия для PPO. Эти baseline предварительно проверялись",
              "на test-сценариях для оценки выполнимости; test обученных PPO до фиксации весов не вычислялся.", "",
              "| Механика | Constant support | Force inverse | PPO |", "| ---: | --- | --- | --- |"]
    for v in ("1", "2", "3", "4"):
        a = audit["variants"][v]
        lines.append(f"| {v} | {metric(a['constant_support'])} | {metric(a['force_inverse'])} | {metric(a['test'])} |")
    lines += ["", "## Реакция на величину команды", "",
              "На тех же test-интервалах оценивается линейный коэффициент реакции",
              "`gain = cov(a_actual, a_due) / var(a_due)` и средняя ошибка. Для точного исполнения",
              "gain близок к 1, ошибка близка к 0. У постоянного ускорения gain близок к 0.",
              "Это описание траекторий, не отдельная приёмка или доказательство механизма обучения.", "",
              "| Механика | Gain PPO | Средняя ошибка PPO, m/s² | Gain force inverse |",
              "| ---: | ---: | ---: | ---: |"]
    for v in ("1", "2", "3", "4"):
        values = []
        for name in ("test_replay", "force_inverse_test"):
            cases = json.loads((args.run/f"variant_{v}"/f"{name}.json").read_text())["cases"]
            data = np.concatenate([np.asarray(c["samples"])[2:, 4:6] for c in cases])
            due, actual = data[:, 0], data[:, 1]
            gain = float(np.mean((due-due.mean())*(actual-actual.mean()))/np.var(due))
            values.append((gain, float(np.mean(actual-due))))
        lines.append(f"| {v} | {values[0][0]:.3f} | {values[0][1]:.3f} | {values[1][0]:.3f} |")
    reused = sum(summary["variants"][v]["trained_steps"] for v in summary.get("reused_variants", {}))
    total = sum(r["trained_steps"] for r in summary["variants"].values())
    prior = Path("runs/vertical_scheduled_threshold_selection")
    prior_steps = sum(json.loads(p.read_text())["history"][-1]["steps"] for p in prior.glob("variant_*/summary.json"))
    lines += ["", "## Исправление отбора в ходе опыта", "",
              "В предварительном запуске приоритет одного нулевого успеха оставил выбранным checkpoint",
              "с RMSE 0,414. Вариант с меньшим RMSE 0,399 был отклонён: ни один его профиль ещё не уложился в порог.",
              "При этом ошибка четырёх движений уменьшилась. Критерий изменён до test обученных PPO:",
              "полное освоение остаётся приоритетом, а частичные policy сравниваются по непрерывной ошибке.",
              "Архив `runs/vertical_scheduled_threshold_selection/` содержит исходные checkpoint,",
              "журналы и точные прежние trainer/orchestrator sources; test в нём не вычислялся.", "",
              "Ветки 1/4 обучены заново. У веток 2/3 число успехов всегда было нулевым, поэтому",
              "порядок best и остановка при новом критерии совпадают с прежними; их завершённое",
              "обучение использовано повторно, исходные metadata сохранены. Сравнение общих префиксов",
              "подтвердило точное совпадение physical validation, PPO logs и rollout audits.", "",
              f"Предварительный запуск выполнил {prior_steps} переходов; финальный дополнительно {total-reused}.",
              f"Суммарно реально выполнено {prior_steps+total-reused} обучающих переходов, включая повтор префиксов.", "",
              "![Обучение и время команд](../runs/vertical_scheduled_ppo/scheduled_comparison.png)", "",
              "![Ошибки отдельных заданий](../runs/vertical_scheduled_ppo/scheduled_cases.png)", "",
              "## Проверки и границы вывода", "",
              f"Четыре итоговых trial содержат {total} переходов среды по 20 ms, из них {reused} переиспользованы.",
              f"Wall time финального запуска: {summary['elapsed_s']:.2f} s; до четырёх локальных процессов по одному torch-потоку.",
              f"Независимо загружены {audit['strict_checkpoints']} initial/best/last checkpoint; повторены",
              "выбор best, initial/last validation, train/validation/test, контракты и счётчики шагов.",
              f"Для {audit['verified_test_trajectories']} test-траекторий заново проверены issued/due задания по индексам,",
              "среднее ускорение по Δv/Δt, завершение очереди и точный native replay q/v/time.", "",
              "Это новая низкоуровневая задача: вместе с таймингом изменились входы, сценарии и reward.",
              "Прямое сравнение процентов успеха с прежним PD-опытом не изолирует эффект одного сдвига.",
              "Один seed и фиксированные случаи не доказывают устойчивость результата. Исполнение",
              "ускорений не заменяет обучение верхнего контроллера положению или равновесию.",
              "Исходники прежних экспериментов не менялись; generated weights/rollouts остаются ignored.", "",
              "## Команды", "", "```bash",
              ".venv/bin/python3 scripts/run_vertical_scheduled.py > runs/vertical_scheduled_ppo.log 2>&1",
              ".venv/bin/python3 scripts/evaluate_vertical_scheduled.py",
              "PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_scheduled.py",
              ".venv/bin/python3 scripts/demo_vertical_scheduled.py --controller runs/vertical_scheduled_ppo/variant_4/best.zip --view",
              "```", "", "Для нового обучения нужен новый каталог `--output-dir`. При наличии архива правки",
              "завершённые ветки с доказанно тем же отбором могут использоваться повторно.",
              "Проверенные команды и ограничения GUI — в [validation.md](validation.md).", ""]
    Path("docs/vertical-scheduled-ppo.md").write_text('\n'.join(lines))

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)
    for i, v in enumerate(("1", "2", "3", "4")):
        r = summary["variants"][v]
        h = [(0, r["initial_validation"])]+[(row["steps"], row["validation"]) for row in r["history"]]
        axes[i, 0].plot([step for step, _ in h], [value["score_m_s2"] for _, value in h], "o-", label="Current validation")
        axes[i, 0].axhline(.25, color="black", ls="--", label="RMSE 0.25")
        axes[i, 0].set_ylabel(f"Механика {v}\nRMSE/score, m/s²")
        axes[i, 0].set_xlabel("Переходы среды (20 ms)")
        samples = np.asarray(json.loads((args.run/f"variant_{v}"/"test_replay.json").read_text())["cases"][0]["samples"])
        axes[i, 1].step(samples[:, 0]-.02, samples[:, 3], where="post", alpha=.6, ls="--", label="Issued for +40 ms")
        axes[i, 1].step(samples[:, 0]-.02, samples[:, 4], where="post", label="Due now")
        axes[i, 1].plot(samples[:, 0]-.01, samples[:, 5], label="Actual interval mean")
        axes[i, 1].set_ylabel("Ускорение, m/s²")
        axes[i, 1].set_xlabel("Время, s")
        for axis in axes[i]:
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    axes[0, 0].set_title("Текущая validation, не best-so-far")
    axes[0, 1].set_title("Test A=0.7: issued → due через 40 ms")
    save(fig, "scheduled_comparison")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, v in zip(axes.ravel(), ("1", "2", "3", "4")):
        for offset, name, label in ((-.25, "constant_support", "Support"), (0., "force_inverse", "Force inverse"), (.25, "test", "PPO")):
            cases = audit["variants"][v][name]["cases"]
            axis.bar(np.arange(5)+offset, [c["acceleration_rmse_m_s2"] if not c["physical_failure"] else 3. for c in cases], width=.25, label=label)
        axis.axhline(.25, color="black", ls="--")
        axis.set_xticks(range(5), ["+0.7/140", "−0.7/140", "+0.8/200", "−0.8/200", "Zero"])
        axis.set_ylabel("RMSE/score, m/s²")
        axis.set_title(f"Механика {v}; A[m/s²] / H[ms]")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.2)
    save(fig, "scheduled_cases")


if __name__ == "__main__":
    main()
