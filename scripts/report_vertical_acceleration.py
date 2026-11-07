"""Russian acceleration-reward PPO results and native closed-loop plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metric(result):
    return f"{result['passed']}/5; {result['score_m']*1000:.3f}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_acceleration_ppo"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    reward_audit = json.loads((args.run/"reward_audit.json").read_text())
    baseline = json.loads(Path(summary["baseline"]).read_text())
    assert summary["status"] == "complete" and audit["status"] == "passed" and reward_audit["status"] == "passed"
    lines = ["# PPO с наградой за совпадение ускорений", "",
             "Опыт 2026-11-06 на прежнем вертикальном стенде, четыре отдельные policy с нуля.",
             "Сравнение — с [исходной наградой за положение](vertical-ppo.md), не с дообученными",
             "победителями [подбора гиперпараметров](vertical-ppo-sweep.md).", "",
             "## Что изменилось", "",
             "Сеть получает прежние состояние, очередь команд, предыдущее возбуждение и цель положения.",
             "На выходе две независимые Beta-excitations. Actor и critic — отдельные Tanh MLP 128×128.",
             "Основное слагаемое reward теперь оценивает соответствие среднего фактического ускорения",
             "за интервал заданному PD-ускорению в начале этого же интервала:", "", "```text",
             "a_ref = 28 * (target - q_before) - 9 * v_before     [m/s²]",
             "a_actual = (v_after - v_before) / elapsed_time     [m/s²]",
             "r_raw = 0.85 * exp(-((a_actual - a_ref) / 2.24)^2)",
             "      + 0.15 * exp(-(v_after / 0.4)^2)",
             "      - 0.01 * mean(u²) - 0.002 * mean((u-u_previous)²)",
             "      - 1.0 * physical_failure",
             "r_training = 0.1 * r_raw", "```", "",
             "PD-коэффициенты взяты из простой части heuristic вертикального стенда, без preview.",
             "Это инженерный эталон, не оптимальное или биологически установленное ускорение.",
             "Вычисление использует истинные q/v симулятора; ошибку сенсоров этот опыт не добавляет.",
             "Масштаб 2,24 = 28×0,08 выбран так, чтобы при v=0 и a=0 совпадала прежняя",
             "ширина позиционной награды 80 mm. Бонус скорости, затраты и штраф падения сохранены.", "",
             "Это настоящий PPO: actor обучается по log-probability, clipping и GAE advantage; critic",
             "по bootstrapped returns новой награды. Градиент через MuJoCo не вычисляется.",
             "Нет regression loss по возбуждениям или ускорениям, размеченных действий, predictor",
             "или скрытого вызова heuristic для команд. Отличие от прежнего supervised-опыта —",
             "ускорение определяет скалярную награду, а не дифференцируемую ошибку actor.", "",
             "## Время и сравнение", "",
             "Все четыре native-механики сохранены: 1 — без упругости/задержек; 2 — упругость мышцы",
             "и отдельного сухожилия; 3 — также транспорт 40 ms; 4 — также активация 15/50 ms.",
             "Груз 1 kg, утомления нет. Физика 1 ms, команды 20 ms. Новая команда в механиках 3/4 не меняет первые два",
             "интервала ускорения. Награда остаётся на фактическом шаге результата; перенос её влияния",
             "к раннему действию выполняет GAE. Замена reward не устраняет физическое запаздывание.", "",
             "Seed 0, исходные PPO hyperparameters, те же девять train и пять validation/test случаев",
             "по 4 s. Наблюдения и action mapping сохранены. До обучения тензоры всех четырёх новых",
             "policy точно сравнены с исходными initial checkpoint без изменения RNG.",
             "Контроль с исходной наградой повторил первые 8192 шага механики 4: validation,",
             "PPO losses и восемь rollout audits совпали точно. Остальные контрольные результаты",
             "взяты из завершённого исходного опыта, его hash сохранён.", "",
             "Блок 8192 перехода среды, минимум 32768, предел 131072 на механику. Остановка при",
             "двух подряд полных успехах либо шести проверках без накопленного улучшения",
             "max(0,5 mm, 2% anchor). У контроля предел был 262144, но все четыре фактически",
             "остановились раньше 131072; он не влиял на их траектории обучения.",
             "Выбор best: меньше падений, больше успешных position-случаев, меньше steady RMSE.",
             "Test открыт после фиксации всех четырёх best. Acceleration RMSE не выбирает веса.", "",
             "## Результаты движения", "",
             "Ячейка — **успехов из 5; средний RMSE последней секунды в mm**. Успех: без падения,",
             "не менее 90% последней секунды внутри 5 mm и 0,03 m/s. Проверки всех моделей используют",
             "общую исходную position-награду для raw_return; основные критерии — физические.", "",
             "| Механика | Контроль Val | Ускорения Val | Контроль Test | Ускорения Test |",
             "| ---: | --- | --- | --- | --- |"]
    for v in ("1", "2", "3", "4"):
        row = audit["variants"][v]
        lines.append(f"| {v} | {metric(row['baseline']['validation'])} | {metric(row['acceleration']['validation'])} | "
                     f"{metric(row['baseline']['test'])} | {metric(row['acceleration']['test'])} |")
    if all(audit["variants"][v]["acceleration"]["test"]["score_m"] > audit["variants"][v]["baseline"]["test"]["score_m"]
           for v in ("1", "2", "3", "4")):
        lines += ["", "В этой настройке новая награда ухудшила test-ошибку всех четырёх механик.",
                  "Ухудшение есть в механике 1 без задержек и упругости; результат нельзя объяснить",
                  "только моторным запаздыванием. У механик с остановкой по бюджету плато не установлено.", ""]
    lines += ["", "## Ускорения и бюджет", "",
              "Acceleration RMSE — среднее по пяти test-случаям от полного RMSE за 4 s, в m/s².",
              "Она вычислена по собственным реальным траекториям каждой policy; состояния между ними",
              "различаются. Поэтому это не ошибка на одинаковом заранее размеченном наборе состояний.", "",
              "| Механика | Acc. RMSE контроль / PPO-acc | Шаги контроль / PPO-acc | Best PPO-acc | Причина остановки |",
              "| ---: | --- | --- | ---: | --- |"]
    for v in ("1", "2", "3", "4"):
        row, record = audit["variants"][v], summary["variants"][v]
        a, b = row["baseline"]["test"]["mean_case_acceleration_rmse_m_s2"], row["acceleration"]["test"]["mean_case_acceleration_rmse_m_s2"]
        lines.append(f"| {v} | {a:.3f} / {b:.3f} | {baseline['variants'][v]['trained_steps']} / {record['trained_steps']} | "
                     f"{record['best_steps']} | {record['stop_reason']} |")
    lines += ["", "## Одна и та же новая награда для обеих policy", "",
              "Команды всех 40 детерминированных test-траекторий повторены в native-среде с новой",
              "наградой: время, положение и скорость совпали точно. Дополнительно выполнены test",
              "со стохастическими действиями, один фиксированный seed 0. Это диагностика без обучения.", "",
              "| Механика | Новая raw-награда mean: контроль / PPO-acc | Та же награда stochastic seed0: контроль / PPO-acc | Steady acc. RMSE: контроль / PPO-acc |",
              "| ---: | --- | --- | --- |"]
    for v in ("1", "2", "3", "4"):
        old, new = reward_audit["variants"][v]["baseline"], reward_audit["variants"][v]["acceleration"]
        lines.append(f"| {v} | {old['mean_raw_acceleration_reward']:.4f} / {new['mean_raw_acceleration_reward']:.4f} | "
                     f"{old['stochastic_seed0']['mean_raw_reward']:.4f} / {new['stochastic_seed0']['mean_raw_reward']:.4f} | "
                     f"{old['steady_acceleration_rmse_m_s2']:.4f} / {new['steady_acceleration_rmse_m_s2']:.4f} |")
    lines += ["", "Старая policy получает больше новой награды при исполнении средних команд. Но при",
              "стохастическом исполнении в проверенном seed 0 выше награда новой policy. PPO обучает",
              "распределение действий; в последних обучающих rollout оно существенно уже у новой policy. Поэтому сравнение только",
              "средних выходов не показывает, какая policy выгоднее в стохастическом обучении.",
              "Все эти стохастические прогоны прошли без физических падений.", "",
              "Полный RMSE ускорения и Gaussian reward также не тождественны: краткие большие",
              "ошибки сильно влияют на RMSE, тогда как exp-награда ограничена. Steady acc. RMSE",
              "у новых policy хуже, несмотря на меньший полный RMSE за эпизод.", ""]
    lines += ["", "## Что видно в обучении", "",
              "Статистика последних обучающих rollout относится к стохастическим действиям и last policy.",
              "Посещённые состояния и число пройденных шагов различаются. Это описание exploration,",
              "не проверка на одном фиксированном наборе входов и не диагноз причины неудачи.", "",
              "| Механика | STD команд контроль: верх/низ | STD PPO-acc: верх/низ | Средняя нижняя команда контроль / PPO-acc | Critic EV PPO-acc |",
              "| ---: | --- | --- | --- | ---: |"]
    for v in ("1", "2", "3", "4"):
        old, new = baseline["variants"][v]["audit"][-1], summary["variants"][v]["audit"][-1]
        fmt = lambda values: " / ".join(f"{x:.4f}" for x in values)
        lines.append(f"| {v} | {fmt(old['excitation_std'])} | {fmt(new['excitation_std'])} | "
                     f"{old['excitation_mean'][1]:.4f} / {new['excitation_mean'][1]:.4f} | {new['critic_ev']:.3f} |")
    lines += ["", "Наблюдение: разброс команд уменьшился, средняя нижняя команда приблизилась к нулю.",
              "Гипотеза для следующей проверки: acceleration reward сильнее наказывает быстрые случайные",
              "изменения силы и поощряет подавление вариативности раньше освоения точного движения.",
              "Пока вклад этого механизма не изолирован. Высокий critic EV в части механик не обеспечил",
              "точное управление; он относится к rollout returns собственной награды, не к прохождению задач.", ""]
    lines += ["", "## Ограничения и артефакты", "",
              "Один seed и фиксированные задачи — не статистический поиск оптимальной награды.",
              "Успехи относятся к вертикальному стенду, не ко всему телу или MyoArm.",
              "Наличие acceleration reward само по себе не доказывает освоение движения или точность",
              "удержания. Новый actor всё ещё получает цель положения, а не команду ускорения от верхней сети.",
              "Сравнение не проверяет иерархический контроллер.", "",
              f"Новых переходов среды: {sum(r['trained_steps'] for r in summary['variants'].values())}; "
              f"wall time оркестратора {summary['elapsed_s']:.2f} s, четыре локальных процесса по одному torch-потоку.",
              f"Независимая проверка: {audit['strict_checkpoints']} строгих загрузок новых и исходных checkpoint; "
              "повторены physical validation/test/switches, best selection, initial weights и контрольный префикс.",
              "Сохранены initial/best/last, metadata с actual reward, физикой, входами, timestep и source hashes.",
              "Журналы и веса ignored: `runs/vertical_acceleration_ppo/`; исходники прежних опытов не менялись.", "",
              "![Обучение и движение](../runs/vertical_acceleration_ppo/comparison.png)", "",
              "![Test по отдельным целям](../runs/vertical_acceleration_ppo/test_cases.png)", "",
              "## Запуск", "", "```bash",
              ".venv/bin/python3 scripts/run_vertical_acceleration.py > runs/vertical_acceleration_ppo.log 2>&1",
              ".venv/bin/python3 scripts/evaluate_vertical_acceleration.py",
              ".venv/bin/python3 scripts/audit_vertical_acceleration_reward.py",
              "PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_acceleration.py",
              ".venv/bin/python3 scripts/demo_vertical_acceleration.py --controller runs/vertical_acceleration_ppo/variant_4/best.zip --view",
              "```", "",
              "Обучение требует нового выходного каталога. Проверенные команды и результаты —",
              "в [validation.md](validation.md).", ""]
    Path("docs/vertical-acceleration-ppo.md").write_text('\n'.join(lines))

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    for index, v in enumerate(("1", "2", "3", "4")):
        control, changed = baseline["variants"][v], summary["variants"][v]
        for record, color, label in ((control, "C0", "Position reward"), (changed, "C1", "Acceleration reward")):
            h = [(0, record["initial_validation"])]+[(r["steps"], r["validation"]) for r in record["history"]]
            axes[index, 0].plot([s for s, _ in h], [m["score_m"]*1000 for _, m in h], "o-", color=color, label=label)
        axes[index, 0].set_yscale("log")
        axes[index, 0].set_ylabel(f"Механика {v}\nVal RMSE, mm")
        axes[index, 0].set_xlabel("Переходы среды (20 ms)")
        for label, color in (("baseline", "C0"), ("acceleration", "C1")):
            case = json.loads((args.run/f"variant_{v}"/f"{label}_test_replay.json").read_text())["cases"][0]
            samples = np.asarray(case["samples"])
            accelerations = np.asarray(case["acceleration_samples"])
            axes[index, 1].plot(samples[:, 0], samples[:, 1]*1000, color=color, label=label)
            axes[index, 2].plot(samples[:, 0], accelerations[:, 1]-accelerations[:, 0], color=color, label=label)
        axes[index, 1].axhline(60., color="black", ls="--", label="Цель")
        axes[index, 1].set_ylabel("Положение, mm")
        axes[index, 1].set_xlabel("Время, s")
        axes[index, 2].set_ylabel("a_actual − a_ref, m/s²")
        axes[index, 2].set_xlabel("Время, s")
        for axis in axes[index]:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    for axis, title in zip(axes[0], ("Текущие validation checkpoint", "Выбранные policy: −30 → +60 mm", "Ошибка ускорения на том же движении")):
        axis.set_title(title)
    save(fig, "comparison")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, v in zip(axes.ravel(), ("1", "2", "3", "4")):
        for offset, label in ((-.18, "baseline"), (.18, "acceleration")):
            cases = audit["variants"][v][label]["test"]["cases"]
            axis.bar(np.arange(len(cases))+offset, [c["last_second_rmse_m"]*1000 for c in cases], width=.36, label=label)
        axis.axhline(5., color="black", ls="--", label="5 mm")
        axis.set_xticks(range(5), ["−30→60", "30→−60", "−30→−30", "30→30", "0→0"], rotation=15)
        axis.set_title(f"Механика {v}")
        axis.set_ylabel("Последняя секунда RMSE, mm")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    save(fig, "test_cases")


if __name__ == "__main__":
    main()
