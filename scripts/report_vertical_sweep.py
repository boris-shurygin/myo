"""Produce a separate Russian results table for every single-factor experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FACTORS = {
    "learning_rate": "Learning rate", "n_steps": "Размер rollout",
    "ent_coef": "Коэффициент энтропии", "gae_lambda": "GAE lambda",
    "gamma": "Discount gamma", "batch_size": "Размер minibatch",
    "n_epochs": "Число epochs", "clip_range": "PPO clip range",
    "target_kl": "Target KL", "position_scale_m": "Масштаб позиционной награды",
}

DESCRIPTIONS = {
    "learning_rate": "Размер шага Adam; бюджет опыта и число epochs сохранены. Фактическое число updates может меняться из-за KL-stop.",
    "n_steps": "Число переходов в одном rollout до обновления PPO. Это не длина памяти сети; границы эпизодов сохраняются.",
    "ent_coef": "Вес entropy bonus. Нулевой коэффициент сохраняет стохастическую выборку действий из Beta.",
    "gae_lambda": "Затухание будущих TD-ошибок в GAE при неизменном gamma. Большая lambda удлиняет их влияние.",
    "gamma": "Дисконтирование будущих наград. Одновременно меняются value targets и веса будущих TD-ошибок в GAE.",
    "batch_size": "Число переходов на градиентный шаг. При том же числе epochs большой batch уменьшает число Adam updates.",
    "n_epochs": "Число повторных проходов по одному rollout, если раньше не сработал KL-stop.",
    "clip_range": "Диапазон PPO clipping для отношения вероятностей действий; это не ограничение самих возбуждений.",
    "target_kl": "Порог раннего прекращения PPO epochs по изменению распределения команд; не отдельное слагаемое reward.",
    "position_scale_m": "Ширина Gaussian-слагаемого exp(-(error/scale)²). Веса положения/скорости/затрат сохранены; физика прежняя.",
}


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_ppo_sweep"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Finish all trials and independent evaluation before reporting")
    variants = summary["arguments"]["variants"]
    trials = summary["trials"]
    names = [name for name, _ in summary["matrix"]]
    controls = {v: trials[f"variant_{v}/control"] for v in variants}
    lines = ["# PPO: отдельное влияние параметров обучения", "",
             "Опыт 2026-11-06 — продолжение [исходного PPO стенда](vertical-ppo.md).",
             "Сравниваются механики 2 (упругость без задержки), 3 (также задержка)",
             "и 4 (также постепенная активация). Задержка 40 ms; период команд 20 ms,",
             "физический шаг 1 ms. В механике 4 постоянные активации/расслабления 15/50 ms.",
             "Механика 1 в этой серии не варьировалась. Во всех трёх есть отдельное упругое сухожилие.", "",
             "## Методика", "",
             "Каждая ветка начинает с одного и того же best checkpoint своей механики,",
             "включая состояние Adam. Изменяется ровно один параметр. Контроль тоже получает",
             f"{summary['arguments']['steps']} новых переходов среды по 20 ms, как каждый кандидат.",
             "Seed 0, эпизоды по 4 s, свежий физический reset и одинаковое начало фиксированного цикла задач.",
             "Actor/critic 128×128, физика, наблюдения, action mapping и исходные веса сохранены.",
             "Ранней остановки нет. Проверки каждые 8192 перехода; исходные веса входят в выбор.",
             "Параметры других таблиц остаются исходными: найденные улучшения не накапливаются.", "",
             "Это сравнение локального дообучения, не подбора каждого режима с нуля.",
             "При изменении gamma/GAE или reward critic начинает с оценок исходного режима и адаптируется.",
             "Параметры оптимизатора меняются без сброса его моментов; gamma/GAE обновлены также",
             "в rollout buffer, размер буфера пересоздан, learning-rate/clip schedules обновлены.", "",
             "Выбор checkpoint: меньше падений, больше успешных validation-случаев, ниже средний",
             "RMSE последней секунды. Test вычислен после фиксации весов всех ветвей и не",
             "использовался для их выбора. В validation/test единый исходный reward даже для",
             "ветвей с другой шириной награды; сравниваем прежде всего физическое исполнение.", "",
             "Train/validation/test используют прежние фиксированные случаи исходного опыта,",
             "не новый внешний набор. Изменение параметра меняет посещаемые состояния; одинаковый",
             "seed не означает побитно одинаковых траекторий разных настроек.", "",
             "В ячейках Val/Test: **успехов из 5; средний RMSE последней секунды в mm**.",
             "Успех каждого случая: без падения, не менее 90% последней секунды в пределах",
             "5 mm и 0,03 m/s. Среднее ниже 5 mm само по себе не означает прохождение всех случаев.",
             "Таблицы относятся к отдельным заданиям на положение; непрерывная смена трёх целей",
             "проверяется дополнительно в `switches.json`, но не участвует в выборе checkpoint.",
             "Best +шаг — число новых переходов до выбранных весов; 0 означает сохранение родителя.",
             "Звёздочка у последнего шага означает best на границе бюджета, не установленное плато.",
             "В каждой таблице исходное значение параметра — продолженный контроль, не старый родитель.",
             "Adam updates — фактическое число шагов action-head Adam за всю ветку, включая влияние KL-stop.",
             "Одинаковый бюджет новых переходов не означает одинаковое число градиентных обновлений",
             "при изменении batch/epochs или при досрочном KL-stop внутри PPO.", ""]

    def metrics(m):
        return f"{m['passed']}/5; {number(m['score_m']*1000)}"

    def setting(value, factor):
        return (f"{value*1000:g} mm" if factor == "position_scale_m" else f"{value:g}").replace(".", ",")

    lines += ["### Исходные параметры остальных ветвей", "",
              "| Параметр | Исходное значение |", "| --- | ---: |"]
    baseline = next(iter(controls.values()))["training"]["parameters"]
    for factor, title in FACTORS.items():
        lines.append(f"| {title} | {setting(baseline[factor], factor)} |")
    lines += ["", "### Родитель и продолженный контроль", "",
              "| Механика | Родитель Val: успехи; mm | Контроль Val best: успехи; mm |",
              "| ---: | --- | --- |"]
    for v in variants:
        row = controls[v]
        lines.append(f"| {v} | {metrics(row['history'][0]['validation'])} | {metrics(row['selected_validation'])} |")
    lines.append("")

    lines += ["## Лучшие ветви по критериям validation", "",
              "Выбор ниже сделан по тому же правилу failures → passed → RMSE, не по test.",
              "Это отдельные изменения относительно исходного режима, не их сочетание.", "",
              "| Механика | Ветка | Val best: успехи; mm | Контроль Val: успехи; mm | Test best: успехи; mm | Best +шаг |",
              "| ---: | --- | --- | --- | --- | ---: |"]
    residuals = []
    for v in variants:
        name = min(names, key=lambda name: (
            trials[f"variant_{v}/{name}"]["selected_validation"]["failures"],
            -trials[f"variant_{v}/{name}"]["selected_validation"]["passed"],
            trials[f"variant_{v}/{name}"]["selected_validation"]["score_m"]))
        row = trials[f"variant_{v}/{name}"]
        best_step = str(row['best_additional_steps'])
        if row['best_additional_steps'] == summary['arguments']['steps']:
            best_step += "*"
        lines.append(f"| {v} | `{name}` | {metrics(row['selected_validation'])} | "
                     f"{metrics(controls[v]['selected_validation'])} | {metrics(row['test'])} | {best_step} |")
        failed = [c for c in row["test"]["cases"] if not c["passed"]]
        if failed:
            details = "; ".join(f"цель {number(c['target_m']*1000,0)} mm — RMSE {number(c['last_second_rmse_m']*1000)} mm"
                                for c in failed)
            residuals.append(f"Непройденные test-случаи `{v}/{name}`: {details}.")
    lines += ["", "Ниже — все проверенные уровни, включая ухудшения и сохранение родителя.", ""]
    lines += ["Максимальное число успешных случаев среди всех проверенных ветвей:"]
    for v in variants:
        rows = [trials[f"variant_{v}/{name}"] for name in names]
        val_passes = max(r["selected_validation"]["passed"] for r in rows)
        test_passes = max(r["test"]["passed"] for r in rows)
        lines.append(f"механика {v} — validation {val_passes}/5, test {test_passes}/5.")
    lines += ["Это максимумы по столбцу, не выбор по test. В полной механике отдельные ветви",
              "проходят больше test-случаев, чем ветка с минимальной validation-ошибкой.",
              "Пороговое число успехов и средний RMSE показывают разные стороны результата.", ""]
    for paragraph in residuals:
        lines += [paragraph, ""]

    for factor, title in FACTORS.items():
        selected_names = ["control"]+[name for name, changes in summary["matrix"] if factor in changes]
        lines += [f"## {title}", "", DESCRIPTIONS[factor], "",
                  "| Значение | Механика | Val best: успехи; mm | Val last: успехи; mm | Test best: успехи; mm | Best +шаг | Всего Adam updates |",
                  "| --- | ---: | --- | --- | --- | ---: | ---: |"]
        for name in selected_names:
            for v in variants:
                key = f"variant_{v}/{name}"
                r = trials[key]
                value = setting(r["training"]["parameters"][factor], factor)
                best_step = str(r["best_additional_steps"])
                if r["best_additional_steps"] == summary["arguments"]["steps"]:
                    best_step += "*"
                lines.append(f"| {value} | {v} | {metrics(r['selected_validation'])} | "
                             f"{metrics(r['history'][-1]['validation'])} | {metrics(r['test'])} | "
                             f"{best_step} | {r['optimizer_updates']} |")
        lines.append("")
        if factor == "ent_coef":
            lines += ["STD ниже измерен на одних и тех же 30 причинных наблюдениях родительской policy",
                      "для каждой механики, не на разных посещённых состояниях кандидатов. Уменьшение",
                      "entropy coefficient не задаёт STD напрямую.", "",
                      "| Коэффициент | Механика | Средний STD верхней / нижней команды |", "| --- | ---: | --- |"]
            for name in selected_names:
                for v in variants:
                    key = f"variant_{v}/{name}"
                    std = audit["trials"][key]["fixed_probe_excitation_std"]
                    lines.append(f"| {setting(trials[key]['training']['parameters'][factor], factor)} | {v} | "
                                 f"{number(std[0],4)} / {number(std[1],4)} |")
            lines.append("")

    elapsed = summary["elapsed_s"]
    transitions = len(trials)*summary["arguments"]["steps"]
    updates = sum(r["optimizer_updates"] for r in trials.values())
    lines += ["## Артефакты и ограничения", "",
              f"Завершено {len(trials)} ветвей, {transitions} новых переходов среды по 20 ms, {updates} Adam updates.",
              f"Wall time оркестратора: {number(elapsed,2)} s; до трёх локальных процессов, по одному torch-потоку.",
              "Параметры проходят последовательно; механики внутри одного параметра выполняются параллельно.",
              "Исходные веса, моменты и сценарии сохранены. Один seed и один набор задач не устанавливают",
              "устойчивость результата на других запусках. Best на конце бюджета не является доказательством",
              "сходимости; особенно это существенно для маленького lr. Сочетания улучшений не обучались.", "",
              "`runs/vertical_ppo_sweep/summary.json` содержит полный журнал и зафиксированную матрицу.",
              "`independent_evaluation.json` повторяет строгую загрузку, настройки PPO/буфера, выбор best,",
              "validation/test/switches и фактические Adam updates. `best.zip`/`last.zip` каждой ветви",
              "хранят metadata с actual reward, родителем, порядком входов, исходниками и hash весов.",
              "Сравнение последовательного preflight и параллельного контроля — в том же отчёте проверки.",
              "Методика и команды проверок — в [validation.md](validation.md).", "",
              "![Отдельное влияние факторов](../runs/vertical_ppo_sweep/factor_comparison.png)", "",
              "Кривые validation: [механика 3](../runs/vertical_ppo_sweep/learning_curves_3.png),",
              "[механика 4](../runs/vertical_ppo_sweep/learning_curves_4.png).", "",
              "## Воспроизведение", "", "```bash",
              ".venv/bin/python3 scripts/run_vertical_sweep.py > runs/vertical_ppo_sweep.log 2>&1",
              ".venv/bin/python3 scripts/evaluate_vertical_sweep.py",
              "PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_sweep.py --output docs/vertical-ppo-sweep.md",
              "```", "",
              "Для продолжения прерванной серии есть `--continue-run`: законченные ветви",
              "сохраняются, незаконченная архивируется и начинается заново с общего родителя.",
              "Точное восстановление частичного физического rollout не заявляется.", "",
              "Демо выбранной ветви (пример для контрольной полной механики):", "", "```bash",
              ".venv/bin/python3 scripts/demo_vertical_sweep.py --controller runs/vertical_ppo_sweep/variant_4/control/best.zip --view",
              "```", "",
              "Этот загрузчик проверяет расширенный контракт опыта, включая реальный reward.",
              "Архитектура и физика исходного стенда не менялись.", ""]
    output = args.output or args.run/"report.md"
    output.write_text('\n'.join(lines))

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    matrix = np.asarray([[np.log10(max(1e-12,trials[f"variant_{v}/{name}"]["selected_validation"]["score_m"])/
                                    max(1e-12,controls[v]["selected_validation"]["score_m"])) for v in variants] for name in names[1:]])
    fig, axis = plt.subplots(figsize=(8, 10), constrained_layout=True)
    panel = axis.imshow(matrix, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    axis.set_xticks(range(len(variants)), [f"Механика {v}" for v in variants])
    axis.set_yticks(range(len(names)-1), names[1:])
    for row, name in enumerate(names[1:]):
        for column, v in enumerate(variants):
            r = trials[f"variant_{v}/{name}"]["selected_validation"]
            percent = 100*(r["score_m"]/controls[v]["selected_validation"]["score_m"]-1)
            axis.text(column,row,f"{r['passed']}/5; {percent:+.1f}%",ha="center",va="center",
                      color="white" if abs(matrix[row,column])>.3 else "black")
    fig.colorbar(panel,ax=axis,label="log10(RMSE / RMSE продолженного контроля)")
    axis.set_title("Val best: успехи и изменение ошибки; синий — ниже ошибка\nКаждая строка меняет только один параметр")
    save(fig,"factor_comparison")

    for v in (3,4):
        if v not in variants:
            continue
        fig,axes = plt.subplots(5,2,figsize=(12,16),constrained_layout=True)
        for axis,(factor,title) in zip(axes.ravel(),FACTORS.items()):
            selected_names = ["control"]+[name for name, changes in summary["matrix"] if factor in changes]
            for name in selected_names:
                r = trials[f"variant_{v}/{name}"]
                h = r["history"]
                axis.plot([x["additional_steps"] for x in h], [x["validation"]["score_m"]*1000 for x in h], "o-",label=name)
            axis.set_title(title)
            axis.set_xlabel("Новые переходы среды (20 ms)")
            axis.set_ylabel("Validation RMSE, mm")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.suptitle(f"Механика {v}: одинаковый родитель и бюджет; показаны текущие, не best-so-far метрики")
        save(fig,f"learning_curves_{v}")


if __name__ == "__main__":
    main()
