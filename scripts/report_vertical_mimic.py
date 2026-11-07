"""Produce measured tables and plots for the CPU MuscleMimic adaptation."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/vertical_mimic")
    summaries = [json.loads((root/f"variant_{v}/summary.json").read_text()) for v in range(1, 5)]
    audit = json.loads((root/"independent_evaluation.json").read_text())
    assert audit["status"] == "passed"
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for variant, (summary, row) in enumerate(zip(summaries, axes), 1):
        h = summary["history"]
        row[0].plot([r["steps"] for r in h], [1000*r["rollout_q_rmse_m"] for r in h], label="Train stochastic")
        vh = [r for r in h if "validation" in r]
        row[0].plot([r["steps"] for r in vh], [1000*r["validation"]["selection_score_m"] for r in vh], "o-", label="Validation RMSE / failure penalty")
        row[0].set(title=f"Mechanics {variant}", ylabel="Position RMSE, mm", xlabel="Native transitions")
        data = json.loads((root/f"variant_{variant}/test.json").read_text())
        s = np.asarray(data["cases"][0]["samples"])
        row[1].plot(s[:, 0], s[:, 3]*1000, label="Reference")
        row[1].plot(s[:, 0], s[:, 1]*1000, label="Policy")
        row[1].set(title="Held-out sine, 75 mm amplitude", ylabel="Position, mm", xlabel="Time, s")
        for ax in row:
            ax.grid(alpha=.25)
            ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root/f"comparison.{ext}", dpi=150)
    plt.close(fig)

    text = """\n## Фактический прогон\n\nВо всех четырёх вариантах использована сеть **4×256** (ширина уменьшена относительно\nзакреплённого config), 128 независимых сред, 128 updates, 327 680 transitions на\nвариант. Все четыре policy обучены с нуля, seed=0. Физические экземпляры независимы,\nно на CPU шагают последовательно; сетевые вычисления используют общий batch.\nКаждый вариант получает одинаковый бюджет. Конечный бюджет не устанавливает\nплато, если кривая ещё улучшается.\n\n"""
    text += "| Механика | Best step | Train strict | Test completion | Test strict | Test q RMSE, mm | Test v RMSE, m/s | Constant q RMSE, mm |\n"
    text += "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for v, (s, a) in enumerate(zip(summaries, audit["variants"]), 1):
        t, b = a["test"], a["constant_support"]
        text += f"| {v} | {s['best_step']} | {s['train']['passed']}/6 | {t['completed']}/5 | {t['passed']}/5 | {t['q_rmse_m']*1000:.2f} | {t['v_rmse_m_s']:.3f} | {b['q_rmse_m']*1000:.2f} |\n"
    text += "\nRMSE считается по прожитой части клипа и при раннем обрыве может быть\nобманчиво низкой — поэтому рядом стоят completion и строгая приёмка.\nConstant support игнорирует reference и выдаёт постоянное исходное возбуждение,\nвычисленное для q=0; начальные activation/очередь при этом тоже нулевые.\n\n"
    text += "По отдельным test-клипам: ячейка — **position RMSE, mm; strict pass**.\n\n"
    text += "| Клип | A, mm | P, s | Offset, mm | Механика 1 | 2 | 3 | 4 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for i, case in enumerate(audit["variants"][0]["test"]["cases"]):
        m = case["motion"]
        cells = []
        for a in audit["variants"]:
            c = a["test"]["cases"][i]
            cells.append(f"{1000*c['q_rmse_m']:.2f}; {'да' if c['passed'] else 'нет'}")
        text += f"| {m['name']} | {1000*m['amplitude_m']:.0f} | {m['period_s']} | {1000*m['offset_m']:.0f} | "+" | ".join(cells)+" |\n"
    text += "\nЕдинственный train-hold находится на q=0; test-hold на q=+20 mm\nпроверяет перенос на другую высоту. Его провал нельзя скрывать усреднением\nс хорошо выполненными синусами.\n\n"
    text += "| Механика | Аналитический reference-контроллер: strict | q RMSE, mm | v RMSE, m/s |\n| ---: | ---: | ---: | ---: |\n"
    for v, a in enumerate(audit["variants"], 1):
        b = a["heuristic"]
        text += f"| {v} | {b['passed']}/5 | {1000*b['q_rmse_m']:.2f} | {b['v_rmse_m_s']:.3f} |\n"
    text += "\nДиагностический controller использует PD 28/9, известное ускорение reference,\npreview на transport delay +25 ms для activation и текущий force–velocity factor.\nОн не инвертирует точно compliance и не задаёт мышечных labels для PPO.\nДля него заранее проверялась выполнимость test-команд; эти данные не были\nскрыты от автора протокола. Обученные PPO до окончания отбора на test не проверялись.\n\n"
    text += "| Механика | Stochastic test completion | Stochastic strict | q RMSE, mm | Конечная std верх/низ |\n| ---: | ---: | ---: | ---: | --- |\n"
    for v, (s, a) in enumerate(zip(summaries, audit["variants"]), 1):
        t = a["stochastic"]
        text += f"| {v} | {t['completed']}/5 | {t['passed']}/5 | {1000*t['q_rmse_m']:.2f} | {s['history'][-1]['std'][0]:.3f} / {s['history'][-1]['std'][1]:.3f} |\n"
    text += "\nStochastic test — один фиксированный поток случайности seed=321; это\nдиагностика шума, не многосидовая оценка устойчивости. Std в таблице относится\nк last, а исполнение оценивает best; эти checkpoints могут отличаться.\n\n"
    text += "![Кривые обучения и test-движение](../runs/vertical_mimic/comparison.png)\n\n"
    text += (f"Независимая проверка: **{audit['strict_checkpoints']}** строгих загрузок и "
             f"**{audit['native_replays']}** test-траекторий, точно воспроизведённых\n"
             "в чистом VerticalMuscles по записанным возбуждениям. Проверены\n"
             "mapping Gaussian→excitation, delivered excitation, activation, q/v и время.\n\n")
    benchmark = json.loads(Path("runs/vertical_mimic_benchmark_256/summary.json").read_text())
    text += (f"Размер основной сети: {summaries[0]['parameter_count']:,} trainable параметров\n"
             "actor+critic для варианта 1. В замере 32 сред сеть 4×1024 тратила\n"
             "136,65 и 140,90 s на первые два updates; 4×256 после компиляции —\n"
             f"{benchmark['history'][1]['seconds']:.2f} и {benchmark['history'][2]['seconds']:.2f} s.\n"
             "Это замер одного laptop CPU, не оценка оригинального GPU pipeline.\n\n")
    text += """Таблицы не являются контролируемой абляцией прежней награды ускорений:\nизменились сама задача, reset, preview, observation, частота, сеть, распределение\nдействий, нормализация, optimizer и PPO batching. Улучшение не позволяет\nприписать эффект одному из этих компонентов. Полная постановка MuscleMimic\nна Warp, 8192 среды, опубликованная 16-слойная сеть и миллиардный бюджет\nздесь не воспроизведены.\n\n```bash\n.venv/bin/python3 scripts/evaluate_vertical_mimic.py\n.venv/bin/python3 scripts/demo_vertical_mimic.py \\\n  --controller runs/vertical_mimic/variant_1/best.msgpack --view\n```\n\nНовое окно в этом опыте отдельно не проверялось; headless-оценка не доказывает\nкачество отображения.\n"""
    doc = Path("docs/vertical-musclemimic.md")
    prefix = doc.read_text().split("\n## Фактический прогон\n")[0]
    prefix = prefix.replace("Результаты обучения будут записаны ниже после фактического запуска.\n", "")
    doc.write_text(prefix.rstrip()+"\n"+text)


if __name__ == "__main__":
    main()
