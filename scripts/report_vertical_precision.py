"""Report physical accuracy of the dense/log PPO experiment without reselecting."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path("runs/vertical_precision/combined")
    summary = json.loads((root/"summary.json").read_text())
    assert summary["status"] == "complete"
    results = {name: json.loads((root/f"{name}_test.json").read_text()) for name in ("baseline", "best", "last")}
    text = ["## Результаты", "", f"Завершено {summary['steps']:,} transitions, seed=0. Best на шаге {summary['best_steps']:,}.",
            "Обучение завершено по фиксированному бюджету; это не доказательство плато.",
            f"Обучение с validation: {summary['elapsed_s']:.1f} s; optimizer steps: {summary['optimizer_steps']}.",
            "Начальные веса совпали с прежней комбинацией, что проверено перед test.", "",
            "| Модель | Старый test, RMSE mm; pass | Расширенный test, RMSE mm; pass | Full RMSE mm | Срывы |",
            "| --- | --- | --- | ---: | ---: |"]
    labels = {"baseline": "Прежняя комбинация best", "best": "Dense/log best", "last": "Dense/log last"}
    for name, result in results.items():
        old = result["cases"][:5]
        score = np.mean([c["last_second_rmse_m"] if not c["physical_failure"] else .3 for c in old])*1000
        full = np.mean([c["position_rmse_m"] for c in result["cases"]])*1000
        text.append(f"| {labels[name]} | {score:.3f}; {sum(c['passed'] for c in old)}/5 | {result['score_m']*1000:.3f}; {result['passed']}/11 | {full:.3f} | {result['failures']} |")
    text.extend(["", "RMSE steady — средняя RMSE последней секунды каждого эпизода; full — всего эпизода.",
                 "Ниже отдельно показано знаковое смещение: отрицательное — ниже цели.", "",
                 "| Старт→цель mm | Прежняя RMSE mm | Новая RMSE mm | Новое смещение mm | Пройдено |",
                 "| --- | ---: | ---: | ---: | --- |"])
    for old, new in zip(results["baseline"]["cases"], results["best"]["cases"], strict=True):
        a = np.asarray(new["samples"])
        tail = a[:, 0] >= 3.
        signed = np.mean(a[tail, 1]-a[tail, 3])*1000 if tail.any() else float("nan")
        text.append(f"| {new['initial_position_m']*1000:+.0f}→{new['target_m']*1000:+.0f} | {old['last_second_rmse_m']*1000:.3f} | {new['last_second_rmse_m']*1000:.3f} | {signed:+.3f} | {'да' if new['passed'] else 'нет'} |")
    val = summary["best_validation"]
    last_val = summary["history"][-1]["validation"]
    text.extend(["", f"Best validation: {val['score_m']*1000:.3f} mm, {val['passed']}/9; last validation: {last_val['score_m']*1000:.3f} mm, {last_val['passed']}/9.",
                 "Checkpoint выбирался по расширенному validation; test не меняет выбор.",
                 "Проверены четыре строгие загрузки (initial/best/last и старый best), воспроизведение",
                 "выбора validation и 11 побитово совпавших повторов действий в native-физике.",
                 "Поведение GUI в этом опыте не проверялось. Один seed и одновременная замена",
                 "заданий и награды не позволяют разделить причины изменений.", ""])
    report = "\n".join(text)
    path = Path("docs/vertical-precision-ppo.md")
    previous = path.read_text()
    conclusion = previous.partition("## Вывод")[2]
    path.write_text(previous.split("## Результаты")[0]+report+("\n## Вывод"+conclusion if conclusion else ""))
    (root/"report.md").write_text(report)
    print(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    h = summary["history"]
    axes[0].plot([x["steps"] for x in h], [x["validation"]["score_m"]*1000 for x in h], marker=".")
    axes[0].axvline(summary["best_steps"], ls="--", c="gray")
    axes[0].set(xlabel="Transitions", ylabel="Steady RMSE, mm", title="Expanded validation (9 cases)", yscale="log")
    axes[1].plot([x["steps"] for x in h], [x["validation"]["passed"] for x in h], marker=".")
    axes[1].set(xlabel="Transitions", ylabel="Passed / 9", title="Validation success", ylim=(-.2, 9.2))
    for name, style in (("baseline", "--"), ("best", "-")):
        c = results[name]["cases"][2]
        a = np.asarray(c["samples"])
        axes[2].plot(a[:, 0], a[:, 1]*1000, style, label=labels[name])
    axes[2].axhline(-30, color="black", lw=1, label="Target")
    axes[2].set(xlabel="Time, s", ylabel="Position, mm", title="Unseen hold at -30 mm")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root/f"comparison.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
