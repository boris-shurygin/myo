"""Export learning-rate and activation comparisons for the original predictor."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_tuning"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Finish training and independent evaluation before plotting")
    colors = ["#367bb5", "#d28328", "#25977f", "#ac5399", "#835e43"]
    labels = {"tanh": "Tanh", "silu": "SiLU", "leaky_relu": "LeakyReLU(0,2)",
              "linear": "Линейный", "tanh_unit": "Tanh ±1 STD", "tanh_scaled": "Tanh с train-диапазоном"}

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, phase, title in zip(axes[0], ("warmup_lr", "bptt_lr"), ("Learning rate одношаговой подготовки", "Learning rate BPTT из общей подготовки")):
        candidates = summary["phases"][phase]
        ordered = sorted(candidates.values(), key=lambda row: row["learning_rate"])
        axis.plot([row["learning_rate"] for row in ordered], [row["mean_validation_score"] for row in ordered], "o-")
        axis.set_xscale("log")
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("Средний validation score четырёх механик")
        axis.set_title(title)
        axis.grid(alpha=.2)
    x = np.arange(4)
    for axis, phase, title in zip(axes[1], ("hidden_activation", "output_activation"),
                                  ("Скрытые активации; линейный выход", "Выходные активации; выбранные скрытые слои")):
        for i, (name, candidate) in enumerate(summary["phases"][phase].items()):
            values = [summary["test"][candidate["models"][str(v)]]["forecast"]["60"]["velocity_rmse_m_s"]*1000 for v in range(1, 5)]
            axis.bar(x+(i-1)*.25, values, .23, label=labels[name], color=colors[i])
        axis.set_xticks(x, [f"Механика {v}" for v in range(1, 5)])
        axis.set_yscale("log")
        axis.set_ylabel("Test RMSE скорости через 60 мс, мм/с")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=2, fontsize=8)
    fig.suptitle("Исходные входы без истории; переход 20 мс, BPTT 60 мс; один seed")
    save(fig, "tuning_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, phase, title in zip(axes.ravel(), ("warmup_lr", "bptt_lr", "hidden_activation", "output_activation"),
                                 ("Learning rate подготовки", "Learning rate BPTT", "Скрытая активация", "Выходная активация")):
        for color, (key, candidate) in zip(colors, summary["phases"][phase].items()):
            record = summary["trials"][candidate["models"]["4"]]
            history = record["history"]
            label = (key if "lr" in phase else labels[key])+f"; best {record['best_update']}"
            axis.plot([row["update"] for row in history], [row["validation"]["score"] for row in history], label=label, color=color)
            best = next(row for row in history if row["update"] == record["best_update"])
            axis.plot(best["update"], best["validation"]["score"], "o", color=color)
        axis.set_title(title)
        axis.set_xlabel("Обновления оптимизатора")
        axis.set_ylabel("Validation score полной механики")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Механика 4: задержка, упругость, постепенная активация; точки — best")
    save(fig, "learning_curves")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    horizons = (20, 40, 60, 120, 400)
    for variant, axis in zip(("1", "2", "3", "4"), axes.ravel()):
        rows = [("Tanh, исходный lr", summary["phases"]["lr_pair"]["default"]["models"][variant]),
                ("SiLU, линейный выход", summary["phases"]["hidden_activation"]["silu"]["models"][variant]),
                ("Общий выбор по validation", summary["selected_models"][variant])]
        for color, (label, path) in zip(colors, rows):
            data = summary["test"][path]["forecast"]
            axis.plot(horizons, [data[str(ms)]["velocity_rmse_m_s"]*1000 for ms in horizons], "o-", label=label, color=color)
        old = audit["variants"][variant]["historical_120ms_training"]["test"]
        axis.plot(horizons, [old[str(ms)]["velocity_rmse_m_s"]*1000 for ms in horizons], "k--", label="Прежний Tanh, BPTT 120 мс")
        axis.set_xscale("log")
        axis.set_xticks(horizons, list(map(str, horizons)))
        axis.set_yscale("log")
        axis.set_title(f"Механика {variant}")
        axis.set_xlabel("Горизонт прогноза, мс")
        axis.set_ylabel("Test RMSE скорости, мм/с")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("20/60 мс — основные горизонты; 120/400 мс — дополнительная диагностика")
    save(fig, "forecast_horizons")


if __name__ == "__main__":
    main()
