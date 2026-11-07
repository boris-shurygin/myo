"""Export sensory-rate comparisons from completed local experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_sampling"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Complete training and independent evaluation before plotting")
    variants = list(summary["variants"])
    x = np.arange(len(variants))
    colors = ["#367bb5", "#d28328", "#25977f"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for index, (ms, color) in enumerate(zip(("20", "5", "1"), colors)):
        offset = (index-1)*.25
        values = [summary["variants"][v]["direct"][ms]["test"]["20"]["fiber_length_rmse_mm"]*1000 for v in variants]
        axes[0].bar(x+offset, values, .23, color=color, label=f"Ощущения через {ms} мс")
        values = [summary["variants"][v]["rollout"][ms]["test"]["60"]["fiber_length_rmse_mm"]*1000 for v in variants]
        axes[1].bar(x+offset, values, .23, color=color, label=f"Шаг прогноза {ms} мс")
        values = [audit["variants"][v]["sensitivities"][ms]["relative_jacobian_rmse"] for v in variants]
        axes[2].bar(x+offset, values, .23, color=color)
    for axis in axes:
        axis.set_xticks(x, [f"Вариант {v}" for v in variants])
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    for axis, ms in zip(axes[:2], ("20", "60")):
        baseline = [audit["variants"][v]["baselines"]["constant_rate"][ms]["fiber_length_rmse_mm"]*1000 for v in variants]
        axis.plot(x, baseline, "kD", label="Постоянная скорость", markersize=5)
        axis.set_yscale("log")
        axis.set_ylabel("Test RMSE длины волокна, мкм")
        axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2)
    axes[0].set_title("Прямой прогноз через 20 мс\nРазличается только частота истории")
    axes[1].set_title("Собственные прогнозы до 60 мс\nИстория и переход: 20 / 5 / 1 мс")
    axes[2].set_title("Влияние новой команды к 60 мс\nШесть состояний на вариант")
    axes[2].axhline(1, color="black", linestyle="--", label="Нулевой отклик")
    axes[2].set_ylabel("Ошибка производной / ошибка нулевого отклика")
    axes[2].legend(fontsize=8)
    fig.suptitle("История ощущений 60 мс; команды каждые 20 мс; один seed", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(args.run/f"sampling_comparison.{suffix}", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, variant in zip(axes.ravel(), variants):
        for ms, color in zip(("20", "5", "1"), colors):
            branch = summary["variants"][variant]["rollout"][ms]
            history = branch["history"]
            axis.plot([row["update"] for row in history], [row["validation_score"] for row in history],
                      color=color, label=f"Шаг {ms} мс")
            best = next(row for row in history if row["update"] == branch["best_update"])
            axis.plot(best["update"], best["validation_score"], "o", color=color)
        axis.set_title(f"Вариант {variant}")
        axis.set_xlabel("Обновления BPTT")
        axis.set_ylabel("Validation RMSE / STD приращений за 20 мс")
        axis.set_yscale("log")
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle("Средний показатель на 20 и 60 мс; точки — выбранные checkpoints")
    for suffix in ("png", "svg"):
        fig.savefig(args.run/f"learning_curves.{suffix}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
