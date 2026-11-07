"""Export learning curves and the distinction between forecasts and sensitivities."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/tissue_capacity"))
    parser.add_argument("--quality", type=Path, default=Path("runs/tissue_predictor/quality_audit.json"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((args.run/"summary.json").read_text())
    quality = json.loads(args.quality.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for (variant, record), ax in zip(summary["variants"].items(), axes.flat):
        for (name, model), color in zip(record["models"].items(), ("C0", "C1", "C2")):
            stage = model["stages"]["bptt"]
            history = stage["history"]
            x = np.array([h["update"] for h in history])
            y = np.array([h["validation_score"] for h in history])
            label = "×".join(map(str, model["contract"]["architecture"]["hidden_sizes"]))
            ax.plot(x, y, color=color, alpha=.25, lw=1)
            ax.plot(x, np.minimum.accumulate(y), color=color, label=label)
            ax.plot(x[-1], min(y), marker="x" if stage["stop_reason"] == "validation_plateau" else "o", color=color)
        ax.set(title=f"Механика {variant}", xlabel="Обновления BPTT (128 окон на обновление)",
               ylabel="Нормированный RMSE скоростей, validation", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Размер сети: бледная линия — текущая ошибка, яркая — лучшая достигнутая\n"
                 "До BPTT: 1000 одношаговых обновлений; × — плато validation, ○ — лимит")
    for suffix in ("png", "svg"):
        fig.savefig(args.run/f"learning_curves.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = ("strain_history", "derivative_inputs", "derivative_loss")
    for i, name in enumerate(names):
        x = np.arange(4)+(i-1)*.24
        forecasts, jacobians = [], []
        for variant in map(str, range(1, 5)):
            record = quality["variants"][variant]
            forecasts.append(record["forecast"][name]["120"]["rate_normalized_rmse"]/
                             record["forecast"]["constant_strain_rate"]["120"]["rate_normalized_rmse"])
            jacobians.append(record["sensitivities"][name]["relative_rmse_to_zero"])
        label = ("История длин", "+ производные во входах", "+ ошибка производных")[i]
        axes[0].bar(x, forecasts, .24, label=label)
        axes[1].bar(x, jacobians, .24, label=label)
    for ax in axes:
        ax.axhline(1., color="black", ls="--", lw=1)
        ax.set(xticks=np.arange(4), xticklabels=["1", "2", "3", "4"], xlabel="Вариант механики")
        ax.grid(axis="y", alpha=.2)
    axes[0].set(title="Ошибка прогноза / ошибка постоянной скорости\n(горизонт 120 мс)", ylabel="Отношение ошибок: меньше — лучше")
    axes[1].set(title="Ошибка производной по команде /\nошибка нулевой производной", ylabel="Отношение ошибок: меньше — лучше")
    axes[0].legend(fontsize=8)
    fig.suptitle("Исходные сети 128×128: точность траектории и точность влияния команд")
    for suffix in ("png", "svg"):
        fig.savefig(args.run/f"forecast_and_sensitivity.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
