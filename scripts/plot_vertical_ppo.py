"""Export PPO learning curves and native closed-loop motion traces."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_ppo"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Finish training and independent evaluation first")
    colors = ["#367bb5", "#d28328", "#25977f", "#ac5399"]

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for color, (variant, record) in zip(colors, summary["variants"].items()):
        rows = record["history"]
        steps = [r["steps"] for r in rows]
        label = f"Механика {variant}"
        axes[0, 0].plot(steps, [r["validation"]["score_m"]*1000 for r in rows], "o-", color=color, label=label)
        axes[0, 1].plot(steps, [r["validation"]["passed"] for r in rows], "o-", color=color, label=label)
        axes[1, 0].plot([r["steps"] for r in record["audit"]], [r["critic_ev"] for r in record["audit"]], color=color, label=label)
        axes[1, 1].plot([r["steps"] for r in record["audit"]], [np.mean(r["excitation_std"]) for r in record["audit"]], color=color, label=label)
    for axis, title, ylabel in zip(axes.ravel(),
            ("Validation: установившееся положение", "Validation: пройденные случаи", "Critic: explained variance на rollout", "Шум исследования действий"),
            ("Средний RMSE последней секунды, мм", "Количество из 5", "Explained variance", "Средний STD возбуждения")):
        axis.set_title(title)
        axis.set_xlabel("Физические переходы PPO по 20 мс")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0, 0].set_yscale("log")
    fig.suptitle("Прямой PPO по результатам движения; отдельные actor/critic 128×128, один seed")
    save(fig, "learning_curves")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, (variant, record) in enumerate(summary["variants"].items()):
        folder = args.run/f"variant_{variant}"
        test = json.loads((folder/"test.json").read_text())
        for case in test["cases"][:2]:
            a = np.asarray(case["samples"])
            axes[0, column].plot(a[:, 0], a[:, 1]*1000, label=f"{case['initial_position_m']*1000:+.0f}→{case['target_m']*1000:+.0f} мм")
            axes[0, column].axhline(case["target_m"]*1000, color="gray", linestyle="--", alpha=.5)
        continuation = json.loads((folder/"continuation.json").read_text())
        a = np.asarray(continuation["cases"][0]["samples"])
        axes[1, column].plot(a[:, 0], a[:, 1]*1000, label="Груз")
        axes[1, column].plot(a[:, 0], a[:, 3]*1000, "k--", label="Цель")
        axes[0, column].set_title(f"Механика {variant}: test")
        axes[1, column].set_title(f"Механика {variant}: без reset через 4 с")
        for row in (0, 1):
            axes[row, column].set_xlabel("Время, с")
            axes[row, column].set_ylabel("Положение, мм")
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(fontsize=8)
    fig.suptitle("Сохранённый best checkpoint; действия — среднее Beta; движение создаёт исходная физика")
    save(fig, "closed_loop")


if __name__ == "__main__":
    main()
