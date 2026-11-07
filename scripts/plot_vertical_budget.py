"""Compare short and extended budgets without selecting on test results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_budget"))
    parser.add_argument("--previous", type=Path, default=Path("runs/vertical_tuning"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    previous = json.loads((args.previous/"summary.json").read_text())
    audit = json.loads((args.run/"independent_evaluation.json").read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Finish training and independent evaluation first")
    colors = ["#367bb5", "#d28328", "#25977f", "#ac5399"]

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(args.run/f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    # Verify that the short prefix of each warm-up actually reproduces the old run.
    # BPTT uses the new selected parent and is compared only to its own prefix.
    for key, candidate in summary["phases"]["warmup_lr"].items():
        for variant, path in candidate["models"].items():
            old_path = previous["phases"]["warmup_lr"][key]["models"][variant]
            old_rows = previous["trials"][old_path]["history"]
            rows = summary["trials"][path]["history"][:len(old_rows)]
            assert rows == old_rows

    comparison = {"warmup_short_prefix_reproduced": True, "phases": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, phase, title in zip(axes, ("warmup_lr", "bptt_lr"),
                                 ("Одношаговая подготовка: 1 000 → 10 000", "BPTT: 2 000 → 20 000; общий новый старт")):
        comparison["phases"][phase] = {}
        candidates = summary["phases"][phase]
        keys = list(candidates)
        short, long = [], []
        for key, candidate in candidates.items():
            records = [summary["trials"][p] for p in candidate["models"].values()]
            prefix_mean = float(np.mean([r["short_budget_prefix"]["best_score"] for r in records]))
            comparison["phases"][phase][key] = {
                "short_prefix_mean": prefix_mean, "long_mean": candidate["mean_validation_score"],
                "best_updates": [r["best_update"] for r in records],
                "best_to_final_validation_ratio": [r["best_score"]/r["history"][-1]["validation"]["score"] for r in records],
            }
            short.append(prefix_mean)
            long.append(candidate["mean_validation_score"])
        x = np.arange(len(keys))
        axis.bar(x-.18, short, .34, label="Короткий префикс той же траектории", color="#a6b4c5")
        axis.bar(x+.18, long, .34, label="Весь увеличенный бюджет", color="#367bb5")
        axis.set_xticks(x, keys)
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("Средний best validation score, меньше лучше")
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.18), fontsize=8)
    fig.suptitle("Исходные четыре механики, Tanh 128×128, линейный выход, без истории")
    save(fig, "budget_comparison")

    for metric, label, filename in (("score", "Validation velocity score", "validation_curves"),
                                    ("normalized_mse", "Train MSE состояния / STD приращений²", "train_curves")):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for row, phase in enumerate(("warmup_lr", "bptt_lr")):
            for col, variant in enumerate(("1", "2", "3", "4")):
                axis = axes[row, col]
                for color, (key, candidate) in zip(colors, summary["phases"][phase].items()):
                    record = summary["trials"][candidate["models"][variant]]
                    history = record["history"]
                    split = "validation" if metric == "score" else "train_fixed"
                    axis.plot([r["update"] for r in history], [r[split][metric] for r in history], color=color, label=key)
                    if metric == "score":
                        selected = next(r for r in history if r["update"] == record["best_update"])
                        axis.plot(selected["update"], selected[split][metric], "o", color=color, markersize=4)
                axis.axvline(1000 if row == 0 else 2000, color="black", linestyle="--", alpha=.4)
                axis.set_title(f"Механика {variant}: {'подготовка' if row == 0 else 'BPTT'}")
                axis.set_xlabel("Обновления Adam")
                axis.set_yscale("log")
                axis.set_ylabel(label, fontsize=8)
                axis.grid(alpha=.2)
                if col == 0:
                    axis.legend(fontsize=8)
        fig.suptitle("Точки — выбранные updates; пунктир — прежний бюджет" if metric == "score"
                     else "Train-диагностика на фиксированных трёхшаговых окнах, не minibatch loss")
        save(fig, filename)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(4)
    old_models = previous["phases"]["lr_pair"]["default"]["models"]
    default_models = summary["phases"]["lr_pair"]["default"]["models"]
    candidates = (("Прежний бюджет, lr 0,001", previous, old_models),
                  ("Бюджет ×10, lr 0,001", summary, default_models),
                  ("Бюджет ×10, общий выбор validation", summary, summary["selected_models"]))
    for axis, horizon in zip(axes, ("20", "60")):
        for index, (label, run, models) in enumerate(candidates):
            values = [run["test"][models[str(v)]]["forecast"][horizon]["velocity_rmse_m_s"]*1000 for v in range(1, 5)]
            axis.bar(x+(index-1)*.25, values, .23, label=label, color=colors[index])
        axis.set_xticks(x, [f"Механика {v}" for v in range(1, 5)])
        axis.set_yscale("log")
        axis.set_ylabel("Test velocity RMSE, мм/с")
        axis.set_title(f"Прогноз через {horizon} мс")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)
    fig.suptitle("Test вычислен после фиксации выбора; все варианты — Tanh/linear")
    save(fig, "test_comparison")
    (args.run/"budget_comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
