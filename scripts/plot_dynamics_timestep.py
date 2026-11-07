"""Plot matched physical-time prediction and derivative errors."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d1_dynamics_timestep"))
    args = parser.parse_args()
    summary = json.loads((args.run/"summary.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ("#637381", "#d68400", "#277da8", "#a44b9c")
    for color, (name, record) in zip(colors, summary["models"].items()):
        selected = record["acceleration"]
        horizons = sorted(map(int, selected["test"]))
        for ax, key, title, unit in ((axes[0, 0], "velocity_rmse_rad_s", "Velocity prediction", "RMSE, rad/s"),
                                     (axes[0, 1], "last_20ms_acceleration_rmse_rad_s2", "Acceleration, last 20 ms", "RMSE, rad/s²")):
            ax.plot(horizons, [selected["test"][str(h)][key] for h in horizons], "o-", label=name, color=color)
            ax.set(title=title, xlabel="Forecast horizon, ms", ylabel=unit)
        history = record["training"]["history"]
        axes[1, 0].plot([h["update"] for h in history], [h["velocity_score"] for h in history], color=color, label=name)
        derivative = selected["derivatives"]["velocity_60_to_120ms"]
        axes[1, 1].bar(name, derivative["global_cosine"], color=color)
    axes[1, 0].set(title="Validation: mean velocity RMSE at 20/60/120 ms", xlabel="Optimizer updates", ylabel="rad/s")
    axes[1, 1].set(title="Velocity Jacobian direction vs native physics", ylabel="Global cosine")
    axes[1, 1].axhline(0, color="black", linewidth=.8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    figure.suptitle("Same 1 ms native physics, 40 ms delay and 120 ms BPTT; acceleration-selected checkpoints")
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(args.run/f"comparison.{extension}", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
