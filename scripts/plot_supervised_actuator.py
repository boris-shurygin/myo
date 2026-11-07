"""Plot saved supervised-control results without running or changing physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d1_supervised_actuator"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    report = json.loads((args.run / "independent_evaluation.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    styles = {"actions": ("Command targets", "tab:blue"),
              "accelerations": ("Acceleration targets", "tab:orange"),
              "heuristic": ("Heuristic", "tab:green")}
    for name, (label, color) in styles.items():
        result = (report["heuristic_hold20s"] if name == "heuristic"
                  else report["branches"][name]["hold20s"])
        trace = np.load(result["trace_path"])["pre_action_samples"].reshape(6, 1000, -1)
        for index, case in enumerate(trace):
            axes[0, 0].plot(case[:, 0], np.rad2deg(case[:, 1]), color=color,
                            alpha=.65, lw=1.2, label=label if index == 0 else None)
        # Fixed trigger=1, initial capacity=.8; mean within each antagonist group.
        case = trace[5]
        axes[0, 1].plot(case[:, 0], case[:, 4:7].mean(1), color=color, label=f"{label}: flexors")
        axes[0, 1].plot(case[:, 0], case[:, 7:10].mean(1), color=color,
                        ls="--", label=f"{label}: extensors")
        axes[1, 1].plot(case[:, 0], case[:, 3], color=color, label=label)
        if name != "heuristic":
            checks = summary["branches"][name]["evaluations"]
            x = [r["updates"] for r in checks]
            y = [r["score"] for r in checks]
            axes[1, 0].plot(x, y, "o-", color=color, ms=4, label=label)
            first_second_round = next(r for r in checks if r["round"] == 2)
            i = checks.index(first_second_round)
            axes[1, 0].axvline(checks[i - 1]["updates"], color=color, ls=":", alpha=.7)
    axes[0, 0].axhspan(27, 33, color="grey", alpha=.12, label="Position tolerance")
    axes[0, 0].axhline(30, color="black", ls=":", lw=1)
    axes[0, 0].set(title="Frozen actors: all six 20 s checks", xlabel="Time (s)", ylabel="Angle (deg)")
    axes[0, 1].set(title="Trigger=1, initial capacity=.8", xlabel="Time (s)", ylabel="Mean muscle command (0..1)")
    axes[1, 0].set(title="4 s closed-loop training checks; dotted: added actor states",
                   xlabel="Supervised optimizer updates", ylabel="Tracking score (lower is better)", yscale="log")
    axes[1, 0].set_yticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
    axes[1, 1].set(title="Trigger=1, initial capacity=.8", xlabel="Time (s)", ylabel="Minimum capacity proxy")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Instant activation, zero motor delay, rigid tendon; 1 kg, target 30 deg\n"
                 "Same initial MLP and shared data; no PPO/GAE; one fixed seed")
    for suffix in ("png", "svg"):
        fig.savefig(args.run / f"comparison.{suffix}", dpi=160)
    print(args.run / "comparison.png")


if __name__ == "__main__":
    main()
