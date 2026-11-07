"""Increase the update budget for the original four-mechanics LR experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.train_vertical_history import read_source
from myo.train_vertical_tuning import (
    TuningPredictor, data_identity, evaluate_test, load_model, output_coverage,
    train_stage, tuning_contract,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("runs/vertical_predictor"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_budget"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .0003, .0001, .00003])
    parser.add_argument("--one-step-updates", type=int, default=10000)
    parser.add_argument("--bptt-updates", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    summary = {
        "schema": "myo.vertical_budget_experiment.v1", "status": "running",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "scope": "Original state, no history; Tanh 128x128, linear output; 20ms transitions, 60ms BPTT",
        "restart": "Same seed from scratch; old checkpoints do not store optimizer state",
        "sources": {}, "trials": {}, "phases": {}, "selection": {},
    }
    contexts = {}

    def report():
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    for variant in range(1, 5):
        source = args.reference/f"variant_{variant}"
        metadata = json.loads((source/"bptt.pt.metadata.json").read_text())
        env = VerticalMuscles(VerticalConfig(**metadata["config"]))
        episodes = read_source(source, env)
        identity = data_identity(source)
        contexts[str(variant)] = env, episodes, identity
        summary["sources"][str(variant)] = {"path": str(source), "data_sha256": identity}

    def new_model(variant):
        env, episodes, _ = contexts[variant]
        torch.manual_seed(0)
        model = TuningPredictor(env.config.physical_size, env.config.queue_steps)
        model.fit_normalization([e for e in episodes if e["split"] == "train"])
        return model

    def read_trial(relative):
        record = summary["trials"][relative]
        env, _, identity = contexts[str(record["variant"])]
        prototype = TuningPredictor(**record["contract"]["architecture"])
        return load_model(args.output_dir/relative, tuning_contract(env, prototype, identity))[0]

    def trial(variant, folder, key, model, stage, lr, parent=None):
        env, episodes, identity = contexts[variant]
        relative = f"variant_{variant}/{folder}/{key}.pt"
        path = args.output_dir/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = train_stage(env, episodes, model, args, path, stage, lr, identity, parent)
        # Compare the short and long budgets on the SAME optimization trajectory.
        # These are validation diagnostics, not extra test-driven selections.
        short_budget = 1000 if stage == "one_step" else 2000
        prefix = [row for row in record["history"] if row["update"] <= short_budget]
        best_prefix = min(prefix, key=lambda row: row["validation"]["score"])
        record["short_budget_prefix"] = {
            "budget": short_budget, "best_update": best_prefix["update"],
            "best_score": best_prefix["validation"]["score"],
        }
        summary["trials"][relative] = record
        report()
        return relative

    def select(phase):
        candidates = summary["phases"][phase]
        for candidate in candidates.values():
            candidate["mean_validation_score"] = float(np.mean([
                summary["trials"][p]["validation"]["score"] for p in candidate["models"].values()
            ]))
        winner = min(candidates, key=lambda key: candidates[key]["mean_validation_score"])
        summary["selection"][phase] = winner
        print(f"SELECT {phase}: {winner}", flush=True)
        report()
        return candidates[winner]["models"]

    report()
    summary["phases"]["warmup_lr"] = {}
    for lr in args.learning_rates:
        key = f"{lr:.0e}"
        models = {}
        summary["phases"]["warmup_lr"][key] = {"learning_rate": lr, "models": models}
        for variant in contexts:
            models[variant] = trial(variant, "lr_warmup", key, new_model(variant), "one_step", lr)
    warm = select("warmup_lr")
    warm_lr = summary["phases"]["warmup_lr"][summary["selection"]["warmup_lr"]]["learning_rate"]

    summary["phases"]["bptt_lr"] = {}
    for lr in args.learning_rates:
        key = f"{lr:.0e}"
        models = {}
        summary["phases"]["bptt_lr"][key] = {"learning_rate": lr, "models": models}
        for variant, parent in warm.items():
            models[variant] = trial(variant, "lr_bptt", key, read_trial(parent), "bptt", lr, parent)
    final = select("bptt_lr")
    bptt_lr = summary["phases"]["bptt_lr"][summary["selection"]["bptt_lr"]]["learning_rate"]

    summary["phases"]["lr_pair"] = {
        "sequential": {"warmup_lr": warm_lr, "bptt_lr": bptt_lr, "models": final}
    }
    # A sequential warm-up choice need not select the best final pair.
    if .001 in args.learning_rates:
        if warm_lr == .001:
            default = summary["phases"]["bptt_lr"]["1e-03"]["models"]
        else:
            default = {}
            for variant, parent in summary["phases"]["warmup_lr"]["1e-03"]["models"].items():
                default[variant] = trial(variant, "default_control", "bptt", read_trial(parent), "bptt", .001, parent)
        summary["phases"]["lr_pair"]["default"] = {
            "warmup_lr": .001, "bptt_lr": .001, "models": default
        }
    summary["selected_models"] = select("lr_pair")
    summary["selection_frozen_before_test"] = json.loads(json.dumps(summary["selection"]))
    report()
    summary["test"] = {}
    candidates = {path for phase in ("bptt_lr", "lr_pair")
                  for candidate in summary["phases"][phase].values() for path in candidate["models"].values()}
    for path in sorted(candidates):
        _, episodes, _ = contexts[str(summary["trials"][path]["variant"])]
        model = read_trial(path)
        summary["test"][path] = {
            "forecast": evaluate_test(model, episodes), "output_bound": model.net[-1].bound.tolist(),
            "coverage": output_coverage(model, episodes),
        }
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    report()


if __name__ == "__main__":
    main()
