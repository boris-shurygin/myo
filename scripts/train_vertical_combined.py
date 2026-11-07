"""Combine independently helpful factors using completed validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/vertical_architecture"))
    args = parser.parse_args()
    from myo.train_vertical_architecture import TRIALS, main as train
    from myo.train_vertical_ppo import write_json
    names = ("control", "silu", "deep", "rmsprop")
    summaries = {name: json.loads((args.output_root/name/"summary.json").read_text()) for name in names}
    assert all(s["status"] == "training_complete" for s in summaries.values())
    rank = lambda s: (s["best_validation"]["failures"], -s["best_validation"]["passed"], s["best_validation"]["score_m"])
    assert all(rank(summaries[name]) < rank(summaries["control"]) for name in names[1:])
    spec = {"hidden": TRIALS["deep"]["hidden"], "activation": TRIALS["silu"]["activation"],
            "optimizer": TRIALS["rmsprop"]["optimizer"]}
    TRIALS["combined"] = spec
    write_json(args.output_root/"combination_protocol.json", {
        "selection_basis": "Only completed validation of independent factors; no test results accessed",
        "sources": {name: {"best_steps": s["best_steps"], "validation": s["best_validation"]}
                    for name, s in summaries.items()},
        "trial": spec, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    budget = summaries["control"]["training"]["transition_budget"]
    interval = summaries["control"]["history"][1]["steps"]
    sys.argv = [sys.argv[0], "--trial", "combined", "--output-root", str(args.output_root),
                "--steps", str(budget), "--eval-every", str(interval), "--defer-test"]
    train()


if __name__ == "__main__":
    main()
