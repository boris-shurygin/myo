"""Reload vertical PPO checkpoints and independently repeat physical cases."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from myo.train_vertical_ppo import compact
from myo.vertical_control import (
    VerticalControlEnv, TRAIN_CASES, VALIDATION_CASES, TEST_CASES, evaluate_policy, load_policy,
)
from myo.vertical_muscles import VerticalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_ppo"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run/"summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Wait for training to finish")
    runner = Path("src/myo/train_vertical_ppo.py")
    assert summary["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    result = {"status": "running", "checkpoints": 0, "variants": {}}
    for variant, record in summary["variants"].items():
        config = VerticalConfig(variant=int(variant))
        env = VerticalControlEnv(config)
        folder = args.run/f"variant_{variant}"
        models = {}
        for name in ("initial", "best", "last"):
            models[name], metadata = load_policy(folder/f"{name}.zip", env)
            assert metadata["training"] == summary["training"]
            result["checkpoints"] += 1
        best = models["best"]
        observed = compact(evaluate_policy(best, config, VALIDATION_CASES))
        assert observed == record["selected_validation"]
        assert compact(evaluate_policy(best, config, TEST_CASES)) == record["test"]
        assert compact(evaluate_policy(best, config, ((0., .06),), seconds=6., switches=True)) == record["switches"]
        candidates = [(record["initial_steps"], record["initial_validation"])]+[
            (row["steps"], row["validation"]) for row in record["history"]]
        selected = min(candidates, key=lambda x: (x[1]["failures"], -x[1]["passed"], x[1]["score_m"]))
        assert selected[0] == record["best_steps"] == best.num_timesteps
        assert record["selection_frozen_before_test"] == {"steps": selected[0], "score_m": selected[1]["score_m"]}
        assert models["last"].num_timesteps == record["trained_steps"]
        assert all(row["out_of_bounds_actions"] == 0 for row in record["audit"])
        if record["stop_reason"] == "validation_ready_twice":
            assert all(row["validation"]["ready"] for row in record["history"][-2:])
            assert observed["ready"]
        # No reset is permitted when a live demo crosses the training time limit.
        continuation = evaluate_policy(best, config, ((0., .06),), seconds=12., switches=True)
        known_cases = evaluate_policy(best, config, TRAIN_CASES)
        long_validation = evaluate_policy(best, config, VALIDATION_CASES, seconds=12.)
        result["variants"][variant] = {"selected_steps": best.num_timesteps,
            "test_passed": record["test"]["passed"], "test_total": record["test"]["total"],
            "continuation_12s": compact(continuation),
            "known_training_cases": compact(known_cases),
            "validation_12s": compact(long_validation),
            "last_critic_ev": record["audit"][-1]["critic_ev"],
            "no_out_of_bounds_actions": True}
        (folder/"continuation.json").write_text(json.dumps(continuation, indent=2, allow_nan=False)+'\n')
        print(f"v{variant}: checkpoints, selection, validation/test/switches and live 12s continuation checked", flush=True)
    result["status"] = "passed"
    (args.run/"independent_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
