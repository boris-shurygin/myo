"""Freeze validation selection, then compare dense/log PPO with the old combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/vertical_precision/combined"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/vertical_architecture/combined"))
    args = parser.parse_args()
    import torch
    from myo.train_vertical_architecture import load_trial, optimizer_steps
    from myo.train_vertical_ppo import compact, write_json
    from myo.vertical_control import VerticalControlEnv, evaluate_policy, TEST_CASES
    from myo.vertical_muscles import VerticalConfig, VerticalMuscles
    from myo.vertical_precision import VerticalPrecisionEnv, evaluate_precision, PRECISION_VALIDATION, PRECISION_TEST
    torch.set_num_threads(1)
    root = args.root
    summary = json.loads((root/"summary.json").read_text())
    baseline_summary = json.loads((args.baseline/"summary.json").read_text())
    assert summary["status"] == "training_complete"
    assert summary["steps"] == summary["training"]["transition_budget"] == baseline_summary["steps"]
    assert summary["training"]["trial"] == baseline_summary["training"]["trial"]
    assert summary["initial_parameter_sha256"] == baseline_summary["initial_parameter_sha256"]
    rank = lambda x: (x["failures"], -x["passed"], x["score_m"])
    selected = min(summary["history"], key=lambda h: rank(h["validation"]))
    assert selected["steps"] == summary["best_steps"]
    write_json(root/"frozen_selection.json", {
        "basis": "Expanded validation only; fixed budget completed before test",
        "best_steps": selected["steps"],
        "checkpoint_sha256": json.loads((root/"best.zip.metadata.json").read_text())["checkpoint_sha256"],
        "baseline_best_steps": baseline_summary["best_steps"],
        "test_note": "Historical five cases already examined in earlier experiments; six additional cases fixed before this run"})
    config = VerticalConfig(variant=4)
    env = VerticalPrecisionEnv(config)
    loaded = {}
    for checkpoint in ("initial", "best", "last"):
        loaded[checkpoint], _ = load_trial(root/f"{checkpoint}.zip", env)
    assert optimizer_steps(loaded["last"]) == summary["optimizer_steps"]
    validation = evaluate_precision(loaded["best"], config, PRECISION_VALIDATION)
    assert compact(validation) == summary["best_validation"]
    baseline, _ = load_trial(args.baseline/"best.zip", VerticalControlEnv(config))
    results = {}
    for name, model, evaluator in (("baseline", baseline, evaluate_policy),
                                    ("best", loaded["best"], evaluate_precision),
                                    ("last", loaded["last"], evaluate_precision)):
        results[name] = evaluator(model, config, PRECISION_TEST)
        write_json(root/f"{name}_test.json", results[name])
        old = results[name]["cases"][:len(TEST_CASES)]
        print(json.dumps({"model": name, "historical_test_mm": float(np.mean([c["last_second_rmse_m"] if not c["physical_failure"] else .3 for c in old]))*1000,
                          "historical_passed": sum(c["passed"] for c in old), "extended_test_mm": results[name]["score_m"]*1000,
                          "extended_passed": results[name]["passed"], "failures": results[name]["failures"]}), flush=True)
    replays = 0
    for case in results["best"]["cases"]:
        physical = VerticalMuscles(config)
        physical.reset(case["initial_position_m"])
        for t, q, v, target, u0, u1 in case["samples"]:
            try:
                physical.step([u0, u1])
            except RuntimeError:
                assert case["physical_failure"]
            np.testing.assert_array_equal([physical.time, physical.position, physical.velocity], [t, q, v])
        replays += 1
    switches = evaluate_precision(loaded["best"], config, ((0., .06),), seconds=6., switches=True)
    write_json(root/"switches.json", switches)
    summary.update(status="complete", test=compact(results["best"]), last_test=compact(results["last"]),
                   baseline_test=compact(results["baseline"]), switches=compact(switches))
    write_json(root/"summary.json", summary)
    write_json(root/"evaluation_audit.json", {"status": "passed", "strict_checkpoints": 4,
        "native_replays": replays, "initial_weights_identical": True, "validation_selection_reproduced": True})


if __name__ == "__main__":
    main()
