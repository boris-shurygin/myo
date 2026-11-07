"""Freeze all trial selections before test, and replay policies in native physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/vertical_architecture"))
    parser.add_argument("--include-combined", action="store_true")
    args = parser.parse_args()
    import torch
    from myo.train_vertical_architecture import TRIALS, load_trial, optimizer_steps
    from myo.train_vertical_ppo import compact, write_json
    from myo.vertical_control import VerticalControlEnv, VALIDATION_CASES, TEST_CASES, evaluate_policy
    from myo.vertical_muscles import VerticalConfig, VerticalMuscles
    torch.set_num_threads(1)
    names = list(TRIALS)+(["combined"] if args.include_combined else [])
    summaries = {name: json.loads((args.root/name/"summary.json").read_text()) for name in names}
    control = summaries["control"]
    for name, summary in summaries.items():
        assert summary["status"] in ("training_complete", "complete")
        assert summary["steps"] == summary["training"]["transition_budget"]
        assert len(summary["history"]) == len(control["history"])
        assert summary["contract"] == control["contract"]
        if summary["training"]["trial"]["hidden"] == control["training"]["trial"]["hidden"]:
            assert summary["initial_parameter_sha256"] == control["initial_parameter_sha256"]
        for key in control["training"]:
            if key not in ("trial", "optimizer_defaults"):
                assert summary["training"][key] == control["training"][key], (name, key)
    if args.include_combined:
        assert summaries["combined"]["initial_parameter_sha256"] == summaries["deep"]["initial_parameter_sha256"]
    write_json(args.root/"frozen_selection.json", {
        "all_training_completed_before_test": True,
        "selections": {name: {"steps": summary["best_steps"],
            "checkpoint_sha256": json.loads((args.root/name/"best.zip.metadata.json").read_text())["checkpoint_sha256"]}
            for name, summary in summaries.items()}})
    audit = {"status": "running", "strict_checkpoints": 0, "native_replays": 0, "trials": {}}
    config = VerticalConfig(variant=4)
    env = VerticalControlEnv(config)
    for name, summary in summaries.items():
        folder = args.root/name
        # Reload optimizer/architecture too, rather than evaluating live weights.
        for checkpoint in ("initial", "last", "best"):
            loaded, metadata = load_trial(folder/f"{checkpoint}.zip", env)
            assert loaded.policy.optimizer.state_dict()["param_groups"]
            audit["strict_checkpoints"] += 1
        model = loaded
        validation = evaluate_policy(model, config, VALIDATION_CASES)
        assert compact(validation) == summary["best_validation"]
        rank = lambda result: (result["failures"], -result["passed"], result["score_m"])
        selected = min(summary["history"], key=lambda row: rank(row["validation"]))
        assert selected["steps"] == summary["best_steps"] == metadata["trained_steps"]
        assert rank(validation) == rank(selected["validation"])
        best = evaluate_policy(model, config, TEST_CASES)
        write_json(folder/"test.json", best)
        for case in best["cases"]:
            native = VerticalMuscles(config)
            native.reset(case["initial_position_m"])
            for t, q, v, target, u0, u1 in case["samples"]:
                try:
                    native.step([u0, u1])
                except RuntimeError:
                    assert case["physical_failure"]
                np.testing.assert_array_equal([native.time, native.position, native.velocity], [t, q, v])
            audit["native_replays"] += 1
        last, _ = load_trial(folder/"last.zip", env)
        assert optimizer_steps(last) == summary["optimizer_steps"]
        last_test = evaluate_policy(last, config, TEST_CASES)
        switches = evaluate_policy(model, config, ((0., .06),), seconds=6., switches=True)
        write_json(folder/"last_test.json", last_test)
        write_json(folder/"switches.json", switches)
        summary.update(status="complete", test=compact(best), last_test=compact(last_test),
                       switches=compact(switches))
        write_json(folder/"summary.json", summary)
        audit["trials"][name] = {"best_steps": summary["best_steps"], "test": compact(best),
                                  "last_test": compact(last_test), "validation": compact(validation)}
        print(json.dumps({"trial": name, "test_steady_mm": best["score_m"]*1000,
                          "passed": best["passed"], "last_test_mm": last_test["score_m"]*1000}), flush=True)
    audit["status"] = "passed"
    write_json(args.root/"independent_evaluation.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
    main()
