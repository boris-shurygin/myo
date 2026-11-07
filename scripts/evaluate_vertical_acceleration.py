"""Strictly replay acceleration PPO and its original position-reward control."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from myo.train_vertical_ppo import compact, write_json
from myo.vertical_acceleration import (
    compact_acceleration, evaluate_acceleration_policy, load_acceleration_policy,
)
from myo.vertical_control import (
    VALIDATION_CASES, TEST_CASES, VerticalControlEnv, load_policy,
)
from myo.vertical_muscles import VerticalConfig


def position_metrics(result):
    return {k: ([{q: v for q, v in c.items() if q != "acceleration_rmse_m_s2"} for c in value]
                if k == "cases" else value)
            for k, value in result.items() if k != "mean_case_acceleration_rmse_m_s2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_acceleration_ppo"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run/"summary.json").read_text())
    assert summary["status"] == "complete" and summary["all_selections_frozen_before_test"]
    assert summary["runner_sha256"] == hashlib.sha256(Path("src/myo/train_vertical_acceleration.py").read_bytes()).hexdigest()
    assert summary["orchestrator_sha256"] == hashlib.sha256(Path("scripts/run_vertical_acceleration.py").read_bytes()).hexdigest()
    baseline_path = Path(summary["baseline"])
    assert summary["baseline_sha256"] == hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    baseline = json.loads(baseline_path.read_text())
    assert baseline["status"] == "complete"
    preflight = json.loads(Path(summary["control_preflight"]).read_text())
    original = baseline["variants"]["4"]
    assert position_metrics(preflight["initial_validation"]) == original["initial_validation"]
    assert position_metrics(preflight["history"][0]["validation"]) == original["history"][0]["validation"]
    assert preflight["history"][0]["ppo"] == original["history"][0]["ppo"]
    for a, b in zip(preflight["audit"], original["audit"]):
        assert all(a[k] == b[k] for k in ("steps", "critic_ev", "value_mean", "return_mean",
                                          "excitation_mean", "excitation_std", "out_of_bounds_actions"))
    result = {"status": "running", "strict_checkpoints": 0, "control_prefix_exact_steps": 8192,
              "variants": {}}
    for variant in ("1", "2", "3", "4"):
        record = summary["variants"][variant]
        folder = args.run/f"variant_{variant}"
        config = VerticalConfig(variant=int(variant))
        models = {}
        for name in ("initial", "best", "last"):
            model, env, metadata = load_acceleration_policy(folder/f"{name}.zip")
            assert metadata["training"] == record["training"] and env.objective == "acceleration"
            assert model.n_steps == 1024 and model.gamma == .99 and model.gae_lambda == .95
            models[name] = model
            result["strict_checkpoints"] += 1
        original_initial, _ = load_policy(baseline_path.parent/f"variant_{variant}"/"initial.zip", VerticalControlEnv(config))
        original_best, _ = load_policy(baseline_path.parent/f"variant_{variant}"/"best.zip", VerticalControlEnv(config))
        result["strict_checkpoints"] += 2
        for key, tensor in models["initial"].policy.state_dict().items():
            torch.testing.assert_close(tensor, original_initial.policy.state_dict()[key], rtol=0, atol=0)
        assert models["initial"].num_timesteps == 0
        assert models["last"].num_timesteps == record["trained_steps"]
        history = [(0, record["initial_validation"])]+[(h["steps"], h["validation"]) for h in record["history"]]
        best_step, best_metrics = min(history, key=lambda x: (x[1]["failures"], -x[1]["passed"], x[1]["score_m"]))
        assert best_step == record["best_steps"] == models["best"].num_timesteps
        assert best_metrics == record["selected_validation"]
        frozen = json.loads((folder/"best.zip.metadata.json").read_text())["checkpoint_sha256"]
        assert frozen == record["selection_frozen_before_test"]["checkpoint_sha256"]
        assert compact_acceleration(evaluate_acceleration_policy(models["initial"], config, VALIDATION_CASES)) == record["initial_validation"]
        assert compact_acceleration(evaluate_acceleration_policy(models["last"], config, VALIDATION_CASES)) == record["history"][-1]["validation"]
        if record["stop_reason"] == "validation_ready_twice":
            assert all(row["validation"]["ready"] for row in record["history"][-2:])
        assert all(a["out_of_bounds_actions"] == 0 for a in record["audit"])
        row = {"baseline": {}, "acceleration": {}}
        for label, model in (("baseline", original_best), ("acceleration", models["best"])):
            for name, cases, options in (("validation", VALIDATION_CASES, {}), ("test", TEST_CASES, {}),
                                         ("switches", ((0., .06),), {"seconds": 6., "switches": True})):
                evaluated = evaluate_acceleration_policy(model, config, cases, **options)
                metrics = compact_acceleration(evaluated)
                row[label][name] = metrics
                field = "selected_validation" if name == "validation" else name
                if label == "baseline":
                    assert position_metrics(metrics) == baseline["variants"][variant][field]
                else:
                    assert metrics == record[field]
                write_json(folder/f"{label}_{name}_replay.json", evaluated)
        result["variants"][variant] = row
        write_json(args.run/"independent_evaluation.json", result)
        print(f"v{variant}: five strict checkpoints, initial weights, best selection and physical replays checked", flush=True)
    result["status"] = "passed"
    write_json(args.run/"independent_evaluation.json", result)
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
