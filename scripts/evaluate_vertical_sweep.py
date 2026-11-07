"""Independently reload one-factor PPO branches and repeat physical metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_vertical_ppo import compact, write_json
from myo.train_vertical_sweep import actual_parameters, candidates, optimizer_steps
from myo.vertical_control import VALIDATION_CASES, TEST_CASES, VerticalControlEnv, evaluate_policy, load_policy
from myo.vertical_muscles import VerticalConfig
from myo.vertical_sweep_env import load_sweep_policy


def probe_bank(model, config):
    """Same causal observations for comparing exploration STD between policies."""
    observations = []
    for position, target in VALIDATION_CASES:
        env = VerticalControlEnv(config)
        obs, _ = env.reset(options={"initial_position_m": position, "target_m": target})
        for step in range(200):
            if step in (0, 25, 50, 100, 150, 190):
                observations.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return torch.tensor(np.asarray(observations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_ppo_sweep"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run/"summary.json").read_text())
    assert summary["status"] == "complete" and summary["all_selections_frozen_before_test"]
    assert summary["runner_sha256"] == hashlib.sha256(Path("src/myo/train_vertical_sweep.py").read_bytes()).hexdigest()
    assert summary["orchestrator_sha256"] == hashlib.sha256(Path("scripts/run_vertical_sweep.py").read_bytes()).hexdigest()
    assert summary["matrix"] == json.loads(json.dumps(candidates()))
    expected = {f"variant_{v}/{name}" for v in summary["arguments"]["variants"] for name, _ in candidates()}
    assert set(summary["trials"]) == expected
    result = {"status": "running", "strict_checkpoints": 0, "trials": {}, "parallel_reproduction": None}
    banks, parent_validation = {}, {}
    for variant, parent in summary["parents"].items():
        env = VerticalControlEnv(VerticalConfig(variant=int(variant)))
        model, metadata = load_policy(parent["path"], env)
        assert metadata["checkpoint_sha256"] == parent["checkpoint_sha256"]
        assert optimizer_steps(model) == parent["optimizer_steps"]
        banks[variant] = probe_bank(model, env.physical.config)
        parent_validation[variant] = compact(evaluate_policy(model, env.physical.config, VALIDATION_CASES))
        np.save(args.run/f"parent_probe_bank_{variant}.npy", banks[variant].numpy())
    for key, record in summary["trials"].items():
        folder = args.run/key
        origin = summary["parents"][str(record["variant"]) ]
        assert record["training"]["parent_checkpoint_sha256"] == origin["checkpoint_sha256"]
        assert record["training"]["parent_optimizer_steps"] == origin["optimizer_steps"]
        assert record["training"]["changes"] == record["changes"]
        assert record["history"][0]["validation"] == parent_validation[str(record["variant"])]
        original = origin["training"]
        baseline = {k: original.get(k, .08 if k == "position_scale_m" else None)
                    for k in record["training"]["parameters"]}
        changed = {k: v for k, v in record["training"]["parameters"].items() if v != baseline[k]}
        assert changed == record["changes"] and len(changed) <= 1
        for name in ("best", "last"):
            model, env, metadata = load_sweep_policy(folder/f"{name}.zip")
            assert metadata["training"] == record["training"]
            assert actual_parameters(model, env) == record["training"]["parameters"]
            assert model.rollout_buffer.gamma == model.gamma
            assert model.rollout_buffer.gae_lambda == model.gae_lambda
            assert model.rollout_buffer.buffer_size == model.n_steps
            result["strict_checkpoints"] += 1
            if name == "last":
                assert model.num_timesteps-origin["trained_steps"] == summary["arguments"]["steps"]
                assert optimizer_steps(model)-origin["optimizer_steps"] == record["optimizer_updates"]
                assert compact(evaluate_policy(model, env.physical.config, VALIDATION_CASES)) == record["history"][-1]["validation"]
                continue
            selected = min(record["history"], key=lambda r: (r["validation"]["failures"], -r["validation"]["passed"], r["validation"]["score_m"]))
            assert selected["additional_steps"] == record["best_additional_steps"]
            assert model.num_timesteps-origin["trained_steps"] == record["best_additional_steps"]
            assert optimizer_steps(model)-origin["optimizer_steps"] == selected["optimizer_updates"]
            assert record["selection_frozen_before_test"]["checkpoint_sha256"] == metadata["checkpoint_sha256"]
            assert compact(evaluate_policy(model, env.physical.config, VALIDATION_CASES)) == record["selected_validation"]
            assert compact(evaluate_policy(model, env.physical.config, TEST_CASES)) == record["test"]
            assert compact(evaluate_policy(model, env.physical.config, ((0., .06),), seconds=6., switches=True)) == record["switches"]
            with torch.no_grad():
                dist = model.policy.get_distribution(banks[str(record["variant"]) ]).distribution
            result["trials"][key] = {"fixed_probe_count": len(banks[str(record["variant"]) ]),
                "fixed_probe_excitation_mean": dist.mean.mean(0).tolist(),
                "fixed_probe_excitation_std": dist.stddev.mean(0).tolist()}
        assert all(row["out_of_bounds_actions"] == 0 for row in record["audit"])
        write_json(args.run/"independent_evaluation.json", result)
        print(f"checked {key}", flush=True)
    preflight = args.run.with_name(args.run.name+"_sequential_preflight")/"summary.json"
    if preflight.exists():
        old = json.loads(preflight.read_text())
        keys = [k for k, r in old["trials"].items() if r["status"] == "complete"]
        for key in keys:
            assert summary["trials"][key]["history"] == old["trials"][key]["history"]
            assert summary["trials"][key]["audit"] == old["trials"][key]["audit"]
        result["parallel_reproduction"] = {"exact_history_and_audit_trials": keys}
    result["status"] = "passed"
    write_json(args.run/"independent_evaluation.json", result)
    print({k: v for k, v in result.items() if k != "trials"})


if __name__ == "__main__":
    main()
