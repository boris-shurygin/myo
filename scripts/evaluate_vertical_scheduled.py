"""Independently check scheduled tasks, native replays and trained PPO policies."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.train_vertical_ppo import write_json
from myo.train_vertical_scheduled import validation_rank
from myo.vertical_muscles import VerticalConfig, VerticalMuscles
from myo.vertical_scheduled import (
    ACCURACY, AccelerationCase, TRAIN_CASES, VALIDATION_CASES, TEST_CASES,
    ScheduledAccelerationEnv, compact_scheduled, evaluate_scheduled, load_scheduled_policy,
)


class ForceInverse:
    """Diagnostic force inverse; no teacher labels or simulated future states."""
    def __init__(self, config):
        self.config = config
        self.physical = VerticalMuscles(config)

    def predict(self, observation, deterministic=True):
        q, velocity = observation[0]*.1, observation[1]*.5
        lengths = observation[2:4]*.1+.48 if self.config.elastic else np.array([.48-q, .48+q])
        desired = observation[-1] if self.config.delayed else observation[-3]
        terms = [m._fiber_forces(float(length), 1., 1., float(v)) for m, length, v in
                 zip(self.physical.muscles, lengths, [-velocity, velocity])]
        coefficient = np.array([r[0] for r in terms])
        passive = np.array([r[1]+r[2] for r in terms]) if self.config.elastic else np.zeros(2)
        force = self.config.mass_kg*(9.81+desired)
        lower = max(passive[1]+5., passive[0]+5.-force)
        return np.clip((np.array([lower+force, lower])-passive)/np.maximum(coefficient, 1e-8), 0., 1.), None


class ConstantSupport:
    def __init__(self, config):
        physical = VerticalMuscles(config)
        physical.reset()
        self.action = physical.excitation.copy()

    def predict(self, observation, deterministic=True):
        return self.action, None


def verify_trace(case, config):
    issued = AccelerationCase(**case["case"]).commands()
    samples = np.asarray(case["samples"])
    lead = round(.04/config.dt)
    due = np.r_[np.zeros(lead), issued][:len(samples)]
    expected_issued = np.r_[issued, np.zeros(lead)][:len(samples)]
    np.testing.assert_array_equal(samples[:, 3], expected_issued)
    np.testing.assert_array_equal(samples[:, 4], due)
    dt = np.diff(np.r_[0., samples[:, 0]])
    actual = np.diff(np.r_[0., samples[:, 2]])/dt
    np.testing.assert_array_equal(actual, samples[:, 5])
    error = actual[lead:]-due[lead:]
    if len(error):
        assert np.isclose(np.sqrt(np.mean(error**2)), case["acceleration_rmse_m_s2"], rtol=0, atol=1e-14)
    if not case["physical_failure"]:
        assert len(samples) == len(issued)+lead and case["scored_steps"] == len(issued)
    # Replay commands through native mechanics: task/reward state cannot affect q/v.
    native = VerticalMuscles(config)
    native.reset(case["case"]["initial_position_m"])
    for sample in samples:
        try:
            native.step(sample[6:8])
        except RuntimeError as exc:
            if "left the benchmark workspace" not in str(exc):
                raise
        np.testing.assert_array_equal([native.time, native.position, native.velocity], sample[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/vertical_scheduled_ppo"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run/"summary.json").read_text())
    assert summary["status"] == "complete" and summary["all_selections_frozen_before_test"]
    assert summary["runner_sha256"] == hashlib.sha256(Path("src/myo/train_vertical_scheduled.py").read_bytes()).hexdigest()
    assert summary["orchestrator_sha256"] == hashlib.sha256(Path("scripts/run_vertical_scheduled.py").read_bytes()).hexdigest()
    result = {"status": "running", "strict_checkpoints": 0, "verified_test_trajectories": 0,
              "selection_revision": {}, "variants": {}}
    for variant in ("1", "2", "3", "4"):
        config = VerticalConfig(variant=int(variant))
        folder, record = args.run/f"variant_{variant}", summary["variants"][variant]
        trainer_source = Path(summary["reused_variants"][variant]["trainer_source"]) if variant in summary.get("reused_variants", {}) else Path("src/myo/train_vertical_scheduled.py")
        assert record["training"]["runner_sha256"] == hashlib.sha256(trainer_source.read_bytes()).hexdigest()
        models = {}
        for name in ("initial", "best", "last"):
            model, env, metadata = load_scheduled_policy(folder/f"{name}.zip")
            assert metadata["training"] == record["training"]
            assert env.observation_names == record["observation_names"]
            assert model.n_steps == 1024 and model.gamma == .99 and model.gae_lambda == .95
            models[name] = model
            result["strict_checkpoints"] += 1
        assert models["initial"].num_timesteps == 0 and models["last"].num_timesteps == record["trained_steps"]
        history = [(0, record["initial_validation"])]+[(r["steps"], r["validation"]) for r in record["history"]]
        selected = min(history, key=lambda r: validation_rank(r[1]))
        assert selected[0] == record["best_steps"] == models["best"].num_timesteps
        assert selected[1] == record["selected_validation"]
        assert record["selection_frozen_before_test"]["checkpoint_sha256"] == json.loads((folder/"best.zip.metadata.json").read_text())["checkpoint_sha256"]
        assert compact_scheduled(evaluate_scheduled(models["initial"], config, VALIDATION_CASES)) == record["initial_validation"]
        assert compact_scheduled(evaluate_scheduled(models["last"], config, VALIDATION_CASES)) == record["history"][-1]["validation"]
        if record["stop_reason"] == "validation_ready_twice":
            assert all(r["validation"]["ready"] for r in record["history"][-2:])
        assert all(r["out_of_bounds_actions"] == 0 for r in record["audit"])
        prior_path = Path("runs/vertical_scheduled_threshold_selection")/f"variant_{variant}"/"summary.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            prior_history = {r["steps"]: r for r in prior["history"]}
            common = [r for r in record["history"] if r["steps"] in prior_history]
            for row in common:
                old = prior_history[row["steps"]]
                assert row["validation"] == old["validation"] and row["ppo"] == old["ppo"]
            n = min(len(record["audit"]), len(prior["audit"]))
            assert record["audit"][:n] == prior["audit"][:n]
            result["selection_revision"][variant] = {
                "reused": variant in summary.get("reused_variants", {}),
                "exact_training_prefix_steps": common[-1]["steps"] if common else 0}
        row = {}
        for name, cases in (("train", TRAIN_CASES), ("validation", VALIDATION_CASES), ("test", TEST_CASES)):
            replay = evaluate_scheduled(models["best"], config, cases)
            row[name] = compact_scheduled(replay)
            if name != "train":
                assert row[name] == record["selected_validation" if name == "validation" else "test"]
            write_json(folder/f"{name}_replay.json", replay)
            if name == "test":
                for case in replay["cases"]:
                    verify_trace(case, config)
                    result["verified_test_trajectories"] += 1
        for name, controller in (("constant_support", ConstantSupport(config)), ("force_inverse", ForceInverse(config))):
            reference = evaluate_scheduled(controller, config, TEST_CASES)
            row[name] = compact_scheduled(reference)
            write_json(folder/f"{name}_test.json", reference)
        assert not row["constant_support"]["ready"]
        result["variants"][variant] = row
        write_json(args.run/"independent_evaluation.json", result)
        print(f"v{variant}: checkpoints, issued/due targets, tail flush and native test replay checked", flush=True)
    result["status"] = "passed"
    write_json(args.run/"independent_evaluation.json", result)
    print({k: v for k, v in result.items() if k != "variants"})


if __name__ == "__main__":
    main()
