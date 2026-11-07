"""Compare completed stage0 actuator experiments using actual saved data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.actuator_ablation import load_ablation_controller
from myo.runtime import dynamics_digest, load_controller


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(directory, experimental):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] == "running":
        raise ValueError("Wait for the experiment to finish")
    record = summary if experimental else summary["stages"][0]
    best = directory / ("best.zip" if experimental else "stage_0_best.zip")
    metadata = json.loads(Path(str(best) + ".metadata.json").read_text())
    if experimental:
        env, _, _ = load_ablation_controller(best)
        env.close()
    else:
        env, _, _ = load_controller(str(best))
        env.close()
    best_source = Path(record["best_controller"])
    evaluation = (directory / ("initial_evaluation.json" if best_source.stem == "initial"
                               else f"evaluation_{best_source.stem.split('_')[-1]}.json")
                  if experimental else directory / f"{best_source.stem}_evaluation.json")
    metrics = json.loads(evaluation.read_text())
    arrays, provenance = [], []
    for path in sorted(directory.glob("rollout_*.npz")):
        with np.load(path) as d:
            physical = d["physical_samples"]
            assert np.isfinite(physical).all()
            if experimental and summary["mode"] == "instant_rigid":
                np.testing.assert_array_equal(d["activations"], d["actions"][:, 0])
                np.testing.assert_array_equal(d["delayed_excitation"], d["actions"][:, 0])
                padding = [i for i, n in enumerate(metadata["observation_names"])
                           if n.startswith("inactive_delay_padding/")]
                assert len(padding) == 240 and np.count_nonzero(d["observations"][:, 0, padding]) == 0
            arrays.append(physical)
        provenance.append({"path": str(path), "sha256": sha(path)})
    samples = np.concatenate(arrays)
    q, v = np.rad2deg(samples[:, 0]), np.rad2deg(samples[:, 1])
    settled = (abs(q - 30) <= 3) & (abs(v) <= 5)
    best_streak, streak, first_above, first_settled = 0, 0, None, None
    for i, good in enumerate(settled):
        if first_above is None and q[i] > 25:
            first_above = i + 1
        if first_settled is None and good:
            first_settled = i + 1
        streak = streak + 1 if good else 0
        best_streak = max(best_streak, streak)
        if samples[i, 6]:
            streak = 0
    return {"directory": str(directory), "status": summary["status"], "actual_steps": len(samples),
        "best_score": record["best_score"], "best_steps": metadata["trained_steps"],
        "best_passed": metrics["passed"], "best_total": metrics["total"],
        "best_final_angles_deg": [r["final_angle_deg"] for r in metrics["scenarios"]],
        "best_tail_rmse_deg": [r["windows"][0]["tail_angle_rmse_deg"] for r in metrics["scenarios"]],
        "best_tail_speed_rms_deg_s": [r["windows"][0]["tail_speed_rms_deg_s"] for r in metrics["scenarios"]],
        "training_max_angle_deg": float(q.max()), "training_mean_raw_reward": float(samples[:, 3].mean()),
        "training_settled_steps": int(settled.sum()), "longest_training_settled_s": best_streak * .02,
        "first_training_step_above25": first_above, "first_training_settled_step": first_settled,
        "training_physical_failures": int(np.count_nonzero(samples[:, 6].astype(bool) & (samples[:, 7] == 0))),
        "score_by_steps": {str(e["steps"]): e["score"] for e in record["evaluations"]},
        "best_checkpoint_sha256": sha(best), "source_rollouts": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple", type=Path, default=Path("runs/d1_actuator_instant_rigid"))
    parser.add_argument("--output", type=Path, default=Path("runs/d1_actuator_ablation_comparison.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline = Path("runs/d1_movement_curriculum")
    full = summarize(baseline, False)
    simple = summarize(args.simple, True)
    initial = PPO.load("runs/d1_architecture_small_tanh/initial.zip", device="cpu").policy.state_dict()
    experimental = PPO.load(args.simple / "initial.zip", device="cpu").policy.state_dict()
    error = max(float(torch.max(abs(initial[k] - experimental[k]))) for k in initial)
    # A resumed run starts with learned weights; exact initial equality is required only for fresh training.
    initial_steps = json.loads((args.simple / "summary.json").read_text())["initial_steps"]
    if initial_steps == 0:
        assert error == 0
    report = {"script_sha256": sha(__file__), "base_dynamics_sha256": dynamics_digest(),
              "initial_policy_difference_from_untrained": error,
              "full": full, "simple": simple,
              "limitations": ["One seed; same first-stage task/gates and PPO hyperparameters.",
                  "Three actuator simplifications change together, including static fiber operating length.",
                  "Dormant queue channels are zero-padded; 773 inputs and neural architecture preserved.",
                  "A negative result does not prove dynamics have no effect or the policy class cannot solve it.",
                  "The control is the saved full run, checked by exact reproduction of its first block."]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {p: report[k][p] for p in (
        "actual_steps", "status", "best_score", "best_passed", "best_final_angles_deg",
        "training_max_angle_deg", "longest_training_settled_s")}
        for k in ("full", "simple")}, indent=2))


if __name__ == "__main__":
    main()
