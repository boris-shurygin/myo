"""Independent closed-loop checks of the two supervised muscle actors."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.actuator_ablation import make_environment
from myo.env import PreacherCurlConfig
from myo.movement_curriculum import MovementProtocol, evaluate_movement
from myo.runtime import dynamics_digest, load_controller
from myo.supervised_actuator import load_supervised_controller, sha


class TrackedController:
    def __init__(self, env, model):
        self.env, self.model, self.rows = env, model, []

    def predict(self, obs, deterministic=True):
        base = self.env.unwrapped
        action = (base.baseline_action() if self.model is None
                  else self.model.predict(obs, deterministic=True)[0])
        info = base.telemetry()
        self.rows.append([base.time, info["q"], info["qvel"], min(info["capacity"]),
                          *action, *info["tendon_force_N"]])
        return action, None


def evaluate(config, model, initial_angle, duration, trace_path):
    env = MovementProtocol(make_environment("instant_rigid", config), 0)
    env.scenarios = tuple(replace(s, initial_angle_deg=initial_angle, duration_s=duration)
                          for s in env.scenarios)
    tracker = TrackedController(env, model)
    try:
        result = evaluate_movement(tracker, env)
    finally:
        env.close()
    traces = np.asarray(tracker.rows)
    np.savez_compressed(trace_path, pre_action_samples=traces,
        columns=np.array(["time_s", "q_rad", "velocity_rad_s", "minimum_capacity",
                          *[f"command/{n}" for n in env.unwrapped.muscle_names],
                          *[f"force_N/{n}" for n in env.unwrapped.muscle_names]]))
    result.update(initial_angle_deg=initial_angle, duration_s=duration,
                  control_steps=len(traces), minimum_capacity=float(traces[:, 3].min()),
                  trace_path=str(trace_path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d1_supervised_actuator"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run / "summary.json").read_text())
    if summary["status"] == "running":
        raise ValueError("Finish training before independent evaluation")
    config = PreacherCurlConfig(**summary["physics"]["actuator"]["source_config"])
    initial = PPO.load(args.run / "actions" / "initial.zip", device="cpu").policy.state_dict()
    report = {"script_sha256": sha(__file__), "base_dynamics_sha256": dynamics_digest(),
              "training_status": summary["status"], "branches": {},
              "scope": "frozen actors; six known trigger/capacity conditions; no optimizer or teacher assistance"}
    sources = []
    for name in ("actions", "accelerations"):
        path = args.run / name / "best.zip"
        env, _, metadata = load_supervised_controller(path)
        env.close()
        model = PPO.load(path, device="cpu")
        before = sha(path)
        sources.append(path)
        current = model.policy.state_dict()
        unchanged = all(torch.equal(initial[k], current[k]) for k in initial
                        if k.startswith(("mlp_extractor.value_net", "value_net", "features_extractor")))
        assert unchanged
        try:
            load_controller(str(path))
        except ValueError as error:
            rejection = str(error)
        else:
            raise AssertionError("Normal runtime accepted experimental actuator")
        row = {"checkpoint_sha256": before, "supervised_updates_at_selection": metadata["trained_steps"],
               "critic_and_normalization_unchanged": unchanged, "normal_runtime_rejection": rejection}
        for label, angle, duration in (("stage0", 20., 4.), ("heldout_start21", 21., 4.), ("hold20s", 20., 20.)):
            row[label] = evaluate(config, model, angle, duration, args.run / f"{name}_{label}_trace.npz")
            print(name, label, "passed", row[label]["passed"], "score", round(row[label]["score"], 5), flush=True)
        assert before == sha(path)
        report["branches"][name] = row
    report["heuristic_hold20s"] = evaluate(config, None, 20., 20., args.run / "heuristic_hold20s_trace.npz")
    training = np.load(args.run / "training_round_1.npz")
    validation = np.load(args.run / "validation.npz")
    assert set(training["episode_ids"]).isdisjoint(validation["episode_ids"])
    report["validation_episodes_disjoint"] = True
    report["checkpoint_hashes_unchanged"] = {str(p): sha(p) for p in sources}
    report["training_data_hashes"] = {str(p): sha(p) for p in sorted(args.run.glob("training_round_*.npz"))}
    (args.run / "independent_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("saved independent_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
