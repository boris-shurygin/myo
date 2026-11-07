"""Independent native-physics checks of the frozen acceleration-trained actor."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.delayed_acceleration import sha
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.movement_curriculum import MovementProtocol, evaluate_movement
from myo.runtime import dynamics_digest, load_controller


def evaluate(config, model, initial_angle, duration, trace_path):
    env = MovementProtocol(PreacherCurlEnv(config), 0)
    env.scenarios = tuple(replace(s, initial_angle_deg=initial_angle, duration_s=duration)
                          for s in env.scenarios)
    rows = []

    class Tracker:
        def predict(self, observation, deterministic=True):
            base = env.unwrapped
            command = (base.baseline_action() if model is None
                       else model.predict(observation, deterministic=True)[0])
            info = base.telemetry()
            rows.append([base.time, info["q"], info["qvel"], min(info["capacity"]), *command,
                         *info["delayed_excitation"], *info["activation"],
                         *info["tendon_force_N"], *info["tendon_strain"]])
            return command, None

    try:
        result = evaluate_movement(Tracker(), env)
        names = list(env.unwrapped.muscle_names)
    finally:
        env.close()
    trace = np.asarray(rows)
    np.savez_compressed(trace_path, pre_action_samples=trace,
        columns=np.array(["time_s", "q_rad", "velocity_rad_s", "minimum_capacity",
                          *[f"{field}/{name}" for field in
                            ("command", "delayed_excitation", "activation", "force_N", "tendon_strain")
                            for name in names]]))
    result.update(initial_angle_deg=initial_angle, duration_s=duration,
                  physical_control_steps=len(rows), minimum_capacity=float(trace[:, 3].min()),
                  trace_path=str(trace_path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/d1_delayed_acceleration_descent"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.run / "summary.json").read_text())
    if summary["status"] == "running":
        raise ValueError("Finish training before independent checks")
    path = args.run / "best.zip"
    env, _, metadata = load_controller(str(path))
    env.close()
    before = sha(path)
    model = PPO.load(path, device="cpu")
    initial = PPO.load(args.run / "initial.zip", device="cpu").policy.state_dict()
    current = model.policy.state_dict()
    unchanged = all(torch.equal(initial[key], current[key]) for key in initial
                    if key.startswith(("mlp_extractor.value_net", "value_net", "features_extractor")))
    assert unchanged
    config = PreacherCurlConfig(**metadata["config"])
    validation = np.load(args.run / "validation.npz")
    for file in args.run.glob("round_*_data.npz"):
        data = np.load(file)
        assert set(data["episode_ids"]).isdisjoint(validation["episode_ids"])
    report = {"script_sha256": sha(__file__), "checkpoint_sha256": before,
              "dynamics_sha256": dynamics_digest(), "training_status": summary["status"],
              "optimizer_updates_at_selection": metadata["trained_steps"],
              "normal_loader_passed": True, "critic_and_normalization_unchanged": unchanged,
              "validation_episodes_disjoint": True, "checks": {},
              "scope": "frozen actor; full native physics; no teacher assistance or optimizer"}
    for label, angle, duration in (("stage0", 20., 4.), ("heldout_start21", 21., 4.), ("hold20s", 20., 20.)):
        result = evaluate(config, model, angle, duration, args.run / f"{label}_trace.npz")
        report["checks"][label] = result
        print(label, "passed", result["passed"], "score", round(result["score"], 5), flush=True)
    report["heuristic_hold20s"] = evaluate(config, None, 20., 20., args.run / "heuristic_hold20s_trace.npz")
    assert before == sha(path)
    report["checkpoint_unchanged"] = True
    (args.run / "independent_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("saved independent_evaluation.json", flush=True)


if __name__ == "__main__":
    main()
