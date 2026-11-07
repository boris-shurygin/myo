"""Compare frozen critic TD ranking with short same-state physical interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.evaluate import CommandEvent, Scenario
from myo.prepared_protocol import PreparedProtocol
from myo.runtime import dynamics_digest, load_controller


DELTAS = {
    "baseline": (0, 0), "flex_plus": (.05, 0), "ext_minus": (0, -.05),
    "combined": (.05, -.05), "inverse": (-.05, .05),
}


def value(model, observation):
    with torch.no_grad():
        return float(model.policy.predict_values(torch.as_tensor(observation[None, :]))[0, 0])


def run_case(physical, actor, model, direction, duration):
    env = PreparedProtocol(physical, (Scenario("hold75", 4, (CommandEvent(0, 75, 1),),
                        1, initial_angle_deg=75),), 0)
    observation, _ = env.reset(seed=0)
    prefix = hashlib.sha256()
    for _ in range(round(2 / physical.control_dt)):
        prefix.update(observation.tobytes())
        observation, _, terminated, _, _ = env.step(actor(observation))
        if terminated:
            raise RuntimeError("Prefix physically terminated")
    prefix.update(observation.tobytes())
    initial_value = value(model, observation)
    initial_command = actor(observation)
    if direction == "heuristic":
        intervention_command = physical.baseline_action()
    else:
        delta = DELTAS[direction]
        intervention_command = np.clip(initial_command + np.repeat(delta, 3), .0001, .9999)
    with torch.no_grad():
        distribution = model.policy.get_distribution(torch.as_tensor(observation[None, :]))
        log_probability = float(distribution.log_prob(torch.as_tensor(
            np.clip(intervention_command, .0001, .9999)[None, :], dtype=torch.float32))[0])
    handoff = {"angle_deg": float(np.rad2deg(physical.data.qpos[physical.qpos_index])),
               "value": initial_value, "initial_policy_command": initial_command.tolist(),
               "intervention_command": intervention_command.tolist(),
               "intervention_initial_log_probability": log_probability}
    rows = []
    before_value = initial_value
    for step in range(round(2 / physical.control_dt)):
        if direction == "heuristic":
            action = physical.baseline_action()
        elif step < round(duration / physical.control_dt):
            action = intervention_command
        else:
            action = actor(observation)
        observation, reward, terminated, _, info = env.step(action)
        after_value = value(model, observation)
        rows.append({"time_s": (step + 1) * physical.control_dt,
                     "reward": float(reward), "scaled_reward": float(.01 * reward),
                     "reward_terms": info["reward_components"],
                     "value_before": before_value, "value_after": after_value,
                     "td": float(.01 * reward + model.gamma * after_value - before_value),
                     "angle_deg": float(np.rad2deg(info["q"])),
                     "velocity_deg_s": float(np.rad2deg(info["qvel"])),
                     "action": np.asarray(action).tolist(),
                     "muscle_torque_Nm": info["muscle_torque_Nm"],
                     "passive_limit_torque_Nm": info["passive_limit_torque_Nm"],
                     "constraint_torque_Nm": float(physical.data.qfrc_constraint[physical.dof_index])})
        before_value = after_value
        if terminated:
            raise RuntimeError("Intervention physically terminated")
    horizons = {}
    for seconds in (.2, .5, 1, 2):
        n = round(seconds / physical.control_dt)
        discounted = float(sum((model.gamma ** i) * row["scaled_reward"]
                               for i, row in enumerate(rows[:n])))
        bootstrap = float(model.gamma ** n * rows[n - 1]["value_after"])
        horizons[str(seconds)] = {"discounted_scaled_actual_reward": discounted,
            "endpoint_value": rows[n - 1]["value_after"],
            "discounted_bootstrap": bootstrap,
            "bootstrapped_advantage": discounted + bootstrap - initial_value,
            "final_angle_deg": rows[n - 1]["angle_deg"]}
    return {"direction": direction, "intervention_seconds": duration,
            "prefix_observation_sha256": prefix.hexdigest(), "handoff": handoff,
            "gamma": model.gamma, "gae_lambda": model.gae_lambda, "reward_scale": .01,
            "finite_horizon_gae": float(sum((model.gamma * model.gae_lambda) ** i * row["td"]
                                            for i, row in enumerate(rows))),
            "first_step": rows[0],
            "horizons": horizons, "trace": rows}


def add_differences(cases):
    for case in cases:
        duration = case["intervention_seconds"]
        reference_duration = .02 if case["direction"] == "heuristic" else duration
        baseline = next(r for r in cases if r["direction"] == "baseline"
                        and np.isclose(r["intervention_seconds"], reference_duration))
        if case["prefix_observation_sha256"] != baseline["prefix_observation_sha256"]:
            raise AssertionError("Intervention prefixes differ")
        case["difference_from_baseline"] = {
            "reference_intervention_seconds": reference_duration,
            "finite_horizon_gae": case["finite_horizon_gae"] - baseline["finite_horizon_gae"],
            "first_step_td": case["first_step"]["td"] - baseline["first_step"]["td"],
            "first_step_scaled_reward": case["first_step"]["scaled_reward"]
                                        - baseline["first_step"]["scaled_reward"],
            "horizons": {key: {metric: values[metric] - baseline["horizons"][key][metric]
                               for metric in values} for key, values in case["horizons"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = {"dynamics_sha256": dynamics_digest(), "policies": {},
        "limitations": ["Frozen weights; no training or optimizer update.",
            "Single deterministic target75 state after2s from initial75 for each policy.",
            "A repeated intervention is a diagnostic external command, not policy semantics.",
            "Actual reward differences omit later consequences after their stated horizon.",
            "Bootstrapped returns depend on the same learned critic being diagnosed.",
            "All values and TD use training reward scaling .01."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in ("small", "wide"):
        path = Path(f"runs/d1_architecture_{size}_tanh/block_004.zip")
        physical, actor, metadata = load_controller(str(path))
        model = PPO.load(path, device="cpu")
        cases = []
        try:
            for duration in (.02, .2):
                for direction in DELTAS:
                    row = run_case(physical, actor, model, direction, duration)
                    cases.append(row)
                    print(size, direction, duration, row["first_step"]["td"],
                          row["horizons"]["2"]["final_angle_deg"], flush=True)
            cases.append(run_case(physical, actor, model, "heuristic", 2))
            add_differences(cases)
            report["policies"][size] = {"checkpoint": str(path),
                "checkpoint_sha256": metadata["checkpoint_sha256"], "cases": cases}
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        finally:
            physical.close()


if __name__ == "__main__":
    main()
