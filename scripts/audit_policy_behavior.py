"""Read fixed evaluations and probe policy behavior without training updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from myo.evaluate import CommandEvent, Scenario
from myo.prepared_protocol import PreparedProtocol
from myo.runtime import dynamics_digest, load_controller


def summarize_saved(directory: Path) -> list[dict]:
    result = []
    for path in sorted(directory.glob("evaluation_*.json")):
        data = json.loads(path.read_text())
        cases = []
        for case in data["scenarios"]:
            trace = case["trace_1hz"]
            cases.append({
                "name": case["scenario"]["name"],
                "final_angle_deg": case["final_angle_deg"],
                "command_responses": case["command_responses"],
                "overall": case["overall"],
                "sampled_angle_range_deg": [min(r["angle_deg"] for r in trace),
                                             max(r["angle_deg"] for r in trace)],
                "final_trace": trace[-1],
            })
        result.append({"evaluation": str(path), "metrics": data["metrics"], "cases": cases})
    return result


def probe(physical, actor, goal: float, preparation: float, mode: str) -> dict:
    duration = 4.0 if goal == 75 and preparation == 0 else 2.0
    scenario = Scenario(f"hold_{goal:g}", duration, (CommandEvent(0, goal, 1),),
                        1.0, initial_angle_deg=goal)
    env = PreparedProtocol(physical, (scenario,), preparation)
    obs, info = env.reset(seed=0)
    handoff = dict(info["preparation"])
    start = physical.time
    prefix = hashlib.sha256()
    rows = []
    while True:
        elapsed = physical.time - start
        if elapsed < 2 - 1e-8:
            prefix.update(obs.tobytes())
        if mode == "zero":
            action = np.zeros(physical.action_space.shape, dtype=np.float32)
        elif mode == "heuristic" or (mode == "switch_at_2s" and elapsed >= 2 - 1e-8):
            action = physical.baseline_action()
        else:
            action = actor(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "time_after_handoff_s": physical.time - start,
            "angle_deg": float(np.rad2deg(info["q"])),
            "velocity_deg_s": float(np.rad2deg(info["qvel"])),
            "reward": reward, "reward_terms": info["reward_components"],
            "excitation": np.asarray(action).tolist(),
            "delayed_excitation": info["delayed_excitation"],
            "activation": info["activation"], "capacity": info["capacity"],
            "muscle_torque_Nm": info["muscle_torque_Nm"],
            "passive_limit_torque_Nm": info["passive_limit_torque_Nm"],
            "constraint_torque_Nm": float(physical.data.qfrc_constraint[physical.dof_index]),
            "bias_torque_Nm": float(physical.data.qfrc_bias[physical.dof_index]),
            "peak_joint_limit_reaction_Nm": info["substep_peaks"]["joint_limit_reaction_Nm"],
        })
        if terminated or truncated:
            break
    tail = rows[-25:]
    angles = np.array([r["angle_deg"] for r in rows])
    speeds = np.array([r["velocity_deg_s"] for r in rows])
    settled = (abs(angles - goal) <= 3) & (abs(speeds) <= 5)
    result = {
        "goal_deg": goal, "preparation_s": preparation, "mode": mode,
        "handoff": handoff, "rollout_seconds": duration,
        "prefix_observation_sha256": prefix.hexdigest(),
        "final_angle_deg": float(angles[-1]),
        "angle_range_deg": [float(angles.min()), float(angles.max())],
        "angle_rmse_deg": float(np.sqrt(np.mean((angles - goal) ** 2))),
        "settled_fraction": float(settled.mean()),
        "tail_settled_fraction": float(settled[-25:].mean()),
        "peak_speed_deg_s": float(abs(speeds).max()),
        "tail_mean_reward": float(np.mean([r["reward"] for r in tail])),
        "tail_mean_reward_terms": {k: float(np.mean([r["reward_terms"][k] for r in tail]))
                                    for k in tail[0]["reward_terms"]},
        "tail_mean_excitation": np.mean([r["excitation"] for r in tail], axis=0).tolist(),
        "final_capacity": rows[-1]["capacity"],
        "tail_mean_torques_Nm": {k: float(np.mean([r[k] for r in tail])) for k in (
            "muscle_torque_Nm", "passive_limit_torque_Nm", "constraint_torque_Nm", "bias_torque_Nm")},
        "trace_5hz": [rows[0], *rows[9::10]],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = {"dynamics_sha256": dynamics_digest(), "saved": {}, "probes": [],
              "limitations": ["No training or optimizer updates.",
                  "Short deterministic diagnostic probes, not generalized skills or endurance.",
                  "Heuristic preparation/switching is only a diagnostic counterfactual.",
                  "5-Hz traces can miss extrema; aggregate extrema use every control step.",
                  "Joint load and capacity are engineering proxies."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in ("small", "wide"):
        directory = Path(f"runs/d1_architecture_{size}_tanh")
        report["saved"][size] = summarize_saved(directory)
        for checkpoint in ("initial.zip", "block_004.zip"):
            path = directory / checkpoint
            physical, actor, metadata = load_controller(str(path))
            try:
                cases = [(75, 0.0, "policy")]
                if checkpoint != "initial.zip":
                    cases = [(goal, prep, "policy") for goal in (20, 45, 75, 120)
                             for prep in (0.0, 2.0)]
                    cases.append((75, 0.0, "switch_at_2s"))
                    if size == "wide":
                        cases.extend(((75, 0.0, "heuristic"), (75, 0.0, "zero"),
                                      (75, 2.0, "zero")))
                for goal, preparation, mode in cases:
                    row = probe(physical, actor, goal, preparation, mode)
                    row.update(checkpoint=str(path), checkpoint_sha256=metadata["checkpoint_sha256"])
                    report["probes"].append(row)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(json.dumps({"size": size, "checkpoint": checkpoint, "goal": goal,
                        "preparation": preparation, "mode": mode,
                        "final_deg": row["final_angle_deg"], "settled": row["settled_fraction"],
                        "tail_reward": row["tail_mean_reward"]}), flush=True)
            finally:
                physical.close()


if __name__ == "__main__":
    main()
