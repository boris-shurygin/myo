"""Compare actual compact rewards for passive rest and frozen controllers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from myo.compact_protocol import CompactProtocol, compact_scenarios, protocol_metadata
from myo.runtime import dynamics_digest, load_controller, read_config


CONTROLLERS = {"zero": "zero", "heuristic": "heuristic",
    "small": "runs/d1_architecture_small_tanh/block_004.zip",
    "wide": "runs/d1_architecture_wide_tanh/block_004.zip"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, term_names):
    data = np.asarray(rows, dtype=float)
    # Columns: episode,time,goal,q,velocity,reward,muscle,passive,constraint,bias,terms.
    goal, q, speed, reward = data[:, 2], data[:, 3], data[:, 4], data[:, 5]
    groups = {}
    for label, mask in (("goal15", np.isclose(goal, 15)), ("goal20", np.isclose(goal, 20)),
                        ("goal_at_least25", goal >= 24.99)):
        if not mask.any():
            continue
        groups[label] = {"steps": int(mask.sum()), "seconds": float(mask.sum() * .02),
            "time_fraction": float(mask.mean()), "mean_raw_reward": float(reward[mask].mean()),
            "sum_raw_reward": float(reward[mask].sum()),
            "positive_reward_fraction": float(np.mean(reward[mask] > 0)),
            "settled_fraction": float(np.mean((abs(q[mask] - goal[mask]) <= 3) & (abs(speed[mask]) <= 5))),
            "mean_angle_deg": float(q[mask].mean()),
            "mean_reward_terms": {name: float(data[mask, 10 + i].mean()) for i, name in enumerate(term_names)}}
    return {"steps": len(data), "seconds": len(data) * .02,
        "mean_raw_reward": float(reward.mean()), "sum_raw_reward": float(reward.sum()),
        "positive_reward_fraction": float(np.mean(reward > 0)),
        "fraction_below25deg": float(np.mean(q < 25)), "groups": groups,
        "mean_reward_terms": {name: float(data[:, 10 + i].mean()) for i, name in enumerate(term_names)}}


def run_controller(label, output):
    checkpoint = CONTROLLERS[label]
    before = file_hash(checkpoint) if label in ("small", "wide") else None
    physical, actor, metadata = load_controller(checkpoint, config_path=Path("configs/d1_train.json"))
    env = CompactProtocol(physical)
    episodes, all_rows, actions = [], [], []
    started = time.perf_counter()
    try:
        for index in range(len(env.scenarios)):
            observation, _ = env.reset(seed=0)
            rows = []
            while True:
                action = actor(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                terms = info["reward_components"]
                assert "requested_angle_cost" in terms and "position_tracking" not in terms
                assert np.isclose(reward, sum(terms.values()), rtol=0, atol=1e-10)
                term_names = list(terms)
                rows.append([index, physical.time, np.rad2deg(info["requested_target_q"]),
                    np.rad2deg(info["q"]), np.rad2deg(info["qvel"]), reward,
                    info["muscle_torque_Nm"], info["passive_limit_torque_Nm"],
                    float(physical.data.qfrc_constraint[physical.dof_index]),
                    float(physical.data.qfrc_bias[physical.dof_index]), *terms.values()])
                actions.append(action)
                if terminated or truncated:
                    break
            stats = summarize(rows, term_names)
            rewards = np.asarray(rows)[:, 5]
            tail = np.asarray(rows)[-25:]
            stats.update(scenario=asdict(env.scenario), terminated=bool(terminated),
                completed=bool(physical.time + 1e-9 >= env.scenario.duration_s),
                discounted_raw_return=float(np.dot(.99 ** np.arange(len(rows)), rewards)),
                final_angle_deg=float(rows[-1][3]),
                final_capacity=info["capacity"],
                tail_mean_angle_deg=float(tail[:, 3].mean()),
                tail_mean_torques_Nm={key: float(tail[:, col].mean()) for key, col in (
                    ("muscle", 6), ("passive", 7), ("constraint", 8), ("gravity_bias", 9))})
            episodes.append(stats)
            all_rows.extend(rows)
            print(label, env.scenario.name, "mean reward", round(stats["mean_raw_reward"], 6),
                  "return", round(stats["discounted_raw_return"], 3), flush=True)
        aggregate = summarize(all_rows, term_names)
        aggregate["episode_weighted_mean_raw_reward"] = float(np.mean([r["mean_raw_reward"] for r in episodes]))
        aggregate["episode_weighted_discounted_raw_return"] = float(np.mean([r["discounted_raw_return"] for r in episodes]))
        aggregate["time_weighted_mean_raw_reward"] = aggregate["mean_raw_reward"]
        npz = output.with_name(output.stem + "_" + label + ".npz")
        np.savez_compressed(npz, data=np.asarray(all_rows), actions=np.asarray(actions),
            columns=np.asarray(["episode", "time_s", "goal_deg", "angle_deg", "velocity_deg_s", "raw_reward",
                "muscle_torque_Nm", "passive_torque_Nm", "constraint_torque_Nm", "bias_torque_Nm", *term_names]))
        if before:
            assert file_hash(checkpoint) == before
        return {"controller": checkpoint, "checkpoint_sha256": before,
            "config": asdict(physical.config), "aggregate": aggregate, "episodes": episodes,
            "step_data": str(npz), "step_data_sha256": file_hash(npz),
            "elapsed_seconds": time.perf_counter() - started}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/d1_rest_reward_audit.json"))
    parser.add_argument("--controllers", nargs="+", choices=list(CONTROLLERS), default=["zero", "heuristic"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    schedule = {}
    for scenario in compact_scenarios(read_config(Path("configs/d1_train.json")).payload_mass):
        for i, command in enumerate(scenario.commands):
            stop = scenario.commands[i + 1].time_s if i + 1 < len(scenario.commands) else scenario.duration_s
            goal = f"{command.target_deg:g}"
            schedule[goal] = schedule.get(goal, 0.0) + stop - command.time_s
    report = {"script_sha256": file_hash(__file__), "dynamics_sha256": dynamics_digest(), "protocol": protocol_metadata(1),
        "scheduled_seconds_by_goal": schedule, "scheduled_seconds_total": sum(schedule.values()),
        "gamma": .99, "models": {}, "limitations": [
            "Frozen deterministic controllers, one seed, full18 fixed exercises; no training or optimizer updates.",
            "Every reward is actual CompactProtocol.result_reward, not the base evaluator tracking reward.",
            "Raw physical rewards before training multiplier .01; no critic bootstrap in finite discounted returns.",
            "Time-weighted and episode-weighted means answer different questions; PPO does not directly maximize either static metric.",
            "Counterexamples can refute global optimality of rest here; no proof of global optimal policy or all alternatives."]}
    bound_terms = {goal: -max(float(goal) - 20, 0) / 30
                   + np.exp(-(max(float(goal) - 20, 0) / 3) ** 2) for goal in schedule}
    report["relaxed_q_at_most20_upper_bound"] = {
        "mean_raw_reward": float(sum(schedule[g] * bound_terms[g] for g in schedule) / sum(schedule.values())),
        "assumptions": "Full fixed-duration episodes; q<=20 throughout; all costs<=0. Relaxation allows instantaneous optimal angle<=20 and zero velocity/costs; not a physical rollout.",
        "best_tracking_plus_bonus_by_goal": bound_terms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.controllers:
        report["models"][label] = run_controller(label, args.output)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("AGGREGATE", label, report["models"][label]["aggregate"]["time_weighted_mean_raw_reward"], flush=True)


if __name__ == "__main__":
    main()
