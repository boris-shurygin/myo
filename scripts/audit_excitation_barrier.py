"""Measure short excitation pulses from matched settled states; no learning.

Payload changes occur only at reset in separate diagnostic episodes. Every
case replays the same zero-command prefix. Existing muscle dynamics, motor
delay, physical forces and reward equations are used without modification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from myo.compact_protocol import initialize_command_at_reset, result_reward
from myo.env import PreacherCurlEnv
from myo.runtime import dynamics_digest, read_config


def state_digest(env):
    state = np.concatenate((env.data.qpos, env.data.qvel, env.observe(),
                            *[muscle.state_vector() for muscle in env.muscles]))
    return hashlib.sha256(state.tobytes()).hexdigest()


def snapshot(env, info, elapsed):
    return {"time_s": elapsed, "angle_deg": float(np.rad2deg(info["q"])),
        "velocity_deg_s": float(np.rad2deg(info["qvel"])),
        "muscle_torque_Nm": info["muscle_torque_Nm"],
        "applied_muscle_torque_Nm": info["applied_muscle_torque_Nm"],
        "bias_compensation_Nm": float(env.data.qfrc_bias[env.dof_index]),
        "passive_limit_torque_Nm": info["passive_limit_torque_Nm"],
        "constraint_torque_Nm": float(env.data.qfrc_constraint[env.dof_index]),
        "summed_tendon_force_N": info["summed_tendon_force_proxy_N"],
        "tendon_force_N": info["tendon_force_N"], "activation": info["activation"],
        "remaining_capacity": info["capacity"]}


def probe(env, mass, flexor_excitation, extensor_excitation, pulse_seconds):
    env.reset(seed=0, options={"q": float(np.deg2rad(15)), "qvel": 0.,
        "target_angle": float(np.deg2rad(75)), "trigger": 1., "payload_mass": mass, "capacity": 1.})
    initialize_command_at_reset(env, 75., 1.)
    zero = np.zeros(len(env.muscles), dtype=np.float64)
    for _ in range(round(1 / env.control_dt)):
        _, _, terminated, truncated, info = env.step(zero)
        assert not terminated and not truncated
    start = snapshot(env, info, 0.)
    assert abs(start["velocity_deg_s"]) < .1
    initial_hash = state_digest(env)
    flexors = np.array([not name.lower().startswith("tri") for name in env.muscle_names])
    action = np.where(flexors, flexor_excitation, extensor_excitation)
    support_edge_deg = float(np.rad2deg(env.angle_limits[0] + env.config.passive_limit_width))
    pulse_steps = round(pulse_seconds / env.control_dt)
    rows, rewards, term_sums = [], [], {}
    failure = None
    for step in range(round(2 / env.control_dt)):
        try:
            _, _, terminated, truncated, info = env.step(action if step < pulse_steps else zero)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failure = f"{type(error).__name__}: {error}"
            break
        reward, terms = result_reward(info)
        row = snapshot(env, info, (step + 1) * env.control_dt)
        row["reward_raw"] = reward
        row["tissue_contact_force_N"] = info["substep_peaks"]["tissue_contact_force_N"]
        row["path_direction_reversal"] = info["path_direction_reversal"]
        rows.append(row)
        rewards.append(reward)
        for key, value in terms.items():
            term_sums[key] = term_sums.get(key, 0.) + value
        if terminated or truncated:
            failure = info.get("termination_reason") or "unexpected time limit"
            break
    assert rows
    angles = np.array([row["angle_deg"] for row in rows])
    progress = angles - start["angle_deg"]
    events = {}
    for name, predicate in (
        ("movement", lambda row: row["angle_deg"] >= start["angle_deg"] + .1 and row["velocity_deg_s"] >= 5),
        ("progress_3deg", lambda row: row["angle_deg"] >= start["angle_deg"] + 3),
        ("progress_5deg", lambda row: row["angle_deg"] >= start["angle_deg"] + 5),
        ("exit_passive_support_region", lambda row: row["angle_deg"] >= support_edge_deg)):
        events[name] = next((row for row in rows if predicate(row)), None)
    return {"mass_kg": mass, "flexor_excitation": flexor_excitation,
        "extensor_excitation_during_pulse": extensor_excitation, "pulse_seconds": pulse_seconds,
        "initial_state_sha256": initial_hash, "initial_state": start,
        "completed_2s": len(rows) == round(2 / env.control_dt) and failure is None,
        "failure": failure, "observed_seconds": len(rows) * env.control_dt,
        "maximum_angle_deg": float(angles.max()), "maximum_progress_deg": float(progress.max()),
        "final_angle_deg": float(angles[-1]), "maximum_angle_state": rows[int(angles.argmax())],
        "first_events": events, "passive_support_edge_deg": support_edge_deg,
        "time_outside_passive_support_region_s": float(np.sum(angles >= support_edge_deg) * env.control_dt),
        "time_with_progress_3deg_s": float(np.sum(progress >= 3) * env.control_dt),
        "time_with_progress_5deg_s": float(np.sum(progress >= 5) * env.control_dt),
        "peak_summed_tendon_force_N": max(row["summed_tendon_force_N"] for row in rows),
        "peak_contact_force_N": max(row["tissue_contact_force_N"] for row in rows),
        "path_direction_reversal": any(row["path_direction_reversal"] for row in rows),
        "undiscounted_scaled_return": float(.01 * np.sum(rewards)),
        "discounted_scaled_return_gamma099": float(.01 * np.dot(.99 ** np.arange(len(rewards)), rewards)),
        "reward_components_raw_sum": term_sums,
        "trace": [{key: row[key] for key in ("time_s", "angle_deg", "velocity_deg_s", "reward_raw",
                    "muscle_torque_Nm", "passive_limit_torque_Nm", "bias_compensation_Nm")}
                  for row in rows]}


def compare(case, baseline):
    assert case["initial_state_sha256"] == baseline["initial_state_sha256"]
    count = min(len(case["trace"]), len(baseline["trace"]))
    difference = np.array([case["trace"][i]["reward_raw"] - baseline["trace"][i]["reward_raw"]
                           for i in range(count)])
    return {"comparison_seconds": case["observed_seconds"],
        "discounted_scaled_return_improvement": float(.01 * np.dot(.99 ** np.arange(count), difference)),
        "undiscounted_scaled_return_improvement": float(.01 * difference.sum()),
        "peak_instantaneous_raw_reward_improvement": float(difference.max()),
        "first_raw_reward_improvement_gt_001_time_s": next(
            (case["trace"][i]["time_s"] for i in range(count) if difference[i] > .01), None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/d1_excitation_barrier_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read_config(args.config)
    env = PreacherCurlEnv(config)
    report = {"status": "running", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": asdict(config), "dynamics_sha256": dynamics_digest(), "muscle_names": list(env.muscle_names),
        "protocol": {"reset_angle_deg": 15., "requested_target_deg": 75., "trigger": 1.,
            "capacity_at_reset": 1., "zero_prefix_seconds": 1., "observation_seconds_after_prefix": 2.,
            "control_dt": env.control_dt, "motor_delay_s": config.motor_delay,
            "after_pulse": "Zero excitation in all muscles; no pose reset or payload change",
            "extensor_baseline": "0 or .1 only during the pulse, then zero; comparisons include both all-zero and matching extensor-only controls",
            "passive_support_exit": "Angle reaches lower joint limit + passive_limit_width, where the modeled spring/damper is zero",
            "movement_event": "At least .1deg progress and 5deg/s positive velocity, sampled every20ms"},
        "limitations": ["Fixed coherent pulses test mechanical responses, not policy discovery probabilities or learned control.",
            "Grid values are sufficient examples, not a search for a true minimum excitation/duration.",
            "All-zero histories are consistent diagnostic states, not necessarily the states visited by a noisy policy.",
            "Payload masses change only at reset in separate probes; no training configuration is changed.",
            "Passive muscle tension and the modeled lower-limit spring contribute to initial support and motion.",
            "Two-second progress/return is not successful arrival or holding; fatigue is a proxy.",
            "Event latencies and outside-region durations are sampled at20ms, not exact continuous-time crossings."],
        "controls": [], "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)
    started = time.perf_counter()
    try:
        for mass in (1., .25, 2.):
            baseline = probe(env, mass, 0., 0., 0.)
            report["controls"].append(baseline)
            extension_durations = (.02, .1, .2, .5) if mass == 1 else (.1, .5)
            ext_controls = {duration: probe(env, mass, 0., .1, duration) for duration in extension_durations}
            report["controls"].extend(ext_controls.values())
            if mass == 1:
                grid = [(amplitude, duration, extensor) for extensor in (0., .1)
                        for amplitude in (.1, .2, .3, .5) for duration in (.02, .1, .2, .5)]
            else:
                grid = [(amplitude, duration, 0.) for amplitude in (.1, .3, .5) for duration in (.02, .2, .5)]
                grid += [(.3, duration, .1) for duration in (.1, .5)]
            for amplitude, duration, extensor in grid:
                case = probe(env, mass, amplitude, extensor, duration)
                case["versus_all_zero"] = compare(case, baseline)
                case["versus_matching_extensor_only"] = compare(case, ext_controls[duration] if extensor else baseline)
                report["cases"].append(case)
                report["elapsed_seconds"] = time.perf_counter() - started
                save()
                print(json.dumps({"case": len(report["cases"]), "mass": mass, "flex": amplitude,
                    "ext": extensor, "seconds": duration, "max_progress_deg": case["maximum_progress_deg"],
                    "return_improvement": case["versus_all_zero"]["discounted_scaled_return_improvement"]}), flush=True)
        report["status"] = "complete"
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
