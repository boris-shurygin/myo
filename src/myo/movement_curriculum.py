"""Cold-start movement curriculum with unchanged physical outcome rewards."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from .compact_protocol import CAPACITIES, TRIGGERS, CompactProtocol, compact_scenarios, protocol_metadata
from .evaluate import CommandEvent, Scenario


STAGE_NAMES = ("raise_30", "raise_35", "reverse_35", "ladder_75", "ladder_135", "compact")
stage_names = STAGE_NAMES


def stage_cases(stage: int, payload_mass: float) -> tuple[Scenario, ...]:
    """Use only fixed commands; upper starting states are never supplied."""
    if not 0 <= stage < len(STAGE_NAMES):
        raise ValueError("Unknown movement curriculum stage")
    if stage == 5:
        return compact_scenarios(payload_mass)
    targets, interval = (
        ((30,), 4.0), ((35,), 4.0), ((35, 20, 35), 4.0),
        ((35, 50, 75, 50, 35, 20), 3.0),
        ((35, 75, 105, 135, 105, 75, 35, 20), 3.0),
    )[stage]
    cases = []
    for capacity in CAPACITIES:
        for trigger in TRIGGERS:
            suffix = f"u{trigger:g}_c{capacity:g}"
            cases.append(Scenario(f"{STAGE_NAMES[stage]}_{suffix}", len(targets) * interval,
                tuple(CommandEvent(i * interval, target, trigger) for i, target in enumerate(targets)),
                payload_mass, capacity=capacity, initial_angle_deg=20))
            if stage:
                retained_target = 30 if stage <= 2 else 35
                cases.append(Scenario(f"retain_{retained_target}_{suffix}", 4.0,
                    (CommandEvent(0, retained_target, trigger),), payload_mass,
                    capacity=capacity, initial_angle_deg=20))
    return tuple(cases)


def curriculum_metadata(payload_mass: float) -> dict:
    return {
        "name": "cold_movement_curriculum_v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compact_protocol_sha256": protocol_metadata(payload_mass)["source_sha256"],
        "stages": [{"index": index, "name": name,
                    "scenarios": [asdict(s) for s in stage_cases(index, payload_mass)]}
                   for index, name in enumerate(STAGE_NAMES)],
        "order": "fixed cyclic scenarios; seed 0; no random initial states",
        "preparation": "none; normal cold reset; no teacher or muscle-action labels",
        "reward": protocol_metadata(payload_mass)["reward"],
        "gate": {"tail_seconds": 1.0, "angle_rmse_deg": 3.0, "final_error_deg": 3.0,
                 "speed_rms_deg_s": 5.0, "joint_reaction_peak_Nm": 20.0,
                 "contact_force_peak_N": 250.0,
                 "scope": "every command window in every scenario, complete durations required"},
        "final_evaluation": "unchanged compact behavior_summary; earlier stages are not demo readiness",
        "limitations": ["Fixed exercises do not establish generalization.",
                        "No trigger release training or established protective behavior.",
                        "Load thresholds are engineering proxies, not human safety limits."],
    }


descriptor = curriculum_metadata


class MovementProtocol(CompactProtocol):
    """Retain the causal command and outcome-reward behavior of CompactProtocol."""

    def __init__(self, env, stage: int):
        super().__init__(env)
        self.stage = stage
        self.scenarios = stage_cases(stage, self.unwrapped.config.payload_mass)
        self.scenario = self.scenarios[0]


def summarize_movement_samples(scenario: Scenario, samples: list[dict], control_dt: float,
                               *, terminated: bool = False) -> dict:
    """Score completed command windows, including the entire final hold second."""
    times = np.asarray([r["time_s"] for r in samples], dtype=float)
    q = np.asarray([r["angle_deg"] for r in samples], dtype=float)
    speed = np.asarray([r["speed_deg_s"] for r in samples], dtype=float)
    goals = np.asarray([r["target_deg"] for r in samples], dtype=float)
    reaction = np.asarray([r["joint_reaction_Nm"] for r in samples], dtype=float)
    contact = np.asarray([r["contact_force_N"] for r in samples], dtype=float)
    expected_steps = round(scenario.duration_s / control_dt)
    completed = len(times) == expected_steps and np.allclose(
        times, np.arange(1, expected_steps + 1) * control_dt, rtol=0, atol=1e-8)
    windows = []
    for index, event in enumerate(scenario.commands):
        end = (scenario.commands[index + 1].time_s if index + 1 < len(scenario.commands)
               else scenario.duration_s)
        mask = (times > event.time_s + 1e-8) & (times <= end + 1e-8)
        tail = mask & (times > max(event.time_s, end - 1.0) + 1e-8)
        window_steps = round((end - event.time_s) / control_dt)
        window_complete = (np.count_nonzero(mask) == window_steps
                           and np.allclose(goals[mask], event.target_deg, rtol=0, atol=1e-8))
        if np.any(tail):
            error = q[tail] - event.target_deg
            rmse = float(np.sqrt(np.mean(error ** 2)))
            speed_rms = float(np.sqrt(np.mean(speed[tail] ** 2)))
            final_error = abs(float(q[mask][-1] - event.target_deg))
        else:
            rmse, speed_rms, final_error = 180.0, 180.0, 180.0
        joint_peak = float(np.max(reaction[mask])) if np.any(mask) else 0.0
        contact_peak = float(np.max(contact[mask])) if np.any(mask) else 0.0
        settled = mask & (np.abs(q - event.target_deg) <= 3) & (np.abs(speed) <= 5)
        passed = bool(completed and window_complete and not terminated and rmse <= 3
                      and final_error <= 3 and speed_rms <= 5 and joint_peak <= 20
                      and contact_peak <= 250)
        windows.append({"target_deg": event.target_deg, "start_s": event.time_s, "end_s": end,
            "completed": bool(window_complete), "passed": passed,
            "tail_angle_rmse_deg": rmse, "tail_speed_rms_deg_s": speed_rms,
            "final_error_deg": final_error, "joint_reaction_peak_Nm": joint_peak,
            "contact_force_peak_N": contact_peak,
            "settled_fraction": float(np.count_nonzero(settled) / window_steps),
            "tail_settled_fraction": float(np.count_nonzero(settled & tail)
                / max(1, round(min(1.0, end - event.time_s) / control_dt))),
            "max_angle_deg": float(np.max(q[mask])) if np.any(mask) else scenario.initial_angle_deg,
            "score": rmse + 0.1 * speed_rms})
    excitation = np.asarray([r["excitation"] for r in samples], dtype=float)
    mean_excitation = excitation.mean(axis=0).tolist() if len(samples) else []
    return {"name": scenario.name, "passed": all(w["passed"] for w in windows),
            "completed_duration": bool(completed), "physical_failure": bool(terminated),
            "sample_count": len(samples), "expected_sample_count": expected_steps,
            "final_angle_deg": float(q[-1]) if len(q) else scenario.initial_angle_deg,
            "max_angle_deg": float(np.max(q)) if len(q) else scenario.initial_angle_deg,
            "time_above_lower_support_s": float(np.count_nonzero(q > 25) * control_dt),
            "raw_reward_sum": float(sum(r["reward"] for r in samples)),
            "raw_reward_mean": float(np.mean([r["reward"] for r in samples])) if samples else 0.0,
            "mean_excitation": mean_excitation, "windows": windows}


def evaluate_movement(model, env: MovementProtocol) -> dict:
    """Evaluate a complete deterministic cycle without changing its ordering."""
    env.next_scenario = 0
    results = []
    for _ in env.scenarios:
        observation, _ = env.reset(seed=0)
        samples = []
        while True:
            action = model.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            samples.append({"time_s": float(env.unwrapped.time),
                "angle_deg": float(np.rad2deg(info["q"])),
                "speed_deg_s": float(np.rad2deg(info["qvel"])),
                "target_deg": float(np.rad2deg(info["requested_target_q"])),
                "joint_reaction_Nm": float(info["substep_peaks"]["joint_limit_reaction_Nm"]),
                "contact_force_N": float(info["substep_peaks"]["tissue_contact_force_N"]),
                "excitation": list(info["excitation"]), "reward": float(reward)})
            if terminated or truncated:
                break
        result = summarize_movement_samples(env.scenario, samples, env.unwrapped.control_dt,
                                            terminated=terminated)
        names = env.unwrapped.muscle_names
        means = result["mean_excitation"]
        result["mean_excitation_by_muscle"] = dict(zip(names, means))
        for group, indices in (("flexor", [i for i, name in enumerate(names) if not name.startswith("TRI")]),
                               ("extensor", [i for i, name in enumerate(names) if name.startswith("TRI")])):
            result[f"mean_{group}_excitation"] = float(np.mean([means[i] for i in indices]))
        results.append(result)
    windows = [window for result in results for window in result["windows"]]
    failures = sum(r["physical_failure"] or not r["completed_duration"] for r in results)
    return {"score": float(np.mean([w["score"] for w in windows])) + 180.0 * failures / len(results),
            "ready": all(r["passed"] for r in results),
            "passed": sum(r["passed"] for r in results), "total": len(results),
            "windows_passed": sum(w["passed"] for w in windows), "windows_total": len(windows),
            "physical_failures": failures, "scenarios": results}
