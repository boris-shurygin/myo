"""Deterministic command/release checks for the D1 arm; no policy quality claims."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np

from .muscle import MuscleEquilibriumError
from .runtime import dynamics_digest, load_controller, policy_contract


@dataclass(frozen=True)
class CommandEvent:
    time_s: float
    target_deg: float
    trigger: float


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    commands: tuple[CommandEvent, ...]
    payload_mass_kg: float
    capacity: float = 1.0
    initial_angle_deg: float = 75.0
    initial_velocity_rad_s: float = 0.0
    branch_time_s: float | None = None
    forced_zero_time_s: float | None = None


def scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    hold = CommandEvent(0.0, 75.0, 1.0)
    return (
        Scenario("lift_hold_lower", 10.0, (hold, CommandEvent(1.5, 105.0, 1.0),
                 CommandEvent(5.5, 45.0, 1.0)), payload_mass),
        Scenario("controlled_lower", 6.0, (hold, CommandEvent(2.0, 45.0, 1.0)),
                 payload_mass, branch_time_s=2.0),
        Scenario("release", 6.0, (hold, CommandEvent(2.0, 75.0, 0.0)),
                 payload_mass, branch_time_s=2.0),
        Scenario("forced_zero", 6.0, (hold, CommandEvent(2.0, 75.0, 0.0)),
                 payload_mass, branch_time_s=2.0, forced_zero_time_s=2.0),
        Scenario("weak_trigger_raise", 6.0, (hold, CommandEvent(2.0, 105.0, 0.1)),
                 payload_mass, branch_time_s=2.0),
        Scenario("fatigued_release", 6.0, (hold, CommandEvent(2.0, 75.0, 0.0)),
                 payload_mass, capacity=0.7, branch_time_s=2.0),
    )


def hold_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    """Long manual holds at nominal load and fresh capacity, as in demo."""
    return tuple(Scenario(f"hold_{angle}", 30.0,
                          (CommandEvent(0.0, 75.0, 1.0), CommandEvent(2.0, angle, 1.0)),
                          payload_mass)
                 for angle in (75, 120, 125))


def extension_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    """Distinguish an exact lower-angle request from passive release/rest."""
    hold = CommandEvent(0.0, 75.0, 1.0)
    return (
        Scenario("extension_command", 30.0, (hold, CommandEvent(2.0, 15.0, 1.0)),
                 payload_mass, branch_time_s=2.0),
        Scenario("extension_release", 30.0, (hold, CommandEvent(2.0, 15.0, 0.0)),
                 payload_mass, branch_time_s=2.0),
        Scenario("extension_forced_zero", 30.0, (hold, CommandEvent(2.0, 15.0, 0.0)),
                 payload_mass, branch_time_s=2.0, forced_zero_time_s=2.0),
        Scenario("passive_rest", 30.0, (CommandEvent(0.0, 15.0, 0.0),),
                 payload_mass, initial_angle_deg=20.0),
    )


def vertical_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    """Track past the old 130-degree boundary and inspect torque sign changes."""
    return tuple(Scenario(f"hold_{angle}", 30.0,
                          (CommandEvent(0.0, 75.0, 1.0), CommandEvent(2.0, angle, 1.0)),
                          payload_mass)
                 for angle in (140, 150, 155, 175)) + (
        Scenario("vertical_return", 30.0,
                 (CommandEvent(0.0, 75.0, 1.0), CommandEvent(2.0, 155.0, 1.0),
                  CommandEvent(10.0, 145.0, 1.0), CommandEvent(18.0, 60.0, 1.0)),
                 payload_mass),
    )


def endurance_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    """Five-minute holds followed by two minutes of uninterrupted recovery."""
    return tuple(Scenario(f"endurance_{angle}", 420.0,
                          (CommandEvent(0.0, 75.0, 1.0),
                           CommandEvent(2.0, angle, 1.0),
                           CommandEvent(300.0, angle, 0.0)),
                          payload_mass, branch_time_s=300.0)
                 for angle in (75, 125))


def contact_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    """Approach, load and leave tissue contact through ordinary muscle motion.

    High targets are deliberately obstructed. Their tracking error alone is
    not a success score: inspect compression, effort and recovery on return.
    """
    hold = CommandEvent(0, 75, 1)
    release = (hold, CommandEvent(2, 155, 1), CommandEvent(10, 155, 0))
    return (
        Scenario("near_contact_hold", 30, (hold, CommandEvent(2, 138, 1)), payload_mass),
        Scenario("contact_return", 30,
                 (hold, CommandEvent(2, 175, 1), CommandEvent(10, 125, 1),
                  CommandEvent(20, 45, 1)), payload_mass),
        Scenario("contact_release", 24, release, payload_mass, branch_time_s=10),
        Scenario("contact_forced_zero", 24, release, payload_mass,
                 branch_time_s=10, forced_zero_time_s=10),
    )


def _state(base) -> dict:
    """Compare all physical muscle states and controller input at the branch."""
    state = {"time_s": base.time, "qpos": base.data.qpos.tolist(),
             "qvel": base.data.qvel.tolist(), "payload_mass_kg": base.payload_mass,
             "muscles": [muscle.state_vector().tolist() for muscle in base.muscles],
             "observation": base.observe().tolist()}
    encoded = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "angle_deg": float(np.rad2deg(base.data.qpos[base.qpos_index])),
            "velocity_rad_s": float(base.data.qvel[base.dof_index]),
            "payload_mass_kg": base.payload_mass,
            "activation": [muscle.activation for muscle in base.muscles],
            "capacity": [muscle.capacity for muscle in base.muscles]}


def _metrics(samples: list[tuple[dict, np.ndarray]], control_dt: float) -> dict:
    if not samples:
        return {"samples": 0, "tracking_samples": 0, "angle_rmse_deg": None,
                "velocity_rmse_rad_s": None, "substep_peaks": {}}
    tracking = [np.asarray(info["tracking_error"]) for info, _ in samples if info["trigger"] > 0]
    errors = np.sqrt(np.mean(np.square(tracking), axis=0)) if tracking else None
    excess = np.asarray([np.maximum(action - info["trigger"], 0.0) for info, action in samples])
    peaks: dict[str, float] = {}
    for info, _ in samples:
        for key, value in info["substep_peaks"].items():
            peaks[key] = max(peaks.get(key, 0.0), value)
    return {"samples": len(samples), "duration_s": len(samples) * control_dt,
            "tracking_samples": len(tracking),
            "angle_rmse_deg": float(np.rad2deg(errors[0])) if errors is not None else None,
            "velocity_rmse_rad_s": float(errors[1]) if errors is not None else None,
            "trigger_excess_mean": float(excess.mean()),
            "trigger_excess_max": float(excess.max()),
            "trigger_excess_step_fraction": float(np.mean(np.any(excess > 1e-6, axis=1))),
            "mean_excitation": float(np.mean([action for _, action in samples])),
            "mean_excitation_by_muscle": np.mean([action for _, action in samples], axis=0).tolist(),
            "mean_activation_by_muscle": np.mean([info["activation"] for info, _ in samples], axis=0).tolist(),
            "velocity_rms_rad_s": float(np.sqrt(np.mean([info["qvel"] ** 2 for info, _ in samples]))),
            "path_direction_reversal_step_fraction": float(np.mean([
                info.get("path_direction_reversal", False) for info, _ in samples])),
            "substep_peaks": peaks,
            "sampled_min_angle_deg": float(np.rad2deg(min(info["q"] for info, _ in samples))),
            "sampled_max_angle_deg": float(np.rad2deg(max(info["q"] for info, _ in samples))),
            "angle_std_deg": float(np.rad2deg(np.std([info["q"] for info, _ in samples]))),
            "sampled_max_speed_rad_s": max(abs(info["qvel"]) for info, _ in samples)}


def run_scenario(env: gym.Env, action: Callable[[np.ndarray], np.ndarray],
                 scenario: Scenario, seed: int = 0, *, continue_after_time_limit: bool = False,
                 measure_commands: bool = False) -> dict:
    """Reveal only the command due now; do not give policies future events."""
    base = env.unwrapped
    first = scenario.commands[0]
    options = {"q": float(np.deg2rad(scenario.initial_angle_deg)),
               "qvel": scenario.initial_velocity_rad_s,
               "target_angle": float(np.deg2rad(first.target_deg)), "trigger": first.trigger,
               "payload_mass": scenario.payload_mass_kg, "capacity": scenario.capacity,
               "reference_phase": 0.0, "reference_amplitude": 0.0, "reference_frequency": 0.0}
    env.reset(seed=seed, options=options)
    if measure_commands:
        from .compact_protocol import initialize_command_at_reset

        initialize_command_at_reset(env, first.target_deg, first.trigger)
    base.set_command(float(np.deg2rad(first.target_deg)), first.trigger)
    initial = _state(base)
    branch = None
    samples: list[tuple[dict, np.ndarray]] = []
    after_branch: list[tuple[dict, np.ndarray]] = []
    hold_tail: list[tuple[dict, np.ndarray]] = []
    step_times, policy_times = [], []
    trace = []
    bias_crossings = []
    previous_bias = float(base.data.qfrc_bias[base.dof_index])
    terminated = truncated = False
    failure = None
    info = base.telemetry()
    start = time.perf_counter()
    while base.time + 1e-10 < scenario.duration_s:
        current_time = base.time
        is_after_branch = scenario.branch_time_s is not None and current_time + 1e-10 >= scenario.branch_time_s
        if is_after_branch and branch is None:
            branch = _state(base)
        event = max((event for event in scenario.commands if event.time_s <= current_time + 1e-10),
                    key=lambda event: event.time_s)
        base.set_command(float(np.deg2rad(event.target_deg)), event.trigger)
        # Input changed after the last env.step. Refresh it before acting,
        # including the frozen predictor if this checkpoint was trained with it.
        observation = base.observe()
        if isinstance(env, gym.ObservationWrapper):
            observation = env.observation(observation)
        before_policy = time.perf_counter()
        forced_zero = scenario.forced_zero_time_s is not None and current_time + 1e-10 >= scenario.forced_zero_time_s
        excitation = (np.zeros(env.action_space.shape, dtype=np.float32) if forced_zero
                      else np.asarray(action(observation), dtype=np.float32))
        before_step = time.perf_counter()
        try:
            _, _, terminated, truncated, info = env.step(excitation)
        except (MuscleEquilibriumError, FloatingPointError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
        after_step = time.perf_counter()
        policy_times.append(1000 * (before_step - before_policy))
        step_times.append(1000 * (after_step - before_step))
        sample = (info, excitation.copy())
        samples.append(sample)
        bias_torque = float(base.data.qfrc_bias[base.dof_index])
        trace_row = {"time_s": base.time, "angle_deg": float(np.rad2deg(info["q"])),
                     "velocity_rad_s": info["qvel"], "trigger": info["trigger"],
                     "requested_target_deg": float(np.rad2deg(info["requested_target_q"])),
                     "excitation": excitation.tolist(), "activation": info["activation"],
                     "capacity": info["capacity"], "bias_torque_Nm": bias_torque,
                     "path_direction_reversal": bool(info.get("path_direction_reversal", False)),
                     "muscle_torque_Nm": info["muscle_torque_Nm"],
                     "tissue_contact": info["tissue_contact"],
                     "passive_limit_torque_Nm": info["passive_limit_torque_Nm"]}
        if len(samples) == 1 or len(samples) % max(1, round(1.0 / base.control_dt)) == 0:
            trace.append(trace_row)
        if previous_bias * bias_torque < 0:
            bias_crossings.append(trace_row)
        previous_bias = bias_torque
        if is_after_branch:
            after_branch.append(sample)
        if continue_after_time_limit and current_time + 1e-10 >= scenario.duration_s - 5.0:
            hold_tail.append(sample)
        if terminated:
            break
        if truncated:
            if not continue_after_time_limit:
                break
            # Explicit long-hold evaluation follows the continuous demo.
            # Reopen only the training time limit: preserve clock, physical
            # state, fatigue, activation, delay queues and observation history.
            base._episode_done = False
    wall_seconds = time.perf_counter() - start
    if samples and trace[-1]["time_s"] != trace_row["time_s"]:
        trace.append(trace_row)
    return {"scenario": asdict(scenario), "seed": seed, "initial_state": initial,
            "command_responses": command_response_metrics(scenario, samples, base.control_dt)
                if measure_commands else None,
            "branch_state": branch, "overall": _metrics(samples, base.control_dt),
            "after_branch": _metrics(after_branch, base.control_dt),
            "continue_after_time_limit": continue_after_time_limit,
            "hold_tail_last_5s": _metrics(hold_tail, base.control_dt) if continue_after_time_limit else None,
            "completed_duration": base.time + 1e-10 >= scenario.duration_s,
            "physical_failure": bool(terminated), "numerical_failure": failure,
            "termination_reason": info.get("termination_reason"), "episode_truncated": bool(truncated),
            "final_angle_deg": float(np.rad2deg(info["q"])),
            "final_velocity_rad_s": info["qvel"], "final_capacity": info["capacity"],
            "trace_1hz": trace, "bias_torque_sign_crossings": bias_crossings,
            "wall_seconds": wall_seconds,
            "simulated_seconds": len(samples) * base.control_dt,
            "policy_mean_ms": float(np.mean(policy_times)) if policy_times else None,
            "env_step_mean_ms": float(np.mean(step_times)) if step_times else None,
            "env_step_p95_ms": float(np.percentile(step_times, 95)) if step_times else None}


def command_response_metrics(scenario, samples, dt):
    """Evaluate every command at control rate; a fly-through is not settling."""
    from .compact_protocol import ANGLE_TOLERANCE_RAD, SPEED_TOLERANCE_RAD_S, SETTLE_DURATION_S

    reports = []
    initial_angle = float(np.deg2rad(scenario.initial_angle_deg))
    for index, event in enumerate(scenario.commands):
        end = scenario.commands[index + 1].time_s if index + 1 < len(scenario.commands) else scenario.duration_s
        # An outcome at t was generated by the command present at t-dt.
        rows = [info for info, _ in samples if event.time_s - 1e-9 <= info["t"] - dt < end - 1e-9]
        if not rows:
            reports.append({"command_time_s": event.time_s, "target_deg": event.target_deg,
                            "samples": 0, "settling_time_s": None, "completed_window": False})
            continue
        angles = np.asarray([r["q"] for r in rows])
        errors = angles - np.deg2rad(event.target_deg)
        times = np.asarray([r["t"] - event.time_s for r in rows])
        in_angle = np.abs(errors) <= ANGLE_TOLERANCE_RAD
        stable = in_angle & (np.abs([r["qvel"] for r in rows]) <= SPEED_TOLERANCE_RAD_S)
        suffix_stable = np.logical_and.accumulate(stable[::-1])[::-1]
        completed = rows[-1]["t"] >= end - 1e-9
        candidates = np.flatnonzero(suffix_stable & (times[-1] - times + dt >= SETTLE_DURATION_S - 1e-9))
        reached = np.flatnonzero(in_angle)
        direction = np.sign(np.deg2rad(event.target_deg) - initial_angle)
        reports.append({
            "command_time_s": event.time_s, "target_deg": event.target_deg, "samples": len(rows),
            "completed_window": completed,
            "first_reach_time_s": float(times[reached[0]]) if len(reached) else None,
            "settling_time_s": float(times[candidates[0]]) if completed and len(candidates) else None,
            "angle_tolerance_deg": float(np.rad2deg(ANGLE_TOLERANCE_RAD)),
            "speed_tolerance_deg_s": float(np.rad2deg(SPEED_TOLERANCE_RAD_S)),
            "minimum_settled_duration_s": SETTLE_DURATION_S,
            "requested_angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors ** 2)))),
            "integrated_absolute_error_deg_s": float(np.rad2deg(np.sum(np.abs(errors))) * dt),
            "final_error_deg": float(np.rad2deg(errors[-1])),
            "overshoot_deg": float(np.rad2deg(max(0.0, np.max(direction * errors)))) if direction else 0.0,
            "contact_force_peak_N": max(r["substep_peaks"]["tissue_contact_force_N"] for r in rows),
            "contact_penetration_peak_m": max(r["substep_peaks"]["tissue_contact_penetration_m"] for r in rows),
        })
        initial_angle = float(angles[-1])
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="heuristic")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", choices=("commands", "holds", "extension", "vertical", "endurance", "contact", "compact"), default="commands",
                        help="commands: short checks; contact: obstructed goals/return/release; endurance: 300 s hold + 120 s rest.")
    parser.add_argument("--output", type=Path, default=Path("runs/d1_evaluation.json"))
    args = parser.parse_args(argv)
    config_path = args.config
    if config_path is None and args.controller in ("heuristic", "zero"):
        config_path = Path("configs/d1_myoarm.json")
    # Learned controllers keep the exact saved configuration (including
    # curriculum/history). set_command disables random future commands.
    env, action, metadata = load_controller(args.controller, config_path, args.predictor)
    try:
        base = env.unwrapped
        if base.config.task_mode == "reference":
            raise ValueError("D1 evaluation requires an interactive/curriculum controller configuration.")
        from .compact_protocol import compact_scenarios, protocol_metadata

        suite = {"commands": scenarios, "holds": hold_scenarios, "compact": compact_scenarios,
                 "extension": extension_scenarios, "vertical": vertical_scenarios,
                 "endurance": endurance_scenarios, "contact": contact_scenarios}[args.suite](base.config.payload_mass)
        results = [run_scenario(env, action, scenario, args.seed,
                               continue_after_time_limit=args.suite != "commands",
                               measure_commands=args.suite == "compact")
                   for scenario in suite]
        branch_names = ({"contact_release", "contact_forced_zero"} if args.suite == "contact" else
                        {"extension_command", "extension_release", "extension_forced_zero"}
                        if args.suite == "extension" else {"controlled_lower", "release", "forced_zero"})
        comparable = {result["scenario"]["name"]: result["branch_state"]["sha256"]
                      for result in results
                      if result["scenario"]["name"] in branch_names
                      and result["branch_state"] is not None}
        report = {"schema_version": 1, "controller": args.controller, "suite": args.suite,
                  "protocol": protocol_metadata(base.config.payload_mass) if args.suite == "compact" else None,
                  "config": asdict(base.config), "contract": policy_contract(base),
                  "muscle_names": list(base.muscle_names),
                  "dynamics_sha256": dynamics_digest(),
                  "training_transitions": metadata.get("trained_steps"),
                  "comparison_branch_states_identical": (len(comparable) == len(branch_names) and len(set(comparable.values())) == 1)
                      if args.suite in ("commands", "extension", "contact") else None,
                  "scenarios": results,
                  "limitations": ["Engineering proxies, not human injury thresholds or probabilities.",
                      "Tracking RMSE uses only frames with trigger > 0; release has no angle objective.",
                      "Substep peaks are environment-reported; angle/speed extrema use control frames.",
                      "Forced zero preserves delayed excitation, activation and passive mechanics.",
                      "Long suites continue across training time limits without resetting; physical failures still stop them.",
                      "Bias torque is MuJoCo qfrc_bias; its sign crossing is sampled at control frames.",
                      "An exact lower-angle command can press against passive support; release has no angle objective.",
                      "Path direction reversals are reported as extrapolation warnings; they do not stop or clamp motion.",
                      "Angle standard deviation includes slow drift and does not alone establish tremor.",
                      "Contact can obstruct a high target; smaller tracking error alone can mean excessive compression.",
                      "These fixed scenarios do not establish general policy success or formal safety."]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output),
                          "comparison_branch_states_identical": report["comparison_branch_states_identical"],
                          "scenarios": [{"name": result["scenario"]["name"],
                              "tracking_rmse_deg": result["overall"]["angle_rmse_deg"],
                              "physical_failure": result["physical_failure"],
                              "numerical_failure": result["numerical_failure"],
                              "final_angle_deg": result["final_angle_deg"],
                              "hold_tail_rmse_deg": result["hold_tail_last_5s"]["angle_rmse_deg"]
                                  if result["hold_tail_last_5s"] else None} for result in results]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
