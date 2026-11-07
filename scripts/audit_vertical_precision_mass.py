"""Explicit mass distribution shift for a frozen, strictly loaded PPO policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from myo.train_vertical_architecture import load_trial
from myo.train_vertical_ppo import write_json
from myo.vertical_control import TEST_CASES
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import VerticalPrecisionEnv, PRECISION_TEST


def run_case(model, config, initial, target):
    env = VerticalPrecisionEnv(config)
    obs, _ = env.reset(options={"initial_position_m": initial, "target_m": target})
    p = env.physical
    initial_activation = p.activation.copy()
    initial_residual = float(p.applied_forces[0]-p.applied_forces[1]-config.mass_kg*9.81)
    peaks = {name: {"upper": {"value": float(values[0]), "time_s": 0.},
                    "lower": {"value": float(values[1]), "time_s": 0.}}
             for name, values in (("activation", p.activation), ("excitation", p.excitation),
                                  ("delivered", p.delivered), ("tendon_force_N", p.applied_forces))}
    physics = [[0., p.position, p.velocity]]
    real_step = mujoco.mj_step

    def record(model, data):
        real_step(model, data)
        physics.append([p.time, p.position, p.velocity])
        for name, values in (("activation", p.activation), ("excitation", p.excitation),
                             ("delivered", p.delivered), ("tendon_force_N", p.applied_forces)):
            for index, muscle in enumerate(("upper", "lower")):
                if values[index] > peaks[name][muscle]["value"]:
                    peaks[name][muscle] = {"value": float(values[index]), "time_s": p.time}

    samples, failed = [], False
    with patch("myo.vertical_muscles.mujoco.mj_step", side_effect=record):
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, failed, truncated, _ = env.step(action)
            samples.append([p.time, p.position, p.velocity, *p.excitation, *p.delivered,
                            *p.activation, *p.applied_forces])
            if failed or truncated:
                break
    a = np.asarray(physics)
    t, q, v = a.T
    tail = t >= 3.
    within = (np.abs(q-target) <= .005) & (np.abs(v) <= .03)
    stationary = np.logical_and.accumulate(within[::-1])[::-1]
    settled = np.flatnonzero(stationary)
    # Keep pass identical to the historical 20 ms test, rather than changing
    # its threshold by switching the sampling rate in this diagnostic.
    controls = np.asarray(samples)
    control_tail = controls[:, 0] >= 3.
    control_within = (np.abs(controls[:, 1]-target) <= .005) & (np.abs(controls[:, 2]) <= .03)
    passed = not failed and bool(np.mean(control_within[control_tail]) >= .9) if control_tail.any() else False
    direction = np.sign(target-initial)
    result = {"initial_position_m": initial, "target_m": target, "duration_s": p.time,
        "physical_failure": bool(failed), "passed": passed, "initial_activation": initial_activation.tolist(),
        "initial_force_balance_residual_N": initial_residual, "peaks_1ms": peaks,
        "last_second_rmse_m": float(np.sqrt(np.mean((controls[control_tail, 1]-target)**2))) if control_tail.any() else None,
        "last_second_signed_error_m": float(np.mean(q[tail]-target)) if tail.any() else None,
        "overshoot_m": max(0., float(np.max(direction*(q-target)))) if direction else None,
        "settling_time_s": float(t[settled[0]]) if len(settled) and not failed else None,
        "final_activation": p.activation.tolist(), "final_excitation": p.excitation.tolist(),
        "last_second_upper_activation_mean": float(np.mean(controls[control_tail, 7])) if control_tail.any() else None,
        "samples_columns": ["time_s", "position_m", "velocity_m_s", "excitation_upper", "excitation_lower",
                            "delivered_upper", "delivered_lower", "activation_upper", "activation_lower",
                            "tendon_force_upper_N", "tendon_force_lower_N"],
        "samples": samples}
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("runs/vertical_precision/combined/best.zip"))
    parser.add_argument("--masses", type=float, nargs="+", default=[.5, 1., 1.5, 2., 3., 5.])
    parser.add_argument("--output", type=Path, default=Path("runs/vertical_precision/mass_audit.json"))
    args = parser.parse_args()
    if any(not np.isfinite(m) or m <= 0 for m in args.masses):
        parser.error("Masses must be positive finite values")
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.controller)+".metadata.json").read_text())
    training_config = VerticalConfig(**metadata["contract"]["physical"]["config"])
    original_env = VerticalPrecisionEnv(training_config)
    if "shared_design" in metadata["contract"]:
        from myo.shared_vertical import load_shared
        original_env.close()
        model, original_env, _ = load_shared(args.controller)
    else:
        model, _ = load_trial(args.controller, original_env)
    original_env.close()
    output = {"checkpoint": str(args.controller), "checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trained_physics_config": metadata["contract"]["physical"]["config"],
        "protocol": "Frozen model; strict load at trained mass, explicit evaluation-only mass changes; no retraining",
        "reset": "Standard mass-specific balanced preload; no mid-episode payload changes or rescue",
        "physics_dt_s": training_config.physics_dt, "control_dt_s": training_config.dt,
        "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mass in args.masses:
        config = replace(training_config, mass_kg=mass)
        cases = [run_case(model, config, initial, target) for initial, target in PRECISION_TEST]
        def peak(name):
            c = max(cases, key=lambda case: case["peaks_1ms"][name]["upper"]["value"])
            return {**c["peaks_1ms"][name]["upper"], "initial_position_m": c["initial_position_m"], "target_m": c["target_m"]}
        score = np.mean([c["last_second_rmse_m"] if not c["physical_failure"] else .3 for c in cases])
        row = {"mass_kg": mass, "evaluation_physics_config": {**metadata["contract"]["physical"]["config"], "mass_kg": mass},
            "passed": sum(c["passed"] for c in cases), "total": len(cases),
            "historical_passed": sum(c["passed"] for c in cases[:len(TEST_CASES)]),
            "failures": sum(c["physical_failure"] for c in cases), "score_m": float(score),
            "max_initial_force_balance_residual_N": max(abs(c["initial_force_balance_residual_N"]) for c in cases),
            "peak_upper_activation": peak("activation"), "peak_upper_excitation": peak("excitation"),
            "peak_upper_tendon_force_N": peak("tendon_force_N"), "cases": cases}
        output["rows"].append(row)
        write_json(args.output, output)
        print(json.dumps({k: v for k, v in row.items() if k not in ("cases", "evaluation_physics_config")}), flush=True)
    output["status"] = "complete"
    # At the trained mass, reproduce all frozen test control samples exactly.
    for row in output["rows"]:
        if row["mass_kg"] == training_config.mass_kg:
            reference_path = args.controller.parent/"best_test.json"
            if reference_path.exists():
                reference = json.loads(reference_path.read_text())["cases"]
                for case, saved in zip(row["cases"], reference, strict=True):
                    a = np.asarray(case["samples"])
                    b = np.asarray(saved["samples"])
                    np.testing.assert_array_equal(a[:, [0, 1, 2, 3, 4]], b[:, [0, 1, 2, 4, 5]])
                    assert case["passed"] == saved["passed"]
                output["trained_mass_control_replay_exact"] = True
    write_json(args.output, output)


if __name__ == "__main__":
    main()
