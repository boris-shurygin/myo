"""Offline isometric arm audit; no controller, motion integration or hidden support.

Each pose is held by the measurement fixture, as in a dynamometer experiment.
Independent copies of the series-elastic muscles solve static equilibrium. The
result is a fresh mechanical capacity, not a prediction of voluntary human MVC.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from .env import PreacherCurlConfig, PreacherCurlEnv
from .muscle import SeriesElasticMuscle
from .runtime import config_digest, dynamics_digest, read_config


def measure_pose(env: PreacherCurlEnv, angle: float, capacity: float = 1.0) -> dict:
    """Measure signed torques at a fixed angle in radians, flexion positive.

    This mutates only the offline fixture's MuJoCo pose. It does not step time,
    fatigue or the environment's live muscles. Named muscle groups remain fixed
    if an extrapolated path reverses its moment arm.
    """
    env.data.qpos[env.qpos_index] = angle
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    lengths = env.muscle_lengths()
    arms = (env.geometry.moment_arms(env.data) if env.geometry else env.moment_arms.copy())
    flexors = np.array([not name.lower().startswith("tri") for name in env.muscle_names])

    def equilibrium(activation):
        outputs = [SeriesElasticMuscle(m.params, env.config.physics_dt).reset(
            float(length), float(a), capacity)
            for m, length, a in zip(env.muscles, lengths, activation)]
        return float(arms @ np.array([o.tendon_force for o in outputs])), outputs

    passive, passive_outputs = equilibrium(np.zeros(len(arms)))
    full_flexion, flexion_outputs = equilibrium(flexors.astype(float))
    full_extension, extension_outputs = equilibrium((~flexors).astype(float))
    gravity = float(env.data.qfrc_bias[env.dof_index])
    limit = env.passive_limit_torque()
    required = gravity - passive - limit
    selected = flexors if required >= 0 else ~flexors
    maximum = full_flexion if required >= 0 else full_extension
    target = gravity - limit
    reachable = min(passive, maximum) <= target <= max(passive, maximum)
    activation = None
    holding_outputs = None
    if reachable:
        lo, hi = 0.0, 1.0
        for _ in range(36):
            mid = (lo + hi) / 2
            torque, holding_outputs = equilibrium(selected.astype(float) * mid)
            if (torque < target) == (maximum > passive):
                lo = mid
            else:
                hi = mid
        activation = (lo + hi) / 2

    def muscle_row(index):
        p = env.muscles[index].params
        return {
            "name": env.muscle_names[index], "length_m": float(lengths[index]),
            "moment_arm_m": float(arms[index]),
            "expected_moment_arm_sign": 1 if flexors[index] else -1,
            "moment_arm_reversed": bool((arms[index] < 0) if flexors[index] else (arms[index] > 0)),
            "passive_tendon_force_N": passive_outputs[index].tendon_force,
            "full_group_tendon_force_N": (flexion_outputs if flexors[index] else extension_outputs)[index].tendon_force,
            "passive_fiber_length_ratio": passive_outputs[index].fiber_length / p.optimal_fiber_length,
            "holding_activation": (activation if selected[index] else 0.0) if reachable else None,
            "holding_tendon_force_N": holding_outputs[index].tendon_force if reachable else None,
            "holding_fiber_length_ratio": holding_outputs[index].fiber_length / p.optimal_fiber_length if reachable else None,
        }

    return {
        "angle_deg": float(np.rad2deg(angle)), "capacity": capacity,
        "gravity_compensation_Nm": gravity,
        "passive_muscle_torque_Nm": passive, "passive_limit_torque_Nm": limit,
        "required_active_torque_Nm": required,
        "full_flexion_net_muscle_torque_Nm": full_flexion,
        "full_extension_net_muscle_torque_Nm": full_extension,
        "flexion_increment_over_passive_Nm": full_flexion - passive,
        "extension_increment_over_passive_Nm": passive - full_extension,
        "holding_group": "flexors" if required >= 0 else "extensors",
        "uniform_holding_activation": activation,
        "holding_reachable": bool(reachable),
        "muscles": [muscle_row(i) for i in range(len(arms))],
    }


def audit(config: PreacherCurlConfig, angles: list[float], capacity: float = 1.0) -> dict:
    env = PreacherCurlEnv(config)
    try:
        rows = [measure_pose(env, float(np.deg2rad(a)), capacity) for a in angles]
        if env.geometry:
            source = json.loads((Path(config.asset_dir) / "muscles.json").read_text())
            for row in rows:
                arms = np.array([m["moment_arm_m"] for m in row["muscles"]])
                active, passive = [], []
                for muscle, imported in zip(row["muscles"], source):
                    length_range = np.array(imported["upstream_lengthrange_m"])
                    gain = np.array(imported["upstream_gainprm"][:9])
                    active.append(-mujoco.mju_muscleGain(muscle["length_m"], 0.0, length_range, 1.0, gain))
                    passive.append(-mujoco.mju_muscleBias(muscle["length_m"], length_range, 1.0, gain))
                flexors = np.array([not m["name"].lower().startswith("tri") for m in row["muscles"]])
                active, passive = np.array(active), np.array(passive)
                row["upstream_rigid_tendon_reference"] = {
                    "passive_muscle_torque_Nm": float(arms @ passive),
                    "full_flexion_net_muscle_torque_Nm": float(arms @ (passive + active * flexors)),
                    "full_extension_net_muscle_torque_Nm": float(arms @ (passive + active * ~flexors)),
                    "note": "source gain and bias at zero velocity; no series elasticity or fatigue; never applied to simulation",
                }
        return {
            "schema": "myo.isometric_audit.v1", "config": asdict(config),
            "config_sha256": config_digest(config), "dynamics_sha256": dynamics_digest(),
            "geometry": env.geometry.fingerprint if env.geometry else "toy",
            "protocol": "fixed_pose_static_series_elastic_equilibrium; no_time_advance; no_fatigue_update",
            "torque_sign": "flexion_positive; gravity_compensation_is_required_motor_torque",
            "limitations": ["mechanical full activation is not measured human voluntary MVC",
                            "no collisions or anatomical validity implied beyond imported ROM",
                            "uniform agonist activation is a static feasibility probe, not a controller"],
            "poses": rows,
        }
    finally:
        env.close()


def fit_active_scale(config: PreacherCurlConfig, reference: dict) -> dict:
    """Fit one bounded group scale while retaining tendon and passive mechanics.

    This single-angle proposal is written only to the report, never applied to
    the user's config. References must distinguish peak from sustained MVC.
    """
    group = reference["group"]
    field = f"{group}_active_force_scale"
    if field not in config.__dataclass_fields__:
        raise ValueError(f"The environment does not expose independent {field}.")
    angle = float(reference["angle_deg"])
    target = float(reference["torque_Nm"])
    lo, hi = reference.get("scale_bounds", [0.5, 3.0])
    key = "full_flexion_net_muscle_torque_Nm" if group == "flexor" else "full_extension_net_muscle_torque_Nm"
    env = PreacherCurlEnv(config)
    originals = [m.params for m in env.muscles]
    indices = [i for i, name in enumerate(env.muscle_names)
               if name.lower().startswith("tri") == (group == "extensor")]

    def at_scale(scale):
        for i in indices:
            env.muscles[i] = SeriesElasticMuscle(replace(originals[i], active_force_scale=scale), config.physics_dt)
        return abs(measure_pose(env, np.deg2rad(angle))[key])

    try:
        low_torque, high_torque = at_scale(lo), at_scale(hi)
        if not low_torque <= target <= high_torque:
            return {"reference": reference, "fit_success": False,
                    "reason": "reference outside bounded scale search", "achievable_Nm": [low_torque, high_torque]}
        for _ in range(32):
            mid = (lo + hi) / 2
            if at_scale(mid) < target:
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
        return {"reference": reference, "fit_success": True, "proposed_config": {field: scale},
                "fitted_torque_Nm": at_scale(scale), "automatic_config_update": False,
                "limitation": "single-angle preliminary reference; not full-curve anatomical validation"}
    finally:
        env.close()


def fit_source_width(config: PreacherCurlConfig) -> dict:
    """Match the imported model's fresh torque-curve shape, not human data.

    Both anatomical muscle groups have equal relative weight. The source model
    is evaluated with its own gain/bias functions; it never drives the fixture.
    Alternate angles are held out from this one-parameter shape fit.
    """
    baseline = replace(config, flexor_active_force_scale=1.0, extensor_active_force_scale=1.0)
    angles = [15, 30, 45, 60, 75, 90, 105, 120, 130]
    fit_mask = np.array([True, False, True, False, True, False, True, False, True])
    if not baseline.geometry == "myoarm":
        raise ValueError("Source curve fitting requires MyoArm geometry.")
    source = audit(baseline, angles)
    keys = ["full_flexion_net_muscle_torque_Nm", "full_extension_net_muscle_torque_Nm"]
    targets = np.array([[p["upstream_rigid_tendon_reference"][k] for k in keys] for p in source["poses"]])
    normalizer = np.max(np.abs(targets[fit_mask]), axis=0)
    env = PreacherCurlEnv(baseline)
    originals = [m.params for m in env.muscles]

    def curve(width):
        env.muscles = [SeriesElasticMuscle(replace(p, force_length_width=width), baseline.physics_dt)
                       for p in originals]
        return np.array([[p[k] for k in keys] for a in angles
                         for p in [measure_pose(env, np.deg2rad(a))]])

    def objective(width):
        return float(np.mean(((curve(width)[fit_mask] - targets[fit_mask]) / normalizer) ** 2))

    try:
        lo, hi = 0.5, 0.95
        for _ in range(30):
            left, right = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if objective(left) < objective(right):
                hi = right
            else:
                lo = left
        width = (lo + hi) / 2
        fitted = curve(width)
        return {
            "proposed_config": {"muscle_force_length_width": width},
            "fit_angles_deg": np.array(angles)[fit_mask].tolist(),
            "held_out_angles_deg": np.array(angles)[~fit_mask].tolist(),
            "torque_columns": keys, "angles_deg": angles,
            "source_torques_Nm": targets.tolist(), "fitted_torques_Nm": fitted.tolist(),
            "fit_rmse_Nm": np.sqrt(np.mean((fitted[fit_mask] - targets[fit_mask]) ** 2, axis=0)).tolist(),
            "held_out_rmse_Nm": np.sqrt(np.mean((fitted[~fit_mask] - targets[~fit_mask]) ** 2, axis=0)).tolist(),
            "objective": "mean_squared_error_per_direction_normalized_by_source_training_peak",
            "search_bounds": [0.5, 0.95], "active_force_scales_during_fit": [1.0, 1.0],
            "limitation": "agreement with an imported rigid-tendon model is not human anatomical validation",
        }
    finally:
        env.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_myoarm.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/arm_calibration.json"))
    parser.add_argument("--angles", type=float, nargs="+", default=[15, 30, 50, 60, 75, 90, 105, 120, 130, 140, 145, 150, 155, 160, 170, 180])
    parser.add_argument("--capacity", type=float, default=1.0)
    parser.add_argument("--fit-source-width", action="store_true")
    parser.add_argument("--fit-reference", type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    report = audit(config, args.angles, args.capacity)
    fit_config = config
    if args.fit_source_width:
        report["source_width_fit"] = fit_source_width(config)
        fit_config = replace(config, **report["source_width_fit"]["proposed_config"])
    if args.fit_reference:
        report["strength_fit"] = fit_active_scale(fit_config, json.loads(args.fit_reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "poses": len(report["poses"]),
                      "reversed_path_angles_deg": [p["angle_deg"] for p in report["poses"]
                          if any(m["moment_arm_reversed"] for m in p["muscles"])]}))


if __name__ == "__main__":
    main()
