#!/usr/bin/env python3
"""Measure the contact solver response in a gravity-free, muscle-free fixture.

This queries fixed poses and velocities; it is not a static holding equilibrium
or a tissue material measurement. Forces depend on the arm's effective inertia.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from myo.env import PreacherCurlEnv
from myo.runtime import config_digest, dynamics_digest, read_config


def audit(config_path: Path) -> dict:
    config = read_config(config_path)
    if config.geometry != "myoarm":
        raise ValueError("The contact audit requires MyoArm geometry.")
    env = PreacherCurlEnv(config)
    try:
        # No step() and no controller: isolate the contact acceleration response
        # from gravity, delayed muscles and the lower joint limit.
        env.model.opt.gravity[:] = 0.0
        env.model.dof_damping[:] = 0.0
        env.data.qfrc_applied[:] = 0.0

        def sample(degrees: float, speed: float = 0.0) -> dict:
            env.data.qpos[env.qpos_index] = np.deg2rad(degrees)
            env.data.qvel[env.dof_index] = speed
            env.data.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            return {"angle_deg": float(degrees), "angular_velocity_rad_s": speed,
                    **env.tissue_contact_metrics()}

        previous = 15.0
        bracket = None
        for degrees in np.arange(15.0, 170.01, 0.25):
            if sample(degrees)["count"]:
                bracket = (previous, float(degrees))
                break
            previous = float(degrees)
        if bracket is None or bracket[0] == bracket[1]:
            raise RuntimeError("Expected first tissue contact between 15 and 170 degrees.")
        lower, upper = bracket
        for _ in range(35):
            middle = (lower + upper) / 2.0
            if sample(middle)["count"]:
                upper = middle
            else:
                lower = middle
        onset = upper
        rows = [sample(onset + offset, speed)
                for speed in (0.0, 0.5, 2.0)
                for offset in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0)]
        return {
            "config_path": str(config_path), "config": asdict(config),
            "config_sha256": config_digest(config), "dynamics_sha256": dynamics_digest(),
            "geometry_sha256": env.geometry.fingerprint,
            "fixture": "fixed poses; gravity=0; joint damping=0; applied force=0; no muscles or time integration",
            "limitations": "Solver response depends on effective inertia; not tissue stress or human calibration.",
            "first_contact_deg": onset, "samples": rows,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_myoarm.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/tissue_contact_response.json"))
    args = parser.parse_args()
    report = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"First contact: {report['first_contact_deg']:.4f} deg; report: {args.output}")
    for row in report["samples"]:
        if row["angular_velocity_rad_s"] == 0:
            print(f"{row['angle_deg']:.3f} deg: {1000 * row['max_penetration_m']:.3f} mm, "
                  f"{row['normal_force_N']:.2f} N, {row['torque_Nm']:.3f} Nm")


if __name__ == "__main__":
    main()
