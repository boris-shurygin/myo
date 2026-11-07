"""Strict checkpoint reload, deterministic native replay and goal-blind baselines."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def heuristic_reference_baseline(config):
    """Known-trajectory inverse-force diagnostic; never used to train PPO."""
    from myo.vertical_mimic import TEST_MOTIONS, VerticalMimicEnv
    cases = []
    for index, motion in enumerate(TEST_MOTIONS):
        env = VerticalMimicEnv(config, TEST_MOTIONS)
        env.reset(options={"motion_index": index, "frame": 0})
        samples = []
        while True:
            p = env.physical
            preview = (config.motor_delay_s if config.delayed else 0.)+(.025 if config.activation_dynamic else 0.)
            qr, vr = env.reference(env.frame+preview/config.dt)
            ar = -(2*np.pi/motion.period_s)**2*(qr-motion.offset_m)
            acceleration = ar+28*(qr-p.position-preview*p.velocity)+9*(vr-p.velocity)
            # Include the current force-velocity factor. The original toy
            # helper inverts its isometric coefficient and biases moving cases.
            lengths = p.fiber_lengths()
            velocities = [-p.velocity, p.velocity]
            coefficients, passive = [], []
            for muscle, length, velocity in zip(p.muscles, lengths, velocities):
                active, spring = muscle._fiber_forces(float(length), 1., 1., float(velocity))[:2]
                coefficients.append(active)
                passive.append(spring)
            net = config.mass_kg*(9.81+acceleration)
            lower = max(passive[1]+5., passive[0]+5.-net)
            command = np.clip((np.array([lower+net, lower])-passive)/np.maximum(coefficients, 1e-8), 0., 1.)
            _, _, terminated, truncated, info = env.step(2*command-1)
            samples.append([p.time, info["error_q_m"], info["error_v_m_s"]])
            if terminated or truncated:
                break
        a = np.asarray(samples)
        qrmse, vrmse = np.sqrt(np.mean(a[:, 1:]**2, axis=0))
        complete = not terminated and env.frame == env.last_frame
        cases.append({"motion": motion.name, "completed": complete,
            "q_rmse_m": float(qrmse), "v_rmse_m_s": float(vrmse),
            "passed": bool(complete and qrmse <= .01 and vrmse <= .05)})
    return {"completed": sum(c["completed"] for c in cases), "passed": sum(c["passed"] for c in cases),
            "total": len(cases), "q_rmse_m": float(np.mean([c["q_rmse_m"] for c in cases])),
            "v_rmse_m_s": float(np.mean([c["v_rmse_m_s"] for c in cases])), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/vertical_mimic"))
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-progress", action="store_true",
                        help="After training, also evaluate initial/last on test; never selects weights")
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    from myo.train_vertical_mimic import MimicPolicy, load_checkpoint, write_json
    from myo.vertical_mimic import (VerticalMimicEnv, TEST_MOTIONS, VALIDATION_MOTIONS,
                                     compact_mimic, evaluate_mimic)
    from myo.vertical_muscles import VerticalMuscles
    report = {"status": "running", "strict_checkpoints": 0, "native_replays": 0, "variants": []}
    for variant in args.variants:
        folder = args.root/f"variant_{variant}"
        summary = json.loads((folder/"summary.json").read_text())
        assert summary["status"] == "complete"
        assert summary["training"]["update_epochs"] == 1
        assert len(summary["history"]) == summary["training"]["planned_updates"]
        for name in ("initial", "last", "best"):
            state, network, env, metadata = load_checkpoint(folder/f"{name}.msgpack")
            assert metadata["optimizer_steps"] == metadata["steps"]//(summary["training"]["num_envs"]*20)*32
            report["strict_checkpoints"] += 1
        config = env.physical.config
        replay = evaluate_mimic(MimicPolicy(network, state), config, TEST_MOTIONS)
        saved = json.loads((folder/"test.json").read_text())
        assert compact_mimic(replay) == compact_mimic(saved)
        if args.checkpoint_progress:
            progress = {"best": {"steps": metadata["steps"], **compact_mimic(replay)}}
            for checkpoint_name in ("initial", "last"):
                checkpoint_state, checkpoint_network, _, checkpoint_metadata = load_checkpoint(
                    folder/f"{checkpoint_name}.msgpack")
                data = evaluate_mimic(MimicPolicy(checkpoint_network, checkpoint_state), config, TEST_MOTIONS)
                write_json(folder/f"{checkpoint_name}_test.json", data)
                progress[checkpoint_name] = {"steps": checkpoint_metadata["steps"], **compact_mimic(data)}
            write_json(folder/"test_progress.json", progress)
        for i, case in enumerate(replay["cases"]):
            np.testing.assert_array_equal(case["samples"], saved["cases"][i]["samples"])
            # Same reset, but then the plain native stand runs the recorded
            # excitations, without any reference observations or reward code.
            fresh = VerticalMimicEnv(config, TEST_MOTIONS)
            fresh.reset(options={"motion_index": i, "frame": 0})
            native = VerticalMuscles(config)
            native.restore(fresh.physical.snapshot())
            for sample in case["samples"]:
                t, q, v, rq, rv, a0, a1, u0, u1, d0, d1, act0, act1, reward = sample
                np.testing.assert_allclose([u0, u1], np.clip((np.array([a0, a1])+1)/2, 0, 1), atol=0, rtol=0)
                try:
                    native.step([u0, u1])
                except RuntimeError:
                    assert case["physical_failure"]
                np.testing.assert_allclose([native.time, native.position, native.velocity], [t, q, v], atol=0, rtol=0)
                np.testing.assert_array_equal(native.delivered, [d0, d1])
                np.testing.assert_array_equal(native.activation, [act0, act1])
            report["native_replays"] += 1
        val = evaluate_mimic(MimicPolicy(network, state), config, VALIDATION_MOTIONS, phases=(0., .35))
        initial_val = summary["initial_validation"]
        candidates = [(initial_val["selection_score_m"], 0)]+[
            (h["validation"]["selection_score_m"], h["steps"]) for h in summary["history"] if "validation" in h]
        assert metadata["steps"] == min(candidates, key=lambda x: x[0])[1]
        assert val["selection_score_m"] == min(x[0] for x in candidates)

        support = VerticalMuscles(config)
        support.reset()
        constant = 2*support.excitation-1
        baseline = evaluate_mimic(lambda obs, **kw: constant, config, TEST_MOTIONS)
        write_json(folder/"constant_support_test.json", baseline)
        stochastic = evaluate_mimic(MimicPolicy(network, state, seed=321), config, TEST_MOTIONS,
                                     stochastic=True)
        write_json(folder/"stochastic_test.json", stochastic)
        heuristic = heuristic_reference_baseline(config)
        write_json(folder/"heuristic_test.json", heuristic)
        report["variants"].append({"variant": variant, "best_step": metadata["steps"],
            "test": compact_mimic(replay), "validation": compact_mimic(val),
            "constant_support": compact_mimic(baseline), "stochastic": compact_mimic(stochastic),
            "heuristic": heuristic})
        print(json.dumps({"variant": variant, "test_passed": replay["passed"],
                          "q_rmse_m": replay["q_rmse_m"]}), flush=True)
    report["status"] = "passed"
    write_json(args.output or args.root/"independent_evaluation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}), flush=True)


if __name__ == "__main__":
    main()
