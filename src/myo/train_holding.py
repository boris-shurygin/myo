"""Result-based curriculum: prepared holds, short moves, then cold compact tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np

from .compact_protocol import CAPACITIES, TRIGGERS, compact_scenarios, protocol_metadata
from .continue_compact import Plateau, behavior_summary
from .env import PreacherCurlConfig, PreacherCurlEnv
from .evaluate import CommandEvent, Scenario, run_scenario
from .prepared_protocol import PreparedProtocol
from .runtime import load_controller
from .train_bounded import RewardScale, write_json


def stage_cases(stage, payload):
    if stage == 3:
        return compact_scenarios(payload)
    cases = []
    for capacity in CAPACITIES:
        for trigger in TRIGGERS:
            for target in (45, 75, 120):
                for offset in ((0,) if stage < 2 else (-10, 10)):
                    initial = target + offset
                    commands = (CommandEvent(0, initial, trigger),)
                    if stage == 2:
                        commands += (CommandEvent(0.25, target, trigger),)
                    cases.append(Scenario(f"stage{stage}_q{target}_d{offset}_u{trigger:g}_c{capacity:g}",
                        (0.5, 2.0, 4.0)[stage], commands, payload,
                        capacity=capacity, initial_angle_deg=initial))
    return tuple(cases)


def evaluate_prepared(model, env):
    results = []
    for _ in env.scenarios:
        obs, info = env.reset(seed=0)
        preparation = info["preparation"]
        samples = []
        while True:
            obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            samples.append((float(np.rad2deg(info["q"] - info["requested_target_q"])),
                            float(np.rad2deg(info["qvel"])),
                            float(info["substep_peaks"]["joint_limit_reaction_Nm"])))
            if terminated or truncated:
                break
        a = np.array(samples)
        # For movement stages, measure the last second; no fly-through credit.
        tail = a[-min(len(a), round(1 / env.unwrapped.control_dt)):]
        rmse = float(np.sqrt(np.mean(tail[:, 0] ** 2)))
        speed_rms = float(np.sqrt(np.mean(tail[:, 1] ** 2)))
        final_error = abs(float(a[-1, 0]))
        passed = not terminated and rmse <= 3 and final_error <= 3 and speed_rms <= 5 and a[:, 2].max() <= 20
        results.append({"name": env.scenario.name, "passed": bool(passed),
            "tail_angle_rmse_deg": rmse, "tail_speed_rms_deg_s": speed_rms,
            "final_error_deg": final_error, "joint_reaction_peak_Nm": float(a[:, 2].max()),
            "preparation": preparation, "physical_failure": bool(terminated)})
    return {"score": float(np.mean([r["tail_angle_rmse_deg"] + 0.1 * r["tail_speed_rms_deg_s"] for r in results])),
            "passed": sum(r["passed"] for r in results), "total": len(results),
            "ready": all(r["passed"] for r in results), "scenarios": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("runs/d1_bounded_persistent_beta/initial.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args(argv)
    if args.patience < 1:
        parser.error("Patience must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    checked, _, parent = load_controller(str(args.controller))
    checked.close()
    if parent.get("reward_scale") != 0.01 or parent.get("action_persistence") != 0.9:
        raise ValueError("This curriculum continues the scaled persistent Beta experiment")
    config = PreacherCurlConfig(**parent["config"])
    descriptor = {"name": "prepared_holding_v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_sha256": hashlib.sha256(Path(__file__).with_name("prepared_protocol.py").read_bytes()).hexdigest(),
        "stages": [{"stage": s, "preparation_seconds": 2.0 if s < 3 else 0.0,
                    "scenarios": [asdict(c) for c in stage_cases(s, config.payload_mass)]} for s in range(4)],
        "reward": protocol_metadata(config.payload_mass)["reward"],
        "teacher_role": "initial physical preparation only; no labels or runtime teacher",
        "final_evaluation": "unchanged cold compact 18 exercises"}
    previous_descriptor = parent.get("prepared_curriculum")
    if previous_descriptor and previous_descriptor != json.loads(json.dumps(descriptor)):
        raise ValueError("Prepared curriculum differs from the saved training contract")
    model = None
    summary = {"status": "running", "parent": str(args.controller), "curriculum": descriptor, "stages": []}
    metadata = {**parent, "prepared_curriculum": descriptor,
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "runner_sha256": descriptor["source_sha256"],
                "training_stage": "prepared_result_curriculum",
                "limitations": "Prepared-stage scores are not cold-start demo readiness; no release training"}
    cold_env = PreacherCurlEnv(config)

    def save(name, stage, protocol):
        checkpoint = args.output_dir / f"{name}.zip"
        model.save(checkpoint)
        sidecar = {**metadata, "trained_steps": model.num_timesteps, "prepared_curriculum_stage": stage,
            "training_protocol": {"curriculum": "prepared_holding_v1", "stage": stage,
                                  "descriptor_sha256": descriptor["source_sha256"]},
            "protocol_completed_episodes": protocol.completed_episodes,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        write_json(str(checkpoint) + ".metadata.json", sidecar)
        return checkpoint

    def evaluate_cold(checkpoint, name):
        results = [run_scenario(cold_env, lambda o: model.predict(o, deterministic=True)[0], c, 0,
                   continue_after_time_limit=True, measure_commands=True) for c in compact_scenarios(config.payload_mass)]
        metrics = behavior_summary(results)
        write_json(args.output_dir / f"{name}_cold.json", {"metrics": metrics, "scenarios": results})
        if "best_cold_score" not in summary or metrics["score"] < summary["best_cold_score"] or metrics["demo_ready"]:
            for suffix in (".zip", ".zip.metadata.json"):
                shutil.copy2(str(checkpoint).removesuffix(".zip") + suffix, args.output_dir / ("best_cold" + suffix))
            summary.update(best_cold_score=metrics["score"], best_cold_controller=str(checkpoint))
        return metrics

    class Audit(BaseCallback):
        def __init__(self):
            super().__init__()
            self.bonuses = []

        def _on_step(self):
            self.bonuses.append(self.locals["infos"][0]["reward_components"]["settled_bonus"])
            return True

    try:
        first_stage = parent.get("prepared_curriculum_stage", 0)
        for stage in range(first_stage, 4):
            cases = stage_cases(stage, config.payload_mass)
            protocol = PreparedProtocol(PreacherCurlEnv(config), cases, 2.0 if stage < 3 else 0.0)
            env = Monitor(RewardScale(protocol))
            evaluation = PreparedProtocol(PreacherCurlEnv(config), cases, 2.0 if stage < 3 else 0.0)
            if model is None:
                model = PPO.load(args.controller, env=env, device="cpu")
                model.set_random_seed(0)
            else:
                model.get_env().close()
                model.set_env(env, force_reset=True)
            save(f"stage_{stage}_initial", stage, protocol)
            plateau = None
            record = {"stage": stage, "evaluations": []}
            summary["stages"].append(record)
            block = 0
            while True:
                block += 1
                audit = Audit()
                # At least one fixed round, rounded to complete PPO rollouts.
                steps = math.ceil(sum(c.duration_s for c in cases) / config.control_dt / model.n_steps) * model.n_steps
                model.learn(steps, callback=audit, reset_num_timesteps=False)
                name = f"stage_{stage}_block_{block:03d}"
                checkpoint = save(name, stage, protocol)
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                try:
                    if stage < 3:
                        metrics = evaluate_prepared(model, evaluation)
                    else:
                        metrics = evaluate_cold(checkpoint, name)
                        metrics.update(ready=metrics["demo_ready"], passed=metrics["moves_passed"], total=12)
                finally:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                write_json(args.output_dir / f"{name}_evaluation.json", metrics)
                row = {"block": block, "steps": model.num_timesteps,
                       "score": metrics["score"], "passed": metrics["passed"], "total": metrics["total"],
                       "train_bonus_mean": float(np.mean(audit.bonuses)),
                       "train_bonus_above_half_fraction": float(np.mean(np.asarray(audit.bonuses) > 0.5))}
                record["evaluations"].append(row)
                print(json.dumps({"stage": stage, **row}), flush=True)
                if "best_score" not in record or metrics["score"] < record["best_score"]:
                    record.update(best_score=metrics["score"], best_controller=str(checkpoint))
                stalled = plateau.update(metrics["score"]) if plateau else False
                if plateau is None:
                    plateau = Plateau(metrics["score"], args.patience)
                if metrics["ready"] or stalled:
                    record["status"] = "ready" if metrics["ready"] else "plateau"
                    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                    try:
                        if stage < 3:
                            record["cold_metrics"] = evaluate_cold(checkpoint, name)
                    finally:
                        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                    if stalled and not metrics["ready"]:
                        summary["status"] = "stage_plateau"
                    elif stage == 3:
                        summary["status"] = "demo_ready"
                    break
                write_json(args.output_dir / "summary.json", summary)
            env.close()
            evaluation.close()
            write_json(args.output_dir / "summary.json", summary)
            if summary["status"] != "running":
                break
    finally:
        if model:
            model.get_env().close()
        cold_env.close()


if __name__ == "__main__":
    main()
