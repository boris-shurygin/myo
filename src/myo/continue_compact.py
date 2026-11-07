"""Continue the fixed PPO experiment until useful behavior or measured plateau."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np

from .compact_protocol import CompactProtocol, compact_scenarios, protocol_metadata
from .env import PreacherCurlConfig, PreacherCurlEnv
from .evaluate import run_scenario
from .runtime import load_controller


def behavior_summary(results: list[dict]) -> dict:
    """Continuous ranking; separate explicit gates decide demo readiness."""
    losses, moves, sweeps = [], [], []
    failures = 0
    max_contact = 0.0
    for result in results:
        failed = (result["physical_failure"] or result["numerical_failure"]
                  or not result["completed_duration"])
        failures += bool(failed)
        peaks = result["overall"]["substep_peaks"]
        max_contact = max(max_contact, peaks.get("tissue_contact_force_N", 0.0))
        sweep = result["scenario"]["name"].startswith("sweep_")
        rows = ([r for r in result["command_responses"] if 25 <= r["target_deg"] <= 135]
                if sweep else result["command_responses"][1:])
        errors = [abs(r.get("final_error_deg", 180.0)) for r in rows]
        rmses = [r.get("requested_angle_rmse_deg", 180.0) for r in rows]
        loss = 0.5 * (float(np.mean(errors)) + float(np.mean(rmses))) if rows else 180.0
        losses.append(loss + (180.0 if failed else 0.0))
        if sweep:
            fraction = float(np.mean(np.asarray(errors) <= 5)) if errors else 0.0
            sweeps.append({"name": result["scenario"]["name"], "fraction_within_5_deg": fraction,
                           "mean_command_rmse_deg": float(np.mean(rmses)) if rmses else 180.0,
                           "passed": not failed and fraction >= 0.8 and float(np.mean(rmses)) <= 8})
        else:
            row = rows[-1] if rows else {}
            settling = row.get("settling_time_s")
            tail = result.get("hold_tail_last_5s") or {}
            passed = (not failed and settling is not None and settling <= 3
                      and abs(row.get("final_error_deg", 180)) <= 3
                      and row.get("overshoot_deg", 180) <= 10
                      and tail.get("angle_std_deg", 180) <= 2
                      and peaks.get("joint_limit_reaction_Nm", float("inf")) <= 20)
            moves.append({"name": result["scenario"]["name"], "passed": bool(passed),
                          "final_angle_deg": result["final_angle_deg"],
                          "settling_time_s": settling, "loss_deg": loss})
    score = float(np.mean(losses)) + 0.01 * max(0.0, max_contact - 250)
    return {"score": score, "physical_or_numerical_failures": failures,
            "moves_passed": sum(m["passed"] for m in moves), "moves_total": len(moves),
            "sweeps_passed": sum(s["passed"] for s in sweeps), "sweeps_total": len(sweeps),
            "contact_force_peak_N": max_contact, "moves": moves, "sweeps": sweeps,
            "demo_ready": len(moves) == 12 and len(sweeps) == 6 and failures == 0
                and all(m["passed"] for m in moves) and all(s["passed"] for s in sweeps)
                and max_contact <= 250}


class Plateau:
    def __init__(self, score: float, patience: int = 3):
        self.anchor = score
        self.patience = patience
        self.stale = 0

    def update(self, score: float) -> bool:
        # Small improvements accumulate against an anchor, rather than being
        # discarded every block or resetting patience on numerical noise.
        if self.anchor - score >= max(0.5, 0.02 * abs(self.anchor)):
            self.anchor, self.stale = score, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-cycles", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args(argv)
    if args.block_cycles < 1 or args.patience < 1:
        parser.error("Block cycles and patience must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory; existing experiments are preserved")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    checked, _, parent = load_controller(str(args.controller))
    checked.close()
    config = PreacherCurlConfig(**parent["config"])
    # Metadata is JSON: normalize tuples before comparing the saved contract.
    protocol = json.loads(json.dumps(protocol_metadata(config.payload_mass)))
    if parent.get("training_protocol") != protocol or parent.get("predictor_sha256"):
        raise ValueError("Continuation requires the identical compact protocol, without a predictor")
    if parent.get("reward_scale", 1.0) != 1.0:
        raise ValueError("This runner cannot restore the checkpoint's reward scale; use its experiment runner")
    seed = parent["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env = CompactProtocol(PreacherCurlEnv(config))
    monitored = Monitor(train_env)
    evaluation_env = PreacherCurlEnv(config)
    # PPO.load restores both actor/critic and Adam's moments. Start a fresh
    # physical episode; a ZIP is not a simulator/RNG snapshot of the last step.
    model = PPO.load(args.controller, env=monitored, device="cpu")
    initial_steps = model.num_timesteps
    block_steps = round(sum(s.duration_s for s in train_env.scenarios) / config.control_dt) * args.block_cycles
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"parent": str(args.controller), "parent_sha256": parent["checkpoint_sha256"],
               "seed": seed, "protocol": protocol, "block_requested_steps": block_steps,
               "stopping": {"patience_blocks": args.patience, "meaningful_gain": "max(0.5, 2% of anchor score)",
                            "score": "mean per-exercise mean of command RMSE and final absolute error; +180 per failed exercise; +0.01 per peak contact N above 250",
                            "sweep_scored_targets_deg": [25, 135],
                            "move_gate": "all 12: settle<=3s, final error<=3deg, overshoot<=10deg, tail std<=2deg, joint reaction<=20Nm",
                            "sweep_gate": "each of 6: >=80% endpoint errors<=5deg; mean command RMSE<=8deg",
                            "contact_gate_N": 250,
                            "limits_are": "engineering demo criteria, not medical safety thresholds"},
               "resume_semantics": "weights, optimizer and timestep counter restored; fresh physical episode and saved seed; blocks preserve live training state and RNG",
               "status": "running", "evaluations": []}

    def write_summary():
        path = args.output_dir / "summary.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temp.replace(path)

    def evaluate(block, checkpoint):
        # Evaluation must not consume the training random stream.
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        try:
            results = [run_scenario(evaluation_env,
                lambda obs: model.predict(obs, deterministic=True)[0], s, seed,
                continue_after_time_limit=True, measure_commands=True)
                for s in compact_scenarios(config.payload_mass)]
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        metrics = behavior_summary(results)
        report = {"controller": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "trained_steps": model.num_timesteps, "seed": seed,
                  "config": asdict(config), "protocol": protocol, "metrics": metrics, "scenarios": results}
        path = args.output_dir / f"evaluation_{block:03d}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        row = {"block": block, "trained_steps": model.num_timesteps, "controller": str(checkpoint),
               "report": str(path), **metrics}
        summary["evaluations"].append(row)
        print(json.dumps({k: row[k] for k in ("block", "trained_steps", "score", "moves_passed", "sweeps_passed", "demo_ready")}), flush=True)
        return metrics

    def select(checkpoint, score):
        shutil.copy2(checkpoint, args.output_dir / "best.zip")
        shutil.copy2(Path(str(checkpoint) + ".metadata.json"), args.output_dir / "best.zip.metadata.json")
        summary["best_controller"] = str(checkpoint)
        summary["best_score"] = score

    try:
        baseline = evaluate(0, args.controller)
        select(args.controller, baseline["score"])
        plateau = Plateau(baseline["score"], args.patience)
        write_summary()
        block, last_hash = 0, parent["checkpoint_sha256"]
        while not baseline["demo_ready"]:
            block += 1
            print(json.dumps({"training_block": block, "additional_steps": block_steps}), flush=True)
            model.learn(total_timesteps=block_steps, reset_num_timesteps=False)
            checkpoint = args.output_dir / f"block_{block:03d}.zip"
            model.save(checkpoint)
            metadata = {**parent, "training_stage": "ppo", "trained_steps": model.num_timesteps,
                        "requested_training_steps": block_steps,
                        "continuation_steps": model.num_timesteps - initial_steps,
                        "parent_checkpoint_sha256": last_hash,
                        "protocol_completed_episodes": train_env.completed_episodes,
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "continuation": summary["resume_semantics"]}
            Path(str(checkpoint) + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            last_hash = metadata["checkpoint_sha256"]
            metrics = evaluate(block, checkpoint)
            if metrics["score"] < summary["best_score"] or metrics["demo_ready"]:
                select(checkpoint, metrics["score"])
            stalled = plateau.update(metrics["score"])
            summary["stale_blocks"] = plateau.stale
            if metrics["demo_ready"] or stalled:
                summary["status"] = "demo_ready" if metrics["demo_ready"] else "plateau"
                write_summary()
                break
            write_summary()
        else:
            summary["status"] = "demo_ready"
            write_summary()
        print(json.dumps({"status": summary["status"], "best_controller": summary["best_controller"],
                          "best_score": summary["best_score"]}), flush=True)
    finally:
        monitored.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
