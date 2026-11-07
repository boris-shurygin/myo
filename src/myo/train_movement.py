"""Train direct muscle excitation through fixed, gated movement stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from .compact_protocol import compact_scenarios
from .continue_compact import behavior_summary
from .env import PreacherCurlConfig, PreacherCurlEnv
from .evaluate import run_scenario
from .movement_curriculum import (
    STAGE_NAMES, MovementProtocol, curriculum_metadata, evaluate_movement,
)
from .runtime import dynamics_digest, load_controller
from .train_bounded import RewardScale, write_json


def validate_source(parent, descriptor, resume):
    if (parent.get("reward_scale") != .01 or parent.get("action_persistence") != 0
            or parent.get("bounded_distribution") != "beta_alpha_beta_softplus_floor_0.05"
            or parent.get("predictor_sha256") is not None):
        raise ValueError("Movement curriculum requires direct Beta, reward scale .01 and no predictor")
    if resume:
        if parent.get("movement_curriculum") != json.loads(json.dumps(descriptor)):
            raise ValueError("Movement curriculum differs from the saved training contract")
        stage = parent.get("movement_curriculum_stage")
        if type(stage) is not int or not 0 <= stage < len(STAGE_NAMES):
            raise ValueError("Movement continuation requires a valid saved curriculum stage")
    elif parent.get("trained_steps") != 0:
        raise ValueError("Start with an untrained direct-Beta initial checkpoint, or use --resume")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--controller", type=Path,
                        default=Path("runs/d1_architecture_small_tanh/initial.zip"))
    source.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-rollouts", type=int, default=2)
    parser.add_argument("--min-stage-rollouts", type=int, default=8)
    parser.add_argument("--max-stage-rollouts", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args(argv)
    if min(args.block_rollouts, args.min_stage_rollouts, args.max_stage_rollouts, args.patience) < 1:
        parser.error("Budgets and patience must be positive")
    if args.min_stage_rollouts > args.max_stage_rollouts:
        parser.error("Minimum stage budget exceeds maximum")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from .policy_beta import BetaExcitationPolicy

    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint_source = args.resume or args.controller
    checked, _, parent = load_controller(str(checkpoint_source))
    checked.close()
    config = PreacherCurlConfig(**parent["config"])
    descriptor = curriculum_metadata(config.payload_mass)
    validate_source(parent, descriptor, bool(args.resume))
    model = PPO.load(checkpoint_source, device="cpu")
    if type(model.policy) is not BetaExcitationPolicy:
        raise ValueError("Movement stages preserve direct excitation, not action persistence")
    if model.num_timesteps != parent["trained_steps"]:
        raise ValueError("Checkpoint step counter differs from metadata")
    model.set_random_seed(0)
    physical_sha = dynamics_digest()
    metadata = {**parent, "movement_curriculum": descriptor,
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "training_stage": "movement_result_curriculum",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resume_semantics": "Weights/Adam restored; fresh physical episode and seed 0; live state continues within each stage. Resume restarts the saved stage with a new rollout budget, plateau history and pass-confirmation count",
        "limitations": "Intermediate stages do not establish original compact task success or protective release"}
    initial_steps = model.num_timesteps
    first_stage = parent.get("movement_curriculum_stage", 0) if args.resume else 0
    summary = {"status": "running", "parent": str(checkpoint_source),
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "curriculum": descriptor, "stages": [], "initial_steps": initial_steps,
        "stopping": {"block_rollouts": args.block_rollouts,
            "min_stage_rollouts": args.min_stage_rollouts, "max_stage_rollouts": args.max_stage_rollouts,
            "patience_blocks": args.patience, "meaningful_gain": "max(.25, .02*abs(anchor))",
            "pass_confirmations": 2, "all_cases_required": True},
        "resume_semantics": metadata["resume_semantics"]}
    cold_env = PreacherCurlEnv(config)
    audit_rows = []
    current_stage = first_stage

    def save(name, stage, protocol):
        path = args.output_dir / f"{name}.zip"
        model.save(path)
        write_json(str(path) + ".metadata.json", {**metadata,
            "trained_steps": model.num_timesteps,
            "movement_curriculum_stage": stage,
            "training_protocol": {"name": descriptor["name"], "stage": stage,
                                  "descriptor": descriptor},
            "protocol_completed_episodes": protocol.completed_episodes,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    def select(path, name):
        for suffix in (".zip", ".zip.metadata.json"):
            shutil.copy2(str(path).removesuffix(".zip") + suffix, args.output_dir / (name + suffix))

    def guarded_evaluate(evaluation, compact=False):
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            if compact:
                rows = [run_scenario(cold_env,
                    lambda obs: model.predict(obs, deterministic=True)[0], case, 0,
                    continue_after_time_limit=True, measure_commands=True)
                    for case in compact_scenarios(config.payload_mass)]
                metrics = behavior_summary(rows)
                return {**metrics, "ready": metrics["demo_ready"],
                    "passed": metrics["moves_passed"] + metrics["sweeps_passed"],
                    "total": 18, "scenarios": rows}
            return evaluate_movement(model, evaluation)
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])

    class Audit(BaseCallback):
        def _on_rollout_start(self):
            self.samples = []

        def _on_step(self):
            info = self.locals["infos"][0]
            self.samples.append((info["q"], info["qvel"], info["requested_target_q"],
                sum(info["reward_components"].values()), info["reward_components"]["settled_bonus"],
                min(info["capacity"]), bool(self.locals["dones"][0]),
                bool(info.get("TimeLimit.truncated", False))))
            return True

        def _on_rollout_end(self):
            b = self.model.rollout_buffer
            samples = np.asarray(self.samples)
            with torch.no_grad():
                obs = torch.as_tensor(b.observations.reshape(-1, *b.obs_shape))
                dist = self.model.policy.get_distribution(obs).distribution
                mean, std = dist.mean.numpy(), dist.stddev.numpy()
            variance = float(np.var(b.returns))
            row = {"stage": current_stage, "steps": self.num_timesteps,
                "elapsed_seconds": time.perf_counter() - started,
                "mean_raw_reward": float(samples[:, 3].mean()),
                "angle_max_deg": float(np.rad2deg(samples[:, 0].max())),
                "fraction_above_25deg": float(np.mean(samples[:, 0] > np.deg2rad(25))),
                "settled_bonus_mean": float(samples[:, 4].mean()),
                "minimum_capacity": float(samples[:, 5].min()),
                "excitation_means": mean.mean(0).tolist(), "median_std": float(np.median(std)),
                "critic_ev_bootstrapped_target": float(1 - np.var(b.returns - b.values) / variance) if variance > 1e-12 else None,
                "actions_outside_bounds": int(((b.actions < 0) | (b.actions > 1)).sum())}
            path = args.output_dir / f"rollout_{self.num_timesteps:09d}.npz"
            np.savez_compressed(path, observations=b.observations, actions=b.actions,
                log_probs=b.log_probs, buffer_rewards=b.rewards, values=b.values,
                returns=b.returns, advantages=b.advantages, episode_starts=b.episode_starts,
                physical_samples=samples, last_observation=self.model._last_obs,
                last_values=self.locals["values"].detach().cpu().numpy(), last_dones=self.locals["dones"])
            row["rollout_path"] = str(path)
            row["physical_sample_columns"] = ["q_rad", "qvel_rad_s", "target_rad", "raw_reward",
                                               "settled_bonus", "minimum_capacity", "done", "timeout"]
            audit_rows.append(row)
            write_json(args.output_dir / "audit.json", audit_rows)
            print(json.dumps({"rollout": row}), flush=True)

    try:
        for stage in range(first_stage, len(STAGE_NAMES)):
            current_stage = stage
            protocol = MovementProtocol(PreacherCurlEnv(config), stage)
            env = Monitor(RewardScale(protocol))
            evaluation = MovementProtocol(PreacherCurlEnv(config), stage)
            if model.get_env() is not None:
                model.get_env().close()
            model.set_env(env, force_reset=True)
            stage_start = model.num_timesteps
            record = {"stage": stage, "name": STAGE_NAMES[stage], "status": "running", "evaluations": []}
            summary["stages"].append(record)
            try:
                path = save(f"stage_{stage}_initial", stage, protocol)
                baseline = guarded_evaluate(evaluation, stage == len(STAGE_NAMES) - 1)
                write_json(args.output_dir / f"stage_{stage}_initial_evaluation.json", baseline)
                record.update(initial_score=baseline["score"], best_score=baseline["score"],
                              best_controller=str(path), best_passed=baseline["passed"], best_ready=baseline["ready"])
                select(path, f"stage_{stage}_best")
                anchor, stale, confirmations = baseline["score"], 0, 0
                write_json(args.output_dir / "summary.json", summary)
                print(json.dumps({"stage": stage, "baseline": {k: baseline[k] for k in ("score", "passed", "total")}}), flush=True)
                block = 0
                while True:
                    block += 1
                    used = (model.num_timesteps - stage_start) // model.n_steps
                    rounds = min(args.block_rollouts, args.max_stage_rollouts - used)
                    if rounds <= 0:
                        record["status"] = summary["status"] = "stage_budget_exhausted"
                        break
                    training_start = time.perf_counter()
                    model.learn(rounds * model.n_steps, callback=Audit(), reset_num_timesteps=False)
                    training_seconds = time.perf_counter() - training_start
                    path = save(f"stage_{stage}_block_{block:03d}", stage, protocol)
                    metrics = guarded_evaluate(evaluation, stage == len(STAGE_NAMES) - 1)
                    write_json(args.output_dir / f"stage_{stage}_block_{block:03d}_evaluation.json", metrics)
                    row = {"block": block, "steps": model.num_timesteps,
                        "stage_steps": model.num_timesteps - stage_start, "controller": str(path),
                        "training_seconds": training_seconds, "score": metrics["score"],
                        "passed": metrics["passed"], "total": metrics["total"], "ready": metrics["ready"],
                        "optimizer": {k: float(v) for k, v in model.logger.name_to_value.items()
                                      if k.startswith("train/") and np.isscalar(v)}}
                    record["evaluations"].append(row)
                    confirmations = confirmations + 1 if metrics["ready"] else 0
                    if (not metrics["ready"], metrics["score"]) < (not record["best_ready"], record["best_score"]):
                        record.update(best_score=metrics["score"], best_controller=str(path),
                                      best_passed=metrics["passed"], best_ready=metrics["ready"])
                        select(path, f"stage_{stage}_best")
                    if anchor - metrics["score"] >= max(.25, .02 * abs(anchor)):
                        anchor, stale = metrics["score"], 0
                    else:
                        stale += 1
                    record.update(stale_blocks=stale, pass_confirmations=confirmations,
                                  last_controller=str(path))
                    used = (model.num_timesteps - stage_start) // model.n_steps
                    if confirmations >= 2:
                        record["status"] = "passed"
                        select(path, f"stage_{stage}_accepted")
                        if stage == len(STAGE_NAMES) - 1:
                            summary["status"] = "compact_ready"
                            select(path, "best")
                    elif used >= args.min_stage_rollouts and stale >= args.patience:
                        record["status"] = summary["status"] = "stage_plateau"
                    elif used >= args.max_stage_rollouts:
                        record["status"] = summary["status"] = "stage_budget_exhausted"
                    summary.update(trained_steps=model.num_timesteps,
                                   elapsed_seconds=time.perf_counter() - started)
                    write_json(args.output_dir / "summary.json", summary)
                    print(json.dumps({"stage": stage, **{k: row[k] for k in ("block", "steps", "score", "passed", "total")},
                                      "stale_blocks": stale, "status": record["status"]}), flush=True)
                    if record["status"] != "running":
                        break
            finally:
                env.close()
                evaluation.close()
            if summary["status"] != "running":
                break
        summary["dynamics_unchanged"] = dynamics_digest() == physical_sha
        assert summary["dynamics_unchanged"]
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "trained_steps": model.num_timesteps,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    except KeyboardInterrupt:
        summary.update(status="interrupted", trained_steps=model.num_timesteps,
                       elapsed_seconds=time.perf_counter() - started)
        write_json(args.output_dir / "summary.json", summary)
        raise
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}",
                       trained_steps=model.num_timesteps, elapsed_seconds=time.perf_counter() - started)
        write_json(args.output_dir / "summary.json", summary)
        raise
    finally:
        if model.get_env() is not None:
            model.get_env().close()
        cold_env.close()


if __name__ == "__main__":
    main()
