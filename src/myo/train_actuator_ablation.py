"""Compare actuator dynamics on the fixed first movement stage, headless/local."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from .actuator_ablation import (
    MODES, contract, descriptor, experiment_digest, load_ablation_controller, make_environment,
)
from .env import PreacherCurlConfig
from .movement_curriculum import MovementProtocol, curriculum_metadata, evaluate_movement
from .runtime import dynamics_digest, load_controller
from .train_bounded import RewardScale, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initial", type=Path, default=Path("runs/d1_architecture_small_tanh/initial.zip"))
    source.add_argument("--resume", type=Path)
    parser.add_argument("--max-rollouts", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args(argv)
    if args.max_rollouts < 2 or args.max_rollouts % 2 or args.patience < 1:
        parser.error("Use an even rollout budget >=2 and positive patience")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from .policy_beta import BetaExcitationPolicy

    torch.set_num_threads(1)
    started = time.perf_counter()
    source_path = args.resume or args.initial
    checked, _, parent = (load_ablation_controller(source_path) if args.resume
                          else load_controller(str(source_path)))
    checked.close()
    config = PreacherCurlConfig(**(parent["actuator_ablation"]["source_config"]
                                   if args.resume else parent["config"]))
    details = descriptor(args.mode, config)
    curriculum = json.loads(json.dumps(curriculum_metadata(config.payload_mass)))
    if args.resume:
        if parent["actuator_ablation"] != details or parent["movement_curriculum"] != curriculum:
            raise ValueError("Resume requires the same actuator and exercise contract")
    elif parent["trained_steps"] != 0:
        raise ValueError("Start from the untrained initial network, not learned motor weights")
    if (parent["reward_scale"] != .01 or parent["action_persistence"] != 0
            or parent["predictor_sha256"] is not None):
        raise ValueError("Comparison requires direct Beta, scale .01 and no predictor")
    model = PPO.load(source_path, device="cpu")
    if type(model.policy) is not BetaExcitationPolicy or model.num_timesteps != parent["trained_steps"]:
        raise ValueError("Policy type or saved training counter mismatch")
    model.set_random_seed(0)
    protocol = MovementProtocol(make_environment(args.mode, config), 0)
    evaluation = MovementProtocol(make_environment(args.mode, config), 0)
    env = Monitor(RewardScale(protocol))
    model.set_env(env, force_reset=True)
    physical_sha = dynamics_digest()
    initial_steps = model.num_timesteps
    metadata = {**parent, "config": asdict(protocol.unwrapped.config),
        "dynamics_sha256": experiment_digest(details), "actuator_ablation": details,
        "contract": contract(protocol.unwrapped, args.mode),
        "observation_names": protocol.unwrapped.observation_names,
        "movement_curriculum": curriculum, "movement_curriculum_stage": 0,
        "parent_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "training_stage": "actuator_ablation_raise_30",
        "training_protocol": {"name": "cold_movement_curriculum_v1", "stage": 0, "descriptor": curriculum},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resume_semantics": "Weights/Adam restored; fresh episode and seed0; new budget/plateau/pass counters",
        "limitations": "Diagnostic actuator; first stage only; one fixed seed; not a full demo or anatomical validation"}
    summary = {"status": "running", "mode": args.mode, "initial_steps": initial_steps,
        "source": str(source_path), "setup": metadata, "evaluations": [],
        "stopping": {"block_rollouts": 2, "min_rollouts": 8, "max_rollouts": args.max_rollouts,
                     "patience": args.patience, "meaningful_gain": "max(.25, .02*abs(anchor))",
                     "pass_confirmations": 2, "scope": "all six conditions; stage0 only"}}
    audit = []

    def save(name):
        path = args.output_dir / f"{name}.zip"
        model.save(path)
        write_json(str(path) + ".metadata.json", {**metadata,
            "trained_steps": model.num_timesteps, "protocol_completed_episodes": protocol.completed_episodes,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    def select(path):
        for suffix in (".zip", ".zip.metadata.json"):
            shutil.copy2(str(path).removesuffix(".zip") + suffix, args.output_dir / ("best" + suffix))

    def measure():
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            return evaluate_movement(model, evaluation)
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])

    class Audit(BaseCallback):
        def _on_rollout_start(self):
            self.samples, self.activations, self.delayed, self.forces, self.terminals = [], [], [], [], []

        def _on_step(self):
            info = self.locals["infos"][0]
            self.samples.append((info["q"], info["qvel"], info["requested_target_q"],
                sum(info["reward_components"].values()), info["reward_components"]["settled_bonus"],
                min(info["capacity"]), bool(self.locals["dones"][0]), bool(info.get("TimeLimit.truncated", False))))
            self.activations.append(info["activation"])
            self.delayed.append(info["delayed_excitation"])
            self.forces.append(info["tendon_force_N"])
            self.terminals.append(info.get("terminal_observation", np.zeros(model.observation_space.shape)))
            return True

        def _on_rollout_end(self):
            b = self.model.rollout_buffer
            samples = np.asarray(self.samples)
            with torch.no_grad():
                dist = model.policy.get_distribution(torch.as_tensor(b.observations[:, 0])).distribution
                mean, std = dist.mean.numpy(), dist.stddev.numpy()
            path = args.output_dir / f"rollout_{self.num_timesteps:09d}.npz"
            np.savez_compressed(path, observations=b.observations, actions=b.actions,
                log_probs=b.log_probs, buffer_rewards=b.rewards, values=b.values, returns=b.returns,
                advantages=b.advantages, episode_starts=b.episode_starts, physical_samples=samples,
                activations=np.asarray(self.activations), delayed_excitation=np.asarray(self.delayed),
                tendon_forces=np.asarray(self.forces), terminal_observations=np.asarray(self.terminals),
                last_observation=model._last_obs, last_values=self.locals["values"].detach().cpu().numpy(),
                last_dones=self.locals["dones"])
            row = {"steps": self.num_timesteps, "rollout_path": str(path),
                "elapsed_seconds": time.perf_counter() - started,
                "mean_raw_reward": float(samples[:, 3].mean()),
                "angle_max_deg": float(np.rad2deg(samples[:, 0].max())),
                "fraction_above_25deg": float(np.mean(samples[:, 0] > np.deg2rad(25))),
                "settled_bonus_mean": float(samples[:, 4].mean()),
                "minimum_capacity": float(samples[:, 5].min()),
                "excitation_means": mean.mean(0).tolist(), "median_std": float(np.median(std))}
            audit.append(row)
            write_json(args.output_dir / "audit.json", audit)
            print(json.dumps({"rollout": row}), flush=True)

    try:
        path = save("initial")
        baseline = measure()
        write_json(args.output_dir / "initial_evaluation.json", baseline)
        select(path)
        summary.update(initial_score=baseline["score"], best_score=baseline["score"],
            best_ready=baseline["ready"], best_passed=baseline["passed"], best_controller=str(path))
        anchor, stale, confirmations = baseline["score"], 0, 0
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"baseline": {k: baseline[k] for k in ("score", "passed", "total")}}), flush=True)
        for block in range(1, args.max_rollouts // 2 + 1):
            model.learn(2 * model.n_steps, callback=Audit(), reset_num_timesteps=False)
            path = save(f"block_{block:03d}")
            metrics = measure()
            write_json(args.output_dir / f"evaluation_{block:03d}.json", metrics)
            row = {"block": block, "steps": model.num_timesteps,
                "stage_steps": model.num_timesteps - initial_steps, "controller": str(path),
                **{k: metrics[k] for k in ("score", "passed", "total", "ready")},
                "optimizer": {k: float(v) for k, v in model.logger.name_to_value.items()
                              if k.startswith("train/") and np.isscalar(v)}}
            summary["evaluations"].append(row)
            confirmations = confirmations + 1 if metrics["ready"] else 0
            if (not metrics["ready"], metrics["score"]) < (not summary["best_ready"], summary["best_score"]):
                select(path)
                summary.update(best_score=metrics["score"], best_ready=metrics["ready"],
                    best_passed=metrics["passed"], best_controller=str(path))
            if anchor - metrics["score"] >= max(.25, .02 * abs(anchor)):
                anchor, stale = metrics["score"], 0
            else:
                stale += 1
            if confirmations >= 2:
                summary["status"] = "stage_ready"
            elif block * 2 >= 8 and stale >= args.patience:
                summary["status"] = "stage_plateau"
            elif block * 2 == args.max_rollouts:
                summary["status"] = "stage_budget_exhausted"
            summary.update(trained_steps=model.num_timesteps, stale_blocks=stale,
                pass_confirmations=confirmations, last_controller=str(path),
                elapsed_seconds=time.perf_counter() - started)
            write_json(args.output_dir / "summary.json", summary)
            print(json.dumps({**{k: row[k] for k in ("block", "steps", "score", "passed", "total")},
                              "status": summary["status"], "stale_blocks": stale}), flush=True)
            if summary["status"] != "running":
                break
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(trained_steps=model.num_timesteps, elapsed_seconds=time.perf_counter() - started,
                       base_dynamics_unchanged=dynamics_digest() == physical_sha)
        write_json(args.output_dir / "summary.json", summary)
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
