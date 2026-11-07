"""Train PPO to execute issued acceleration tasks forty milliseconds later."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from .train_vertical_ppo import write_json
from .vertical_control import save_policy
from .vertical_muscles import VerticalConfig
from .vertical_scheduled import (
    ACCURACY, REWARD, TRAIN_CASES, VALIDATION_CASES,
    ScheduledAccelerationEnv, compact_scheduled, evaluate_scheduled,
)


def validation_rank(result):
    # One successful zero task must not hide progress on the four moving tasks.
    return (result["failures"], not result["ready"], result["score_m_s2"], -result["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=131072)
    parser.add_argument("--min-steps", type=int, default=32768)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if args.max_steps % 8192:
        parser.error("Use complete 8192-step blocks")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from .policy_beta import BetaExcitationPolicy
    torch.set_num_threads(1)
    started = time.monotonic()
    config = VerticalConfig(variant=args.variant)
    env = ScheduledAccelerationEnv(config)
    parameters = dict(learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
                      gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.001, vf_coef=.5,
                      max_grad_norm=.5, target_kl=.03)
    model = PPO(BetaExcitationPolicy, Monitor(env), seed=0, device="cpu", verbose=0,
                **parameters, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                            "activation_fn": torch.nn.Tanh})
    training = {"algorithm": "stable_baselines3.PPO", "seed": 0, "initialization": "fresh, no old checkpoint",
                "task": "acceleration issued 40ms before its target interval", "task_lead_s": .04,
                "train_cases": [asdict(c) for c in TRAIN_CASES], "validation_cases": [asdict(c) for c in VALIDATION_CASES],
                "network": {"pi": [128, 128], "vf": [128, 128], "activation": "Tanh", "shared_trainable_weights": False},
                **parameters, "reward": REWARD, "accuracy": ACCURACY, "teacher": None, "predictor": None,
                "gradient_through_physics": False,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {k: version(k) for k in ("torch", "stable-baselines3", "gymnasium", "mujoco", "numpy")}}
    training = json.loads(json.dumps(training))
    summary = {"schema": "myo.vertical_scheduled_trial.v1", "status": "running", "variant": args.variant,
               "training": training, "history": [], "audit": [], "observation_names": env.observation_names,
               "stopping": {"max_steps": args.max_steps, "min_steps": args.min_steps, "block_steps": 8192,
                            "patience": args.patience, "meaningful_improvement": "max(.01m/s2,2% anchor)",
                            "ready": "all five validation acceleration profiles passed twice"},
               "selection": "fewer physical failures, full readiness, lower mean RMSE, passed cases as tie break; includes initial"}

    def report():
        summary["elapsed_s"] = time.monotonic()-started
        write_json(args.output_dir/"summary.json", summary)

    class Audit(BaseCallback):
        def _on_rollout_start(self):
            self.rewards, self.errors = [], []

        def _on_step(self):
            info = self.locals["infos"][0]
            self.rewards.append(info["raw_reward"])
            self.errors.append(info["acceleration_error_m_s2"])
            return True

        def _on_rollout_end(self):
            buffer = self.model.rollout_buffer
            with torch.no_grad():
                dist = self.model.policy.get_distribution(torch.tensor(buffer.observations[:, 0])).distribution
            variance = float(np.var(buffer.returns))
            summary["audit"].append({"steps": self.num_timesteps,
                "raw_reward_mean": float(np.mean(self.rewards)),
                "rollout_acceleration_rmse_m_s2": float(np.sqrt(np.mean(np.square(self.errors)))),
                "critic_ev": float(1-np.var(buffer.returns-buffer.values)/variance) if variance > 1e-12 else None,
                "excitation_mean": dist.mean.mean(0).tolist(), "excitation_std": dist.stddev.mean(0).tolist(),
                "out_of_bounds_actions": int(((buffer.actions < 0) | (buffer.actions > 1)).sum())})

    def evaluate():
        result = evaluate_scheduled(model, config, VALIDATION_CASES)
        return result, validation_rank(result)

    result, best_key = evaluate()
    anchor, anchor_group, stale, confirmations = best_key[2], best_key[:2], 0, 0
    summary.update(initial_validation=compact_scheduled(result), best_steps=0)
    save_policy(model, env, args.output_dir/"initial.zip", training)
    save_policy(model, env, args.output_dir/"best.zip", training)
    write_json(args.output_dir/"best_validation.json", result)
    report()
    audit = Audit()
    for _ in range(args.max_steps//8192):
        model.learn(8192, reset_num_timesteps=False, callback=audit)
        result, key = evaluate()
        if key < best_key:
            best_key = key
            summary["best_steps"] = model.num_timesteps
            save_policy(model, env, args.output_dir/"best.zip", training)
            write_json(args.output_dir/"best_validation.json", result)
        save_policy(model, env, args.output_dir/"last.zip", training)
        meaningful = best_key[:2] < anchor_group or anchor-best_key[2] >= max(.01, .02*abs(anchor))
        if meaningful:
            anchor, anchor_group, stale = best_key[2], best_key[:2], 0
        else:
            stale += 1
        confirmations = confirmations+1 if result["ready"] else 0
        summary["history"].append({"steps": model.num_timesteps, "validation": compact_scheduled(result),
            "stale_blocks": stale,
            "ppo": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}})
        print(f"v{args.variant} {model.num_timesteps} val={result['score_m_s2']:.4f}m/s² "
              f"passed={result['passed']}/5 failures={result['failures']} stale={stale}", flush=True)
        report()
        if model.num_timesteps >= args.min_steps and (confirmations >= 2 or stale >= args.patience):
            summary["stop_reason"] = "validation_ready_twice" if confirmations >= 2 else "validation_plateau"
            break
    else:
        summary["stop_reason"] = "fixed_step_budget"
    summary.update(status="training_complete", trained_steps=model.num_timesteps,
                   selected_validation=compact_scheduled(json.loads((args.output_dir/"best_validation.json").read_text())),
                   selection_frozen_before_test={"steps": summary["best_steps"],
                       "checkpoint_sha256": json.loads((args.output_dir/"best.zip.metadata.json").read_text())["checkpoint_sha256"]})
    report()


if __name__ == "__main__":
    main()
