"""Train PPO from physical movement rewards on the four vertical mechanics."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from .vertical_control import (
    REWARD, TRAIN_CASES, VALIDATION_CASES, TEST_CASES, VerticalControlEnv,
    evaluate_policy, load_policy, save_policy,
)
from .vertical_muscles import VerticalConfig


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compact(result):
    return {**result, "cases": [{k: v for k, v in case.items() if k != "samples"} for case in result["cases"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vertical_ppo"))
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=262144)
    parser.add_argument("--block-steps", type=int, default=8192)
    parser.add_argument("--min-steps", type=int, default=32768)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.block_steps % 1024 or args.max_steps % args.block_steps:
        parser.error("Use complete 1024-step rollouts and complete blocks")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from .policy_beta import BetaExcitationPolicy
    torch.set_num_threads(1)
    started = time.monotonic()
    training = {
        "algorithm": "stable_baselines3.PPO", "seed": 0, "episode_seconds": 4.,
        "train_cases": TRAIN_CASES, "validation_cases": VALIDATION_CASES,
        "network": {"pi": [128, 128], "vf": [128, 128], "activation": "Tanh", "shared_trainable_weights": False},
        "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 128, "n_epochs": 10,
        "gamma": .99, "gae_lambda": .95, "clip_range": .2, "ent_coef": .001,
        "vf_coef": .5, "max_grad_norm": .5, "target_kl": .03,
        "reward": REWARD, "teacher": None, "predictor": None,
        "versions": {k: version(k) for k in ("torch", "stable-baselines3", "gymnasium", "mujoco", "numpy")},
    }
    # Normalize tuples to the same JSON representation used by metadata.
    training = json.loads(json.dumps(training))
    summary = {"schema": "myo.vertical_ppo_experiment.v1", "status": "running", "training": training,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "stopping": {"maximum_additional_steps": args.max_steps, "block_steps": args.block_steps,
                            "min_steps": args.min_steps, "patience": args.patience,
                            "meaningful_improvement": "max(0.0005m, 2% of anchor score)",
                            "ready": "all validation cases settled >=90% of last second within 5mm and 0.03m/s; twice"},
               "selection": "lexicographic: fewer physical failures, more passed cases, smaller mean steady position RMSE",
               "variants": {}}
    if args.resume:
        parent = json.loads(Path(str(args.resume)+".metadata.json").read_text())
        if parent["training"] != training:
            raise ValueError("Resume requires identical PPO parameters, cases and physical task")
        args.variants = [parent["contract"]["physical"]["config"]["variant"]]
        summary["resume"] = {"checkpoint": str(args.resume), "checkpoint_sha256": parent["checkpoint_sha256"],
                             "semantics": "Weights and Adam restored; fresh physical episode, case cycle and seed 0"}

    for variant in args.variants:
        folder = args.output_dir/f"variant_{variant}"
        folder.mkdir()
        config = VerticalConfig(variant=variant)
        env = VerticalControlEnv(config)
        monitor = Monitor(env)
        if args.resume:
            model, _ = load_policy(args.resume, env)
            model.set_env(monitor)
            model.set_random_seed(0)
        else:
            model = PPO(BetaExcitationPolicy, monitor, seed=0, device="cpu", verbose=0,
                        learning_rate=training["learning_rate"], n_steps=1024, batch_size=128, n_epochs=10,
                        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.001, vf_coef=.5,
                        max_grad_norm=.5, target_kl=.03,
                        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                       "activation_fn": torch.nn.Tanh})
        initial_steps = model.num_timesteps
        record = {"status": "running", "initial_steps": initial_steps, "history": [], "audit": [],
                  "parameters": sum(p.numel() for p in model.policy.parameters()),
                  "observation_names": env.observation_names}
        summary["variants"][str(variant)] = record

        class Audit(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                buffer = self.model.rollout_buffer
                with torch.no_grad():
                    dist = self.model.policy.get_distribution(torch.tensor(buffer.observations[:, 0])).distribution
                variance = float(np.var(buffer.returns))
                record["audit"].append({
                    "steps": self.num_timesteps, "reward_mean": float(buffer.rewards.mean()),
                    "critic_ev": float(1-np.var(buffer.returns-buffer.values)/variance) if variance > 1e-12 else None,
                    "value_mean": float(buffer.values.mean()), "return_mean": float(buffer.returns.mean()),
                    "excitation_mean": dist.mean.mean(0).tolist(), "excitation_std": dist.stddev.mean(0).tolist(),
                    "out_of_bounds_actions": int(((buffer.actions < 0) | (buffer.actions > 1)).sum()),
                })

        audit = Audit()
        best, anchor, stale, confirmations = float("inf"), float("inf"), 0, 0
        initial = evaluate_policy(model, config, VALIDATION_CASES)
        record["initial_validation"] = compact(initial)
        save_policy(model, env, folder/"initial.zip", training)
        save_policy(model, env, folder/"best.zip", training)
        best = anchor = initial["score_m"]
        best_key = (initial["failures"], -initial["passed"], best)
        anchor_group = best_key[:2]
        record["best_steps"] = model.num_timesteps
        write_json(folder/"best_validation.json", initial)
        write_json(args.output_dir/"summary.json", summary)
        for block in range(args.max_steps//args.block_steps):
            model.learn(args.block_steps, reset_num_timesteps=False, callback=audit)
            result = evaluate_policy(model, config, VALIDATION_CASES)
            key = (result["failures"], -result["passed"], result["score_m"])
            if key < best_key:
                best_key = key
                best = result["score_m"]
                record["best_steps"] = model.num_timesteps
                save_policy(model, env, folder/"best.zip", training)
                write_json(folder/"best_validation.json", result)
            save_policy(model, env, folder/"last.zip", training)
            meaningful = best_key[:2] < anchor_group or anchor-best >= max(.0005, .02*abs(anchor))
            if meaningful:
                anchor, stale, anchor_group = best, 0, best_key[:2]
            else:
                stale += 1
            confirmations = confirmations+1 if result["ready"] else 0
            record["history"].append({"steps": model.num_timesteps, "validation": compact(result),
                "best_score_m": best, "stale_blocks": stale,
                "ppo": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}})
            print(f"v{variant} steps={model.num_timesteps} val={result['score_m']*1000:.3f}mm "
                  f"passed={result['passed']}/{result['total']} failures={result['failures']} "
                  f"best={best*1000:.3f}mm stale={stale}", flush=True)
            write_json(args.output_dir/"summary.json", summary)
            enough = model.num_timesteps-initial_steps >= args.min_steps
            if enough and (confirmations >= 2 or stale >= args.patience):
                record["stop_reason"] = "validation_ready_twice" if confirmations >= 2 else "validation_plateau"
                break
        else:
            record["stop_reason"] = "fixed_step_budget"
        record["trained_steps"] = model.num_timesteps
        selected, _ = load_policy(folder/"best.zip", env)
        record["selected_validation"] = compact(evaluate_policy(selected, config, VALIDATION_CASES))
        record["selection_frozen_before_test"] = {"steps": record["best_steps"], "score_m": best}
        test = evaluate_policy(selected, config, TEST_CASES)
        switches = evaluate_policy(selected, config, ((0., .06),), seconds=6., switches=True)
        record["test"], record["switches"] = compact(test), compact(switches)
        record["status"] = "complete"
        write_json(folder/"test.json", test)
        write_json(folder/"switches.json", switches)
        monitor.close()
        write_json(args.output_dir/"summary.json", summary)
    summary.update(status="complete", elapsed_s=time.monotonic()-started)
    write_json(args.output_dir/"summary.json", summary)


if __name__ == "__main__":
    main()
