"""Matched PPO transfer trials using separately pretrained body features."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from .pretrained_vertical import PretrainedVerticalEnv, make_model, TRIALS, predictor_identity
from .body_predictor import load_predictor, load_dataset, indices, forecast_metrics
from .train_shared_vertical import PredictionLabels, prediction_metrics
from .train_vertical_architecture import parameter_hash
from .train_vertical_ppo import compact, write_json
from .vertical_control import save_policy
from .vertical_muscles import VerticalConfig
from .vertical_precision import (DENSE_CASES, PRECISION_VALIDATION, PRECISION_TEST,
                                PRECISION_REWARD, evaluate_precision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--root", type=Path, default=Path("runs/pretrained_vertical"))
    parser.add_argument("--predictor", type=Path, default=Path("runs/body_predictor/best.pt"))
    parser.add_argument("--body-data", type=Path, default=Path("runs/body_predictor"))
    parser.add_argument("--steps", type=int, default=131072)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        args.root.mkdir(parents=True, exist_ok=True)
        for filename in ("prediction_validation.npz", "prediction_validation.json"):
            destination = args.root/filename
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(Path("runs/shared_vertical")/filename, destination)
        metadata = json.loads((args.root/"prediction_validation.json").read_text())
        assert metadata["dataset_sha256"] == hashlib.sha256((args.root/"prediction_validation.npz").read_bytes()).hexdigest()
        return
    if args.trial is None or args.steps <= 0 or args.steps % 8192:
        parser.error("Choose trial and positive complete 8192-step budget")
    folder = args.root/args.trial
    folder.mkdir(parents=True, exist_ok=False)
    dataset_path = args.root/"prediction_validation.npz"
    dataset = dict(np.load(dataset_path))
    predictor, _, predictor_metadata = load_predictor(args.predictor)
    identity = predictor_identity(predictor_metadata)
    body_data, body_metadata = load_dataset(args.body_data)
    if body_metadata['dataset_sha256'] != identity['dataset_sha256']:
        raise ValueError('Predictor training dataset identity mismatch')
    body_validation = indices(body_metadata, 'validation')
    env = PretrainedVerticalEnv(identity, trial=args.trial)
    model = make_model(Monitor(env), args.trial, predictor, identity)
    training = {"trial": args.trial, "seed": 0, "transition_budget": args.steps,
        "learning_rate": 3e-4, "rollout_steps": 1024, "batch_size": 128, "epochs": 10,
        "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .001, "vf_coef": .5,
        "max_grad_norm": .5, "ppo_exact_kl_limit": .03, "predictor_kl_guard": None,
        "predictor_epochs": 10, "predictor_batch_size": 128, "predictor_rng_seed": 1701,
        "predictor_learning_rate": 3e-4, "reward": PRECISION_REWARD,
        "train_cases": DENSE_CASES, "validation_cases": PRECISION_VALIDATION, "test_cases": PRECISION_TEST,
        "episode_seconds": 4., "optimizer": "RMSprop", "optimizer_kwargs": {"alpha": .99, "eps": 1e-5},
        "normalization": "pretraining whole-train statistics, frozen", "teacher": None,
        "pretraining": identity, "predictor_optimizer_initialization": "fresh moments",
        "additional_pretraining_cost": "19200 unique train transitions; selected update44500 of50000; separate from PPO budget",
        "prediction_validation_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {k: version(k) for k in ("torch", "stable-baselines3", "mujoco", "numpy", "gymnasium")}}
    started = time.monotonic()
    summary = {"status": "running", "training": training, "history": [], "updates": [],
        "contract": env.contract(), "initial_parameter_sha256": parameter_hash(model),
        "parameter_count": sum(p.numel() for p in model.policy.parameters()),
        "predictor_parameter_count": sum(p.numel() for p in model.policy.prediction_parameters()),
        "initial_private_sha256": private_hash(model),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "stopping": "fixed budget; select by tracking validation only"}
    save_policy(model, env, folder/"initial.zip", training)
    best = None

    class Labels(PredictionLabels):
        def _on_rollout_end(self):
            if self.num_timesteps == 1024:
                b = self.model.rollout_buffer
                digest = hashlib.sha256()
                for array in (b.observations, b.actions, b.next_observations, b.prediction_valid):
                    digest.update(array.tobytes())
                summary["first_rollout_sha256"] = digest.hexdigest()

        def _on_rollout_start(self):
            if self.model.last_audit:
                summary["updates"].append({"steps": self.num_timesteps, **self.model.last_audit})

    def check():
        nonlocal best
        result = evaluate_precision(model, VerticalConfig(variant=4), PRECISION_VALIDATION)
        rank = (result["failures"], -result["passed"], result["score_m"])
        if best is None or rank < best:
            best = rank
            summary.update(best_steps=model.num_timesteps, best_validation=compact(result))
            save_policy(model, env, folder/"best.zip", training)
        summary["history"].append({"steps": model.num_timesteps, "validation": compact(result),
            "prediction_validation": prediction_metrics(model.policy, dataset),
            "body_validation": forecast_metrics(model.policy, body_data, body_validation),
            "accepted_optimizer_steps": model.accepted_steps, "rejected_optimizer_steps": model.rejected_steps,
            "predictor_steps": model.predictor_steps})
        summary["elapsed_s"] = time.monotonic()-started
        write_json(folder/"summary.json", summary)
        print(json.dumps({"trial": args.trial, "steps": model.num_timesteps,
            "validation_mm": result["score_m"]*1000, "passed": result["passed"], "best_steps": summary["best_steps"],
            "predictor_policy_kl": model.last_audit.get("predictor_phase_kl")}), flush=True)

    check()
    for _ in range(args.steps//8192):
        model.learn(8192, reset_num_timesteps=False, callback=Labels())
        check()
    summary["updates"].append({"steps": model.num_timesteps, **model.last_audit})
    save_policy(model, env, folder/"last.zip", training)
    summary.update(status="training_complete", steps=model.num_timesteps,
        accepted_optimizer_steps=model.accepted_steps, rejected_optimizer_steps=model.rejected_steps,
        predictor_steps=model.predictor_steps, elapsed_s=time.monotonic()-started)
    write_json(folder/"summary.json", summary)
    env.close()


def private_hash(model):
    excluded = {id(p) for p in model.policy.prediction_parameters()}
    digest = hashlib.sha256()
    for name, parameter in model.policy.named_parameters():
        if id(parameter) not in excluded:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
