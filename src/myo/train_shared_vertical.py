"""Controlled shared-encoder PPO trials on the unchanged precision task."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .shared_vertical import SharedVerticalEnv, SharedPolicy, make_model
from .train_vertical_architecture import parameter_hash
from .train_vertical_ppo import compact, write_json
from .vertical_control import load_policy, save_policy
from .vertical_muscles import VerticalConfig
from .vertical_precision import (VerticalPrecisionEnv, PRECISION_REWARD, DENSE_CASES,
                                PRECISION_VALIDATION, PRECISION_TEST, evaluate_precision)


class PredictionLabels(BaseCallback):
    def _on_step(self):
        self.model.rollout_buffer.record_next(self.locals["new_obs"], self.locals["infos"])
        return True


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root/"prediction_validation.npz"
    if path.exists():
        raise FileExistsError(path)
    checkpoint = Path("runs/vertical_precision/combined/best.zip")
    model, metadata = load_policy(checkpoint, VerticalPrecisionEnv(VerticalConfig(variant=4)))
    observations, actions, next_observations = [], [], []
    for initial, target in PRECISION_VALIDATION:
        env = VerticalPrecisionEnv(VerticalConfig(variant=4))
        obs, _ = env.reset(options={"initial_position_m": initial, "target_m": target})
        o, a, n = [], [], []
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            following, _, terminated, _, _ = env.step(action)
            assert not terminated
            o.append(obs); a.append(action); n.append(following)
            obs = following
        observations.append(o); actions.append(a); next_observations.append(n)
        env.close()
    np.savez_compressed(path, observations=np.array(observations), actions=np.array(actions),
                        next_observations=np.array(next_observations))
    write_json(root/"prediction_validation.json", {"cases": PRECISION_VALIDATION,
        "source_policy_sha256": metadata["checkpoint_sha256"],
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "role": "fixed validation episodes only; never optimizer examples or normalization data"})


def prediction_metrics(policy, dataset):
    if not isinstance(policy, SharedPolicy) or not bool(policy.delta_calibrated):
        return None
    o, actions, labels = (dataset[k] for k in ("observations", "actions", "next_observations"))
    result = {}
    with torch.no_grad():
        for horizon in (1, 3):
            count = o.shape[1]-horizon+1
            current = torch.as_tensor(o[:, :count].reshape(-1, 13))
            for step in range(horizon):
                command = torch.as_tensor(actions[:, step:step+count].reshape(-1, 2))
                current = policy.next_observation(current, command)
            actual = torch.as_tensor(labels[:, horizon-1:horizon-1+count].reshape(-1, 13))
            residual = current[:, :6]-actual[:, :6]
            rms = residual.square().mean(0).sqrt().cpu().numpy()
            physical_scales = np.array([.1, .5, .1, .1, .5, .5])
            result[str(horizon*20)] = {"samples": len(current),
                "channel_rmse_physical": (rms*physical_scales).tolist(),
                "position_rmse_mm": float(rms[0]*100), "velocity_rmse_mm_s": float(rms[1]*500),
                "normalized_mse": float((residual/policy.delta_std).square().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--trial", choices=("independent", "shared", "shared_prediction"))
    parser.add_argument("--root", type=Path, default=Path("runs/shared_vertical"))
    parser.add_argument("--steps", type=int, default=131072)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare(args.root)
        return
    if args.trial is None or args.steps % 8192:
        parser.error("Choose trial and a complete 8192-step budget")
    folder = args.root/args.trial
    folder.mkdir(parents=True, exist_ok=False)
    dataset_path = args.root/"prediction_validation.npz"
    dataset = dict(np.load(dataset_path))
    env = SharedVerticalEnv(trial=args.trial)
    model = make_model(Monitor(env), args.trial)
    training = {"trial": args.trial, "seed": 0, "transition_budget": args.steps,
        "learning_rate": 3e-4, "rollout_steps": 1024, "batch_size": 128, "epochs": 10,
        "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .001,
        "vf_coef": .5, "max_grad_norm": .5, "exact_kl_limit": .03,
        "prediction_coef": model.prediction_coef, "reward": PRECISION_REWARD,
        "train_cases": DENSE_CASES, "validation_cases": PRECISION_VALIDATION, "test_cases": PRECISION_TEST,
        "episode_seconds": 4., "optimizer": "RMSprop", "optimizer_kwargs": {"alpha": .99, "eps": 1e-5, "momentum": 0., "centered": False, "weight_decay": 0.},
        "normalization": "first valid training rollout only, frozen", "teacher": None,
        "prediction_validation_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {k: version(k) for k in ("torch", "stable-baselines3", "mujoco", "numpy", "gymnasium")}}
    started = time.monotonic()
    summary = {"status": "running", "training": training, "history": [], "updates": [],
        "contract": env.contract(), "initial_parameter_sha256": parameter_hash(model),
        "parameter_count": sum(p.numel() for p in model.policy.parameters()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "stopping": "fixed budget; selection only tracking validation"}
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
            "accepted_optimizer_steps": model.accepted_steps, "rejected_optimizer_steps": model.rejected_steps,
            "ppo": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")} if model.num_timesteps else {}})
        summary["elapsed_s"] = time.monotonic()-started
        write_json(folder/"summary.json", summary)
        print(json.dumps({"trial": args.trial, "steps": model.num_timesteps,
                          "validation_mm": result["score_m"]*1000, "passed": result["passed"],
                          "best_steps": summary["best_steps"]}), flush=True)

    check()
    for _ in range(args.steps//8192):
        model.learn(8192, reset_num_timesteps=False, callback=Labels())
        check()
    summary["updates"].append({"steps": model.num_timesteps, **model.last_audit})
    save_policy(model, env, folder/"last.zip", training)
    summary.update(status="training_complete", steps=model.num_timesteps,
        accepted_optimizer_steps=model.accepted_steps, rejected_optimizer_steps=model.rejected_steps,
        elapsed_s=time.monotonic()-started)
    write_json(folder/"summary.json", summary)
    env.close()


if __name__ == "__main__":
    main()
