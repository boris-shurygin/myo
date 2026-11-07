"""Fixed-budget single-factor architecture/optimizer trials on native PPO."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np

from .train_vertical_ppo import compact, write_json
from .vertical_control import (REWARD, TRAIN_CASES, VALIDATION_CASES, TEST_CASES,
    VerticalControlEnv, evaluate_policy, load_policy, save_policy)
from .vertical_muscles import VerticalConfig


TRIALS = {
    "control": {"hidden": [128, 128], "activation": "Tanh", "optimizer": "Adam"},
    "silu": {"hidden": [128, 128], "activation": "SiLU", "optimizer": "Adam"},
    "deep": {"hidden": [128]*4, "activation": "Tanh", "optimizer": "Adam"},
    "radam": {"hidden": [128, 128], "activation": "Tanh", "optimizer": "RAdam"},
    "rmsprop": {"hidden": [128, 128], "activation": "Tanh", "optimizer": "RMSprop"},
    "muon": {"hidden": [128, 128], "activation": "Tanh", "optimizer": "MuonAdam"},
}


def model_for_trial(name, env, *, rollout_steps=1024, minibatch=128, epochs=10):
    import torch
    from stable_baselines3 import PPO
    from .optimizer_muon import MuonAdam
    from .policy_beta import BetaExcitationPolicy
    trial = TRIALS[name]
    optimizers = {"Adam": torch.optim.Adam, "RAdam": torch.optim.RAdam,
                  "RMSprop": torch.optim.RMSprop, "MuonAdam": MuonAdam}
    # Match SB3's baseline Adam epsilon; RMSprop otherwise uses alpha=.99,
    # no momentum, no centering. All weight decay is zero.
    optimizer_kwargs = {} if name == "muon" else {"eps": 1e-5, "weight_decay": 0.}
    return PPO(BetaExcitationPolicy, env, seed=0, device="cpu", verbose=0,
        learning_rate=3e-4, n_steps=rollout_steps, batch_size=minibatch, n_epochs=epochs,
        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.001, vf_coef=.5,
        max_grad_norm=.5, target_kl=.03,
        policy_kwargs={"net_arch": {"pi": trial["hidden"], "vf": trial["hidden"]},
            "activation_fn": getattr(torch.nn, trial["activation"]),
            "optimizer_class": optimizers[trial["optimizer"]], "optimizer_kwargs": optimizer_kwargs})


def parameter_hash(model):
    checksum = hashlib.sha256()
    for name, value in model.policy.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().numpy().tobytes())
    return checksum.hexdigest()


def optimizer_steps(model):
    return max((int(s["step"]) for s in model.policy.optimizer.state.values() if "step" in s), default=0)


def load_trial(path, env):
    model, metadata = load_policy(path, env)
    training = metadata["training"]
    source = Path(__file__).with_name("optimizer_muon.py")
    if training["optimizer_source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Optimizer source mismatch")
    trial = training["trial"]
    if model.policy.activation_fn.__name__ != trial["activation"]:
        raise ValueError("Checkpoint activation mismatch")
    if model.policy.optimizer.__class__.__name__ != trial["optimizer"]:
        raise ValueError("Checkpoint optimizer mismatch")
    if model.policy.net_arch != {"pi": trial["hidden"], "vf": trial["hidden"]}:
        raise ValueError("Checkpoint network shape mismatch")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=tuple(dict.fromkeys((*TRIALS, "combined"))), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/vertical_architecture"))
    parser.add_argument("--steps", type=int, default=131072)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--protocol", choices=("original", "precision"), default="original")
    args = parser.parse_args()
    if args.protocol == "precision" and args.trial != "combined":
        parser.error("Precision experiment only uses combined 4x128 SiLU RMSprop")
    if args.trial == "combined":
        TRIALS["combined"] = {"hidden": [128]*4, "activation": "SiLU", "optimizer": "RMSprop"}
    if args.steps % args.eval_every or args.eval_every % 1024:
        parser.error("Use complete 1024-step rollouts and validation blocks")
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    folder = args.output_root/args.trial
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = VerticalConfig(variant=4)
    environment, evaluate = VerticalControlEnv, evaluate_policy
    train_cases, validation_cases, test_cases = TRAIN_CASES, VALIDATION_CASES, TEST_CASES
    if args.protocol == "precision":
        from .vertical_precision import (VerticalPrecisionEnv, evaluate_precision,
            DENSE_CASES, PRECISION_VALIDATION, PRECISION_TEST)
        environment, evaluate = VerticalPrecisionEnv, evaluate_precision
        train_cases, validation_cases, test_cases = DENSE_CASES, PRECISION_VALIDATION, PRECISION_TEST
    env = environment(config)
    model = model_for_trial(args.trial, Monitor(env))
    training = {"trial": TRIALS[args.trial], "seed": 0, "learning_rate": 3e-4,
        "transition_budget": args.steps, "rollout_steps": 1024, "batch_size": 128, "epochs": 10,
        "gamma": .99, "gae_lambda": .95, "clip": .2, "entropy_coef": .001,
        "vf_coef": .5, "max_grad_norm": .5, "target_kl": .03,
        "protocol": args.protocol, "reward": env.contract()["reward"],
        "train_cases": train_cases, "validation_cases": validation_cases, "test_cases": test_cases,
        "episode_seconds": 4., "teacher": None, "predictor": None,
        "optimizer_defaults": [{k: v for k, v in g.items() if k != "params"}
                               for g in model.policy.optimizer.param_groups],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimizer_source_sha256": hashlib.sha256(Path(__file__).with_name("optimizer_muon.py").read_bytes()).hexdigest(),
        "versions": {k: version(k) for k in ("torch", "stable-baselines3", "gymnasium", "mujoco", "numpy")}}
    training = json.loads(json.dumps(training))
    summary = {"status": "running", "training": training, "contract": env.contract(),
        "parameter_count": sum(p.numel() for p in model.policy.parameters()),
        "initial_parameter_sha256": parameter_hash(model), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "history": [], "rollouts": [], "selection": "fewer failures, more passed, lower steady RMSE; includes initial",
        "stopping": "fixed transition budget; no plateau/ready early stop"}
    save_policy(model, env, folder/"initial.zip", training)
    best_key = None

    def check():
        nonlocal best_key
        tick = time.monotonic()
        result = evaluate(model, config, validation_cases)
        key = (result["failures"], -result["passed"], result["score_m"])
        if best_key is None or key < best_key:
            best_key = key
            summary["best_steps"] = model.num_timesteps
            summary["best_validation"] = compact(result)
            save_policy(model, env, folder/"best.zip", training)
        summary["history"].append({"steps": model.num_timesteps, "validation": compact(result),
            "optimizer_steps": optimizer_steps(model), "validation_seconds": time.monotonic()-tick,
            "ppo": {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}
                   if model.num_timesteps else {}})
        summary["elapsed_s"] = time.monotonic()-started
        write_json(folder/"summary.json", summary)
        print(json.dumps({"trial": args.trial, "steps": model.num_timesteps,
            "validation_mm": result["score_m"]*1000, "passed": result["passed"],
            "best_steps": summary["best_steps"]}), flush=True)

    class Audit(BaseCallback):
        def _on_rollout_start(self):
            self.tick = time.monotonic()
            self.raw_rewards = []

        def _on_step(self):
            self.raw_rewards.append(self.locals["infos"][0]["raw_reward"])
            return True

        def _on_rollout_end(self):
            buffer = self.model.rollout_buffer
            variance = float(np.var(buffer.returns))
            with torch.no_grad():
                dist = model.policy.get_distribution(torch.as_tensor(buffer.observations[:, 0])).distribution
            summary["rollouts"].append({"steps": self.num_timesteps,
                "collection_seconds": time.monotonic()-self.tick,
                "raw_reward_mean": float(np.mean(self.raw_rewards)),
                "critic_ev": float(1-np.var(buffer.returns-buffer.values)/variance) if variance > 1e-12 else None,
                "excitation_mean": dist.mean.mean(0).tolist(), "excitation_std": dist.stddev.mean(0).tolist(),
                "out_of_bounds_actions": int(((buffer.actions < 0) | (buffer.actions > 1)).sum())})

    check()
    callback = Audit()
    summary["training_seconds"] = 0.
    for _ in range(args.steps//args.eval_every):
        tick = time.monotonic()
        model.learn(args.eval_every, reset_num_timesteps=False, callback=callback)
        seconds = time.monotonic()-tick
        summary["training_seconds"] += seconds
        check()
        summary["history"][-1]["training_block_seconds"] = seconds
    save_policy(model, env, folder/"last.zip", training)
    summary.update(status="training_complete", steps=model.num_timesteps,
                   optimizer_steps=optimizer_steps(model), elapsed_s=time.monotonic()-started)
    write_json(folder/"summary.json", summary)
    # Test is normally deferred until all six selections have been frozen.
    if not args.defer_test:
        selected, _ = load_trial(folder/"best.zip", env)
        result = evaluate(selected, config, test_cases)
        summary.update(test=compact(result), status="complete")
        write_json(folder/"test.json", result)
        write_json(folder/"summary.json", summary)
    model.get_env().close()


if __name__ == "__main__":
    main()
