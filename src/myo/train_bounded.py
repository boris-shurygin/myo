"""Train bounded muscle commands by physical outcomes, stopping at readiness/plateau."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import time

import gymnasium as gym
import numpy as np

from .compact_protocol import CompactProtocol, compact_scenarios, protocol_metadata
from .continue_compact import Plateau, behavior_summary
from .env import PreacherCurlEnv
from .evaluate import run_scenario
from .runtime import dynamics_digest, load_controller, policy_contract, read_config


class RewardScale(gym.RewardWrapper):
    def reward(self, reward):
        return 0.01 * reward


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--block-cycles", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--persistence", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128],
                        help="Hidden widths of both actor and critic; activation remains Tanh")
    parser.add_argument("--statistics", type=Path, default=Path("runs/d1_ppo_diagnosis/frozen_rollouts.npz"))
    args = parser.parse_args(argv)
    if args.block_cycles < 1 or args.patience < 1 or args.epochs < 1:
        parser.error("Block cycles, patience and epochs must be positive")
    if not 0 <= args.persistence < 1:
        parser.error("Persistence must lie in [0, 1)")
    if any(width < 1 for width in args.hidden_layers):
        parser.error("Hidden layer widths must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from .policy_beta import BetaExcitationPolicy
    from .policy_persistent_beta import PersistentBetaPolicy
    from .policy_features import NormalizedFeaturesExtractor

    torch.set_num_threads(1)
    started = time.perf_counter()
    config = read_config(args.config)
    physical = PreacherCurlEnv(config)
    protocol = CompactProtocol(physical)
    env = Monitor(RewardScale(protocol))
    evaluation = PreacherCurlEnv(config)
    root = Path(__file__).parent
    parent = None
    if args.resume:
        checked, _, parent = load_controller(str(args.resume))
        checked.close()
        expected = {"config": asdict(config), "reward_scale": 0.01,
                    "bounded_distribution": "beta_alpha_beta_softplus_floor_0.05",
                    "training_protocol": json.loads(json.dumps(protocol_metadata(config.payload_mass)))}
        if any(parent.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume requires identical physical model, reward, protocol and bounded distribution")
        if parent.get("action_persistence", 0.0) != args.persistence:
            raise ValueError("Resume requires identical conditional action persistence")
        if parent.get("n_epochs") != args.epochs:
            raise ValueError("Resume requires identical PPO epochs")
        model = PPO.load(args.resume, env=env, device="cpu")
        if model.policy.net_arch != args.hidden_layers:
            raise ValueError("Resume requires the same hidden layers as the saved policy")
        model.set_random_seed(0)
        metadata = dict(parent)
        metadata["parent_checkpoint_sha256"] = parent["checkpoint_sha256"]
    else:
        # Reuse only causal observation statistics, not teacher commands or
        # learned actions. Match the previous diagnostic's complete-episode split.
        collection = json.loads(args.statistics.with_name("collection.json").read_text())
        if collection["config"] != asdict(config) or collection["dynamics_sha256"] != dynamics_digest():
            raise ValueError("Observation statistics belong to another physical model")
        data = np.load(args.statistics)
        training = (~np.isin(data["episodes"], [3, 4, 5, 12, 13, 14])) & (data["episodes"] < 18)
        mean = data["observations"][training].mean(axis=0)
        policy_kwargs = {"net_arch": args.hidden_layers, "activation_fn": torch.nn.Tanh,
                         "features_extractor_class": NormalizedFeaturesExtractor}
        if args.persistence:
            policy_kwargs.update(persistence=args.persistence, previous_action_indices=[
                physical.observation_names.index(f"history/1/excitation/{name}") for name in physical.muscle_names])
        model = PPO(PersistentBetaPolicy if args.persistence else BetaExcitationPolicy,
            env, seed=0, device="cpu", n_steps=1472,
            batch_size=64, n_epochs=args.epochs, learning_rate=3e-4, target_kl=0.02,
            policy_kwargs=policy_kwargs, verbose=0)
        model.policy.features_extractor.set_normalization(mean, np.ones_like(mean))
        if args.persistence:
            model.policy.synchronize_previous_action_normalization()
        with torch.no_grad():
            model.policy.value_net.weight.mul_(0.01)
            model.policy.value_net.bias.mul_(0.01)
        metadata = {"schema_version": 2, "task": "curriculum_preacher_curl",
            "algorithm": "stable_baselines3.PPO", "training_stage": "ppo", "seed": 0,
            "config": asdict(config), "dynamics_sha256": dynamics_digest(),
            "contract": policy_contract(physical), "observation_names": list(physical.observation_names),
            "features_sha256": hashlib.sha256((root / "policy_features.py").read_bytes()).hexdigest(),
            "beta_policy_sha256": hashlib.sha256((root / "policy_beta.py").read_bytes()).hexdigest(),
            "persistent_beta_sha256": (hashlib.sha256((root / "policy_persistent_beta.py").read_bytes()).hexdigest()
                                       if args.persistence else None),
            "action_persistence": args.persistence,
            "bounded_distribution": "beta_alpha_beta_softplus_floor_0.05",
            "deterministic_action": "Beta mean; no Gaussian clipping or teacher",
            "predictor_sha256": None, "predictor_path": None, "warm_start": None,
            "training_protocol": protocol_metadata(config.payload_mass), "reward_scale": 0.01,
            "input_transform": "frozen training-episode mean subtraction; scale=1",
            "statistics_sha256": hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
            "torch_threads": 1, "n_steps": model.n_steps, "batch_size": model.batch_size,
            "n_epochs": model.n_epochs, "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "learning_rate": 3e-4, "target_kl": model.target_kl,
            "package_versions": {n: version(n) for n in ("mujoco", "gymnasium", "numpy", "stable-baselines3", "torch")}}
    metadata["network_architecture"] = {
        "input_features": physical.observation_space.shape[0],
        "actor_hidden_layers": list(model.policy.net_arch),
        "critic_hidden_layers": list(model.policy.net_arch),
        "activation": model.policy.activation_fn.__name__,
        "trainable_parameters": sum(p.numel() for p in model.policy.parameters() if p.requires_grad)}
    metadata["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["resume_semantics"] = "Weights/Adam restored; new physical episode and seed 0" if parent else None

    class Audit(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            b = self.model.rollout_buffer
            with torch.no_grad():
                obs = torch.as_tensor(b.observations.reshape(-1, *physical.observation_space.shape))
                dist = self.model.policy.get_distribution(obs).distribution
                ev = 1 - np.var(b.returns - b.values) / np.var(b.returns)
                row = {"steps": self.num_timesteps, "elapsed_seconds": time.perf_counter() - started,
                    "critic_ev": float(ev),
                    "mean_excitation": float(dist.mean.mean()), "median_std": float(dist.stddev.median()),
                    "actions_outside_bounds": int(((b.actions < 0) | (b.actions > 1)).sum())}
            self.rows.append(row)
            write_json(args.output_dir / "audit.json", self.rows)
            if len(self.rows) % 10 == 0:
                print(json.dumps(row), flush=True)

    audit = Audit()
    summary = {"status": "running", "setup": metadata, "evaluations": [],
               "stopping": {"patience": args.patience, "meaningful_gain": "max(0.5, 2% of anchor)",
                            "gates": "unchanged myo.continue_compact.behavior_summary"}}

    def save(name):
        checkpoint = args.output_dir / f"{name}.zip"
        model.save(checkpoint)
        write_json(str(checkpoint) + ".metadata.json", {**metadata, "trained_steps": model.num_timesteps,
            "protocol_completed_episodes": protocol.completed_episodes,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        return checkpoint

    try:
        save("initial")
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"network_architecture": metadata["network_architecture"],
                          "initialization_seconds": time.perf_counter() - started}), flush=True)
        plateau = None
        block = 0
        while True:
            block += 1
            steps = round(sum(s.duration_s for s in protocol.scenarios) / config.control_dt) * args.block_cycles
            training_started = time.perf_counter()
            model.learn(steps, callback=audit, reset_num_timesteps=False)
            training_seconds = time.perf_counter() - training_started
            checkpoint = save(f"block_{block:03d}")
            evaluation_started = time.perf_counter()
            rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
            try:
                results = [run_scenario(evaluation,
                    lambda obs: model.predict(obs, deterministic=True)[0], s, 0,
                    continue_after_time_limit=True, measure_commands=True)
                    for s in compact_scenarios(config.payload_mass)]
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            metrics = behavior_summary(results)
            write_json(args.output_dir / f"evaluation_{block:03d}.json", {"metrics": metrics, "scenarios": results})
            row = {"block": block, "steps": model.num_timesteps, "controller": str(checkpoint),
                   "training_seconds": training_seconds,
                   "evaluation_seconds": time.perf_counter() - evaluation_started,
                   "elapsed_seconds": time.perf_counter() - started, **metrics}
            summary["evaluations"].append(row)
            if "best_score" not in summary or metrics["score"] < summary["best_score"] or metrics["demo_ready"]:
                for suffix in (".zip", ".zip.metadata.json"):
                    shutil.copy2(str(checkpoint).removesuffix(".zip") + suffix, args.output_dir / ("best" + suffix))
                summary.update(best_score=metrics["score"], best_controller=str(checkpoint))
            stalled = plateau.update(metrics["score"]) if plateau else False
            if plateau is None:
                plateau = Plateau(metrics["score"], args.patience)
            summary["stale_blocks"] = plateau.stale
            if metrics["demo_ready"] or stalled:
                summary["status"] = "demo_ready" if metrics["demo_ready"] else "plateau"
            write_json(args.output_dir / "summary.json", summary)
            print(json.dumps({k: row[k] for k in ("block", "steps", "score", "moves_passed", "sweeps_passed", "demo_ready")}), flush=True)
            if summary["status"] != "running":
                break
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
