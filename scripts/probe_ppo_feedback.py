"""Compare fast/persistent gSDE with fixed centered inputs and value units."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from diagnose_compact_ppo import ScaledReward, write_json
from myo.compact_protocol import CompactProtocol, compact_scenarios, protocol_metadata
from myo.continue_compact import behavior_summary
from myo.env import PreacherCurlEnv
from myo.evaluate import run_scenario
from myo.policy_features import NormalizedFeaturesExtractor
from myo.policy_exploration import UnitLatentSDEPolicy
from myo.runtime import dynamics_digest, load_controller, policy_contract, read_config


class Audit(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows, self.episodes, self.window = [], [], []

    def _on_step(self):
        info = self.locals["infos"][0]
        self.window.append((float(np.rad2deg(info["q"])),
                            sum(info["reward_components"].values())))
        if self.locals["dones"][0]:
            self.episodes.append({"steps": self.num_timesteps,
                "scenario": info["protocol_scenario"],
                "angle_min_deg": min(x[0] for x in self.window),
                "angle_max_deg": max(x[0] for x in self.window),
                "raw_reward_per_step": float(np.mean([x[1] for x in self.window]))})
            self.window.clear()
        return True

    def _on_rollout_end(self):
        b = self.model.rollout_buffer
        obs = torch.as_tensor(b.observations.reshape(-1, *self.model.observation_space.shape))
        with torch.no_grad():
            features = self.model.policy.extract_features(obs)
            hidden = self.model.policy.mlp_extractor.value_net(features)
            dist = self.model.policy.get_distribution(obs).distribution
            ev = 1 - np.var(b.returns - b.values) / np.var(b.returns)
        self.rows.append({"steps": self.num_timesteps, "critic_ev": float(ev),
                          "critic_saturation": float((hidden.abs() > .99).float().mean()),
                          "action_std_median": float(dist.stddev.median()),
                          "clipped_fraction": float(((b.actions < 0) | (b.actions > 1)).mean())})
        if len(self.rows) % 30 == 0:
            print(json.dumps(self.rows[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, choices=[1, 10], required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--unit-sde", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gae-lambda", type=float, choices=[0.95, 1.0], default=0.95)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = read_config(args.config)
    previous = Path("runs/d1_ppo_diagnosis")
    data = np.load(previous / "frozen_rollouts.npz")
    collection = json.loads((previous / "collection.json").read_text())
    if collection["dynamics_sha256"] != dynamics_digest() or collection["config"] != asdict(config):
        raise ValueError("Frozen observations belong to another physical model/configuration")
    training = (~np.isin(data["episodes"], [3, 4, 5, 12, 13, 14])) & (data["episodes"] < 18)
    mean = data["observations"][training].mean(axis=0)
    physical = PreacherCurlEnv(config)
    protocol = CompactProtocol(physical)
    env = Monitor(ScaledReward(protocol, 0.01))
    evaluation = PreacherCurlEnv(config)
    model = PPO(UnitLatentSDEPolicy if args.unit_sde else "MlpPolicy", env, device="cpu", seed=0,
        policy_kwargs={"net_arch": [128, 128], "log_std_init": -2,
                       "features_extractor_class": NormalizedFeaturesExtractor},
        n_steps=args.rollout_steps, batch_size=64, learning_rate=3e-4,
        target_kl=0.02, gae_lambda=args.gae_lambda,
        use_sde=True, sde_sample_freq=args.frequency, verbose=0)
    initial = PPO.load(previous / "initial.zip", device="cpu")
    state = model.policy.state_dict()
    for name, value in initial.policy.state_dict().items():
        if name != "log_std" and name in state:
            state[name] = value.clone()
    model.policy.load_state_dict(state)
    model.policy.features_extractor.set_normalization(mean, np.ones_like(mean))
    with torch.no_grad():
        model.policy.value_net.weight.mul_(0.01)
        model.policy.value_net.bias.mul_(0.01)
        calibration = torch.as_tensor(data["observations"][training][::8])
        old_std = model.policy.get_distribution(calibration).distribution.stddev.median()
        model.policy.log_std.add_(torch.log(0.54 / old_std))
        actual_std = model.policy.get_distribution(calibration).distribution.stddev
    parent = None
    if args.resume:
        checked, _, parent = load_controller(str(args.resume))
        checked.close()
        expected = {"config": asdict(config), "n_steps": args.rollout_steps,
                    "sde_sample_freq": args.frequency, "unit_latent_sde": args.unit_sde,
                    "reward_scale": 0.01, "gae_lambda": args.gae_lambda}
        if any(parent.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume requires identical physics, preprocessing, exploration and rollout settings")
        model = PPO.load(args.resume, env=env, device="cpu")
        with torch.no_grad():
            actual_std = model.policy.get_distribution(calibration).distribution.stddev
    if args.noise_std is not None:
        if not args.unit_sde or not np.isfinite(args.noise_std) or args.noise_std <= 0:
            raise ValueError("A positive noise override requires unit-latent SDE")
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.noise_std)))
            actual_std = model.policy.get_distribution(calibration).distribution.stddev
    model.set_random_seed(0)
    audit = Audit()
    metadata = {"schema_version": 2, "task": "curriculum_preacher_curl",
        "algorithm": "stable_baselines3.PPO", "training_stage": "ppo", "seed": 0,
        "torch_threads": 1, "warm_start": None, "config": asdict(config),
        "dynamics_sha256": dynamics_digest(), "contract": policy_contract(physical),
        "observation_names": list(physical.observation_names),
        "features_sha256": hashlib.sha256(Path("src/myo/policy_features.py").read_bytes()).hexdigest(),
        "exploration_sha256": (hashlib.sha256(Path("src/myo/policy_exploration.py").read_bytes()).hexdigest()
                               if args.unit_sde else None),
        "unit_latent_sde": args.unit_sde,
        "predictor_sha256": None, "predictor_path": None,
        "training_protocol": protocol_metadata(config.payload_mass),
        "reward_scale": 0.01, "initial_value_head_scale": 0.01,
        "input_transform": "frozen training-episode mean subtraction; scale=1",
        "use_sde": True, "sde_sample_freq": args.frequency,
        "n_steps": args.rollout_steps, "batch_size": 64, "n_epochs": 10,
        "gamma": model.gamma, "gae_lambda": model.gae_lambda,
        "learning_rate": 3e-4, "target_kl": 0.02,
        "initial_effective_std_min_median_max": [float(actual_std.min()), float(actual_std.median()), float(actual_std.max())],
        "initial_source_sha256": collection["initial_sha256"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"] if parent else None,
        "initial_noise_std_override": args.noise_std,
        "resume_semantics": "weights and Adam restored; fresh physical episode and seed 0" if parent else None,
        "frozen_data_sha256": hashlib.sha256((previous / "frozen_rollouts.npz").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_versions": {n: version(n) for n in ("mujoco", "gymnasium", "numpy", "stable-baselines3", "torch")},
        "limitations": "Controlled diagnostic experiment; fixed exercises do not establish generalization or protection"}
    summary = {"setup": metadata, "status": "running", "evaluations": []}

    def save(name):
        checkpoint = out / f"{name}.zip"
        model.save(checkpoint)
        sidecar = {**metadata, "trained_steps": model.num_timesteps,
            "protocol_completed_episodes": protocol.completed_episodes,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        write_json(str(checkpoint) + ".metadata.json", sidecar)
        return checkpoint

    try:
        save("initial")
        print(json.dumps({"frequency": args.frequency, "initial_std": metadata["initial_effective_std_min_median_max"]}), flush=True)
        for cycle in range(1, args.cycles + 1):
            model.learn(total_timesteps=14700, callback=audit, reset_num_timesteps=False)
            checkpoint = save(f"cycle_{cycle:02d}")
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            cases = compact_scenarios(config.payload_mass)
            if cycle < args.cycles:
                cases = [cases[i] for i in [6, 7, 8]]
            try:
                results = [run_scenario(evaluation,
                    lambda obs: model.predict(obs, deterministic=True)[0], case, 0,
                    continue_after_time_limit=True, measure_commands=True) for case in cases]
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            metrics = behavior_summary(results)
            row = {"cycle": cycle, "steps": model.num_timesteps, "checkpoint": str(checkpoint),
                   "full_suite": cycle == args.cycles, "metrics": metrics}
            summary["evaluations"].append(row)
            write_json(out / f"evaluation_{cycle:02d}.json", {**row, "scenarios": results})
            write_json(out / "audit.json", {"rollouts": audit.rows, "episodes": audit.episodes})
            write_json(out / "summary.json", summary)
            print(json.dumps({"cycle": cycle, "steps": model.num_timesteps,
                              "score": metrics["score"], "moves_passed": metrics["moves_passed"],
                              "sweeps_passed": metrics["sweeps_passed"]}), flush=True)
        summary["status"] = "completed_diagnostic_budget"
        write_json(out / "summary.json", summary)
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
