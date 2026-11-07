"""Recover original episode totals and reproduce initial collection without learning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import json_to_data

from myo.compact_protocol import CompactProtocol, compact_scenarios
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.policy_beta import BetaExcitationPolicy
from myo.policy_features import NormalizedFeaturesExtractor
from myo.runtime import dynamics_digest
from myo.train_bounded import RewardScale


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(policy):
    digest = hashlib.sha256()
    for key, tensor in policy.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def read_episode_records(directory):
    records, sources = {}, []
    for path in sorted(directory.glob("block_*.zip")):
        metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
        assert file_digest(path) == metadata["checkpoint_sha256"]
        assert metadata["dynamics_sha256"] == dynamics_digest()
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
            # Load only the known local checkpoint's monitor records.
            recent = json_to_data(json.dumps({"ep_info_buffer": data["ep_info_buffer"]}))["ep_info_buffer"]
            assert "rollout_buffer" not in data
        completed = metadata["protocol_completed_episodes"]
        scenarios = compact_scenarios(metadata["config"]["payload_mass"])
        for episode, record in enumerate(recent, start=completed - len(recent)):
            scenario = scenarios[episode % len(scenarios)]
            expected_length = round(scenario.duration_s / metadata["config"]["physics_dt"]
                                    / metadata["config"]["frame_skip"])
            assert record["l"] == expected_length, (episode, record, scenario.name)
            row = {"episode_index": episode, "cycle": episode // len(scenarios) + 1,
                   "scenario": scenario.name, "steps": record["l"],
                   "scaled_episode_reward": record["r"],
                   "mean_raw_reward": record["r"] / metadata["reward_scale"] / record["l"],
                   "recorded_wall_seconds": record["t"]}
            if episode in records:
                assert records[episode] == row
            records[episode] = row
        sources.append({"checkpoint": str(path), "sha256": metadata["checkpoint_sha256"],
                        "completed_episodes": completed, "stored_recent_episodes": len(recent),
                        "trained_steps": metadata["trained_steps"]})
    assert sorted(records) == list(range(completed))
    rows = [records[i] for i in sorted(records)]
    by_scenario = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        group = [r for r in rows if r["scenario"] == name]
        by_scenario[name] = {"episodes": len(group),
            "mean_raw_reward_by_cycle": [r["mean_raw_reward"] for r in group],
            "positive_episode_totals": sum(r["scaled_episode_reward"] > 0 for r in group)}
    return {"sources": sources, "completed_episodes": len(rows), "episode_records": rows,
            "completed_steps": sum(r["steps"] for r in rows), "scenarios": by_scenario,
            "limitations": ["Original Monitor totals, not step-level trajectories.",
                "A negative episode total does not exclude useful or positive individual steps.",
                "Scenario identities recovered from fixed cyclic order and validated episode lengths.",
                "Monitor rewards are scaled by .01 and rounded to six decimals."]}


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        info = self.locals["infos"][0]
        self.rows.append({"scenario": info["protocol_scenario"],
            "angle_deg": float(np.rad2deg(info["q"])),
            "velocity_deg_s": float(np.rad2deg(info["qvel"])),
            "target_deg": float(np.rad2deg(info["requested_target_q"])),
            "reward_terms_raw": dict(info["reward_components"]),
            "actual_reward_scaled": float(self.locals["rewards"][0]),
            "done": bool(self.locals["dones"][0]),
            "timeout": bool(info.get("TimeLimit.truncated", False))})
        return True


def replay_first(directory, output):
    initial = directory / "initial.zip"
    metadata = json.loads(Path(str(initial) + ".metadata.json").read_text())
    assert file_digest(initial) == metadata["checkpoint_sha256"]
    assert metadata["dynamics_sha256"] == dynamics_digest()
    reference = PPO.load(initial, device="cpu")
    reference_state = {k: v.clone() for k, v in reference.policy.state_dict().items()}
    config = PreacherCurlConfig(**metadata["config"])
    physical = PreacherCurlEnv(config)
    protocol = CompactProtocol(physical)
    env = Monitor(RewardScale(protocol))
    try:
        # Same initialization path and RNG consumption as the archived runner.
        model = PPO(BetaExcitationPolicy, env, seed=0, device="cpu", n_steps=1472,
            batch_size=64, n_epochs=metadata["n_epochs"], learning_rate=3e-4,
            target_kl=.02, policy_kwargs={"net_arch": metadata["network_architecture"]["actor_hidden_layers"],
                "activation_fn": torch.nn.Tanh,
                "features_extractor_class": NormalizedFeaturesExtractor}, verbose=0)
        statistics = Path("runs/d1_ppo_diagnosis/frozen_rollouts.npz")
        assert file_digest(statistics) == metadata["statistics_sha256"]
        with np.load(statistics) as dataset:
            mask = (~np.isin(dataset["episodes"], [3, 4, 5, 12, 13, 14])) & (dataset["episodes"] < 18)
            mean = dataset["observations"][mask].mean(axis=0)
        model.policy.features_extractor.set_normalization(mean, np.ones_like(mean))
        with torch.no_grad():
            model.policy.value_net.weight.mul_(.01)
            model.policy.value_net.bias.mul_(.01)
        assert all(torch.equal(reference_state[k], v) for k, v in model.policy.state_dict().items())
        before = parameter_digest(model.policy)
        capture = Capture()
        _, callback = model._setup_learn(total_timesteps=1472, callback=capture,
                                         reset_num_timesteps=False)
        callback.on_training_start({}, {})
        assert model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=1472)
        callback.on_training_end()
        assert before == parameter_digest(model.policy)
        assert model._n_updates == 0 and len(model.policy.optimizer.state) == 0
        buffer = model.rollout_buffer
        with torch.no_grad():
            distribution = model.policy.get_distribution(torch.as_tensor(buffer.observations[:, 0])).distribution
        actual = {"steps": model.num_timesteps,
            "critic_ev": float(1 - np.var(buffer.returns - buffer.values) / np.var(buffer.returns)),
            "mean_excitation": float(distribution.mean.mean()),
            "median_std": float(distribution.stddev.median()),
            "actions_outside_bounds": int(((buffer.actions < 0) | (buffer.actions > 1)).sum())}
        original = json.loads((directory / "audit.json").read_text())[0]
        differences = {key: actual[key] - original[key] for key in actual}
        assert all(value == 0 for value in differences.values()), differences
        arrays = {"observations": buffer.observations[:, 0], "actions": buffer.actions[:, 0],
                  "values": buffer.values[:, 0], "gae": buffer.advantages[:, 0],
                  "returns": buffer.returns[:, 0], "buffer_rewards_scaled": buffer.rewards[:, 0],
                  "episode_starts": buffer.episode_starts[:, 0],
                  "actual_rewards_scaled": np.asarray([r["actual_reward_scaled"] for r in capture.rows]),
                  "angles_deg": np.asarray([r["angle_deg"] for r in capture.rows]),
                  "velocities_deg_s": np.asarray([r["velocity_deg_s"] for r in capture.rows]),
                  "targets_deg": np.asarray([r["target_deg"] for r in capture.rows]),
                  "dones": np.asarray([r["done"] for r in capture.rows]),
                  "timeouts": np.asarray([r["timeout"] for r in capture.rows])}
        np.savez_compressed(output, **arrays)
        episodes = {}
        for scenario in dict.fromkeys(r["scenario"] for r in capture.rows):
            group = [r for r in capture.rows if r["scenario"] == scenario]
            q = np.asarray([r["angle_deg"] for r in group])
            target = np.asarray([r["target_deg"] for r in group])
            reward = np.asarray([r["actual_reward_scaled"] for r in group])
            episodes[scenario] = {"steps": len(group), "angle_min_deg": float(q.min()),
                "angle_max_deg": float(q.max()), "positive_reward_steps": int(sum(reward > 0)),
                "positive_reward_steps_target_above_15": int(sum((reward > 0) & (target > 15.1))),
                "highest_angle_target_above_15": float(q[target > 15.1].max()) if any(target > 15.1) else None}
        return {"initial_checkpoint": str(initial), "initial_state_equal": True,
            "weights_unchanged": True, "optimizer_updates": 0,
            "original_first_audit": {k: original[k] for k in actual},
            "replayed_first_audit": actual, "audit_differences": differences,
            "arrays": str(output), "scenarios": episodes,
            "limitations": ["Reconstruction of the initial collection, not recovery of a saved original trace.",
                "Initial tensors and every saved first-rollout audit statistic match exactly.",
                "Later training trajectories cannot be reconstructed from sparse checkpoints without updates."]}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/d1_training_records_audit.json"))
    parser.add_argument("--replay-first", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = {"script_sha256": file_digest(__file__), "dynamics_sha256": dynamics_digest(), "models": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("small", "wide"):
        directory = Path("runs") / f"d1_architecture_{name}_tanh"
        result = read_episode_records(directory)
        if args.replay_first:
            result["first_collection_reconstruction"] = replay_first(directory,
                args.output.with_name(args.output.stem + f"_{name}_first.npz"))
        report["models"][name] = result
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, result["completed_episodes"], result["completed_steps"],
              result.get("first_collection_reconstruction", {}).get("scenarios"), flush=True)


if __name__ == "__main__":
    main()
