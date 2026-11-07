"""Read-only checkpoint audit: weights, representations and command sensitivity.

This script never updates a network. Historical observations are excluded from
normalization by whole episode and are off-policy; their scenario conditions
also occur in PPO training. Feature ranks are descriptive, not proofs of control
capability. The command probe replays identical open-loop
excitation histories with different commands, leaving physical states equal.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.compact_protocol import initialize_command_at_reset
from myo.env import PreacherCurlEnv
from myo.runtime import dynamics_digest, read_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def participation_rank(matrix):
    """(sum eigenvalues)^2 / sum eigenvalues^2 of the unnormalized Gram matrix."""
    matrix = matrix.double()
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    trace = gram.trace()
    return float(trace.square() / gram.square().sum()) if trace > 0 else 0.0


@torch.no_grad()
def parameter_summary(policy, initial, spectral=False):
    rows = {}
    totals = {key: [0.0, 0.0, 0.0] for key in ("actor", "critic")}
    for name, value in policy.named_parameters():
        old = initial[name]
        delta = value.detach() - old
        norm, old_norm, change = (float(x.norm()) for x in (value, old, delta))
        row = {"shape": list(value.shape), "norm": norm, "initial_norm": old_norm,
               "delta_l2": change, "relative_delta": change / old_norm if old_norm else None,
               "max_abs": float(value.detach().abs().max())}
        if spectral and value.ndim == 2:
            row["weight_participation_rank"] = participation_rank(value.detach())
        rows[name] = row
        branch = "actor" if "policy_net" in name or name.startswith("action_net") else "critic"
        for index, number in enumerate((norm, old_norm, change)):
            totals[branch][index] += number ** 2
    aggregate = {branch: {"norm": numbers[0] ** .5, "initial_norm": numbers[1] ** .5,
                          "delta_l2": numbers[2] ** .5,
                          "relative_delta": (numbers[2] / numbers[1]) ** .5}
                 for branch, numbers in totals.items()}
    return {"layers": rows, "branches": aggregate}


@torch.no_grad()
def functional_summary(policy, observations):
    features = policy.extract_features(observations)
    hidden = {}
    for branch in ("policy_net", "value_net"):
        values = features
        for index, layer in enumerate(getattr(policy.mlp_extractor, branch)):
            values = layer(values)
            if isinstance(layer, torch.nn.Tanh):
                centered = values - values.mean(dim=0)
                # The fixed 256-row subset bounds the cost and rank ceiling.
                subset = centered[torch.linspace(0, len(centered) - 1, 256).long()]
                unit_std = values.std(dim=0)
                hidden[f"{branch}/{index}"] = {
                    "width": values.shape[1],
                    "saturated_fraction_abs_gt_0_99": float((values.abs() > .99).float().mean()),
                    "mean_tanh_derivative": float((1 - values.square()).mean()),
                    "unit_std_median": float(unit_std.median()),
                    "unit_fraction_std_lt_1e_3": float((unit_std < .001).float().mean()),
                    "centered_feature_participation_rank_256_rows": participation_rank(subset),
                }
    distribution = policy.get_distribution(observations).distribution
    critic = policy.predict_values(observations)
    mean = distribution.mean
    std = distribution.stddev
    return {
        "hidden_layers": hidden,
        "mean_excitation_per_muscle": mean.mean(dim=0).tolist(),
        "between_state_mean_std_per_muscle": mean.std(dim=0).tolist(),
        "mean_excitation_p05_per_muscle": mean.quantile(.05, dim=0).tolist(),
        "mean_excitation_p95_per_muscle": mean.quantile(.95, dim=0).tolist(),
        "exploration_std_median_per_muscle": std.median(dim=0).values.tolist(),
        "concentration_median_per_muscle": (distribution.concentration1 + distribution.concentration0).median(dim=0).values.tolist(),
        "alpha_median_per_muscle": distribution.concentration1.median(dim=0).values.tolist(),
        "beta_median_per_muscle": distribution.concentration0.median(dim=0).values.tolist(),
        "differential_entropy_mean_per_muscle": distribution.entropy().mean(dim=0).tolist(),
        "critic_mean": float(critic.mean()), "critic_std": float(critic.std()),
        "deterministic_excitation_rms": float(mean.square().mean().sqrt()),
        "state_variation_rms": float(mean.var(dim=0).mean().sqrt()),
        "centered_action_mean_participation_rank": participation_rank(mean - mean.mean(dim=0)),
    }


def command_probe(config):
    env = PreacherCurlEnv(config)
    observations, labels = [], []
    max_physical_difference = 0.0
    try:
        sensor_indices = [env.observation_names.index(name) for name in env._sensor_names]
        for angle in (20, 45, 75, 120):
            reference = {}
            for target in (15, 45, 75, 105, 150):
                env.reset(seed=0, options={"q": float(np.deg2rad(angle)), "qvel": 0,
                    "target_angle": float(np.deg2rad(angle)), "trigger": 1.,
                    "payload_mass": 1., "capacity": 1.})
                initialize_command_at_reset(env, angle, 1.)
                env.set_command(float(np.deg2rad(target)), 1.)
                for step in range(11):
                    observation = env.observe()
                    if step in (0, 10):
                        observations.append(observation)
                        labels.append({"initial_angle_deg": angle, "requested_target_deg": target,
                                       "time_since_request_s": step * env.control_dt,
                                       "actual_angle_deg": float(np.rad2deg(env.data.qpos[env.qpos_index])),
                                       "smoothed_target_deg": float(np.rad2deg(env.command.target))})
                        if step not in reference:
                            reference[step] = observation[sensor_indices]
                        max_physical_difference = max(max_physical_difference,
                            float(np.max(np.abs(observation[sensor_indices] - reference[step]))))
                    if step < 10:
                        env.step(np.full(len(env.muscles), .1, dtype=np.float32))
    finally:
        env.close()
    return torch.as_tensor(np.array(observations)), labels, max_physical_difference


@torch.no_grad()
def probe_summary(policy, observations, labels):
    means = policy.get_distribution(observations).distribution.mean.numpy()
    rows = [{**label, "mean_excitation": mean.tolist()} for label, mean in zip(labels, means)]
    spans = []
    for angle in (20, 45, 75, 120):
        for time in (0., .2):
            indices = [i for i, label in enumerate(labels)
                       if label["initial_angle_deg"] == angle and np.isclose(label["time_since_request_s"], time)]
            spans.append({"initial_angle_deg": angle, "time_since_request_s": time,
                          "target_induced_excitation_span_per_muscle": np.ptp(means[indices], axis=0).tolist()})
    return {"cases": rows, "spans": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", nargs="+", type=Path,
        default=[Path("runs/d1_architecture_small_tanh"), Path("runs/d1_architecture_wide_tanh")])
    parser.add_argument("--observations", type=Path, default=Path("runs/d1_ppo_diagnosis/frozen_rollouts.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/d1_architecture_weight_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_config(args.config)
    collection = json.loads(args.observations.with_name("collection.json").read_text())
    if collection["config"] != asdict(config) or collection["dynamics_sha256"] != dynamics_digest():
        raise ValueError("Historical observations are incompatible with current physics")
    data = np.load(args.observations)
    indices = np.flatnonzero(np.isin(data["episodes"], [3, 4, 5, 12, 13, 14]))
    indices = indices[np.linspace(0, len(indices) - 1, 1024, dtype=int)]
    observations = torch.as_tensor(data["observations"][indices], dtype=torch.float32)
    probe_observations, labels, physical_difference = command_probe(config)
    report = {
        "script_sha256": sha256(__file__), "observations_sha256": sha256(args.observations),
        "cohort": "1024 fixed historical off-policy observations; whole trigger=0.6 episodes excluded from input normalization; these scenario conditions also occur in PPO training, so this is not a held-out behavioral test or a control-success evaluation",
        "historical_row_indices": indices.tolist(),
        "feature_rank_definition": "Participation ratio of centered activation covariance; same fixed 256-row subset for every checkpoint; maximum 255; low rank can reflect correlated inputs",
        "raw_input_participation_rank_256_rows": participation_rank(
            (observations - observations.mean(dim=0))[torch.linspace(0, len(observations) - 1, 256).long()]),
        "weight_rank_definition": "Participation ratio of uncentered W W^T; scale invariant; not algebraic rank",
        "command_probe": "Same physical initialization and 10 identical open-loop actions (.1 per muscle), only requested targets differ; evaluate immediately and at .2 s; no physics state edits or target substitutions inside observation arrays",
        "command_probe_max_physical_sensor_difference": physical_difference,
        "models": {},
    }
    for directory in args.directories:
        checkpoints = [directory / "initial.zip"] + sorted(directory.glob("block_*.zip"))
        initial_model = PPO.load(checkpoints[0], device="cpu")
        initial = {name: value.detach().clone() for name, value in initial_model.policy.named_parameters()}
        rows = {}
        for index, checkpoint in enumerate(checkpoints):
            model = initial_model if index == 0 else PPO.load(checkpoint, device="cpu")
            metadata = json.loads(Path(str(checkpoint) + ".metadata.json").read_text())
            if metadata["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
            if metadata["config"] != asdict(config) or metadata["dynamics_sha256"] != dynamics_digest():
                raise ValueError(f"Checkpoint physics mismatch: {checkpoint}")
            for key, source in (("beta_policy_sha256", "src/myo/policy_beta.py"),
                                ("features_sha256", "src/myo/policy_features.py")):
                if metadata[key] != sha256(source):
                    raise ValueError(f"Checkpoint policy source mismatch: {checkpoint}: {source}")
            model.policy.set_training_mode(False)
            rows[checkpoint.name] = {
                "sha256": metadata["checkpoint_sha256"], "trained_steps": model.num_timesteps,
                "weights": parameter_summary(model.policy, initial, spectral=index in (0, len(checkpoints) - 1)),
                "functions": functional_summary(model.policy, observations),
                "command_response": probe_summary(model.policy, probe_observations, labels),
            }
            print(json.dumps({"directory": str(directory), "checkpoint": checkpoint.name,
                              "actor_relative_delta": rows[checkpoint.name]["weights"]["branches"]["actor"]["relative_delta"],
                              "critic_relative_delta": rows[checkpoint.name]["weights"]["branches"]["critic"]["relative_delta"]}), flush=True)
        report["models"][directory.name] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
