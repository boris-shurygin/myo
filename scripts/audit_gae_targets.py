"""Recompute saved GAE labels and verify SB3 boundaries without training.

Uses existing trajectories and checkpoints. A six-transition scripted fixture
exercises the real SB3 collector; it is not a muscle simulation or a learning
experiment. No learn(), optimizer update or checkpoint save occurs here.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from myo.train_bounded import RewardScale


LAMBDAS = (.90, .95, .99, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_digest(policy):
    result = hashlib.sha256()
    for name, value in policy.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def explained_variance(values, targets):
    variance = np.var(targets)
    return float(1 - np.var(targets - values) / variance) if variance > 1e-12 else None


def correlation(first, second):
    if min(np.std(first), np.std(second)) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def normalized(values):
    return (values - values.mean()) / (values.std(ddof=1) + 1e-8)


def advantages(rewards, values, following, gamma, lam):
    residuals = rewards + gamma * following - values
    output = np.zeros(len(rewards))
    tail = 0.
    for index in reversed(range(len(rewards))):
        tail = residuals[index] + gamma * lam * tail
        output[index] = tail
    return output


def saved_trajectories(summary_path):
    summary = json.loads(summary_path.read_text())
    output = {}
    for label, metadata in summary["models"].items():
        path = summary_path.with_name(summary_path.stem + "_" + label + ".npz")
        data = np.load(path)
        scenarios, start = [], 0
        for case in metadata["scenarios"]:
            end = start + case["steps"]
            scenarios.append((case["scenario"], slice(start, end)))
            start = end
        assert start == len(data["rewards"])
        gamma = metadata["gamma"]
        by_lambda = {}
        for lam in LAMBDAS:
            vectors = [advantages(data["rewards"][part], data["values"][part],
                                  data["next_values"][part], gamma, lam)
                       for _, part in scenarios]
            merged = np.concatenate(vectors)
            global_labels = normalized(merged)
            # Illustrate minibatch normalization on a single reproducible
            # permutation. These are not recovered training minibatches.
            permutation = np.random.default_rng(0).permutation(len(merged))
            minibatch_labels = np.empty_like(merged)
            for offset in range(0, len(merged), 64):
                indices = permutation[offset:offset + 64]
                minibatch_labels[indices] = normalized(merged[indices])
            rows = []
            for (name, part), vector in zip(scenarios, vectors):
                values, mc = data["values"][part], data["mc"][part]
                rows.append({"scenario": name, "steps": len(vector),
                    "advantage_mean": float(vector.mean()), "advantage_std": float(vector.std()),
                    "target_ev": explained_variance(values, values + vector),
                    "advantage_correlation_with_boot_mc": correlation(vector, mc - values),
                    "raw_positive_fraction": float(np.mean(vector > 0)),
                    "global_normalized_positive_fraction": float(np.mean(global_labels[part] > 0)),
                    "within_scenario_normalized_positive_fraction": float(np.mean(normalized(vector) > 0)),
                    "illustrative_shuffled_64_normalized_positive_fraction": float(np.mean(minibatch_labels[part] > 0))})
            by_lambda[str(lam)] = {
                "global_target_ev": explained_variance(data["values"], data["values"] + merged),
                "global_advantage_mean": float(merged.mean()), "global_advantage_std": float(merged.std()),
                "global_advantage_correlation_with_boot_mc": correlation(merged, data["mc"] - data["values"]),
                "global_normalized_sign_disagreement_with_lambda095": float(np.mean(
                    (global_labels > 0) != (normalized(data["gae"]) > 0))),
                "scenarios": rows}
            if lam == .95:
                by_lambda[str(lam)]["max_error_against_saved_gae"] = float(np.max(np.abs(merged - data["gae"])))
                assert np.allclose(merged, data["gae"], atol=1e-10, rtol=0)
            if lam == 1:
                by_lambda[str(lam)]["telescoping_max_error"] = float(np.max(np.abs(
                    merged - (data["mc"] - data["values"]))))
                assert np.allclose(merged, data["mc"] - data["values"], atol=1e-10, rtol=0)
        continuity = max(float(np.max(np.abs(data["next_values"][part][:-1] - data["values"][part][1:])))
                         for _, part in scenarios)
        output[label] = {"source": str(path), "source_sha256": digest(path),
                         "trained_steps": metadata["trained_steps"], "gamma": gamma,
                         "max_within_episode_value_discontinuity": continuity, "lambda_variants": by_lambda}
    return output


def paired_counterfactual(path):
    source = json.loads(path.read_text())
    gamma = source["gamma"]
    assert len({case["prefix_observation_sha256"] for case in source["cases"]}) == 1
    for case in source["cases"]:
        baseline = next(row for row in source["cases"]
                        if row["continuation_seed"] == case["continuation_seed"] and row["direction"] == "baseline")
        assert case["uniform_tape_sha256"] == baseline["uniform_tape_sha256"]
        assert case["handoff"]["value"] == baseline["handoff"]["value"]
    results, maximum_td_error = [], 0.
    for case in source["cases"]:
        rows = case["trace"]
        rewards = np.array([row["scaled_reward"] for row in rows])
        following = np.array([row["value_after"] for row in rows])
        values = np.r_[case["handoff"]["value"], following[:-1]]
        residuals = rewards + gamma * following - values
        maximum_td_error = max(maximum_td_error, float(np.max(np.abs(
            residuals - np.array([row["td"] for row in rows])))))
        horizons = {}
        for seconds in (.5, 1., 2.):
            n = round(seconds / rows[0]["time_s"])
            discounted = float(np.dot(gamma ** np.arange(n), rewards[:n]))
            bootstrap = float(gamma ** n * following[n - 1])
            gae = {str(lam): float(np.dot((gamma * lam) ** np.arange(n), residuals[:n]))
                   for lam in LAMBDAS}
            decomposition = {str(lam): {
                "weighted_real_rewards": float(np.dot((gamma * lam) ** np.arange(n), rewards[:n])),
                "intermediate_values": float(gamma * (1 - lam) * np.dot(
                    (gamma * lam) ** np.arange(n - 1), values[1:n])),
                "terminal_value": float(gamma * (gamma * lam) ** (n - 1) * following[n - 1]),
                "negative_initial_value": float(-values[0])} for lam in LAMBDAS}
            for lam in LAMBDAS:
                assert abs(sum(decomposition[str(lam)].values()) - gae[str(lam)]) < 1e-10
            exact = discounted + bootstrap - values[0]
            assert abs(gae["1.0"] - exact) < 1e-10
            horizons[str(seconds)] = {"discounted_actual_reward": discounted, "discounted_bootstrap": bootstrap,
                "bootstrapped_advantage": exact, "finite_gae": gae, "gae_decomposition": decomposition,
                "final_angle_deg": rows[n - 1]["angle_deg"]}
        results.append({"seed": case["continuation_seed"], "direction": case["direction"],
                        "horizons": horizons})
    paired = []
    for case in results:
        if case["direction"] == "baseline":
            continue
        baseline = next(row for row in results if row["seed"] == case["seed"] and row["direction"] == "baseline")
        horizons = {}
        for horizon, row in case["horizons"].items():
            base = baseline["horizons"][horizon]
            horizons[horizon] = {key: row[key] - base[key] for key in (
                "discounted_actual_reward", "discounted_bootstrap", "bootstrapped_advantage", "final_angle_deg")}
            horizons[horizon]["finite_gae"] = {key: value - base["finite_gae"][key]
                                               for key, value in row["finite_gae"].items()}
            horizons[horizon]["gae_decomposition"] = {
                key: {component: value - base["gae_decomposition"][key][component]
                      for component, value in terms.items()}
                for key, terms in row["gae_decomposition"].items()}
        paired.append({"seed": case["seed"], "direction": case["direction"], "differences": horizons})
    counts = {}
    for horizon in ("0.5", "1.0", "2.0"):
        counts[horizon] = {str(lam): {"opposite_actual_reward_rank_count": sum(
            row["differences"][horizon]["finite_gae"][str(lam)] * row["differences"][horizon]["discounted_actual_reward"] < 0
            for row in paired), "paired_comparisons": len(paired)} for lam in LAMBDAS}
    return {"source": str(path), "source_sha256": digest(path), "gamma": gamma,
            "max_stored_td_error": maximum_td_error, "paired_differences": paired,
            "ranking_counts": counts,
            "decomposition_identity": "GAE(lambda)=-V_0 + sum((gamma*lambda)^i*r_i) + gamma*(1-lambda)*sum_{k=1}^{H-1}((gamma*lambda)^(k-1)*V_k) + gamma*(gamma*lambda)^(H-1)*V_H",
            "identity": "sum(gamma^i * delta_i) = sum(gamma^i * r_i) + gamma^H V_H - V_0; lambda=1 agreement with bootstrapped return is algebra, not independent evidence of a learning fix"}


def reconstructed_training_buffers():
    output = {}
    for name in ("small", "wide"):
        path = Path(f"runs/d1_training_records_audit_{name}_first.npz")
        if not path.exists():
            continue
        data = np.load(path)
        gamma, lam = .99, .95
        continues = 1 - data["episode_starts"][1:]
        expected = (data["buffer_rewards_scaled"][:-1].astype(float)
                    + gamma * data["values"][1:] * continues - data["values"][:-1]
                    + gamma * lam * data["gae"][1:] * continues)
        error = float(np.max(np.abs(expected - data["gae"][:-1])))
        assert error < 1e-5
        assert np.array_equal(data["episode_starts"][1:] != 0, data["dones"][:-1])
        difference = data["buffer_rewards_scaled"] - data["actual_rewards_scaled"]
        assert np.all(difference[~data["timeouts"]] == 0)
        targets = data["targets_deg"]
        original = data["gae"]
        global_labels = normalized(original)
        permutation = np.random.default_rng(0).permutation(len(original))
        minibatch_labels = np.empty_like(original)
        for offset in range(0, len(original), 64):
            indices = permutation[offset:offset + 64]
            minibatch_labels[indices] = normalized(original[indices])
        groups = {}
        for label, mask in (("target_15", np.isclose(targets, 15)), ("target_above_15", targets > 15.1)):
            groups[label] = {"steps": int(mask.sum()), "actual_positive_reward_steps": int(np.sum(data["actual_rewards_scaled"][mask] > 0)),
                "raw_positive_advantage_fraction": float(np.mean(original[mask] > 0)),
                "global_normalized_positive_fraction": float(np.mean(global_labels[mask] > 0)),
                "illustrative_shuffled_64_positive_fraction": float(np.mean(minibatch_labels[mask] > 0))}
        output[name] = {"source": str(path), "source_sha256": digest(path),
            "interior_recurrences_checked": len(expected), "max_interior_recurrence_error": error,
            "timeout_indices": np.flatnonzero(data["timeouts"]).tolist(),
            "stored_reward_adjustments_at_timeouts": difference[data["timeouts"]].tolist(),
            "raw_reward_unchanged_on_other_steps": True, "episode_start_flags_match_previous_done": True,
            "last_timestep_limitation": "Last observation/value was not retained, so final bootstrap is not independently rechecked from this file; separate collector fixture verifies that path",
            "label_groups": groups}
    return output


class BoundaryFixture(gym.Env):
    """Two-step terminal, two-step timeout, then a continuing episode."""
    def __init__(self, observation_space, action_space):
        self.observation_space, self.action_space = observation_space, action_space
        self.episode, self.local_step = -1, 0
        self.events = []

    def observation(self):
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[0] = self.episode + self.local_step / 10
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.local_step = 0
        return self.observation(), {}

    def step(self, action):
        self.local_step += 1
        terminated = self.episode == 0 and self.local_step == 2
        truncated = self.episode == 1 and self.local_step == 2
        raw_reward = float(len(self.events) + 1)
        observation = self.observation()
        self.events.append({"episode": self.episode, "raw_reward": raw_reward,
                            "terminated": terminated, "truncated": truncated, "following": observation.copy()})
        return observation, raw_reward, terminated, truncated, {}


def check_collector(checkpoint):
    model = PPO.load(checkpoint, device="cpu")
    before = weights_digest(model.policy)
    fixture = BoundaryFixture(model.observation_space, model.action_space)
    env = DummyVecEnv([lambda: Monitor(RewardScale(fixture))])
    model.set_env(env)
    model._last_obs = env.reset()
    model._last_episode_starts = np.ones(1, dtype=bool)
    model.ep_info_buffer, model.ep_success_buffer = deque(maxlen=100), deque(maxlen=100)
    class Callback(BaseCallback):
        def _on_step(self):
            return True
    callback = Callback()
    callback.init_callback(model)
    buffer = RolloutBuffer(6, model.observation_space, model.action_space,
                          device="cpu", gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
    try:
        model.collect_rollouts(env, callback, buffer, n_rollout_steps=6)
        expected, tail = np.zeros(6), 0.
        details = []
        for index in reversed(range(6)):
            row = fixture.events[index]
            with torch.no_grad():
                following_value = float(model.policy.predict_values(torch.as_tensor(row["following"][None]))[0, 0])
            bootstrap = 0. if row["terminated"] else following_value
            residual = row["raw_reward"] * .01 + model.gamma * bootstrap - float(buffer.values[index, 0])
            continues = not (row["terminated"] or row["truncated"])
            tail = residual + model.gamma * model.gae_lambda * tail * continues
            expected[index] = tail
            adjusted_reward = row["raw_reward"] * .01 + (model.gamma * following_value if row["truncated"] else 0.)
            details.append({"step": index, "episode": row["episode"], "terminated": row["terminated"],
                "truncated": row["truncated"], "raw_reward": row["raw_reward"],
                "expected_stored_reward": adjusted_reward, "actual_stored_reward": float(buffer.rewards[index, 0]),
                "expected_advantage": tail, "actual_advantage": float(buffer.advantages[index, 0])})
            assert np.isclose(buffer.rewards[index, 0], adjusted_reward, atol=1e-6)
        error = float(np.max(np.abs(expected - buffer.advantages[:, 0])))
        assert error < 1e-5
        assert before == weights_digest(model.policy)
        return {"fixture": "Real SB3 collect_rollouts with existing policy, 6 synthetic steps; no muscle physics, training or optimizer updates",
            "checkpoint": str(checkpoint), "weights_unchanged": True, "max_advantage_error": error,
            "loaded_settings": {"gamma": model.gamma, "gae_lambda": model.gae_lambda,
                "training_rollout_steps": model.n_steps, "training_batch_size": model.batch_size,
                "normalize_advantage": model.normalize_advantage},
            "episode_starts": buffer.episode_starts[:, 0].tolist(), "transitions": list(reversed(details))}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/d1_gae_target_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [Path("src/myo/train_bounded.py"), Path("src/myo/compact_protocol.py"),
               Path(inspect.getfile(OnPolicyAlgorithm)), Path(inspect.getfile(RolloutBuffer)), Path(inspect.getfile(PPO))]
    report = {"script_sha256": digest(__file__), "source_sha256": {str(path): digest(path) for path in sources},
        "limitations": ["No learning, optimizer updates or new weights.",
            "Lambda comparisons reuse identical frozen-policy trajectories and values; they do not test a trained lambda variant.",
            "Historical NPZ splits follow complete scenario boundaries, not the original 1472-step training-buffer boundaries.",
            "The bootstrapped return uses the same diagnosed critic at the final horizon and one stochastic trajectory per exercise.",
            "Global/within-scenario/illustrative minibatch labels are descriptive; only per-minibatch normalization is used by the training runner.",
            "Paired counterfactuals cover one state and three continuation tapes, not statistical confidence or broad policy quality."],
        "training_path": {"reward_scale": .01, "batch_size": 64, "advantage_normalization": "Each shuffled PPO minibatch, sample standard deviation",
            "command_alignment": "CompactProtocol scores the action with old command; only then reveals next command in returned observation",
            "timeout": "SB3 adds gamma*terminal_value to stored reward, then resets GAE propagation across the episode boundary",
            "termination": "No value bootstrap; GAE propagation stops",
            "rollout_cutoff": "Bootstrap final nonterminal observation; reset GAE tail at buffer end"},
        "collector_boundary_check": check_collector(Path("runs/d1_architecture_small_tanh/block_004.zip")),
        "saved_trajectories": saved_trajectories(Path("runs/d1_critic_signal_audit.json")),
        "reconstructed_training_buffers": reconstructed_training_buffers(),
        "paired_counterfactual": paired_counterfactual(Path("runs/d1_critic_stochastic_counterfactual_audit.json"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "boundary_max_error": report["collector_boundary_check"]["max_advantage_error"],
                      "counterfactual_rankings_2s": report["paired_counterfactual"]["ranking_counts"]["2.0"]}))


if __name__ == "__main__":
    main()
