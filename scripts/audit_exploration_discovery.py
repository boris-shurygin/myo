"""Frozen-checkpoint exploration audit; these probes are not historical training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.compact_protocol import CompactProtocol
from myo.runtime import dynamics_digest, load_controller


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_digest(policy):
    result = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def normalize(values):
    return (values - values.mean()) / (values.std(ddof=1) + 1e-8)


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def episode_arrays(data):
    lengths = (1450, 500, 500)
    data["episodes"] = np.repeat(np.arange(3), lengths)
    data["times_s"] = np.concatenate([(np.arange(n) + 1) * .02 for n in lengths])
    data["episode_starts"] = np.r_[True, np.diff(data["episodes"]) != 0]
    data["episode_ends"] = np.r_[np.diff(data["episodes"]) != 0, True]
    data["gae_normalized_all_probes"] = normalize(data["gae"])
    data["gae_normalized_episode"] = np.concatenate([
        normalize(data["gae"][data["episodes"] == ep]) for ep in range(3)])
    return data


def reused_arrays(path):
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    data = episode_arrays(data)
    # Saved observation[t+1] contains the physical post-step velocity of t.
    # The last observation of each episode was not saved; do not invent it.
    velocity = np.full(len(data["rewards"]), np.nan)
    valid = ~data["episode_ends"]
    velocity[valid] = np.rad2deg(data["observations"][np.flatnonzero(valid) + 1, 1] * 4)
    data["velocities_deg_s"] = velocity
    data["velocity_known"] = np.isfinite(velocity)
    error = data["angles_deg"] - data["targets_deg"]
    data["settled_bonus"] = np.exp(-(error / 3) ** 2 - (velocity / 5) ** 2)
    return data


def collect(checkpoint):
    checkpoint_before = digest(checkpoint)
    physical, _, _ = load_controller(str(checkpoint))
    env = CompactProtocol(physical)
    model = PPO.load(checkpoint, device="cpu")
    model.policy.set_training_mode(False)
    weights_before = weights_digest(model.policy)
    torch.manual_seed(0)
    env.next_scenario = 6
    episodes = []
    try:
        for _ in range(3):
            observation, _ = env.reset(seed=0)
            rows = []
            while True:
                with torch.no_grad():
                    inputs = torch.as_tensor(observation[None])
                    distribution = model.policy.get_distribution(inputs).distribution
                    action = distribution.sample()[0].numpy()
                    mean = distribution.mean[0].numpy()
                    std = distribution.stddev[0].numpy()
                    value = float(model.policy.predict_values(inputs)[0, 0])
                following, reward, terminated, truncated, info = env.step(action)
                with torch.no_grad():
                    next_value = 0.0 if terminated else float(model.policy.predict_values(
                        torch.as_tensor(following[None]))[0, 0])
                rows.append({"observations": observation.copy(), "actions": action,
                    "means": mean, "stds": std, "values": value, "next_values": next_value,
                    "rewards": .01 * reward, "angles_deg": float(np.rad2deg(info["q"])),
                    "targets_deg": float(np.rad2deg(info["requested_target_q"])),
                    "velocities_deg_s": float(np.rad2deg(info["qvel"])),
                    "settled_bonus": info["reward_components"]["settled_bonus"]})
                observation = following
                if terminated or truncated:
                    break
            data = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            data["gae"] = np.zeros(len(rows))
            data["mc"] = np.zeros(len(rows))
            gae_tail, mc_tail = 0.0, data["next_values"][-1]
            for index in reversed(range(len(rows))):
                td = data["rewards"][index] + model.gamma * data["next_values"][index] - data["values"][index]
                gae_tail = td + model.gamma * model.gae_lambda * gae_tail
                mc_tail = data["rewards"][index] + model.gamma * mc_tail
                data["gae"][index], data["mc"][index] = gae_tail, mc_tail
            episodes.append(data)
            print(checkpoint.parent.name, checkpoint.name, env.scenario.name,
                  "max angle", float(data["angles_deg"].max()), flush=True)
        merged = {key: np.concatenate([ep[key] for ep in episodes]) for key in episodes[0]}
        assert len(merged["rewards"]) == 2450
        merged["velocity_known"] = np.ones(2450, dtype=bool)
        assert weights_before == weights_digest(model.policy)
        assert checkpoint_before == digest(checkpoint)
        return episode_arrays(merged)
    finally:
        env.close()


def summarize(data):
    results, candidates = [], []
    for ep, name in enumerate(("sweep_u1_c1", "raise_u1_c1", "lower_u1_c1")):
        indices = np.flatnonzero(data["episodes"] == ep)
        q, target, time = (data[key][indices] for key in ("angles_deg", "targets_deg", "times_s"))
        reward = data["rewards"][indices] / .01
        eligible = (time > .5 + 1e-8) & (target > 15.01)
        goal_change = np.r_[0, np.flatnonzero(np.abs(np.diff(target)) > 1e-6) + 1, len(indices)]
        thresholds = {str(amount): {"episodes": 0, "duration_s": 0.0,
                                   "qualifying_windows": []} for amount in (1, 3, 5)}
        sustained_gains = {str(amount): {"windows": 0, "duration_s": 0.0,
                                        "qualifying_windows": []} for amount in (1, 3, 5)}
        settled_windows = []
        for start, stop in zip(goal_change[:-1], goal_change[1:]):
            valid = np.flatnonzero(eligible[start:stop]) + start
            if not len(valid):
                continue
            first = int(valid[0])
            # Starts at a local minimum within the SAME requested-goal window.
            minimum = first
            best = None
            for peak in range(first + 1, stop):
                if q[peak] < q[minimum]:
                    minimum = peak
                progress = q[peak] - q[minimum]
                if target[peak] > q[minimum] and (best is None or progress > best[0]):
                    best = (float(progress), minimum, peak)
            if best is not None:
                progress, low, high = best
                global_low, global_high = int(indices[low]), int(indices[high])
                prior = q[max(0, low - 25):low + 1]
                rebound = bool(len(prior) and float(prior.max() - q[low]) > 10)
                candidate = {"scenario": name, "goal_deg": float(target[high]),
                    "minimum_index": global_low, "peak_index": global_high,
                    "minimum_time_s": float(time[low]), "peak_time_s": float(time[high]),
                    "upward_progress_deg": progress, "minimum_angle_deg": float(q[low]),
                    "peak_angle_deg": float(q[high]),
                    "source_classification": "recent_fall_rebound" if rebound else "rise_during_fixed_goal",
                    "raw_reward_at_minimum": float(reward[low]), "raw_reward_at_peak": float(reward[high]),
                    "raw_reward_change": float(reward[high] - reward[low])}
                candidates.append(candidate)
            for amount in (1, 3, 5):
                above = (q[first:stop] >= 15 + amount) & (q[first:stop] <= target[first:stop] + 3)
                for lo, hi in spans(above):
                    if hi - lo >= 5:
                        thresholds[str(amount)]["episodes"] += 1
                        thresholds[str(amount)]["duration_s"] += (hi - lo) * .02
                        thresholds[str(amount)]["qualifying_windows"].append([
                            float(time[first + lo] - .02), float(time[first + hi - 1])])
                minimum_so_far = np.minimum.accumulate(q[first:stop])
                progress_mask = ((q[first:stop] - minimum_so_far >= amount)
                                 & (target[first:stop] > minimum_so_far)
                                 & (q[first:stop] <= target[first:stop] + 3))
                for lo, hi in spans(progress_mask):
                    if hi - lo >= 5:
                        sustained_gains[str(amount)]["windows"] += 1
                        sustained_gains[str(amount)]["duration_s"] += (hi - lo) * .02
                        sustained_gains[str(amount)]["qualifying_windows"].append([
                            float(time[first + lo] - .02), float(time[first + hi - 1])])
            settled = (np.abs(q[first:stop] - target[first:stop]) <= 3) & (
                np.abs(data["velocities_deg_s"][indices[first:stop]]) <= 5)
            for lo, hi in spans(settled):
                if hi - lo >= 15:
                    settled_windows.append({"start_s": float(time[first + lo] - .02),
                                            "duration_s": (hi - lo) * .02,
                                            "goal_deg": float(target[first + lo])})
        known = eligible & data["velocity_known"][indices]
        local_candidates = [c for c in candidates if c["scenario"] == name]
        results.append({"scenario": name, "steps": len(indices), "eligible_steps": int(eligible.sum()),
            "excluded": "goal15 and first0.5s of reset only; goal changes remain but windows do not cross them",
            "maximum_angle_deg_all": float(q.max()),
            "maximum_angle_deg_eligible": float(q[eligible].max()),
            "positive_raw_reward_count_eligible": int(np.sum(eligible & (reward > 0))),
            "eligible_mean_raw_reward": float(reward[eligible].mean()),
            "eligible_max_settled_bonus": float(np.max(data["settled_bonus"][indices][known])),
            "settled_windows_at_least_0_3s": settled_windows,
            "above_lower_limit_sustained_at_least_0_1s": thresholds,
            "upward_progress_from_window_minimum_sustained_at_least_0_1s": sustained_gains,
            "upward_progress_window_counts": {str(t): sum(c["upward_progress_deg"] >= t for c in local_candidates)
                                               for t in (1, 3, 5)},
            "best_fixed_goal_progress": max(local_candidates, key=lambda c: c["upward_progress_deg"], default=None)})
    best = max(candidates, key=lambda c: c["upward_progress_deg"])
    ep_indices = np.flatnonzero(data["episodes"] == data["episodes"][best["peak_index"]])
    start = max(int(ep_indices[0]), best["minimum_index"] - 15)
    stop = min(int(ep_indices[-1]) + 1, best["peak_index"] + 16)
    selected = np.arange(start, stop)
    best["causal_window_indices"] = selected.tolist()
    best["causal_window"] = [{"index": int(i), "time_s": float(data["times_s"][i]),
        "angle_deg": float(data["angles_deg"][i]), "goal_deg": float(data["targets_deg"][i]),
        "raw_reward": float(data["rewards"][i] / .01), "action": data["actions"][i].tolist(),
        "policy_mean": data["means"][i].tolist(), "gae_raw_scaled": float(data["gae"][i]),
        "gae_normalized_probe_cohort": float(data["gae_normalized_all_probes"][i]),
        "gae_normalized_episode": float(data["gae_normalized_episode"][i])} for i in selected]
    best["causal_window_has_goal_change"] = bool(np.any(np.diff(data["targets_deg"][selected]) != 0))
    best["causal_window_note"] = "0.3s lead-in/aftermath; goal changes flagged, this is temporal association not isolated action causality"
    rise = np.arange(best["minimum_index"], best["peak_index"] + 1)
    best["fixed_goal_rise_summary"] = {"steps": len(rise), "indices": rise.tolist(),
        "mean_raw_reward": float(np.mean(data["rewards"][rise]) / .01),
        "gae": {key: {"mean": float(data[key][rise].mean()),
                       "positive_fraction": float(np.mean(data[key][rise] > 0))}
                for key in ("gae", "gae_normalized_all_probes", "gae_normalized_episode")}}
    return results, best


def compare_probabilities(directory, examples, datasets):
    comparison = {label: [] for label in examples}
    for filename in ("initial.zip", "block_001.zip", "block_002.zip", "block_003.zip", "block_004.zip"):
        path = directory / filename
        before = digest(path)
        model = PPO.load(path, device="cpu")
        before_weights = weights_digest(model.policy)
        for label, best in examples.items():
            data = datasets[label]
            indices = np.asarray(best["causal_window_indices"])
            with torch.no_grad():
                inputs = torch.as_tensor(data["observations"][indices])
                actions = torch.as_tensor(data["actions"][indices])
                logp = model.policy.get_distribution(inputs).log_prob(actions).numpy()
            comparison[label].append({"evaluated_checkpoint": str(path),
                "mean_log_probability": float(logp.mean()), "summed_log_probability": float(logp.sum()),
                "rise_mean_log_probability": float(logp[(indices >= best["minimum_index"])
                                                         & (indices <= best["peak_index"])].mean()),
                "log_probabilities": logp.tolist()})
        assert before_weights == weights_digest(model.policy)
        assert before == digest(path)
    for label, evaluations in comparison.items():
        source_filename = examples[label]["collecting_checkpoint_filename"]
        source = next(e for e in evaluations if Path(e["evaluated_checkpoint"]).name == source_filename)
        for row in evaluations:
            differences = np.asarray(row["log_probabilities"]) - np.asarray(source["log_probabilities"])
            row["mean_log_probability_change_from_collecting_policy"] = float(differences.mean())
            row["rise_mean_log_probability_change_from_collecting_policy"] = (
                row["rise_mean_log_probability"] - source["rise_mean_log_probability"])
            row["fraction_actions_more_probable_than_collecting_policy"] = float(np.mean(differences > 0))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/d1_exploration_discovery_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    report = {"dynamics_sha256": dynamics_digest(), "models": {}, "same_observation_action_probabilities": {},
        "limitations": ["Frozen-policy probes, not original training trajectories or an exact PPO update replay.",
            "One seed and three nominal scenarios per checkpoint; no additional seed grid.",
            "Initial/final probes reused from critic-signal audit, only intermediate1/2/3 collected now.",
            "Raw rewards may improve while remaining negative; no requirement of positive reward for useful progress.",
            "Normalized GAE uses full probe cohort or episode, not actual training minibatches.",
            "GAE is episode-wide with critic bootstrap and omits training1472-step rollout boundaries.",
            "Same-state action likelihood changes are off-policy comparisons, not evidence those actions occurred in training.",
            "Progress windows do not cross goal changes; lead-in causal windows flag any goal change.",
            "Excursions above15 may include passive-limit rebound; recent large falls are explicitly labeled."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in ("small", "wide"):
        directory = Path(f"runs/d1_architecture_{size}_tanh")
        datasets, examples = {}, {}
        for block in range(5):
            filename = "initial.zip" if block == 0 else f"block_{block:03}.zip"
            checkpoint = directory / filename
            label = f"{size}_{block:03}"
            output_npz = args.output.with_name(args.output.stem + "_" + label + ".npz")
            if block in (0, 4):
                source = Path(f"runs/d1_critic_signal_audit_{size}_{'initial' if block == 0 else 'final'}.npz")
                data = reused_arrays(source)
                provenance = {"kind": "reused frozen probe", "source": str(source), "sha256": digest(source)}
            else:
                data = collect(checkpoint)
                provenance = {"kind": "new frozen probe", "seed": 0}
            np.savez_compressed(output_npz, **data)
            scenarios, best = summarize(data)
            best["collecting_checkpoint_filename"] = filename
            report["models"][label] = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "probe": str(output_npz), "probe_sha256": digest(output_npz), "provenance": provenance,
                "weights_unchanged": True, "scenarios": scenarios, "best_spontaneous_excursion": best}
            datasets[label], examples[label] = data, best
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["same_observation_action_probabilities"][size] = compare_probabilities(directory, examples, datasets)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
