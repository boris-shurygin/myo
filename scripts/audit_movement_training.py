"""Audit actual movement-curriculum buffers and checkpoints without learning.

No new rollouts, optimizer steps or weight files. Finite physical returns and
lambda=1 returns are comparisons, not independent estimates of true advantage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.runtime import dynamics_digest, load_controller


DIRECTORIES = (Path("runs/d1_movement_curriculum"),
               Path("runs/d1_movement_curriculum_continued"))
OUTPUT = Path("runs/d1_movement_training_audit.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())} if len(a) else {"n": 0}


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-12 else None


def ev(values, targets):
    return float(1 - np.var(targets - values) / np.var(targets)) if np.var(targets) > 1e-12 else None


def recalculate(data, gamma, lam):
    rewards = data["buffer_rewards"].ravel().astype(float)
    values = data["values"].ravel().astype(float)
    dones = data["physical_samples"][:, 6].astype(bool)
    following = np.r_[values[1:], data["last_values"].item()]
    # SB3 already folds gamma*V(terminal_observation) into timeout rewards.
    residual = rewards + gamma * following * ~dones - values
    result, tail = np.empty_like(residual, dtype=float), 0.
    for i in reversed(range(len(result))):
        tail = residual[i] + gamma * lam * (not dones[i]) * tail
        result[i] = tail
    return result


def label_groups(q, speed, progress):
    return {"all": np.ones(len(q), dtype=bool), "below25": q < 25,
            "upward_below_goal": (q < 27) & (speed > 5),
            "near_goal_fast": (abs(q - 30) <= 5) & (abs(speed) > 5),
            "settled": (abs(q - 30) <= 3) & (abs(speed) <= 5),
            "overshoot_above40": q > 40,
            "toward_goal_gt_point1deg": progress > .1,
            "away_from_goal_gt_point1deg": progress < -.1}


def expected_normalized_positive(advantage):
    # Illustrative random minibatches, not recovered historical PPO permutations.
    rng, counts = np.random.default_rng(0), np.zeros(len(advantage))
    for _ in range(64):
        permutation = rng.permutation(len(advantage))
        for start in range(0, len(advantage), 64):
            indices = permutation[start:start + 64]
            counts[indices] += advantage[indices] > advantage[indices].mean()
    return counts / 64


def inspect_run(directory):
    summary = json.loads((directory / "summary.json").read_text())
    initial = json.loads((directory / "stage_0_initial.zip.metadata.json").read_text())
    gamma, lam = initial["gamma"], initial["gae_lambda"]
    arrays, rows, cohort, provenance = [], [], [], []
    for rollout_index, path in enumerate(sorted(directory.glob("rollout_*.npz"))):
        with np.load(path) as source:
            d = {k: source[k] for k in source.files}
        assert all(np.isfinite(a).all() for a in d.values())
        obs, physical = d["observations"][:, 0], d["physical_samples"]
        assert np.array_equal(d["episode_starts"][1:, 0].astype(bool), physical[:-1, 6].astype(bool))
        assert bool(d["last_dones"].item()) == bool(physical[-1, 6])
        q = np.rad2deg(obs[:, 0] + initial["config"]["reference_center"])
        speed = np.rad2deg(obs[:, 1] * 4)
        progress = abs(q - 30) - abs(np.rad2deg(physical[:, 0]) - 30)
        original, full = d["advantages"].ravel(), recalculate(d, gamma, 1.)
        error = float(np.max(abs(recalculate(d, gamma, lam) - original)))
        assert error < 2e-6
        assert np.max(abs(d["returns"].ravel() - (original + d["values"].ravel()))) < 2e-6
        normalized_probability = expected_normalized_positive(original)
        row = {"rollout": rollout_index + 1, "checkpoint_steps": int(path.stem.split("_")[-1]),
               "gae_max_error": error, "raw_reward": stats(physical[:, 3]),
               "advantages": stats(original), "values": stats(d["values"].ravel()),
               "value_ev_gae_target": ev(d["values"].ravel(), d["returns"].ravel()),
               "value_ev_lambda1_target": ev(d["values"].ravel(), full + d["values"].ravel()),
               "gae_lambda1_correlation": correlation(original, full),
               "global_centered_gae_lambda1_sign_disagreement": float(np.mean(
                   (original > original.mean()) != (full > full.mean()))),
               "groups": {}}
        for name, mask in label_groups(q, speed, progress).items():
            row["groups"][name] = {"n": int(mask.sum())}
            if mask.any():
                row["groups"][name].update(
                    raw_reward=float(physical[mask, 3].mean()),
                    raw_gae=float(original[mask].mean()),
                    centered_gae=float((original[mask] - original.mean()).mean()),
                    raw_gae_positive_fraction=float(np.mean(original[mask] > 0)),
                    illustrative_minibatch_positive_probability=float(normalized_probability[mask].mean()),
                    lambda1_advantage=float(full[mask].mean()))
        rows.append(row)
        arrays.append({"observations": obs, "physical": physical,
                       "actions": d["actions"][:, 0], "gae": original,
                       "lambda1": full, "values": d["values"].ravel(),
                       "centered_gae": original - original.mean(),
                       "normalized_positive_probability": normalized_probability})
        cohort.append(obs[np.linspace(0, len(obs) - 1, 48, dtype=int)])
        provenance.append({"path": str(path), "sha256": digest(path)})
    merged = {key: np.concatenate([d[key] for d in arrays]) for key in arrays[0]}
    physical = merged["physical"]
    starts = np.r_[0, np.flatnonzero(physical[:-1, 6]) + 1]
    ends = np.r_[np.flatnonzero(physical[:, 6]) + 1, len(physical)]
    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    episodes = []
    for eid, (lo, hi) in enumerate(zip(starts, ends)):
        if hi == lo:
            continue
        p = physical[lo:hi]
        q, speed = np.rad2deg(p[:, 0]), np.rad2deg(p[:, 1])
        finite_return, tail = np.zeros(len(p)), 0.
        for i in reversed(range(len(p))):
            tail = .01 * p[i, 3] + gamma * tail
            finite_return[i] = tail
        settled = (abs(q - 30) <= 3) & (abs(speed) <= 5)
        boundaries = np.flatnonzero(np.diff(np.r_[False, settled, False]))
        longest = max((b - a for a, b in zip(boundaries[::2], boundaries[1::2])), default=0)
        complete = bool(p[-1, 6])
        assert not complete or hi - lo == 200
        trigger = merged["observations"][lo, initial["observation_names"].index("trigger")]
        episode = {"episode": eid, "start_index": int(lo), "stop_index": int(hi),
                   "complete": complete, "trigger": float(trigger),
                   "capacity_at_reset": float(merged["observations"][lo, 8]),
                   "q_max": float(q.max()), "q_final": float(q[-1]),
                   "raw_reward_mean": float(p[:, 3].mean()),
                   "discounted_4s_physical_return_terminal_zero": float(finite_return[0]),
                   "critic_at_start": float(merged["values"][lo]),
                   "raw_gae_mean": float(merged["gae"][lo:hi].mean()),
                   "centered_gae_mean": float(merged["centered_gae"][lo:hi].mean()),
                   "illustrative_minibatch_positive_probability": float(
                       merged["normalized_positive_probability"][lo:hi].mean()),
                   "longest_settled_s": longest * .02, "settled_steps": int(sum(settled))}
        episodes.append(episode)
    conditions = []
    for capacity in (1., .8):
        for trigger in (.3, .6, 1.):
            selected = [e for e in episodes if e["complete"]
                        and abs(e["capacity_at_reset"] - capacity) < 1e-6
                        and abs(e["trigger"] - trigger) < 1e-6]
            conditions.append({"trigger": trigger, "capacity": capacity, "episodes": len(selected),
                               "overshoot_above40": sum(e["q_max"] > 40 for e in selected),
                               "overshoot_above100": sum(e["q_max"] > 100 for e in selected),
                               "with_settled_sample": sum(e["settled_steps"] > 0 for e in selected)})
    return {"summary_status": summary["status"], "actual_steps": len(physical),
            "rollouts": rows, "episodes": episodes, "conditions": conditions,
            "provenance": provenance}, merged, np.concatenate(cohort)


def parameter_vector(model, prefix):
    return np.concatenate([p.detach().cpu().numpy().ravel()
                           for n, p in model.policy.named_parameters() if n.startswith(prefix)])


def checkpoint_audit(cohort):
    paths = [DIRECTORIES[0] / "stage_0_initial.zip"]
    paths += [DIRECTORIES[0] / f"stage_0_block_{b:03d}.zip" for b in (8, 12, 13, 14, 16)]
    paths += [DIRECTORIES[1] / f"stage_0_block_{b:03d}.zip" for b in (3, 6)]
    baseline, output = None, []
    for path in paths:
        env, _, metadata = load_controller(str(path))
        env.close()
        model = PPO.load(path, device="cpu")
        assert digest(path) == metadata["checkpoint_sha256"]
        vectors = {key: parameter_vector(model, prefix) for key, prefix in
                   (("actor", "mlp_extractor.policy_net"), ("critic", "mlp_extractor.value_net"),
                    ("action_head", "action_net"), ("value_head", "value_net"))}
        if baseline is None:
            baseline = vectors
            normalization = model.policy.features_extractor.state_dict()
        assert all(torch.equal(v, model.policy.features_extractor.state_dict()[k])
                   for k, v in normalization.items())
        with torch.no_grad():
            obs = torch.as_tensor(cohort)
            features = model.policy.extract_features(obs)
            pi, vf = model.policy.mlp_extractor(features)
            distribution = model.policy.get_distribution(obs).distribution
        latent = pi.numpy()
        covariance = np.cov(latent, rowvar=False)
        rank = np.trace(covariance) ** 2 / np.square(covariance).sum()
        row = {"path": str(path), "sha256": digest(path), "steps": model.num_timesteps,
               "weights": {k: {"norm": float(np.linalg.norm(v)),
                               "relative_change_from_initial": float(np.linalg.norm(v - baseline[k]) / np.linalg.norm(baseline[k]))}
                           for k, v in vectors.items()},
               "actor_tanh_saturated_fraction": float(np.mean(abs(latent) > .99)),
               "critic_tanh_saturated_fraction": float(np.mean(abs(vf.numpy()) > .99)),
               "actor_centered_participation_rank": float(rank),
               "mean_commands_on_fixed_cohort": distribution.mean.mean(0).tolist(),
               "median_std_on_fixed_cohort": float(distribution.stddev.median()),
               "fixed_cohort_command_std_by_muscle": distribution.mean.std(0).tolist(),
               "normalized_abs_input_p99": float(np.quantile(abs(features.numpy()), .99)),
               "ent_coef": model.ent_coef, "vf_coef": model.vf_coef,
               "gamma": model.gamma, "gae_lambda": model.gae_lambda,
               "actor_critic_trainable_weights_shared": bool(
                   {id(p) for p in model.policy.mlp_extractor.policy_net.parameters()} &
                   {id(p) for p in model.policy.mlp_extractor.value_net.parameters()})}
        output.append(row)
    return output


def updated_action_likelihoods(directory, data):
    rows = []
    summary = json.loads((directory / "summary.json").read_text())
    for e in summary["stages"][0]["evaluations"]:
        model = PPO.load(e["controller"], device="cpu")
        end = e["stage_steps"]
        start = end - 2944
        local = []
        files = sorted(directory.glob("rollout_*.npz"))[(e["block"] - 1) * 2:e["block"] * 2]
        for path in files:
            with np.load(path) as d:
                obs = torch.as_tensor(d["observations"][:, 0])
                with torch.no_grad():
                    _, log_prob, _ = model.policy.evaluate_actions(obs, torch.as_tensor(d["actions"][:, 0]))
                local.append(log_prob.numpy() - d["log_probs"].ravel())
        difference = np.concatenate(local)
        centered = data["centered_gae"][start:end]
        rows.append({"block": e["block"], "delta_logprob": stats(difference),
                     "correlation_delta_logprob_with_centered_gae": correlation(difference, centered),
                     "positive_centered_gae_delta_logprob": float(difference[centered > 0].mean()),
                     "negative_centered_gae_delta_logprob": float(difference[centered < 0].mean())})
    return rows


def main():
    torch.set_num_threads(1)
    physical_before = dynamics_digest()
    report = {"script_sha256": digest(__file__), "dynamics_sha256": physical_before,
              "runs": {}, "limitations": [
                  "Saved buffers, not newly sampled probes; no optimizer updates.",
                  "Groups describe temporal associations, not isolated action causality.",
                  "Minibatch sign probabilities use 64 diagnostic permutations, not historical minibatches.",
                  "Lambda1 uses original timeout/final bootstrap; it is not a critic-independent target.",
                  "Finite physical episode return ends at4s with zero; PPO trained with timeout bootstrap.",
                  "Post-block logprobs include two PPO updates and are not per-minibatch loss or entropy changes.",
                  "Tanh/rank measured on the same diagnostic state cohort, not uniform state coverage."]}
    cohorts = []
    for directory in DIRECTORIES:
        row, data, cohort = inspect_run(directory)
        row["post_block_action_likelihoods"] = updated_action_likelihoods(directory, data)
        report["runs"][directory.name] = row
        cohorts.append(cohort)
        print(directory.name, "actual steps", row["actual_steps"], "conditions", row["conditions"], flush=True)
    report["checkpoints"] = checkpoint_audit(np.concatenate(cohorts))
    report["gae_reward_weight_by_delay_s"] = {str(t): (.99 * .95) ** round(t / .02)
                                            for t in (.04, .1, .2, .5, 1., 2.)}
    assert physical_before == dynamics_digest()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
