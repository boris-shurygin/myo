"""One actor-only epoch on copied policies: GAE versus completed-task returns.

This is a finite ten-second diagnostic objective, not resumed controller training.
Original physics, policy sources, optimizer snapshots and weight files stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.compact_protocol import CompactProtocol
from myo.runtime import dynamics_digest, load_controller


CHECKPOINT = Path("runs/d1_architecture_small_tanh/block_003.zip")
SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (10, 11, 12)
ACTOR_PREFIXES = ("mlp_extractor.policy_net", "action_net")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(policy, actor=None):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        if actor is not None and name.startswith(ACTOR_PREFIXES) != actor:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def normalized(values):
    return (values - values.mean()) / (values.std() + 1e-8)


def cosine(left, right):
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-20))


def collect(model, env, seed, deterministic=False):
    env.next_scenario = 7  # Nominal raise: 20 deg, then 75 deg after 0.5 s.
    observation, _ = env.reset(seed=0)
    rows = []
    model.policy.set_training_mode(False)
    for step in range(500):
        # Common random bits restart each step, so variable Beta rejection work
        # cannot desynchronize future steps. This is NOT common Beta quantiles.
        torch.manual_seed(100000 + seed * 1000 + step)
        with torch.no_grad():
            obs = torch.as_tensor(observation[None])
            distribution = model.policy.get_distribution(obs)
            action = distribution.mode() if deterministic else distribution.sample()
            log_prob = float(distribution.log_prob(action)[0])
            value = float(model.policy.predict_values(obs)[0, 0])
            means = distribution.distribution.mean[0].numpy().copy()
        following, reward, terminated, truncated, info = env.step(action[0].numpy())
        rows.append({"observations": observation.copy(), "actions": action[0].numpy().copy(),
                     "old_log_prob": log_prob, "values": value, "rewards": .01 * reward,
                     "angles_deg": float(np.rad2deg(info["q"])),
                     "velocity_deg_s": float(np.rad2deg(info["qvel"])),
                     "targets_deg": float(np.rad2deg(info["requested_target_q"])),
                     "means": means})
        observation = following
        if terminated:
            raise RuntimeError("Physical termination invalidates the planned complete diagnostic episode")
        if truncated:
            assert step == 499
            break
    assert len(rows) == 500 and truncated
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    following_values = np.r_[arrays["values"][1:], 0.0]
    gae_tail = mc_tail = 0.0
    arrays["gae"] = np.empty(500)
    arrays["mc_return"] = np.empty(500)
    for step in reversed(range(500)):
        delta = arrays["rewards"][step] + model.gamma * following_values[step] - arrays["values"][step]
        gae_tail = delta + model.gamma * model.gae_lambda * gae_tail
        mc_tail = arrays["rewards"][step] + model.gamma * mc_tail
        arrays["gae"][step] = gae_tail
        arrays["mc_return"][step] = mc_tail
    arrays["mc"] = arrays["mc_return"] - arrays["values"]
    # Algebraic boundary check: lambda=1 cancels all intermediate values.
    tail = 0.0
    for step in reversed(range(500)):
        tail = arrays["rewards"][step] + model.gamma * following_values[step] - arrays["values"][step] + model.gamma * tail
        assert abs(tail - arrays["mc"][step]) < 1e-10
    return arrays


def summary(data):
    q, target, velocity = (data[k] for k in ("angles_deg", "targets_deg", "velocity_deg_s"))
    eligible = np.arange(500) >= 25
    settled = eligible & (abs(q - target) <= 3) & (abs(velocity) <= 5)
    edges = np.diff(np.r_[False, settled, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {"steps": 500, "mean_raw_reward": float(data["rewards"].mean() / .01),
            "discounted_scaled_return": float(data["mc_return"][0]),
            "max_angle_deg_after_command": float(q[eligible].max()),
            "final_angle_deg": float(q[-1]),
            "settled_duration_s": float(settled.sum() * .02),
            "longest_settled_duration_s": float(max(lengths, default=0) * .02),
            "mean_excitation_by_muscle": data["actions"].mean(0).tolist()}


def best_progress_segment(data):
    # One predefined diagnostic selection per episode, after the goal change.
    # This is association, not a causal label for every action in the window.
    q = data["angles_deg"]
    low = 25
    best = (0.0, 25, 25)
    for high in range(26, 500):
        if q[high] < q[low]:
            low = high
        gain = min(q[high], 75) - q[low]
        if gain > best[0]:
            best = (float(gain), low, high)
    gain, low, high = best
    return {"progress_deg": gain, "angle_start_deg": float(q[low]),
            "angle_end_deg": float(q[high]), "start_s": (low + 1) * .02,
            "end_s": (high + 1) * .02,
            "indices": list(range(max(25, low - 5), high + 1)),
            "selection": "largest forward excursion within goal75; include preceding100ms for delayed actions"}


def actor_parameters(model):
    return [p for name, p in model.policy.named_parameters() if name.startswith(ACTOR_PREFIXES)]


def gradient(model, data, advantage, batches):
    model.policy.zero_grad(set_to_none=True)
    for indices in batches:
        obs = torch.as_tensor(data["observations"][indices])
        action = torch.as_tensor(data["actions"][indices])
        adv = normalized(torch.as_tensor(advantage[indices], dtype=torch.float32))
        loss = -(adv * model.policy.get_distribution(obs).log_prob(action)).sum() / len(advantage)
        loss.backward()
    result = torch.cat([p.grad.detach().flatten() for p in actor_parameters(model)]).clone()
    model.policy.zero_grad(set_to_none=True)
    return result


def log_probs(model, data):
    with torch.no_grad():
        return model.policy.get_distribution(torch.as_tensor(data["observations"])).log_prob(
            torch.as_tensor(data["actions"])).numpy().copy()


def update(model, data, advantage, batches):
    critic_before = tensor_digest(model.policy, actor=False)
    actor = actor_parameters(model)
    actor_before = torch.cat([p.detach().flatten() for p in actor]).clone()
    for name, parameter in model.policy.named_parameters():
        parameter.requires_grad_(name.startswith(ACTOR_PREFIXES))
    records = []
    model.policy.set_training_mode(True)
    clip_range = float(model.clip_range(model._current_progress_remaining))
    for indices in batches:
        inputs = torch.as_tensor(data["observations"][indices])
        actions = torch.as_tensor(data["actions"][indices])
        old_log = torch.as_tensor(data["old_log_prob"][indices], dtype=torch.float32)
        adv = normalized(torch.as_tensor(advantage[indices], dtype=torch.float32))
        current_log = model.policy.get_distribution(inputs).log_prob(actions)
        log_ratio = current_log - old_log
        ratio = log_ratio.exp()
        loss = -torch.minimum(adv * ratio, adv * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
        with torch.no_grad():
            kl = float(((ratio - 1) - log_ratio).mean())
        # Match the saved PPO KL rule, including checking before optimizer.step.
        if model.target_kl is not None and kl > 1.5 * model.target_kl:
            records.append({"early_stop": True, "kl": kl})
            break
        model.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(actor, model.max_grad_norm))
        model.policy.optimizer.step()
        records.append({"steps": len(indices), "loss": float(loss.detach()),
                        "kl": kl, "pre_clip_actor_gradient_norm": norm})
    model.policy.set_training_mode(False)
    assert tensor_digest(model.policy, actor=False) == critic_before
    change = torch.cat([p.detach().flatten() for p in actor]) - actor_before
    return {"minibatches": records, "optimizer_updates": sum("steps" in r for r in records),
            "actor_parameter_change_l2": float(change.norm()), "critic_and_buffers_unchanged": True,
            "actor_sha256": tensor_digest(model.policy, actor=True)}, change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/d1_mc_actor_update_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    torch.set_num_threads(1)
    files = [CHECKPOINT, Path(str(CHECKPOINT) + ".metadata.json")]
    originals = {str(p): file_digest(p) for p in files}
    physical_before = dynamics_digest()
    physical, _, _ = load_controller(str(CHECKPOINT))
    env = CompactProtocol(physical)
    models = {name: PPO.load(CHECKPOINT, device="cpu") for name in ("baseline", "gae", "mc")}
    original_weights = tensor_digest(models["baseline"].policy)
    assert all(tensor_digest(m.policy) == original_weights for m in models.values())
    report = {"checkpoint": str(CHECKPOINT), "checkpoint_files_sha256": originals,
              "dynamics_sha256": physical_before, "script_sha256": file_digest(__file__),
              "selection": "Small block003 chosen before collection because prior independent audit found a spontaneous rise",
              "objective": "Finite complete10s raise20to75 task: explicit terminal V=0 for both estimators",
              "gamma": .99, "gae_lambda": .95, "reward_scale": .01,
              "collection_seeds": list(SEEDS), "evaluation_seeds": list(EVALUATION_SEEDS),
              "collection": [], "updates": {}, "evaluation": {},
              "limitations": [
                  "At most one actor-only epoch; no model.learn, critic update, checkpoint save or long training.",
                  "Same original reward, commands, physical state dynamics and 1kg mass; only diagnostic terminal semantics differ.",
                  "Stored continuing-task critic is a frozen baseline and may poorly fit this finite horizon.",
                  "PPO minibatch advantage normalization, timestep averaging and Adam momentum retained; this is not an unbiased finite-horizon policy-gradient proof.",
                  "Common per-step RNG seeds give exact Beta samples; variable rejection paths mean they are not matched Beta quantiles.",
                  "Three fixed stochastic probes plus deterministic run are diagnostics, not statistical generalization evidence.",
                  "Progress-window likelihood is associative; delayed physical effects do not identify each action as causal.",
                  "Actor gradients are clipped alone because critic is frozen, so this is not the original coupled actor/critic update."]}
    try:
        episodes = []
        for seed in SEEDS:
            data = collect(models["baseline"], env, seed)
            episodes.append(data)
            report["collection"].append({"seed": seed, **summary(data), "progress_segment": best_progress_segment(data)})
            print("collected", seed, summary(data), flush=True)
        merged = {key: np.concatenate([d[key] for d in episodes]) for key in episodes[0]}
        npz = args.output.with_suffix(".npz")
        if npz.exists():
            raise FileExistsError(npz)
        np.savez_compressed(npz, **merged)
        report["fresh_on_policy_data"] = {"path": str(npz), "sha256": file_digest(npz), "steps": 1500}
        order = np.random.default_rng(20261103).permutation(1500)
        batches = [order[i:i + 64] for i in range(0, 1500, 64)]
        report["minibatch_order_sha256"] = hashlib.sha256(order.tobytes()).hexdigest()
        report["optimizer"] = {k: v for k, v in models["baseline"].policy.optimizer.param_groups[0].items() if k != "params"}
        report["optimizer"]["initial_state"] = "Restored checkpoint Adam moments, identical in both copies"
        gradients = {name: gradient(models["baseline"], merged, merged[name], batches) for name in ("gae", "mc")}
        normalized_adv = {}
        for name in ("gae", "mc"):
            normalized_adv[name] = np.empty(1500)
            for indices in batches:
                normalized_adv[name][indices] = normalized(torch.as_tensor(merged[name][indices])).numpy()
        report["comparison"] = {"raw_advantage_correlation": float(np.corrcoef(merged["gae"], merged["mc"])[0, 1]),
            "minibatch_normalized_sign_disagreement_fraction": float(np.mean(normalized_adv["gae"] * normalized_adv["mc"] < 0)),
            "actor_loss_gradient_cosine": cosine(gradients["gae"], gradients["mc"]),
            "actor_loss_gradient_norm": {name: float(g.norm()) for name, g in gradients.items()},
            "advantage_vs_episode_step_correlation": {name: float(np.corrcoef(merged[name], np.tile(np.arange(500), 3))[0, 1]) for name in ("gae", "mc")}}
        reference_log = log_probs(models["baseline"], merged)
        assert np.max(abs(reference_log - merged["old_log_prob"])) < 1e-4
        for name in ("gae", "mc"):
            result, change = update(models[name], merged, merged[name], batches)
            new_log = log_probs(models[name], merged)
            difference = new_log - reference_log
            result["mean_log_probability_change_all"] = float(difference.mean())
            result["parameter_change_cosine_with_negative_gradient"] = {key: cosine(change, -g) for key, g in gradients.items()}
            result["progress_segments"] = []
            for episode_index, collection in enumerate(report["collection"]):
                indices = np.asarray(collection["progress_segment"]["indices"]) + 500 * episode_index
                result["progress_segments"].append({"seed": collection["seed"],
                    "mean_log_probability_change": float(difference[indices].mean()),
                    "fraction_action_log_prob_increased": float(np.mean(difference[indices] > 0)),
                    "mean_advantage_raw": float(merged[name][indices].mean()),
                    "mean_advantage_minibatch_normalized": float(normalized_adv[name][indices].mean())})
            report["updates"][name] = result
            print("updated", name, result, flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name, model in models.items():
            report["evaluation"][name] = []
            for seed, deterministic in ((0, True), *((s, False) for s in EVALUATION_SEEDS)):
                data = collect(model, env, seed, deterministic)
                row = {"seed": seed, "deterministic": deterministic, **summary(data)}
                report["evaluation"][name].append(row)
                print("evaluated", name, row, flush=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["paired_changes"] = {name: [{"seed": item["seed"], "deterministic": item["deterministic"],
            **{metric: item[metric] - baseline[metric] for metric in ("mean_raw_reward", "discounted_scaled_return", "max_angle_deg_after_command", "settled_duration_s")}}
            for item, baseline in zip(report["evaluation"][name], report["evaluation"]["baseline"])] for name in ("gae", "mc")}
        assert tensor_digest(models["baseline"].policy) == original_weights
        assert originals == {str(p): file_digest(p) for p in files}
        assert dynamics_digest() == physical_before
        report["validation"] = {"original_checkpoint_files_unchanged": True, "original_in_memory_weights_unchanged": True,
            "dynamics_unchanged": True, "complete_episodes": 15, "physics_control_steps": 7500,
            "no_weight_files_saved": True, "elapsed_seconds": time.monotonic() - started}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
