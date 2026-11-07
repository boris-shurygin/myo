"""Controlled, local PPO diagnostics on the existing fixed physical exercises.

No teacher labels, changed physics or production checkpoint promotion. Frozen
rollout targets are a regression diagnostic, not an off-policy PPO update.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from myo.compact_protocol import CompactProtocol, compact_scenarios, protocol_metadata
from myo.evaluate import run_scenario
from myo.runtime import dynamics_digest, read_config
from myo.env import PreacherCurlEnv


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fresh_model(env):
    return PPO("MlpPolicy", env, device="cpu", seed=0,
               policy_kwargs={"net_arch": [128, 128], "log_std_init": -2},
               n_steps=128, batch_size=64, learning_rate=3e-4,
               target_kl=0.02, verbose=0)


class Record(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode = 0
        self.episodes, self.raw_rewards, self.buffers = [], [], []

    def _on_step(self):
        self.episodes.append(self.episode)
        self.raw_rewards.append(float(self.locals["rewards"][0]))
        self.episode += int(self.locals["dones"][0])
        return True

    def _on_rollout_end(self):
        b = self.model.rollout_buffer
        self.buffers.append({name: getattr(b, name).copy().reshape(
            (-1, *getattr(b, name).shape[2:])) for name in
            ("observations", "actions", "returns", "advantages", "values", "log_probs")})


def collect(out, config, steps):
    env = Monitor(CompactProtocol(PreacherCurlEnv(config)))
    model = fresh_model(env)
    recorder = Record()
    _, callback = model._setup_learn(total_timesteps=steps, callback=recorder)
    model.save(out / "initial.zip")
    try:
        while model.num_timesteps < steps:
            model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
            if len(recorder.buffers) % 20 == 0:
                print(json.dumps({"collected_steps": model.num_timesteps}), flush=True)
        data = {k: np.concatenate([b[k] for b in recorder.buffers]) for k in recorder.buffers[0]}
        data.update(episodes=np.asarray(recorder.episodes), raw_rewards=np.asarray(recorder.raw_rewards))
        np.savez_compressed(out / "frozen_rollouts.npz", **data)
        write_json(out / "collection.json", {
            "steps": model.num_timesteps, "completed_episodes": recorder.episode,
            "seed": 0, "config": asdict(config), "protocol": protocol_metadata(config.payload_mass),
            "dynamics_sha256": dynamics_digest(),
            "initial_sha256": hashlib.sha256((out / "initial.zip").read_bytes()).hexdigest(),
            "sb3_version": __import__("stable_baselines3").__version__,
            "target_definition": "Stock SB3 128-step rollout GAE targets from an unchanged initial policy; no optimizer updates",
            "raw_reward_range": [float(data["raw_rewards"].min()), float(data["raw_rewards"].max())],
            "returns_range": [float(data["returns"].min()), float(data["returns"].max())],
            "observation_dim": data["observations"].shape[1],
            "action_clipped_fraction": float(((data["actions"] < 0) | (data["actions"] > 1)).mean()),
        })
    finally:
        env.close()


def fit_critics(out, updates):
    data = np.load(out / "frozen_rollouts.npz")
    model = PPO.load(out / "initial.zip", device="cpu")
    x = torch.as_tensor(data["observations"])
    y = torch.as_tensor(data["returns"]).reshape(-1, 1)
    # Entire trigger=0.6 episodes are held out, not neighboring timestamps.
    valid = np.isin(data["episodes"], [3, 4, 5, 12, 13, 14])
    train = (~valid) & (data["episodes"] < 18)
    train_ids, valid_ids = np.flatnonzero(train), np.flatnonzero(valid)
    mean = x[train_ids].mean(0)
    batches = np.random.default_rng(0).choice(train_ids, size=(updates, 64))
    results = []
    for name, centered, scale in [("baseline", False, 1.0), ("target_scale_0.01", False, 0.01),
                                   ("center_observations", True, 1.0),
                                   ("center_and_scale", True, 0.01)]:
        net = nn.Sequential(copy.deepcopy(model.policy.mlp_extractor.value_net),
                            copy.deepcopy(model.policy.value_net))
        # Change the value unit consistently for both initial predictions/targets.
        with torch.no_grad():
            net[1].weight.mul_(scale)
            net[1].bias.mul_(scale)
        inputs = x - mean if centered else x
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
        history = []
        for step in range(updates + 1):
            if step in {0, 50, 200, 600, updates}:
                with torch.no_grad():
                    pred = net(inputs) / scale
                    def metrics(ids):
                        target = y[ids]
                        residual = target - pred[ids]
                        return {"mse_raw_units": float(residual.square().mean()),
                                "explained_variance": float(1 - residual.var() / target.var()),
                                "prediction_std_raw_units": float(pred[ids].std())}
                    hidden = net[0](inputs[train_ids])
                    history.append({"updates": step, "train": metrics(train_ids), "held_out": metrics(valid_ids),
                                    "last_tanh_saturation": float((hidden.abs() > .99).float().mean())})
            if step == updates:
                break
            batch = batches[step]
            loss = 0.5 * (net(inputs[batch]) - y[batch] * scale).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
        result = {"case": name, "history": history}
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(out / "critic_fit.json", {
        "description": "Identical frozen GAE labels, initial weights, minibatches and optimizer; no policy learning",
        "train_episodes": sorted(set(data["episodes"][train].tolist())),
        "held_out_episodes": sorted(set(data["episodes"][valid].tolist())),
        "target_std": float(y[train_ids].std()), "observation_mean_l2": float(mean.norm()),
        "results": results})


def gradient_probe(out):
    data = np.load(out / "frozen_rollouts.npz")
    results = []
    for scale in [1.0, 0.01]:
        model = PPO.load(out / "initial.zip", device="cpu")
        with torch.no_grad():
            model.policy.value_net.weight.mul_(scale)
            model.policy.value_net.bias.mul_(scale)
        obs = torch.as_tensor(data["observations"][:64])
        actions = torch.as_tensor(data["actions"][:64])
        returns = torch.as_tensor(data["returns"][:64]) * scale
        advantages = torch.as_tensor(data["advantages"][:64]) * scale
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        values, log_prob, _ = model.policy.evaluate_actions(obs, actions)
        ratio = torch.exp(log_prob - torch.as_tensor(data["log_probs"][:64]))
        loss = -(advantages * ratio).mean() + 0.5 * (values.flatten() - returns).square().mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        actor = [(n, p) for n, p in model.policy.named_parameters()
                 if n.startswith(("mlp_extractor.policy_net", "action_net")) or n == "log_std"]
        critic = [(n, p) for n, p in model.policy.named_parameters()
                  if n.startswith(("mlp_extractor.value_net", "value_net"))]
        def norm(group):
            return float(torch.stack([p.grad.square().sum() for _, p in group if p.grad is not None]).sum().sqrt())
        actor_norm, critic_norm = norm(actor), norm(critic)
        before = [p.detach().clone() for _, p in actor]
        total = float(torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5))
        model.policy.optimizer.step()
        delta = float(torch.stack([(p.detach() - old).square().sum()
                                  for (_, p), old in zip(actor, before)]).sum().sqrt())
        results.append({"reward_and_value_scale": scale, "actor_grad_norm": actor_norm,
                        "critic_grad_norm": critic_norm, "total_grad_norm": total,
                        "clip_factor": min(1.0, 0.5 / (total + 1e-6)), "actor_adam_step_l2": delta})
    write_json(out / "gradient_probe.json", results)
    print(json.dumps({"gradient_probe": results}), flush=True)


class ScaledReward(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return self.scale * reward


def online(out, config, steps, case):
    """One-factor PPO comparison, including consistent initial value units."""
    dest = out / f"online_{case}"
    dest.mkdir(exist_ok=False)
    scale = 1.0 if case == "baseline" else 0.01
    env = Monitor(ScaledReward(CompactProtocol(PreacherCurlEnv(config)), scale))
    model = PPO.load(out / "initial.zip", env=env, device="cpu")
    with torch.no_grad():
        model.policy.value_net.weight.mul_(scale)
        model.policy.value_net.bias.mul_(scale)
    gradients, history = [], []
    named = list(model.policy.named_parameters())
    actor_params = [p for n, p in named if n.startswith(("mlp_extractor.policy_net", "action_net")) or n == "log_std"]
    critic_params = [p for n, p in named if n.startswith(("mlp_extractor.value_net", "value_net"))]

    def before_step(optimizer, args, kwargs):
        def norm(params):
            return float(torch.stack([p.grad.square().sum() for p in params if p.grad is not None]).sum().sqrt())
        gradients.append({"actor": norm(actor_params), "critic": norm(critic_params)})

    handle = model.policy.optimizer.register_step_pre_hook(before_step)
    stock_train = model.train

    def audited_train():
        start = len(gradients)
        stock_train()
        b = model.rollout_buffer
        obs = torch.as_tensor(b.observations.reshape(-1, *model.observation_space.shape))
        target = torch.as_tensor(b.returns.reshape(-1))
        with torch.no_grad():
            features = model.policy.extract_features(obs)
            hidden = model.policy.mlp_extractor.value_net(features)
            values = model.policy.value_net(hidden).flatten()
            ev = float(1 - (target - values).var() / target.var()) if target.var() > 1e-10 else None
        row = {"steps": model.num_timesteps, "optimizer_steps": len(gradients) - start,
               "old_explained_variance": float(model.logger.name_to_value["train/explained_variance"]),
               "post_fit_explained_variance": ev,
               "last_tanh_saturation": float((hidden.abs() > .99).float().mean()),
               "raw_value_mean": float(values.mean() / scale),
               "raw_value_std": float(values.std() / scale),
               "raw_target_std": float(target.std() / scale),
               "raw_action_clipped_fraction": float(((b.actions < 0) | (b.actions > 1)).mean())}
        history.append(row)
        if len(history) % 20 == 0:
            print(json.dumps({"case": case, **row}), flush=True)

    model.train = audited_train
    try:
        model.learn(total_timesteps=steps)
        handle.remove()
        # Diagnostic artifact; deliberately not a deployable v2 controller.
        model.save(dest / "diagnostic.zip", exclude=["train"])
        evaluation_env = PreacherCurlEnv(config)
        try:
            scenarios = compact_scenarios(config.payload_mass)
            moves = [run_scenario(evaluation_env,
                lambda obs: model.predict(obs, deterministic=True)[0], scenarios[i], 0,
                continue_after_time_limit=True, measure_commands=True) for i in [7, 8]]
        finally:
            evaluation_env.close()
        write_json(dest / "audit.json", {"case": case, "reward_scale": scale,
            "initial_value_head_scale": scale, "steps": model.num_timesteps,
            "seed": 0, "config": asdict(config), "dynamics_sha256": dynamics_digest(),
            "initial_sha256": hashlib.sha256((out / "initial.zip").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "description": "Fresh identical policy; stock PPO optimizer; one fixed cycle, no teacher; optimizer pre-hook records gradients after stock global clipping",
            "history": history, "optimizer_gradient_norms_after_clip": gradients,
            "diagnostic_moves": moves})
        print(json.dumps({"case": case, "finished_steps": model.num_timesteps,
                          "last_saturation": history[-1]["last_tanh_saturation"],
                          "moves": [(m["scenario"]["name"], m["final_angle_deg"]) for m in moves]}), flush=True)
    finally:
        handle.remove()
        env.close()


def exploration(out, config, mode="temporal"):
    """Match latent noise samples, varying only their persistence; no fitting."""
    model = PPO.load(out / "initial.zip", device="cpu")
    # All three cases consume the same table. Held noise repeats the sample at
    # the start of a window; it never depends on a future command or state.
    noise = np.random.default_rng(0).standard_normal((2450, 6)).astype(np.float32)
    results = []
    cases = {"temporal": [(1, 1), (10, 1), (50, 1)],
             "wide": [(50, 4), (50, 8)], "fast_wide": [(1, 4), (10, 4)]}[mode]
    for hold, amplitude in cases:
        env = CompactProtocol(PreacherCurlEnv(config))
        index = 0
        try:
            for _ in range(3):
                obs, _ = env.reset(seed=0)
                angles, activations, actions, rewards = [], [], [], []
                while True:
                    with torch.no_grad():
                        dist = model.policy.get_distribution(torch.as_tensor(obs[None, :])).distribution
                        raw = dist.mean[0].numpy() + amplitude * dist.stddev[0].numpy() * noise[index - index % hold]
                    action = np.clip(raw, 0, 1)
                    obs, reward, terminated, truncated, info = env.step(action)
                    index += 1
                    angles.append(float(np.rad2deg(info["q"])))
                    activations.append(info["activation"])
                    actions.append(action)
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                result = {"noise_hold_steps": hold, "noise_hold_s": hold * config.control_dt,
                          "noise_std_multiplier": amplitude,
                          "scenario": env.scenario.name, "steps": len(angles),
                          "physical_failure": bool(terminated),
                          "angle_min_deg": min(angles), "angle_max_deg": max(angles),
                          "final_angle_deg": angles[-1], "fraction_above_45_deg": float(np.mean(np.array(angles) > 45)),
                          "excitation_std_mean": float(np.asarray(actions).std(axis=0).mean()),
                          "activation_std_mean": float(np.asarray(activations).std(axis=0).mean()),
                          "mean_reward": float(np.mean(rewards))}
                results.append(result)
                print(json.dumps(result), flush=True)
        finally:
            env.close()
    write_json(out / ("exploration.json" if mode == "temporal" else f"exploration_{mode}.json"), {
        "seed": 0, "initial_sha256": hashlib.sha256((out / "initial.zip").read_bytes()).hexdigest(),
        "description": "Initial actor, matched standard-normal noise table, three fixed exercises at trigger0.3/capacity1; no optimizer, teacher or changed actuator timing",
        "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["collect", "critic", "gradients", "online",
                        "exploration", "exploration-wide", "exploration-fast-wide", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/d1_ppo_diagnosis"))
    parser.add_argument("--config", type=Path, default=Path("configs/d1_train.json"))
    parser.add_argument("--steps", type=int, default=14720)
    parser.add_argument("--fit-updates", type=int, default=1200)
    parser.add_argument("--case", choices=["baseline", "scaled"], default="baseline")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in {"collect", "all"}:
        if (out / "frozen_rollouts.npz").exists():
            raise ValueError("Preserve the existing dataset; use --phase critic/gradients or a fresh output directory")
        collect(out, read_config(args.config), args.steps)
    if args.phase in {"critic", "all"}:
        fit_critics(out, args.fit_updates)
    if args.phase in {"gradients", "all"}:
        gradient_probe(out)
    if args.phase == "online":
        online(out, read_config(args.config), args.steps, args.case)
    if args.phase == "exploration":
        exploration(out, read_config(args.config))
    if args.phase == "exploration-wide":
        exploration(out, read_config(args.config), mode="wide")
    if args.phase == "exploration-fast-wide":
        exploration(out, read_config(args.config), mode="fast_wide")


if __name__ == "__main__":
    main()
