"""Replay two recorded episodes and compare frozen physical continuations.

No training. Interventions use ordinary delayed muscle commands. Deterministic
continuations are diagnostic alternatives, not the stochastic policy's true Q.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.movement_curriculum import MovementProtocol
from myo.runtime import dynamics_digest, load_controller


DIRECTORY = Path("runs/d1_movement_curriculum")
CHECKPOINT = DIRECTORY / "stage_0_block_013.zip"
OUTPUT = Path("runs/d1_movement_replay_audit.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(model, observation):
    with torch.no_grad():
        return float(model.policy.predict_values(torch.as_tensor(observation[None]))[0, 0])


def main():
    torch.set_num_threads(1)
    checked, _, metadata = load_controller(str(CHECKPOINT))
    checked.close()
    model = PPO.load(CHECKPOINT, device="cpu")
    before_zip, before_physics = digest(CHECKPOINT), dynamics_digest()
    config = PreacherCurlConfig(**metadata["config"])
    report = {"script_sha256": digest(__file__), "checkpoint_sha256": before_zip,
              "dynamics_sha256": before_physics, "physical_steps": 0, "episodes": [],
              "limitations": ["No optimizer steps or new learned checkpoints.",
                  "Two recorded states of block13; not a state-space or seed search.",
                  "Branches follow the deterministic mean, while the critic predicts stochastic returns.",
                  "Intervention pulse is an external experiment, not a new policy parameterization.",
                  "Finite2s physical reward omits later consequences; bootstrap depends on the critic.",
                  "Raw GAE is not actual minibatch-normalized PPO advantage."]}
    with np.load(DIRECTORY / "rollout_000039744.npz") as d:
        observations = d["observations"][:, 0].copy()
        commands = d["actions"][:, 0].copy()
        rewards = d["physical_samples"][:, 3].copy()
        saved_advantages = d["advantages"][:, 0].copy()
        saved_logprob = d["log_probs"][:, 0].copy()
    report["source_rollout_sha256"] = digest(DIRECTORY / "rollout_000039744.npz")
    with torch.no_grad():
        logprob = model.policy.get_distribution(torch.as_tensor(observations)).log_prob(torch.as_tensor(commands)).numpy()
    report["old_policy_logprob_reconstruction_max_error"] = float(max(abs(logprob - saved_logprob)))
    assert report["old_policy_logprob_reconstruction_max_error"] < 3e-5

    def start(eid, n):
        offset = eid * 200 - 38272
        env = MovementProtocol(PreacherCurlEnv(config), 0)
        env.next_scenario = eid % 6
        observation, _ = env.reset(seed=0)
        error = 0.
        for i in range(n):
            error = max(error, float(np.max(abs(observation - observations[offset + i]))))
            observation, reward, terminated, truncated, _ = env.step(commands[offset + i])
            report["physical_steps"] += 1
            assert not terminated and not truncated
            assert abs(reward - rewards[offset + i]) < 2e-6
        error = max(error, float(np.max(abs(observation - observations[offset + n]))))
        assert error < 2e-6
        return env, observation, error

    for eid in (193, 195):
        offset = eid * 200 - 38272
        episode_observations = observations[offset:offset + 200]
        q = np.rad2deg(episode_observations[:, 0] + config.reference_center)
        speed = np.rad2deg(episode_observations[:, 1] * 4)
        if eid == 193:
            at = int(np.flatnonzero((q >= 25) & (q <= 29) & (speed > 10))[0])
        else:
            at = int(np.flatnonzero((abs(q - 30) <= 3) & (abs(speed) <= 5))[0])
        episode = {"episode": eid, "branch_step": at, "branch_time_s": at * .02,
                   "angle_deg": float(q[at]), "velocity_deg_s": float(speed[at]),
                   "recorded_replay": [], "branches": []}
        env = MovementProtocol(PreacherCurlEnv(config), 0)
        env.next_scenario = eid % 6
        obs, _ = env.reset(seed=0)
        max_error = 0.
        try:
            for i in range(200):
                max_error = max(max_error, float(np.max(abs(obs - episode_observations[i]))))
                mean = model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(commands[offset + i])
                report["physical_steps"] += 1
                assert abs(reward - rewards[offset + i]) < 2e-6
                episode["recorded_replay"].append({"time_s": (i + 1) * .02,
                    "angle_deg": float(np.rad2deg(info["q"])),
                    "speed_deg_s": float(np.rad2deg(info["qvel"])),
                    "reward": reward, "reward_terms": info["reward_components"],
                    "mean_command": mean.tolist(), "recorded_command": commands[offset + i].tolist(),
                    "activation": list(info["activation"]), "delayed_excitation": list(info["delayed_excitation"]),
                    "muscle_torque_Nm": info["muscle_torque_Nm"],
                    "passive_limit_torque_Nm": info["passive_limit_torque_Nm"],
                    "raw_saved_gae": float(saved_advantages[offset + i])})
                assert not terminated
                assert truncated == (i == 199)
        finally:
            env.close()
        assert max_error < 2e-6
        episode["max_observation_replay_error"] = max_error
        variants = (("mean", 0.), ("flex_less", .02), ("ext_more", .02),
                    ("flex_less", .2), ("ext_more", .2), ("heuristic", 2.))
        for variant, duration in variants:
            env, obs, replay_error = start(eid, at)
            initial_hash = hashlib.sha256(obs.tobytes()).hexdigest()
            initial_value = value(model, obs)
            first = model.predict(obs, deterministic=True)[0]
            initial_action = first.copy()
            if variant == "flex_less":
                initial_action[:3] -= .05
            elif variant == "ext_more":
                initial_action[3:] += .05
            initial_action = np.clip(initial_action, .0001, .9999)
            trace, preceding_value = [], initial_value
            try:
                for i in range(100):
                    if variant == "heuristic":
                        action = env.unwrapped.baseline_action()
                    elif i < round(duration / .02):
                        action = initial_action
                    else:
                        action = model.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    report["physical_steps"] += 1
                    assert not terminated and not truncated
                    following = value(model, obs)
                    trace.append({"time_s": (i + 1) * .02, "raw_reward": reward,
                                  "reward_terms": info["reward_components"],
                                  "q_deg": float(np.rad2deg(info["q"])),
                                  "v_deg_s": float(np.rad2deg(info["qvel"])),
                                  "action": action.tolist(), "value_after": following,
                                  "td": .01 * reward + model.gamma * following - preceding_value})
                    preceding_value = following
            finally:
                env.close()
            horizons = {}
            for duration_s in (.2, .5, 1., 2.):
                n = round(duration_s / .02)
                actual = float(sum(model.gamma ** i * trace[i]["raw_reward"] * .01 for i in range(n)))
                bootstrap = model.gamma ** n * trace[n - 1]["value_after"]
                gae = float(sum((model.gamma * model.gae_lambda) ** i * trace[i]["td"] for i in range(n)))
                horizons[str(duration_s)] = {"discounted_physical_reward": actual,
                    "bootstrap": bootstrap, "bootstrapped_return": actual + bootstrap,
                    "finite_gae": gae, "q_deg": trace[n - 1]["q_deg"], "v_deg_s": trace[n - 1]["v_deg_s"]}
            episode["branches"].append({"variant": variant, "pulse_s": duration,
                "initial_observation_sha256": initial_hash, "prefix_replay_error": replay_error,
                "initial_value": initial_value, "first_action": trace[0]["action"],
                "first_td": trace[0]["td"], "horizons": horizons, "trace": trace})
            print(eid, variant, duration, "q_after2s", round(trace[-1]["q_deg"], 3),
                  "return", round(horizons["2.0"]["discounted_physical_reward"], 5), flush=True)
        assert len({b["initial_observation_sha256"] for b in episode["branches"]}) == 1
        report["episodes"].append(episode)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert before_zip == digest(CHECKPOINT)
    assert before_physics == dynamics_digest()
    print("saved", OUTPUT, "steps", report["physical_steps"], flush=True)


if __name__ == "__main__":
    main()
