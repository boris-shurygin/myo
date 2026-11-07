"""Same-state one-action interventions with paired stochastic Beta continuations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath
import numpy as np
import torch
from stable_baselines3 import PPO

from myo.evaluate import CommandEvent, Scenario
from myo.prepared_protocol import PreparedProtocol
from myo.runtime import dynamics_digest, load_controller


class BetaQuantiles:
    """Invert an installed independent CDF implementation; do not add scipy."""

    def __init__(self):
        self.max_cdf_residual = 0.0
        self.count = 0

    def scalar(self, alpha, beta, uniform):
        lower, upper = 0.0, 1.0
        for _ in range(56):
            middle = (lower + upper) / 2
            cdf = float(mpmath.betainc(alpha, beta, 0, middle, regularized=True))
            if abs(cdf - uniform) < 1e-12:
                break
            if cdf < uniform:
                lower = middle
            else:
                upper = middle
        self.max_cdf_residual = max(self.max_cdf_residual, abs(cdf - uniform))
        self.count += 1
        return middle

    def action(self, model, observation, uniforms):
        with torch.no_grad():
            distribution = model.policy.get_distribution(torch.as_tensor(observation[None])).distribution
            alpha = distribution.concentration1[0].numpy().astype(float)
            beta = distribution.concentration0[0].numpy().astype(float)
        return np.asarray([self.scalar(a, b, u) for a, b, u in zip(alpha, beta, uniforms)],
                          dtype=np.float32)


def value(model, observation):
    with torch.no_grad():
        return float(model.policy.predict_values(torch.as_tensor(observation[None]))[0, 0])


def reset(physical):
    env = PreparedProtocol(physical, (Scenario("hold75", 4, (CommandEvent(0, 75, 1),),
                        1, initial_angle_deg=75),), 0)
    observation, _ = env.reset(seed=0)
    return env, observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = "runs/d1_architecture_wide_tanh/block_004.zip"
    physical, actor, metadata = load_controller(checkpoint)
    model = PPO.load(checkpoint, device="cpu")
    quantiles = BetaQuantiles()
    validation = []
    for a, b, u in ((1., 9., .01), (1., 9., .5), (1., 9., .99), (.7, 1., .1), (.7, 1., .9)):
        actual = quantiles.scalar(a, b, u)
        expected = 1 - (1 - u) ** (1 / b) if a == 1 else u ** (1 / a)
        validation.append({"alpha": a, "beta": b, "uniform": u,
                           "absolute_quantile_error": abs(actual - expected)})
        assert abs(actual - expected) < 1e-10
    prefix_seed = 1729
    prefix_uniforms = np.random.default_rng(prefix_seed).uniform(size=(100, 6))
    prefix_actions = []
    env, observation = reset(physical)
    for uniforms in prefix_uniforms:
        action = quantiles.action(model, observation, uniforms)
        prefix_actions.append(action)
        observation, _, terminated, _, _ = env.step(action)
        if terminated:
            raise RuntimeError("Prefix terminated")
    report = {"checkpoint": checkpoint, "checkpoint_sha256": metadata["checkpoint_sha256"],
        "dynamics_sha256": dynamics_digest(), "gamma": model.gamma, "gae_lambda": model.gae_lambda,
        "reward_scale": .01, "prefix_seed": prefix_seed,
        "prefix_actions_sha256": hashlib.sha256(np.asarray(prefix_actions).tobytes()).hexdigest(),
        "prefix_actions": np.asarray(prefix_actions).tolist(), "quantile_validation": validation,
        "cases": [], "limitations": [
            "Frozen wide policy only; no learning or optimizer updates.",
            "One state from a stochastic 2s prefix; three paired continuation tapes are diagnostics, not confidence intervals.",
            "First action is the mean or its intervention; subsequent actions use the actual Beta policy quantiles.",
            "Common uniform variates couple branches even when state-dependent Beta parameters differ.",
            "Reward horizons omit outcomes beyond the cutoff; bootstrap depends on the diagnosed critic.",
            "mpmath regularized beta CDF inversion used because scipy is not installed."]}
    try:
        for seed in (0, 1, 2):
            tape = np.random.default_rng(seed).uniform(size=(99, 6))
            for name, delta in (("baseline", (0, 0)), ("flex_plus", (.05, 0)), ("inverse", (-.05, .05))):
                env, observation = reset(physical)
                prefix = hashlib.sha256()
                for action in prefix_actions:
                    prefix.update(observation.tobytes())
                    observation, _, terminated, _, _ = env.step(action)
                    if terminated:
                        raise RuntimeError("Replay terminated")
                prefix.update(observation.tobytes())
                initial_value = value(model, observation)
                command = np.clip(actor(observation) + np.repeat(delta, 3), .0001, .9999)
                handoff = {"angle_deg": float(np.rad2deg(physical.data.qpos[physical.qpos_index])),
                           "value": initial_value, "action": command.tolist()}
                rows = []
                before_value = initial_value
                for step in range(100):
                    action = command if step == 0 else quantiles.action(model, observation, tape[step - 1])
                    observation, reward, terminated, _, info = env.step(action)
                    after_value = value(model, observation)
                    rows.append({"time_s": (step + 1) * physical.control_dt,
                        "scaled_reward": .01 * reward, "value_after": after_value,
                        "td": .01 * reward + model.gamma * after_value - before_value,
                        "angle_deg": float(np.rad2deg(info["q"])), "action": np.asarray(action).tolist()})
                    before_value = after_value
                    if terminated:
                        raise RuntimeError("Continuation physically terminated")
                horizons = {}
                for seconds in (.5, 1, 2):
                    n = round(seconds / physical.control_dt)
                    actual = float(sum(model.gamma ** i * row["scaled_reward"] for i, row in enumerate(rows[:n])))
                    bootstrap = model.gamma ** n * rows[n - 1]["value_after"]
                    horizons[str(seconds)] = {"discounted_scaled_actual_reward": actual,
                        "discounted_bootstrap": bootstrap,
                        "bootstrapped_advantage": actual + bootstrap - initial_value,
                        "final_angle_deg": rows[n - 1]["angle_deg"]}
                case = {"continuation_seed": seed, "direction": name, "handoff": handoff,
                    "prefix_observation_sha256": prefix.hexdigest(),
                    "uniform_tape_sha256": hashlib.sha256(tape.tobytes()).hexdigest(),
                    "first_step_td": rows[0]["td"],
                    "finite_horizon_gae": sum((model.gamma * model.gae_lambda) ** i * row["td"]
                                              for i, row in enumerate(rows)),
                    "horizons": horizons, "trace": rows}
                report["cases"].append(case)
                print(seed, name, handoff["angle_deg"], rows[0]["td"], horizons["2"], flush=True)
        reference_hash = report["cases"][0]["prefix_observation_sha256"]
        for case in report["cases"]:
            baseline = next(c for c in report["cases"] if c["continuation_seed"] == case["continuation_seed"]
                            and c["direction"] == "baseline")
            assert case["prefix_observation_sha256"] == reference_hash
            assert case["uniform_tape_sha256"] == baseline["uniform_tape_sha256"]
            case["difference_from_paired_baseline"] = {
                "first_step_td": case["first_step_td"] - baseline["first_step_td"],
                "finite_horizon_gae": case["finite_horizon_gae"] - baseline["finite_horizon_gae"],
                "horizons": {key: {metric: values[metric] - baseline["horizons"][key][metric]
                                   for metric in values} for key, values in case["horizons"].items()}}
        report["quantile_max_cdf_residual"] = quantiles.max_cdf_residual
        report["quantile_count"] = quantiles.count
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        physical.close()


if __name__ == "__main__":
    main()
