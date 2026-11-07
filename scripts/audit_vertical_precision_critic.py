"""Compare a frozen critic with independent Monte Carlo policy returns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from myo.train_vertical_architecture import load_trial
from myo.train_vertical_ppo import write_json
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import VerticalPrecisionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path("runs/vertical_precision/combined/best.zip"))
    parser.add_argument("--state-controller", type=Path, help="Optional fixed policy to generate identical evaluation states")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=16.)
    parser.add_argument("--output", type=Path, default=Path("runs/vertical_precision/critic_audit.json"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    metadata = json.loads(Path(str(args.controller)+".metadata.json").read_text())
    config = VerticalConfig(**metadata["contract"]["physical"]["config"])
    checked_env = VerticalPrecisionEnv(config)
    if "pretrained_design" in metadata["contract"]:
        from myo.pretrained_vertical import load_pretrained
        checked_env.close()
        model, checked_env, _ = load_pretrained(args.controller)
    elif "detached_design" in metadata["contract"]:
        from myo.detached_vertical import load_detached
        checked_env.close()
        model, checked_env, _ = load_detached(args.controller)
    elif "shared_design" in metadata["contract"]:
        from myo.shared_vertical import load_shared
        checked_env.close()
        model, checked_env, _ = load_shared(args.controller)
    else:
        model, _ = load_trial(args.controller, checked_env)
    state_model = model
    state_source_sha256 = metadata["checkpoint_sha256"]
    if args.state_controller is not None:
        from myo.vertical_control import load_policy
        state_model, state_metadata = load_policy(args.state_controller, VerticalPrecisionEnv(config))
        state_source_sha256 = state_metadata["checkpoint_sha256"]
    model.policy.set_training_mode(False)
    gamma = float(model.gamma)
    steps = round(args.seconds/config.dt)
    states = []
    # Actual states: starts, 200 ms into deterministic reaches, and balanced holds.
    movements = ((-.03, .06), (.03, -.06), (-.025, .055), (.025, -.055))
    for initial, target in movements:
        source = VerticalPrecisionEnv(config, episode_seconds=20.)
        obs, _ = source.reset(options={"initial_position_m": initial, "target_m": target})
        for prefix in (0, 10):
            if prefix:
                for _ in range(prefix):
                    action, _ = state_model.predict(obs, deterministic=True)
                    obs, *_ = source.step(action)
            states.append({"name": f"reach {initial*1000:+g} to {target*1000:+g} mm at {prefix*20} ms",
                "position_m": source.physical.position, "velocity_m_s": source.physical.velocity,
                "target_m": target, "observation": obs.copy(),
                "physical_snapshot": source.physical.snapshot(), "previous": source.previous.copy()})
        source.close()
    for target in (-.03, .03, 0., .07):
        source = VerticalPrecisionEnv(config, episode_seconds=20.)
        obs, _ = source.reset(options={"initial_position_m": target, "target_m": target})
        states.append({"name": f"hold {target*1000:+g} mm", "position_m": target,
            "velocity_m_s": 0., "target_m": target, "observation": obs.copy(),
            "physical_snapshot": source.physical.snapshot(), "previous": source.previous.copy()})
        source.close()
    obs_states = torch.as_tensor(np.stack([s["observation"] for s in states]))
    with torch.no_grad():
        predictions = model.policy.predict_values(obs_states).flatten().cpu().numpy()
    envs = []
    for state in states:
        for _ in range(args.repeats):
            env = VerticalPrecisionEnv(config, episode_seconds=args.seconds+2.)
            env.reset(options={"initial_position_m": state["position_m"], "target_m": state["target_m"]})
            env.physical.restore(state["physical_snapshot"])
            env.previous = state["previous"].copy()
            np.testing.assert_array_equal(env.observation(), state["observation"])
            envs.append(env)
    obs = np.stack([e.observation() for e in envs])
    active = np.ones(len(envs), dtype=bool)
    returns = np.zeros(len(envs))
    failures = np.zeros(len(envs), dtype=bool)
    torch.manual_seed(917)
    with torch.no_grad():
        for step in range(steps):
            # Stochastic policy, as required by V^pi, rather than the demo's mean actions.
            actions = model.policy.get_distribution(torch.as_tensor(obs)).sample().cpu().numpy()
            for index in np.flatnonzero(active):
                obs[index], reward, terminated, truncated, _ = envs[index].step(actions[index])
                returns[index] += gamma**step*reward
                if terminated or truncated:
                    active[index] = False
                    failures[index] = terminated
            if (step+1) % 200 == 0:
                print(json.dumps({"steps": step+1, "of": steps, "active": int(active.sum())}), flush=True)
    rows = []
    for i, state in enumerate(states):
        mc = returns[i*args.repeats:(i+1)*args.repeats]
        rows.append({k: state[k] for k in ("name", "position_m", "velocity_m_s", "target_m")})
        rows[-1].update(predicted_value=float(predictions[i]), mc_mean=float(np.mean(mc)),
            mc_std=float(np.std(mc, ddof=1)), mc_standard_error=float(np.std(mc, ddof=1)/np.sqrt(args.repeats)),
            bias=float(predictions[i]-np.mean(mc)), sampled_returns=mc.tolist())
    means = np.array([r["mc_mean"] for r in rows])
    errors = predictions-means
    result = {"checkpoint": str(args.controller), "checkpoint_sha256": metadata["checkpoint_sha256"],
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_source_policy_sha256": state_source_sha256,
        "state_observations_sha256": hashlib.sha256(np.stack([state["observation"] for state in states]).tobytes()).hexdigest(),
        "gamma": gamma, "horizon_s": args.seconds, "steps": steps, "repeats": args.repeats,
        "sample_seed": 917, "training_multiplier": metadata["contract"]["reward"]["training_multiplier"],
        "policy": "Independent stochastic Beta actions; frozen best, no update",
        "time_limits": "Continue fixed goal beyond 4s training time limit, matching value bootstrapping; physical failure terminates",
        "bootstrap": False, "discount_at_horizon": gamma**steps,
        "positive_return_remainder_upper_bound": .1*gamma**steps/(1-gamma),
        "failures": int(failures.sum()), "mean_bias": float(np.mean(errors)),
        "rmse_against_mc_means": float(np.sqrt(np.mean(errors**2))),
        "explained_variance_against_mc_means": float(1-np.var(errors)/np.var(means)),
        "mc_mean_range": [float(means.min()), float(means.max())], "rows": rows}
    write_json(args.output, result)
    for env in envs:
        env.close()
    checked_env.close()
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
