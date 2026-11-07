"""Audit reward and exploration at failed holds without updating any policy."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    model = PPO.load(args.controller, device="cpu")
    results = []
    for scenario_index in (7, 8):
        for mode in ("policy", "switch_to_heuristic", "heuristic"):
            physical, actor, metadata = load_controller(str(args.controller))
            env = CompactProtocol(physical)
            env.next_scenario = scenario_index
            obs, _ = env.reset(seed=0)
            prefix = hashlib.sha256()
            rows = []
            try:
                while True:
                    time = physical.time
                    if time < 5 - 1e-8:
                        prefix.update(obs.tobytes())
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(
                            torch.as_tensor(obs[None, :])).distribution
                        mu = distribution.mean[0]
                        sd = distribution.stddev[0]
                        # Per-step marginal, not a probability for a correlated sequence.
                        activation_probability = 0.5 * torch.erfc(
                            (0.05 - mu) / (sd * np.sqrt(2)))
                    use_heuristic = mode == "heuristic" or (
                        mode == "switch_to_heuristic" and time >= 5 - 1e-8)
                    action = physical.baseline_action() if use_heuristic else actor(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rows.append({"time": physical.time, "q": float(info["q"]),
                        "velocity": float(info["qvel"]), "reward": reward,
                        "terms": info["reward_components"], "action": action.tolist(),
                        "mu": mu.numpy().tolist(), "sd": sd.numpy().tolist(),
                        "probability_above_005": activation_probability.numpy().tolist(),
                        "reaction": info["substep_peaks"]["joint_limit_reaction_Nm"]})
                    if terminated or truncated:
                        break
                tail = [r for r in rows if r["time"] > 7.5 + 1e-8]
                result = {"scenario": env.scenario.name, "mode": mode,
                    "prefix_observation_sha256": prefix.hexdigest(),
                    "final_angle_deg": float(np.rad2deg(rows[-1]["q"])),
                    "tail_mean_reward": float(np.mean([r["reward"] for r in tail])),
                    "tail_angle_std_deg": float(np.rad2deg(np.std([r["q"] for r in tail]))),
                    "tail_mean_terms": {k: float(np.mean([r["terms"][k] for r in tail]))
                                        for k in tail[0]["terms"]},
                    "tail_mean_action": np.mean([r["action"] for r in tail], axis=0).tolist(),
                    "tail_mean_raw_mean": np.mean([r["mu"] for r in tail], axis=0).tolist(),
                    "tail_mean_std": np.mean([r["sd"] for r in tail], axis=0).tolist(),
                    "tail_mean_probability_above_005": np.mean(
                        [r["probability_above_005"] for r in tail], axis=0).tolist(),
                    "tail_peak_joint_reaction_Nm": max(r["reaction"] for r in tail),
                    "trace_1hz": rows[::50]}
                results.append(result)
                print(json.dumps({k: result[k] for k in
                    ("scenario", "mode", "final_angle_deg", "tail_mean_reward")}), flush=True)
            finally:
                env.close()
    report = {"controller": str(args.controller),
              "checkpoint_sha256": metadata["checkpoint_sha256"],
              "dynamics_sha256": dynamics_digest(), "results": results,
              "limitations": "No training; deterministic closed-loop comparison. Heuristic is a diagnostic counterfactual, not teacher data. Noise probabilities are marginal at visited states."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
