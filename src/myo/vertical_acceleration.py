"""PPO acceleration matching rewards over unchanged native muscle transitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .vertical_control import REWARD, VerticalControlEnv, evaluate_policy, load_policy
from .vertical_muscles import VerticalConfig


REFERENCE = {"position_gain_s2": 28., "velocity_gain_s": 9.,
             "scale_m_s2": 28.*.08,
             "timing": "PD at interval start; actual delta velocity / actual elapsed time"}


def desired_acceleration(position, velocity, target):
    return REFERENCE["position_gain_s2"]*(target-position)-REFERENCE["velocity_gain_s"]*velocity


class VerticalAccelerationEnv(VerticalControlEnv):
    def __init__(self, config=VerticalConfig(), episode_seconds=4., objective="acceleration"):
        if objective not in ("position", "acceleration"):
            raise ValueError("Choose position or acceleration reward")
        super().__init__(config, episode_seconds)
        self.objective = objective

    def step(self, action):
        before_v, before_t = self.physical.velocity, self.physical.time
        reference = desired_acceleration(self.physical.position, before_v, self.target)
        obs, reward, terminated, truncated, info = super().step(action)
        actual = (self.physical.velocity-before_v)/(self.physical.time-before_t)
        if self.objective == "acceleration":
            parts = info["reward_components"]
            parts.pop("position")
            parts["acceleration"] = REWARD["position_weight"]*np.exp(-((actual-reference)/REFERENCE["scale_m_s2"])**2)
            info["raw_reward"] = float(sum(parts.values()))
            reward = REWARD["training_multiplier"]*info["raw_reward"]
        info.update(desired_acceleration_m_s2=float(reference), actual_acceleration_m_s2=float(actual),
                    acceleration_error_m_s2=float(actual-reference))
        return obs, reward, terminated, truncated, info

    def contract(self):
        return {**super().contract(),
                "reward": {**REWARD, "objective": self.objective, "acceleration_reference": REFERENCE,
                           "replacement": "position component only; velocity, effort and failure unchanged"},
                "acceleration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_acceleration_policy(path, episode_seconds=4.):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    contract = metadata["contract"]
    env = VerticalAccelerationEnv(VerticalConfig(**contract["physical"]["config"]), episode_seconds,
                                  contract["reward"]["objective"])
    model, metadata = load_policy(path, env)
    return model, env, metadata


def evaluate_acceleration_policy(model, config, cases, seconds=4., switches=False):
    """Use common position metrics/reward and reconstruct causal acceleration labels."""
    result = evaluate_policy(model, config, cases, seconds, switches)
    for case in result["cases"]:
        samples = np.asarray(case["samples"])
        dt = np.diff(np.r_[0., samples[:, 0]])
        previous_q = np.r_[case["initial_position_m"], samples[:-1, 1]]
        previous_v = np.r_[0., samples[:-1, 2]]
        reference = desired_acceleration(previous_q, previous_v, samples[:, 3])
        actual = np.diff(np.r_[0., samples[:, 2]])/dt
        case["acceleration_rmse_m_s2"] = float(np.sqrt(np.mean((actual-reference)**2)))
        case["acceleration_samples"] = np.stack([reference, actual], axis=-1).tolist()
    result["mean_case_acceleration_rmse_m_s2"] = float(np.mean([c["acceleration_rmse_m_s2"] for c in result["cases"]]))
    return result


def compact_acceleration(result):
    return {**result, "cases": [{k: v for k, v in case.items() if k not in ("samples", "acceleration_samples")}
                               for case in result["cases"]]}
