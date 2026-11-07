"""Reward-only ablation preserving the vertical control and physics contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .vertical_control import REWARD, VerticalControlEnv, load_policy
from .vertical_muscles import VerticalConfig


class VerticalSweepEnv(VerticalControlEnv):
    def __init__(self, config=VerticalConfig(), position_scale_m=.08, episode_seconds=4.):
        super().__init__(config, episode_seconds)
        self.position_scale_m = position_scale_m

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        parts = info["reward_components"]
        parts["position"] = REWARD["position_weight"]*np.exp(-(info["error_m"]/self.position_scale_m)**2)
        info["raw_reward"] = float(sum(parts.values()))
        return obs, REWARD["training_multiplier"]*info["raw_reward"], terminated, truncated, info

    def contract(self):
        return {**super().contract(), "reward": {**REWARD, "position_scale_m": self.position_scale_m},
                "sweep_environment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_sweep_policy(path, episode_seconds=4.):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    contract = metadata["contract"]
    env = VerticalSweepEnv(VerticalConfig(**contract["physical"]["config"]),
                           contract["reward"]["position_scale_m"], episode_seconds)
    model, metadata = load_policy(path, env)
    return model, env, metadata
