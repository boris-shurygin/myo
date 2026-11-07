"""Prepare physically held reset states; rewards still train actor actions only."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .compact_protocol import CompactProtocol


class PreparedProtocol(CompactProtocol):
    """Use heuristic only before handoff, through normal muscle commands.

    MuJoCo time, fatigue, activation, tendons, delay and history continue across
    handoff. The actor never receives future outcomes or teacher-action labels.
    """

    def __init__(self, env, scenarios, preparation_seconds=2.0):
        super().__init__(env)
        count = preparation_seconds / self.unwrapped.control_dt
        if preparation_seconds < 0 or not np.isclose(count, round(count)):
            raise ValueError("Preparation must be nonnegative whole control steps")
        self.scenarios = tuple(scenarios)
        self.preparation_seconds = float(preparation_seconds)
        self.preparation_steps = round(count)
        self.preparation = {}

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        original = self.scenario
        # The initial goal is present during preparation. Later commands and
        # the episode end are shifted without resetting the physical clock.
        self.scenario = replace(original,
            duration_s=original.duration_s + self.preparation_seconds,
            commands=(original.commands[0], *(replace(e, time_s=e.time_s + self.preparation_seconds)
                                               for e in original.commands[1:])))
        for _ in range(self.preparation_steps):
            observation, _, terminated, truncated, info = self.env.step(self.unwrapped.baseline_action())
            if terminated:
                raise RuntimeError("Physical preparation failed before policy handoff")
            if truncated:
                self.unwrapped._episode_done = False
        self.preparation = {"duration_s": self.preparation_seconds,
            "handoff_time_s": self.unwrapped.time,
            "handoff_angle_deg": float(np.rad2deg(info["q"])),
            "handoff_velocity_rad_s": float(info["qvel"]),
            "handoff_capacity": list(info["capacity"]),
            "method": "heuristic muscle commands before reset returns; no teacher labels"}
        info["protocol_scenario"] = original.name
        info["preparation"] = self.preparation
        return observation, info
