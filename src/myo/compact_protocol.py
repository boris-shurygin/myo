"""Fixed D1 exercises and result-based reward; no teacher actions or pose rescue."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np

from .evaluate import CommandEvent, Scenario


TRIGGERS = (0.3, 0.6, 1.0)
CAPACITIES = (1.0, 0.8)
ANGLE_GRID_DEG = (*range(15, 150, 10), 150)
ANGLE_TOLERANCE_RAD = float(np.deg2rad(3))
SPEED_TOLERANCE_RAD_S = float(np.deg2rad(5))
SETTLE_DURATION_S = 0.3


def compact_scenarios(payload_mass: float) -> tuple[Scenario, ...]:
    # The descent starts from the actual end of the ascent, never a reset
    # into intersecting contact shells at 150 degrees.
    angles = (*ANGLE_GRID_DEG, *reversed(ANGLE_GRID_DEG[:-1]))
    result = []
    for capacity in CAPACITIES:
        for trigger in TRIGGERS:
            suffix = f"u{trigger:g}_c{capacity:g}"
            result.extend((
                Scenario(f"sweep_{suffix}", float(len(angles)),
                         tuple(CommandEvent(float(i), float(q), trigger) for i, q in enumerate(angles)),
                         payload_mass, capacity=capacity, initial_angle_deg=15),
                Scenario(f"raise_{suffix}", 10,
                         (CommandEvent(0, 20, trigger), CommandEvent(0.5, 75, trigger)),
                         payload_mass, capacity=capacity, initial_angle_deg=20),
                Scenario(f"lower_{suffix}", 10,
                         (CommandEvent(0, 75, trigger), CommandEvent(0.5, 20, trigger)),
                         payload_mass, capacity=capacity, initial_angle_deg=75),
            ))
    return tuple(result)


def protocol_metadata(payload_mass: float) -> dict:
    return {
        "name": "compact_v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": [asdict(s) for s in compact_scenarios(payload_mass)],
        "order": "fixed cyclic; no random initial states or command sampling",
        "reward": {"tracking": "-abs(q-requested_target)/angle_scale_rad",
                   "angle_scale_rad": float(np.deg2rad(30)),
                   "settled_bonus": "exp(-(error/angle_tolerance)^2-(velocity/speed_tolerance)^2)",
                   "angle_tolerance_rad": ANGLE_TOLERANCE_RAD,
                   "speed_tolerance_rad_s": SPEED_TOLERANCE_RAD_S,
                   "near_goal_braking_weight": 0.05,
                   "near_goal_scale_rad": float(np.deg2rad(10)),
                   "contact_force_scale_N": 100.0, "contact_cost_weight": 0.1,
                   "other_costs": "unchanged base environment costs; original tracking terms removed"},
        "settle_duration_s": SETTLE_DURATION_S,
        "limitations": ["Fixed exercises are not independent generalization evidence.",
                        "150-degree commands may be obstructed by contact; never force the pose.",
                        "Contact penalty scales are engineering choices, not human safety thresholds.",
                        "This block does not train trigger release or establish long-term endurance."],
    }


def result_reward(info: dict) -> tuple[float, dict]:
    """Reward fast, accurate arrival through its accumulated state outcomes.

    The final requested goal is used, so motion ahead of the smoothed reference
    is not itself penalized. Speed near the goal discourages fly-throughs.
    """
    error = float(info["q"] - info["requested_target_q"])
    velocity = float(info["qvel"])
    terms = {k: float(v) for k, v in info["reward_components"].items()
             if k not in ("position_tracking", "velocity_tracking")}
    tracking = float(info["trigger"] > 0)
    terms["requested_angle_cost"] = -tracking * abs(error) / np.deg2rad(30)
    terms["settled_bonus"] = tracking * float(np.exp(
        -(error / ANGLE_TOLERANCE_RAD) ** 2 - (velocity / SPEED_TOLERANCE_RAD_S) ** 2))
    terms["near_goal_speed_cost"] = -tracking * 0.05 * float(np.exp(
        -(error / np.deg2rad(10)) ** 2)) * velocity ** 2
    force = info["substep_peaks"]["tissue_contact_force_N"]
    terms["tissue_contact_force_proxy_cost"] = -0.1 * float(force / 100.0) ** 2
    return float(sum(terms.values())), terms


def initialize_command_at_reset(env, target_deg: float, trigger: float) -> None:
    """Start the command filter at the initial goal, with consistent history."""
    base = env.unwrapped
    if not base._has_reset or base.time != 0 or base._step_count != 0:
        raise RuntimeError("Initial command history can only be set immediately after reset")
    base.set_command(float(np.deg2rad(target_deg)), trigger)
    base.command.target = base.command.requested
    base.command.velocity = 0.0
    base._command_history.clear()
    for _ in range(base.config.history_steps):
        base._command_history.append(base._command_values())


class CompactProtocol(gym.Wrapper):
    """Cycle exact exercises; change only commands, reset conditions and reward."""

    def __init__(self, env):
        super().__init__(env)
        if env.unwrapped.config.task_mode == "reference":
            raise ValueError("Compact protocol requires command observations.")
        if env.unwrapped.geometry is None:
            raise ValueError("The 15–150 degree compact protocol requires MyoArm geometry.")
        self.scenarios = compact_scenarios(env.unwrapped.config.payload_mass)
        self.next_scenario = 0
        self.completed_episodes = 0
        self.scenario = self.scenarios[0]

    def _command_observation(self):
        base = self.unwrapped
        event = max((e for e in self.scenario.commands if e.time_s <= base.time + 1e-10),
                    key=lambda e: e.time_s)
        base.set_command(float(np.deg2rad(event.target_deg)), event.trigger)
        observation = base.observe()
        return self.env.observation(observation) if isinstance(self.env, gym.ObservationWrapper) else observation

    def reset(self, *, seed=None, options=None):
        self.scenario = self.scenarios[self.next_scenario]
        self.next_scenario = (self.next_scenario + 1) % len(self.scenarios)
        s = self.scenario
        first = s.commands[0]
        self.env.reset(seed=seed, options={
            "q": float(np.deg2rad(s.initial_angle_deg)), "qvel": 0.0,
            "target_angle": float(np.deg2rad(first.target_deg)), "trigger": first.trigger,
            "payload_mass": s.payload_mass_kg, "capacity": s.capacity,
            "reference_phase": 0.0, "reference_amplitude": 0.0, "reference_frequency": 0.0})
        initialize_command_at_reset(self.env, first.target_deg, first.trigger)
        info = self.unwrapped.telemetry()
        info["protocol_scenario"] = s.name
        return self._command_observation(), info

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward, terms = result_reward(info)
        info["reward_components"] = terms
        info["protocol_scenario"] = self.scenario.name
        finished = self.unwrapped.time + 1e-10 >= self.scenario.duration_s
        if not terminated and not finished:
            if truncated:
                self.unwrapped._episode_done = False
            truncated = False
            # Reveal the next command only after scoring the previous action.
            observation = self._command_observation()
        elif not terminated:
            truncated = True
            self.unwrapped._episode_done = True
        if terminated or truncated:
            self.completed_episodes += 1
        return observation, reward, terminated, truncated, info
