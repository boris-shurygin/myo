"""Native delayed/compliant rollouts for short-horizon acceleration supervision.

Only the first 20 ms command is varied. Later commands are identical recorded
teacher commands in every branch, so their consequences are not attributed to
the first command. No production physics is replaced or differentiated by hand.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from .env import PreacherCurlEnv
from .movement_curriculum import MovementProtocol


STATE_FIELDS = (
    "payload_mass", "_previous_excitation", "_muscle_outputs", "_muscle_torque",
    "_step_count", "_phase", "_amplitude", "_frequency", "_reward_components",
    "_episode_done", "_has_reset", "command", "_manual_command", "_next_command_time",
    "_sensor_history", "_action_history", "_command_history", "_passive_torque",
    "_step_peaks", "moment_arms", "_np_random", "_np_random_seed",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class NativeState:
    integration: np.ndarray
    fields: dict
    muscles: list


def capture(env):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, specification))
    mujoco.mj_getState(env.model, env.data, state, specification)
    return NativeState(state,
        copy.deepcopy({name: getattr(env, name) for name in STATE_FIELDS}),
        copy.deepcopy(env.muscles))


class NativeResponse:
    """Reusable independent native simulator with complete state restoration."""

    def __init__(self, config, horizon_steps=6):
        if horizon_steps * config.control_dt <= config.motor_delay:
            raise ValueError("Response horizon must extend beyond the motor delay")
        self.env = PreacherCurlEnv(config)
        self.env.reset(seed=0, options={"payload_mass": config.payload_mass, "capacity": 1.})
        self.horizon_steps = horizon_steps
        self.control_steps = 0

    def restore(self, state):
        if state.fields["payload_mass"] != self.env.config.payload_mass:
            raise ValueError("Native response payload differs from the sampled state")
        mujoco.mj_setState(self.env.model, self.env.data, state.integration,
                          mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.env.model, self.env.data)
        for name, value in copy.deepcopy(state.fields).items():
            setattr(self.env, name, value)
        self.env.muscles = copy.deepcopy(state.muscles)

    def step(self, command):
        before = float(self.env.data.qvel[self.env.dof_index])
        _, _, terminated, truncated, _ = self.env.step(command)
        self.control_steps += 1
        if terminated:
            raise RuntimeError("Native response physically terminated")
        if truncated:
            self.env._episode_done = False  # Time limit only; preserve all physical state.
        return (float(self.env.data.qvel[self.env.dof_index]) - before) / self.env.control_dt

    def reference(self, state):
        self.restore(state)
        commands, accelerations = [], []
        for _ in range(self.horizon_steps):
            command = self.env.baseline_action()
            commands.append(command.copy())
            accelerations.append(self.step(command))
        return np.asarray(commands), np.asarray(accelerations)

    def response(self, state, continuation, first_command):
        self.restore(state)
        return np.asarray([self.step(first_command if i == 0 else command)
                           for i, command in enumerate(continuation)])

    def linearize(self, state, continuation, anchor, epsilon=.001):
        anchor = np.asarray(anchor, dtype=np.float64)
        base = self.response(state, continuation, anchor)
        jacobian = np.empty((self.horizon_steps, len(anchor)))
        for i in range(len(anchor)):
            high, low = anchor.copy(), anchor.copy()
            high[i] = min(1., anchor[i] + epsilon)
            low[i] = max(0., anchor[i] - epsilon)
            jacobian[:, i] = (self.response(state, continuation, high)
                              - self.response(state, continuation, low)) / (high[i] - low[i])
        return base, jacobian

    def close(self):
        self.env.close()


def linear_response(commands, anchors, accelerations, jacobians):
    """First-order surrogate at measured anchors; refreshed between rounds."""
    return accelerations + torch.einsum("bhi,bi->bh", jacobians, commands - anchors)


def collect(config, kernel, model=None, *, initial_angle=20., stride=5, episode_offset=0,
            sample_steps=None):
    env = MovementProtocol(PreacherCurlEnv(config), 0)
    env.scenarios = tuple(replace(s, initial_angle_deg=initial_angle) for s in env.scenarios)
    rows, states, steps, failures = [], [], 0, 0
    try:
        for episode in range(len(env.scenarios)):
            observation, _ = env.reset(seed=0)
            if sample_steps is not None:
                points = np.asarray(sample_steps) * config.control_dt
                edges = np.r_[0., (points[:-1] + points[1:]) / 2, env.scenario.duration_s]
                weights = np.diff(edges) * len(points) / env.scenario.duration_s
            index = 0
            while True:
                base = env.unwrapped
                if (index % stride == 0 if sample_steps is None else index in sample_steps):
                    state = capture(base)
                    continuation, target = kernel.reference(state)
                    rows.append({"observations": observation.copy(), "continuations": continuation,
                                 "targets": target, "episode_ids": episode_offset + episode,
                                 "sample_weights": 1. if sample_steps is None else weights[sample_steps.index(index)]})
                    states.append(state)
                command = (base.baseline_action() if model is None
                           else model.predict(observation, deterministic=True)[0])
                observation, _, terminated, truncated, _ = env.step(command)
                steps += 1
                index += 1
                if terminated or truncated:
                    failures += int(terminated)
                    break
    finally:
        env.close()
    data = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    data["observations"] = data["observations"].astype(np.float32)
    return data, states, {"physical_control_steps": steps, "physical_failures": failures,
                         "samples": len(states), "episodes": len(env.scenarios),
                         "initial_angle_deg": initial_angle,
                         "driver": "heuristic" if model is None else "learned_actor"}


def mean_commands(model, observations):
    return model.policy.get_distribution(torch.as_tensor(observations, dtype=torch.float32)).distribution.mean


def linearize_dataset(model, data, states, kernel):
    with torch.no_grad():
        anchors = mean_commands(model, data["observations"]).numpy()
    values, jacobians = [], []
    for i, (state, continuation, anchor) in enumerate(zip(states, data["continuations"], anchors)):
        value, jacobian = kernel.linearize(state, continuation, anchor)
        values.append(value)
        jacobians.append(jacobian)
        if (i + 1) % 60 == 0:
            print(f"native linearizations {i + 1}/{len(states)}", flush=True)
    return {"anchors": anchors, "accelerations": np.asarray(values),
            "jacobians": np.asarray(jacobians)}


def exact_validation(model, data, states, kernel, linear=None):
    with torch.no_grad():
        commands = mean_commands(model, data["observations"]).numpy()
    actual = np.asarray([kernel.response(state, continuation, command)
                         for state, continuation, command in zip(states, data["continuations"], commands)])
    start = round(kernel.env.config.motor_delay / kernel.env.control_dt)
    error = actual[:, start:] - data["targets"][:, start:]
    result = {"native_acceleration_rmse_rad_s2": float(np.sqrt(np.mean(error ** 2))),
              "native_acceleration_bias_rad_s2": float(np.mean(error))}
    if linear is not None:
        prediction = linear["accelerations"] + np.einsum("bhi,bi->bh", linear["jacobians"],
            commands - linear["anchors"])
        result["linearization_rmse_rad_s2"] = float(np.sqrt(np.mean((prediction[:, start:] - actual[:, start:]) ** 2)))
        result["maximum_anchor_command_distance"] = float(np.max(np.abs(commands - linear["anchors"])))
    return result
