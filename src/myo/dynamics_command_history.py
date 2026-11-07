"""Optional physical-time lags of delivered excitation, with an exact FIFO.

This history precedes activation, but follows the transport delay. It is not
the queue of commands still travelling toward the muscles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dynamics_sensors import SensorTransitionModel, sensor_contract


def history_contract(env, lags_s=(.02, .04, .06)):
    base = sensor_contract(env)
    steps = [round(lag/env.control_dt) for lag in lags_s]
    if any(step < 1 or not np.isclose(step*env.control_dt, lag) for step, lag in zip(steps, lags_s)):
        raise ValueError("Delivered-command lags must be positive integer control intervals")
    return {**base, "schema": "myo.action_dynamics_command_history.v1",
        "sensor_source_sha256": base["source_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "history_parameters": {"lags_s": list(lags_s), "stored_steps": max(steps),
            "order": "newest delivered first; samples at interval start",
            "reset": "constant prehistory equal to reset transport queue"},
        "neural_input_names": [*base["neural_input_names"], *(f"delivered_excitation_lag_{lag:g}s/{name}"
            for lag in lags_s for name in env.muscle_names)],
        "state_names": [*base["state_names"], *(f"delivered_history/{age}/{name}"
            for age in range(1, max(steps)+1) for name in env.muscle_names)]}


def add_delivered_history(episode, dt, max_lag_s=.06, muscles=6):
    count = round(max_lag_s/dt)
    states = episode["states"]
    physical = 2+3*muscles
    # The oldest pending block is delivered during the outgoing interval.
    delivered = states[:-1, physical:physical+muscles]
    prehistory = np.repeat(states[:1, physical:physical+muscles], count, axis=0)
    source = np.concatenate((prehistory, delivered))
    previous = np.stack([source[count-age:len(source)-age+1] for age in range(1, count+1)], axis=1)
    return {**episode, "states": np.concatenate((states, previous.reshape(len(states), -1)), axis=1)}


class HistoryTransitionModel(SensorTransitionModel):
    def __init__(self, muscles=6, queue_steps=2, hidden_layers=(1024, 1024, 512),
                 control_dt=.02, geometry_points=2049, lags_s=(.02, .04, .06)):
        super().__init__(muscles, queue_steps, hidden_layers, control_dt, geometry_points)
        self.lags_s = tuple(lags_s)
        self.lag_steps = tuple(round(lag/control_dt) for lag in lags_s)
        self.history_steps = max(self.lag_steps)
        self.sensor_state_size = self.base_state_size+muscles
        old = self.net[0]
        widened = nn.Linear(old.in_features+len(lags_s)*muscles, old.out_features)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, :old.in_features].copy_(old.weight)
            widened.bias.copy_(old.bias)
        self.net[0] = widened
        self.input_mean = torch.zeros(widened.in_features)
        self.input_scale = torch.ones(widened.in_features)

    def physical_input(self, state):
        current = super().physical_input(state[..., :self.sensor_state_size])
        history = state[..., self.sensor_state_size:].reshape(*state.shape[:-1], self.history_steps, self.muscles)
        return torch.cat((current, *(history[..., step-1, :] for step in self.lag_steps)), dim=-1)

    def forward(self, state, action):
        x = (self.physical_input(state)-self.input_mean)/self.input_scale
        physical = state[..., :self.physical_size]+self.delta_mean+self.delta_scale*self.net(x)
        pending = state[..., self.physical_size:self.base_state_size].reshape(*state.shape[:-1], self.queue_steps, self.muscles)
        queue = torch.cat((pending[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        previous = state[..., self.sensor_state_size:].reshape(*state.shape[:-1], self.history_steps, self.muscles)
        history = torch.cat((pending[..., :1, :], previous[..., :-1, :]), dim=-2)
        return torch.cat((physical, queue.flatten(-2), state[..., self.fiber_slice], history.flatten(-2)), dim=-1)


def load_history_predictor(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Predictor weights differ from metadata")
    for key in ("schema", "config_sha256", "dynamics_sha256", "geometry_sha256", "state_names",
                "neural_input_names", "sensor_parameters", "history_parameters", "action_mapping",
                "action_semantics", "physics_dt_s", "control_dt_s", "payload_mass_kg", "queue_steps",
                "source_sha256", "base_source_sha256", "wide_source_sha256", "sensor_source_sha256"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"Predictor contract mismatch: {key}")
    if metadata["schema"] != "myo.action_dynamics_command_history.v1":
        raise ValueError("Not a delivered-history predictor")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Predictor architecture differs from metadata")
    model = HistoryTransitionModel(**saved["architecture"], control_dt=expected["control_dt_s"],
        geometry_points=expected["sensor_parameters"]["geometry_points"], lags_s=expected["history_parameters"]["lags_s"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata
