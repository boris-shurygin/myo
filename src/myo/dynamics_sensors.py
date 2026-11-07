"""Causal fiber-velocity and tendon-force features for the dynamics predictor.

The learned output remains the same 20 physical increments. Past fiber lengths
are copied explicitly; future force features are reconstructed from predicted
pose/fiber length and a differentiable table of the fixed MyoArm paths. This
sensor calculation never advances or replaces native physics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dynamics_prediction import compact_state, contract
from .train_dynamics_wide import WideTransitionModel, wide_contract


def sensor_contract(env):
    base = wide_contract(contract(env))
    muscles = len(env.muscles)
    return {**base, "schema": "myo.action_dynamics_sensors.v1",
        "wide_source_sha256": base["source_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_names": [*base["state_names"], *(f"previous_fiber_length_m/{n}" for n in env.muscle_names)],
        "neural_input_names": [*base["state_names"][:2 + 3*muscles],
            *(f"delivered_excitation/{n}" for n in env.muscle_names),
            *(f"fiber_velocity_mean_m_s/{n}" for n in env.muscle_names),
            *(f"tendon_force_N/{n}" for n in env.muscle_names)],
        "sensor_parameters": {"velocity_window_s": env.control_dt,
            "reset_velocity": "zero; previous fiber length equals current length",
            "geometry_points": 2049, "angle_interval_rad": [0., float(np.deg2rad(160.))],
            "outside_interval": "linear extrapolation, no clipping of predicted pose",
            "force": "stiffness * max(path_length(angle) - fiber_length - slack_length, 0)",
            "parameters": [{"tendon_slack_length_m": m.params.tendon_slack_length,
                            "tendon_stiffness_N_m": m.params.tendon_stiffness} for m in env.muscles]}}


def augment_episodes(episodes, muscles=6):
    start = 2 + muscles
    result = []
    for episode in episodes:
        states = episode["states"]
        previous = np.concatenate((states[:1, start:start + muscles], states[:-1, start:start + muscles]))
        result.append({**episode, "states": np.concatenate((states, previous), axis=1),
                       "actions": episode["actions"].copy()})
    return result


def sensor_state(env, previous_fiber_lengths=None):
    state = compact_state(env)
    muscles = len(env.muscles)
    current = state[2 + muscles:2 + 2*muscles]
    previous = current if previous_fiber_lengths is None else np.asarray(previous_fiber_lengths, dtype=np.float32)
    return np.concatenate((state, previous)).astype(np.float32)


class SensorTransitionModel(WideTransitionModel):
    def __init__(self, muscles=6, queue_steps=2, hidden_layers=(1024, 1024, 512),
                 control_dt=.02, geometry_points=2049):
        super().__init__(muscles, queue_steps, hidden_layers)
        self.control_dt = control_dt
        self.base_state_size = self.physical_size + queue_steps * muscles
        self.fiber_slice = slice(2 + muscles, 2 + 2*muscles)
        old = self.net[0]
        widened = nn.Linear(old.in_features + 2*muscles, old.out_features)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, :old.in_features].copy_(old.weight)
            widened.bias.copy_(old.bias)
        self.net[0] = widened
        # New columns start at zero, preserving the original initial function.
        self.input_mean = torch.zeros(widened.in_features)
        self.input_scale = torch.ones(widened.in_features)
        self.register_buffer("angle_grid", torch.linspace(0., float(np.deg2rad(160.)), geometry_points))
        self.register_buffer("path_grid", torch.zeros(geometry_points, muscles))
        self.register_buffer("tendon_stiffness", torch.ones(muscles))
        self.register_buffer("tendon_slack_length", torch.zeros(muscles))

    def configure_geometry(self, env):
        angles = self.angle_grid.numpy()
        lengths = np.asarray([env.geometry.lengths_at(float(q)) if env.geometry else env.muscle_lengths(float(q))
                              for q in angles])
        with torch.no_grad():
            self.path_grid.copy_(torch.tensor(lengths, dtype=torch.float32))
            self.tendon_stiffness.copy_(torch.tensor([m.params.tendon_stiffness for m in env.muscles]))
            self.tendon_slack_length.copy_(torch.tensor([m.params.tendon_slack_length for m in env.muscles]))

    def tendon_forces(self, state):
        angle = state[..., 0].contiguous()
        right = torch.searchsorted(self.angle_grid, angle).clamp(1, len(self.angle_grid) - 1)
        left = right - 1
        fraction = ((angle - self.angle_grid[left]) / (self.angle_grid[right] - self.angle_grid[left])).unsqueeze(-1)
        path = self.path_grid[left] + fraction * (self.path_grid[right] - self.path_grid[left])
        extension = path - state[..., self.fiber_slice] - self.tendon_slack_length
        return self.tendon_stiffness * torch.relu(extension)

    def physical_input(self, state):
        velocity = (state[..., self.fiber_slice] - state[..., self.base_state_size:]) / self.control_dt
        return torch.cat((state[..., :self.physical_size + self.muscles], velocity, self.tendon_forces(state)), dim=-1)

    def fit_normalization(self, episodes):
        states = np.concatenate([e["states"][:-1] for e in episodes])
        following = np.concatenate([e["states"][1:] for e in episodes])
        with torch.no_grad():
            x = self.physical_input(torch.tensor(states)).numpy()
        delta = following[:, :self.physical_size] - states[:, :self.physical_size]
        floors = np.r_[1e-4, 1e-3, np.full(self.muscles, 1e-4),
                       np.full(self.muscles, 1e-6), np.full(self.muscles, 1e-6)]
        with torch.no_grad():
            for name, value in (("input_mean", x.mean(0)), ("input_scale", np.maximum(x.std(0), 1e-5)),
                                ("delta_mean", delta.mean(0)), ("delta_scale", np.maximum(delta.std(0), floors))):
                getattr(self, name).copy_(torch.tensor(value, dtype=torch.float32))

    def forward(self, state, action):
        x = (self.physical_input(state) - self.input_mean) / self.input_scale
        physical = state[..., :self.physical_size] + self.delta_mean + self.delta_scale * self.net(x)
        pending = state[..., self.physical_size:self.base_state_size].reshape(*state.shape[:-1], self.queue_steps, self.muscles)
        queue = torch.cat((pending[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        return torch.cat((physical, queue.flatten(-2), state[..., self.fiber_slice]), dim=-1)


def load_sensor_predictor(path, expected):
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Predictor weights differ from metadata")
    for key in ("schema", "config_sha256", "dynamics_sha256", "geometry_sha256", "state_names",
                "neural_input_names", "sensor_parameters", "action_mapping", "action_semantics",
                "physics_dt_s", "control_dt_s", "payload_mass_kg", "queue_steps", "source_sha256",
                "base_source_sha256", "wide_source_sha256"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"Predictor contract mismatch: {key}")
    if metadata["schema"] != "myo.action_dynamics_sensors.v1":
        raise ValueError("Not a sensor-augmented predictor")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != metadata["architecture"]:
        raise ValueError("Predictor architecture differs from metadata")
    model = SensorTransitionModel(**saved["architecture"], control_dt=expected["control_dt_s"],
                                  geometry_points=expected["sensor_parameters"]["geometry_points"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata
