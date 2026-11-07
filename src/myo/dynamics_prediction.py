"""Action-conditioned 20 ms dynamics experiment; never replaces native physics.

The transport queue is explicit and shifted exactly. A residual MLP learns the
physical transition from the current compact state and the excitation arriving
during this interval. Future commands are inputs to subsequent transitions,
not observations available to the encoder at the initial time.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .runtime import config_digest, dynamics_digest


def state_names(muscle_names, queue_steps):
    return ["angle_rad", "angular_velocity_rad_s", *(
        f"{field}/{name}" for field in ("activation", "fiber_length_m", "capacity_proxy")
        for name in muscle_names), *(
        f"pending/{age}/{name}" for age in range(queue_steps) for name in muscle_names)]


def compact_state(env):
    """SI physical state plus control-interval queues, oldest block first.

    This experiment holds each excitation constant over a control interval;
    compressing the physics-rate queue is lossless only at those boundaries.
    """
    blocks = round(env.config.motor_delay / env.control_dt)
    if blocks < 1 or not np.isclose(blocks * env.control_dt, env.config.motor_delay):
        raise ValueError("Experiment requires an integer, positive control-step delay")
    queues = []
    for muscle in env.muscles:
        pending = muscle.delay.pending.reshape(blocks, env.config.frame_skip)
        if not np.all(pending == pending[:, :1]):
            raise ValueError("Queue contains commands changing inside a control interval")
        queues.append(pending[:, 0])
    return np.concatenate((
        [env.data.qpos[env.qpos_index], env.data.qvel[env.dof_index]],
        [m.activation for m in env.muscles], [m.fiber_length for m in env.muscles],
        [m.capacity for m in env.muscles], np.asarray(queues).T.ravel(),
    )).astype(np.float32)


def contract(env):
    queue_steps = round(env.config.motor_delay / env.control_dt)
    return {
        "schema": "myo.action_dynamics.v1", "config": asdict(env.config),
        "config_sha256": config_digest(env.config), "dynamics_sha256": dynamics_digest(),
        "geometry_sha256": env.geometry.fingerprint if env.geometry else None,
        "state_names": state_names(env.muscle_names, queue_steps),
        "action_mapping": list(env.muscle_names),
        "action_semantics": "excitation_0_1_held_for_control_dt_before_transport_delay",
        "physics_dt_s": env.config.physics_dt, "control_dt_s": env.control_dt,
        "payload_mass_kg": env.payload_mass, "queue_steps": queue_steps,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "fixed payload and mount; compact simulator state, not biological sensors",
    }


def sequence_windows(episodes, horizon):
    """Whole windows inside each episode, with no hidden reset transitions."""
    initial, actions, targets, ids = [], [], [], []
    for episode in episodes:
        states, commands = episode["states"], episode["actions"]
        for start in range(len(commands) - horizon + 1):
            initial.append(states[start])
            actions.append(commands[start:start + horizon])
            targets.append(states[start + 1:start + horizon + 1])
            ids.append(episode["id"])
    return {"initial": np.asarray(initial, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "targets": np.asarray(targets, dtype=np.float32), "episode_ids": np.asarray(ids)}


class TransitionModel(nn.Module):
    """Learn muscle/body dynamics; retain known transport timing structurally."""

    def __init__(self, muscles=6, queue_steps=2, hidden=128):
        super().__init__()
        self.muscles, self.queue_steps, self.hidden = muscles, queue_steps, hidden
        self.physical_size = 2 + 3 * muscles
        width = self.physical_size + muscles
        self.register_buffer("input_mean", torch.zeros(width))
        self.register_buffer("input_scale", torch.ones(width))
        self.register_buffer("delta_mean", torch.zeros(self.physical_size))
        self.register_buffer("delta_scale", torch.ones(self.physical_size))
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, self.physical_size))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def physical_input(self, state):
        # Only the oldest pending block reaches the muscles in this interval.
        return state[..., :self.physical_size + self.muscles]

    def fit_normalization(self, episodes):
        states = np.concatenate([e["states"][:-1] for e in episodes])
        next_states = np.concatenate([e["states"][1:] for e in episodes])
        x = states[:, :self.physical_size + self.muscles]
        delta = next_states[:, :self.physical_size] - states[:, :self.physical_size]
        # Floors keep nearly constant channels numerically well conditioned.
        floors = np.r_[1e-4, 1e-3, np.full(self.muscles, 1e-4),
                       np.full(self.muscles, 1e-6), np.full(self.muscles, 1e-6)]
        with torch.no_grad():
            for name, value in (("input_mean", x.mean(0)),
                                ("input_scale", np.maximum(x.std(0), 1e-5)),
                                ("delta_mean", delta.mean(0)),
                                ("delta_scale", np.maximum(delta.std(0), floors))):
                getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def forward(self, state, action):
        x = (self.physical_input(state) - self.input_mean) / self.input_scale
        physical = state[..., :self.physical_size] + self.delta_mean + self.delta_scale * self.net(x)
        pending = state[..., self.physical_size:].reshape(*state.shape[:-1], self.queue_steps, self.muscles)
        queue = torch.cat((pending[..., 1:, :], action.unsqueeze(-2)), dim=-2)
        return torch.cat((physical, queue.flatten(-2)), dim=-1)

    def rollout(self, state, actions, *, detach_between_steps=False):
        predictions = []
        for step in range(actions.shape[-2]):
            state = self(state, actions[..., step, :])
            predictions.append(state)
            if detach_between_steps:
                state = state.detach()
        return torch.stack(predictions, dim=-2)


def save_predictor(path, model, metadata):
    path = Path(path)
    torch.save({"state_dict": model.state_dict(), "architecture": {
        "muscles": model.muscles, "queue_steps": model.queue_steps, "hidden": model.hidden}}, path)
    sidecar = {**metadata, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(str(path) + ".metadata.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")


def load_predictor(path, expected_contract):
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Predictor weights differ from metadata")
    for key in ("schema", "config_sha256", "dynamics_sha256", "geometry_sha256", "state_names",
                "action_mapping", "action_semantics", "physics_dt_s", "control_dt_s",
                "payload_mass_kg", "queue_steps", "source_sha256"):
        if metadata.get(key) != expected_contract.get(key):
            raise ValueError(f"Predictor contract mismatch: {key}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = TransitionModel(**saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), metadata
