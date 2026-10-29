"""Versioned, SI-unit render snapshots. No physics runs in the transport."""

from __future__ import annotations

import json
from numbers import Integral
import socket

from .env import PreacherCurlEnv
from .runtime import config_digest, dynamics_digest


def make_packet(env: PreacherCurlEnv, sequence: int, episode: int, run_id: str,
                prediction: dict | None = None) -> dict:
    if (any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in (sequence, episode)) or not isinstance(run_id, str) or not run_id):
        raise ValueError("A packet needs nonnegative sequence/episode and a nonempty run_id.")
    state = env.telemetry()
    muscles = []
    for i, muscle in enumerate(env.muscles):
        muscles.append({
            "name": state["muscle_names"][i],
            "excitation_command": state["excitation"][i],
            "excitation_delayed": state["delayed_excitation"][i],
            "activation": state["activation"][i],
            "tendon_force_N": state["tendon_force_N"][i],
            "reference_max_force_N": muscle.params.max_isometric_force,
            "force_normalized": state["tendon_force_N"][i] / muscle.params.max_isometric_force,
            "tendon_strain": state["tendon_strain"][i],
            "fiber_length_m": state["fiber_length_m"][i],
            "fiber_velocity_m_s": state["fiber_velocity_m_s"][i],
            "remaining_capacity": state["capacity"][i],
            "fatigue_proxy": state["fatigue"][i],
        })
    packet = {
        "schema": "myo.character_state.v1",
        "model": "preacher_curl_1dof_v0",
        "config_sha256": config_digest(env.config),
        "dynamics_sha256": dynamics_digest(),
        "run_id": run_id,
        "sequence": int(sequence),
        "episode": int(episode),
        "simulation_time_s": state["t"],
        "physics_dt_s": env.config.physics_dt,
        "control_dt_s": env.control_dt,
        "coordinates": {"handedness": "right", "up": "Z", "length_unit": "m", "angle_unit": "rad"},
        "joints": [{
            "name": "elbow",
            "angle_rad": state["q"],
            "angular_velocity_rad_s": state["qvel"],
            "target_angle_rad": state["target_q"],
            "target_angular_velocity_rad_s": state["target_qvel"],
        }],
        "muscles": muscles,
        "prediction": prediction,
        "metrics": {
            "tracking_error_rad": state["tracking_error"][0],
            "tracking_error_rad_s": state["tracking_error"][1],
            "muscle_torque_Nm": state["muscle_torque_Nm"],
            "summed_tendon_force_proxy_N": state["summed_tendon_force_proxy_N"],
            "pain_proxy_components": state.get("pain_proxy_components", {}),
            "reward_components": state["reward_components"],
            "equilibrium_residual_N": state["equilibrium_residual_N"],
        },
    }
    # Validate numeric finiteness at the serialization boundary.
    encode_packet(packet)
    return packet


def encode_packet(packet: dict) -> bytes:
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class LoopbackPublisher:
    """Optional one-way UDP to a local renderer; no external network by default."""

    def __init__(self, port: int):
        if not 1 <= port <= 65535:
            raise ValueError("UDP port must be in [1, 65535].")
        self.address = ("127.0.0.1", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, packet: dict) -> None:
        payload = encode_packet(packet)
        if len(payload) > 65507:
            raise ValueError("Snapshot exceeds the UDP datagram limit; use a different transport.")
        self.socket.sendto(payload, self.address)

    def close(self) -> None:
        self.socket.close()

