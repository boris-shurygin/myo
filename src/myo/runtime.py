"""Load a controller together with its simulation and observation contract."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

from .env import PreacherCurlConfig, PreacherCurlEnv


def read_config(path: Path | None) -> PreacherCurlConfig:
    return PreacherCurlConfig(**json.loads(path.read_text(encoding="utf-8"))) if path else PreacherCurlConfig()


@lru_cache(maxsize=32)
def config_digest(config: PreacherCurlConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def dynamics_digest() -> str:
    """V0 reproducibility guard for its original model and physical implementation."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    digest.update(f"mujoco={version('mujoco')}\n".encode())
    for name in ("env.py", "commands.py", "geometry.py", "muscle.py", "fatigue.py", "delay.py", "assets/preacher_curl.xml"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def policy_contract(env: PreacherCurlEnv) -> dict:
    return {
        "action_mapping": list(env.muscle_names),
        "action_semantics": "excitation_0_1_before_motor_delay; trigger_is_not_output_multiplier",
        "physics_dt_s": env.config.physics_dt,
        "control_dt_s": env.control_dt,
        "history_steps": env.config.history_steps if env.config.task_mode != "reference" else 0,
        "history_initialization": "repeat_reset_sensor_command_and_initial_excitation",
        "geometry_sha256": env.geometry.fingerprint if env.geometry else "toy_v0_in_dynamics_digest",
    }


def load_controller(
    controller: str = "heuristic",
    config_path: Path | None = None,
    predictor_path: Path | None = None,
    render_mode: str | None = None,
) -> tuple[gym.Env, Callable[[np.ndarray], np.ndarray], dict]:
    """Return env, causal action function, and provenance.

    A learned policy must have the metadata emitted by ``myo.train``. Explicit
    configuration/predictor mismatches fail instead of silently changing physics.
    """
    metadata: dict = {"controller": controller}
    checkpoint = None
    if controller in ("heuristic", "zero"):
        config = read_config(config_path)
    else:
        checkpoint = Path(controller)
        sidecar = Path(str(checkpoint) + ".metadata.json")
        if not checkpoint.is_file() or not sidecar.is_file():
            raise ValueError("A PPO .zip checkpoint and its .zip.metadata.json sidecar are required.")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError("Unsupported policy metadata version.")
        if metadata.get("dynamics_sha256") != dynamics_digest():
            raise ValueError("Policy dynamics source/model fingerprint does not match this installation.")
        if metadata.get("features_sha256") is not None:
            features = Path(__file__).with_name("policy_features.py")
            if hashlib.sha256(features.read_bytes()).hexdigest() != metadata["features_sha256"]:
                raise ValueError("Policy feature preprocessing differs from its saved implementation.")
        config = PreacherCurlConfig(**metadata["config"])
        if config_path and asdict(read_config(config_path)) != asdict(config):
            raise ValueError("Runtime configuration differs from the policy's saved configuration.")
        expected_predictor = metadata.get("predictor_sha256")
        if expected_predictor:
            if predictor_path is None:
                stored = Path(metadata["predictor_path"])
                predictor_path = stored if stored.is_absolute() else sidecar.parent / stored
            if not predictor_path.is_file():
                raise ValueError("The policy's predictor is missing; provide its file with --predictor.")
            if hashlib.sha256(predictor_path.read_bytes()).hexdigest() != expected_predictor:
                raise ValueError("Predictor contents differ from the artifact used by this policy.")
        elif predictor_path is not None:
            raise ValueError("This policy was trained without predictor observations.")
    env: gym.Env = PreacherCurlEnv(config=config, render_mode=render_mode)
    try:
        if predictor_path:
            from .predictor import PredictorObservationWrapper

            env = PredictorObservationWrapper(env, predictor_path)
        if checkpoint:
            if metadata.get("contract") != policy_contract(env.unwrapped):
                raise ValueError("Runtime action/timing/geometry contract differs from the checkpoint.")
            if metadata.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError("Policy checkpoint contents differ from its metadata.")
            if metadata["observation_names"] != list(env.observation_names):
                raise ValueError("Runtime observation order differs from the policy's saved contract.")
            try:
                import torch
                from stable_baselines3 import PPO
            except ImportError as exc:
                raise RuntimeError("Install the train extra to load a PPO controller.") from exc
            torch.set_num_threads(1)
            model = PPO.load(checkpoint, device="cpu")

            def act(observation: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)

        elif controller == "heuristic":
            def act(observation: np.ndarray) -> np.ndarray:
                return env.unwrapped.baseline_action()
        else:
            def act(observation: np.ndarray) -> np.ndarray:
                return np.zeros(env.action_space.shape, dtype=np.float32)
        return env, act, metadata
    except Exception:
        env.close()
        raise

