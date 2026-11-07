"""Explicit diagnostic actuators; the production series-elastic arm is unchanged."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import numpy as np

from .env import PreacherCurlConfig, PreacherCurlEnv
from .fatigue import advance_capacity
from .muscle import MuscleOutput, SeriesElasticMuscle
from .runtime import dynamics_digest, policy_contract


MODES = ("full", "instant_rigid")


class InstantRigidMuscle(SeriesElasticMuscle):
    """Taut tendon has fixed length; activation equals the current command.

    Reuse the original active/passive force-length and force-velocity curves,
    damping and capacity law. There is no tendon energy, fiber equilibrium
    solve, transport queue or activation filter. A short path can leave the
    tensile-only tendon slack; compressive fiber force is not transmitted.
    """

    def reset(self, mt_length, activation=0., capacity=1.):
        self._check_length(mt_length)
        self._check_unit(activation, "activation")
        self._check_unit(capacity, "capacity")
        self.activation, self.capacity = float(activation), float(capacity)
        self.delay.reset(activation)
        self._initialized = True
        self.last_output = self._rigid_output(activation, mt_length, 0.)
        self.fiber_length = self.last_output.fiber_length
        return self.last_output

    def _rigid_output(self, command, mt_length, mt_velocity):
        self._check_length(mt_length)
        if not np.isfinite(mt_velocity):
            raise ValueError("Muscle path velocity must be finite")
        p = self.params
        raw_fiber = mt_length - p.tendon_slack_length
        fiber = max(self.minimum_fiber_length, raw_fiber)
        taut = raw_fiber >= self.minimum_fiber_length
        velocity = float(mt_velocity) if taut else 0.
        active, passive, damping, *_ = self._fiber_forces(
            fiber, self.activation, self.capacity, velocity)
        force = max(0., active + passive + damping) if taut else 0.
        stretch = max(0., fiber / p.optimal_fiber_length - 1.)
        return MuscleOutput(
            excitation=float(command), delayed_excitation=float(command),
            activation=self.activation, capacity=self.capacity,
            fiber_length=float(fiber), fiber_velocity=velocity,
            tendon_length=float(mt_length - fiber), tendon_strain=0.,
            tendon_force=float(force), active_force=float(active), passive_force=float(passive),
            damping_force=float(damping), tendon_energy=0.,
            passive_energy=p.max_isometric_force * p.passive_stiffness
                * p.optimal_fiber_length * stretch ** 3 / 3,
            damping_power=float(damping * velocity), mt_power=float(-force * mt_velocity),
            equilibrium_residual=float(force - active - passive - damping),
            solver_iterations=0, at_fiber_bound=not taut)

    def step(self, excitation, mt_length, mt_velocity=0., dt=None):
        if not self._initialized:
            raise RuntimeError("Reset muscle before stepping")
        self._check_unit(excitation, "excitation")
        if dt is not None and not np.isclose(dt, self.dt, rtol=1e-12, atol=0):
            raise ValueError("Muscle timestep mismatch")
        self.activation = float(excitation)
        p = self.params
        self.capacity = advance_capacity(self.capacity, self.activation, self.dt,
            p.fatigue_rate, p.recovery_rate, p.rest_recovery_multiplier, p.rest_activation_threshold)
        self.last_output = self._rigid_output(excitation, mt_length, mt_velocity)
        self.fiber_length = self.last_output.fiber_length
        return self.last_output

    def sample(self, mt_length, mt_velocity=0.):
        if self.last_output is None:
            raise RuntimeError("Reset muscle before sampling")
        return self._rigid_output(self.last_output.excitation, mt_length, mt_velocity)


class InstantRigidArmEnv(PreacherCurlEnv):
    """Same skeleton/task with explicit zero padding for removed delay inputs."""

    def __init__(self, source_config, render_mode=None):
        if source_config.task_mode == "reference":
            raise ValueError("Actuator comparison requires command observations")
        self.padding_count = 0
        super().__init__(replace(source_config, motor_delay=0.), render_mode=render_mode)
        self.muscles = [InstantRigidMuscle(m.params, self.config.physics_dt) for m in self.muscles]
        slots = round(source_config.motor_delay / source_config.physics_dt)
        self.padding_count = slots * len(self.muscles)
        self.observation_names.extend(f"inactive_delay_padding/{name}/{i}"
                                     for name in self.muscle_names for i in range(slots))
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(len(self.observation_names),), dtype=np.float32)

    def _get_obs(self):
        obs = super()._get_obs()
        return np.concatenate((obs, np.zeros(self.padding_count, dtype=np.float32)))

    def telemetry(self):
        info = super().telemetry()
        info["fiber_length_m"] = [o.fiber_length for o in self._state_outputs()]
        info["diagnostic_actuator"] = "instant_rigid"
        return info


def make_environment(mode, source_config, render_mode=None):
    if mode not in MODES:
        raise ValueError("Unknown actuator comparison mode")
    cls = PreacherCurlEnv if mode == "full" else InstantRigidArmEnv
    return cls(source_config, render_mode=render_mode)


def descriptor(mode, source_config):
    if mode not in MODES:
        raise ValueError("Unknown actuator comparison mode")
    return {"name": "actuator_ablation_v1", "mode": mode,
            "source_config": asdict(source_config), "base_dynamics_sha256": dynamics_digest(),
            "actuator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "transport_delay_s": source_config.motor_delay if mode == "full" else 0.,
            "activation_time_constant_s": .015 if mode == "full" else 0.,
            "deactivation_time_constant_s": .05 if mode == "full" else 0.,
            "tendon": "independent_series_elastic_state" if mode == "full" else "inextensible_tensile_only",
            "observation_padding": "none" if mode == "full" else "zero slots replacing the removed motor queue",
            "scope": "diagnostic physics; joint driven only by muscle forces; no teacher during learning"}


def experiment_digest(details):
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()


def contract(env, mode):
    result = policy_contract(env)
    if mode != "full":
        result["action_semantics"] = "excitation_equals_instant_activation_0_1; rigid_tendon_diagnostic"
        result["inactive_delay_padding"] = env.padding_count
    return result


def load_ablation_controller(path, render_mode=None):
    """Load only with the saved experimental physics; normal loader rejects it."""
    import torch
    from stable_baselines3 import PPO
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".metadata.json").read_text())
    details = metadata["actuator_ablation"]
    config = PreacherCurlConfig(**details["source_config"])
    if (metadata.get("schema_version") != 2 or details != descriptor(details["mode"], config)
            or metadata["dynamics_sha256"] != experiment_digest(details)
            or metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Actuator experiment/checkpoint differs from the saved contract")
    for name, key in (("policy_features.py", "features_sha256"), ("policy_beta.py", "beta_policy_sha256")):
        if metadata[key] != hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest():
            raise ValueError("Actuator policy implementation mismatch")
    env = make_environment(details["mode"], config, render_mode)
    try:
        if (metadata["config"] != asdict(env.config) or metadata["contract"] != contract(env, details["mode"])
                or metadata["observation_names"] != env.observation_names):
            raise ValueError("Actuator observation/action/physical contract mismatch")
        torch.set_num_threads(1)
        model = PPO.load(path, device="cpu")
        if model.num_timesteps != metadata["trained_steps"]:
            raise ValueError("Actuator checkpoint step mismatch")
        return env, lambda obs: model.predict(obs, deterministic=True)[0], metadata
    except Exception:
        env.close()
        raise
