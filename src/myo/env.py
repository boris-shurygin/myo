"""A small muscle-driven MuJoCo task with an analytic curl reference.

This is an imitation *pipeline baseline*, not a reproduction of MuscleMimic,
a calibrated human arm, or a biomechanical assessment. Four external
series-elastic muscles pull on one elbow DOF. Constant moment arms are an
explicit simplification; their length and torque maps obey virtual work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from myo.muscle import MuscleParameters, SeriesElasticMuscle
from myo.commands import ArmCommand


@dataclass(frozen=True)
class PreacherCurlConfig:
    """SI units; parameters are illustrative, not fitted to human data."""

    physics_dt: float = 0.001
    frame_skip: int = 20
    episode_seconds: float = 10.0
    reference_center: float = float(np.deg2rad(75.0))
    reference_amplitude: float = float(np.deg2rad(30.0))
    reference_frequency: float = 0.18
    randomize_reference: bool = True
    initial_angle_noise: float = 0.015
    initial_activation: float = 0.03
    payload_mass: float = 1.0
    motor_delay: float = 0.04
    fatigue_rate: float = 0.08
    recovery_rate: float = 0.02
    rest_recovery_multiplier: float = 1.0
    rest_activation_threshold: float = 0.01
    flexor_active_force_scale: float = 1.0
    extensor_active_force_scale: float = 1.0
    muscle_force_length_width: float = 0.5
    # Chosen engineering penalty thresholds, not physiological safety limits.
    rom_margin: float = float(np.deg2rad(10.0))
    joint_load_force_threshold: float = 600.0  # N, summed tendon tension proxy.
    fiber_stretch_threshold: float = 1.15  # Fiber length / optimal length.
    tendon_strain_threshold: float = 0.04
    rom_cost_weight: float = 0.02
    joint_load_cost_weight: float = 0.01
    fiber_stretch_cost_weight: float = 0.01
    tendon_stretch_cost_weight: float = 0.01
    render_width: int = 960
    render_height: int = 720
    geometry: str = "toy"
    asset_dir: str = "data/myoarm"
    arm_mount_tilt: float = float(np.deg2rad(30.0))  # MyoArm: rotation from vertical-down, radians.
    # MuJoCo contact proxies, not measured tissue material constants.
    tissue_contact_time_constant: float = 0.03  # s; larger means softer response.
    tissue_contact_transition_depth: float = 0.015  # m; depth over which impedance increases.
    task_mode: str = "reference"
    history_steps: int = 10
    target_max_speed: float = 1.0
    target_max_acceleration: float = 3.0
    initial_trigger: float = 1.0
    payload_mass_min: float = 0.5
    payload_mass_max: float = 2.0
    initial_capacity_min: float = 0.7
    curriculum_target_max: float = float(np.deg2rad(158.0))  # Sampling coverage, not a joint limit.
    passive_limit_width: float = 0.1745329252
    passive_limit_stiffness: float = 200.0  # Nm / rad^2, engineering fixture proxy.
    passive_limit_damping: float = 3.0  # Nm s / rad at the numerical boundary.
    trigger_excess_cost_weight: float = 0.25
    limit_velocity_cost_weight: float = 0.1

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.frame_skip

    def __post_init__(self) -> None:
        finite_fields = (
            self.physics_dt, self.episode_seconds, self.reference_center,
            self.reference_amplitude, self.reference_frequency,
            self.initial_angle_noise, self.initial_activation,
            self.payload_mass, self.motor_delay, self.fatigue_rate,
            self.recovery_rate, self.rom_margin, self.joint_load_force_threshold,
            self.fiber_stretch_threshold, self.tendon_strain_threshold,
            self.rom_cost_weight, self.joint_load_cost_weight,
            self.fiber_stretch_cost_weight, self.tendon_stretch_cost_weight, self.arm_mount_tilt,
        )
        if not all(np.isfinite(v) for v in finite_fields):
            raise ValueError("Environment settings must be finite.")
        if self.physics_dt <= 0 or self.frame_skip < 1 or int(self.frame_skip) != self.frame_skip:
            raise ValueError("physics_dt must be positive and frame_skip a positive integer.")
        if self.episode_seconds <= 0 or self.payload_mass <= 0:
            raise ValueError("episode_seconds and payload_mass must be positive.")
        if self.reference_amplitude < 0 or self.reference_frequency < 0:
            raise ValueError("Reference amplitude and frequency must be nonnegative.")
        if self.initial_angle_noise < 0 or not 0 <= self.initial_activation <= 1:
            raise ValueError("Invalid initial angle noise or activation.")
        if min(self.motor_delay, self.fatigue_rate, self.recovery_rate) < 0:
            raise ValueError("Delay and fatigue/recovery rates must be nonnegative.")
        if min(self.rom_margin, self.joint_load_force_threshold, self.tendon_strain_threshold) <= 0:
            raise ValueError("Engineering proxy thresholds must be positive.")
        if self.fiber_stretch_threshold <= 1:
            raise ValueError("fiber_stretch_threshold must exceed the optimal-length ratio of 1.")
        if min(self.rom_cost_weight, self.joint_load_cost_weight,
               self.fiber_stretch_cost_weight, self.tendon_stretch_cost_weight) < 0:
            raise ValueError("Engineering proxy cost weights must be nonnegative.")
        if self.geometry not in ("toy", "myoarm") or self.task_mode not in ("reference", "interactive", "curriculum"):
            raise ValueError("Unknown geometry or task_mode.")
        if not all(np.isfinite(x) and x > 0 for x in (
            self.tissue_contact_time_constant, self.tissue_contact_transition_depth,
        )):
            raise ValueError("Tissue contact time constant and transition depth must be finite and positive.")
        if self.geometry == "myoarm" and self.tissue_contact_time_constant < 2 * self.physics_dt:
            raise ValueError("Tissue contact time constant must be at least two physics timesteps.")
        if self.history_steps < 0 or int(self.history_steps) != self.history_steps:
            raise ValueError("history_steps must be a nonnegative integer.")
        positive = (self.target_max_speed, self.target_max_acceleration,
                    self.payload_mass_min, self.payload_mass_max, self.passive_limit_width,
                    self.rest_recovery_multiplier, self.flexor_active_force_scale,
                    self.extensor_active_force_scale, self.curriculum_target_max)
        nonnegative = (self.passive_limit_stiffness, self.passive_limit_damping,
                       self.trigger_excess_cost_weight, self.limit_velocity_cost_weight)
        if not all(np.isfinite(x) and x > 0 for x in positive) or not all(np.isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError("Invalid command, payload or passive resistance parameters.")
        if not 0 <= self.initial_trigger <= 1 or not 0 < self.initial_capacity_min <= 1:
            raise ValueError("Invalid initial trigger or capacity.")
        if self.payload_mass_min > self.payload_mass_max:
            raise ValueError("Payload mass bounds are reversed.")
        if not np.isfinite(self.rest_activation_threshold) or not 0 <= self.rest_activation_threshold <= 1:
            raise ValueError("Rest activation threshold must be in [0, 1].")
        if not np.isfinite(self.muscle_force_length_width) or not 0 < self.muscle_force_length_width < 1:
            raise ValueError("Muscle force-length width must be in (0, 1).")


class PreacherCurlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Control four excitations in [0, 1] at 50 Hz by default.

    The observation contains the command generator state, current joint state,
    previous excitation, and every muscle's activation, fiber length, fatigue,
    and complete pending motor-delay queue. The default task is fully observed
    in simulation. This does not claim these signals are measured in a person.

    ``tracking_error`` always means actual minus commanded, in radians and
    radians/second. Reference previews are commands, never future observations.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    muscle_names = ("biceps_long", "biceps_short", "brachialis", "triceps")
    # Signed flexion moment arms: l_mtu = l_ref - r * (q - q_ref), tau = r * F.
    moment_arms = np.array([0.035, 0.030, 0.025, -0.025], dtype=np.float64)

    def __init__(
        self,
        config: PreacherCurlConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PreacherCurlConfig()
        self.control_dt = self.config.control_dt
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": round(1.0 / self.control_dt)}
        self.geometry = None
        if self.config.geometry == "myoarm":
            from myo.geometry import MyoArmGeometry
            self.geometry = MyoArmGeometry(self.config.asset_dir, physics_dt=self.config.physics_dt,
                                          payload_mass=self.config.payload_mass,
                                          arm_mount_tilt=self.config.arm_mount_tilt,
                                          tissue_contact_time_constant=self.config.tissue_contact_time_constant,
                                          tissue_contact_transition_depth=self.config.tissue_contact_transition_depth)
            self.model = self.geometry.model
            self.muscle_names = self.geometry.muscle_names
        else:
            xml = Path(__file__).parent / "assets" / "preacher_curl.xml"
            self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = self.config.physics_dt
        self.data = mujoco.MjData(self.model)
        self.elbow_id = self.geometry.elbow_id if self.geometry else self.model.joint("elbow").id
        self.qpos_index = int(self.model.jnt_qposadr[self.elbow_id])
        self.dof_index = int(self.model.jnt_dofadr[self.elbow_id])
        self.angle_limits = self.model.jnt_range[self.elbow_id].copy()
        largest_amplitude = self.config.reference_amplitude * (1.1 if self.config.randomize_reference else 1)
        if (self.config.reference_center - largest_amplitude < self.angle_limits[0]
                or self.config.reference_center + largest_amplitude > self.angle_limits[1]):
            raise ValueError("The full reference range must stay inside the elbow limits.")

        payload = self.geometry.payload_id if self.geometry else self.model.body("payload").id
        self.payload_id = payload
        self.payload_mass = self.config.payload_mass
        self.model.body_mass[payload] = self.config.payload_mass
        self.model.body_inertia[payload] = 0.4 * self.config.payload_mass * 0.055**2
        mujoco.mj_setConst(self.model, self.data)
        maxima = (350.0, 300.0, 450.0, 600.0)
        fibers = (0.110, 0.105, 0.090, 0.100)
        tendons = (0.200, 0.180, 0.120, 0.170)
        parameters = self.geometry.muscle_parameters if self.geometry else [
                MuscleParameters(
                    name=name,
                    max_isometric_force=force,
                    optimal_fiber_length=fiber,
                    tendon_slack_length=tendon,
                    motor_delay=self.config.motor_delay,
                    fatigue_rate=self.config.fatigue_rate,
                    recovery_rate=self.config.recovery_rate,
                )
            for name, force, fiber, tendon in zip(self.muscle_names, maxima, fibers, tendons)
        ]
        self.muscles = [SeriesElasticMuscle(replace(p, motor_delay=self.config.motor_delay,
                        fatigue_rate=self.config.fatigue_rate, recovery_rate=self.config.recovery_rate,
                        rest_recovery_multiplier=self.config.rest_recovery_multiplier,
                        rest_activation_threshold=self.config.rest_activation_threshold,
                        force_length_width=self.config.muscle_force_length_width,
                        active_force_scale=(self.config.extensor_active_force_scale
                            if p.name.startswith("TRI") or p.name == "triceps"
                            else self.config.flexor_active_force_scale)),
                        dt=self.config.physics_dt) for p in parameters]
        self._reference_lengths = np.asarray(fibers) + np.asarray(tendons)
        self._length_reference_angle = float(np.deg2rad(75.0))
        self.action_space = spaces.Box(0.0, 1.0, shape=(len(self.muscles),), dtype=np.float32)
        self._previous_excitation = np.zeros(len(self.muscles), dtype=np.float64)
        self._muscle_outputs: list[Any] = []
        self._muscle_torque = 0.0
        self._step_count = 0
        self._phase = 0.0
        self._amplitude = self.config.reference_amplitude
        self._frequency = self.config.reference_frequency
        self._reward_components: dict[str, float] = {}
        self._renderer: Any = None
        self._viewer: Any = None
        self._episode_done = False
        self._has_reset = False
        self.command = ArmCommand(self.config.reference_center, self.config.reference_center,
                                  self.config.initial_trigger)
        self._manual_command = self.config.task_mode == "interactive"
        self._next_command_time = 0.0
        self._sensor_history: deque = deque(maxlen=self.config.history_steps)
        self._action_history: deque = deque(maxlen=self.config.history_steps)
        self._command_history: deque = deque(maxlen=self.config.history_steps)
        self._passive_torque = 0.0
        self._step_peaks: dict[str, float] = {}

        self.observation_names = [
            "angle_centered_rad", "angular_velocity_scaled", "target_angle_centered_rad",
            "target_velocity_scaled", "reference_phase_sin", "reference_phase_cos",
            "reference_amplitude_rad", "reference_frequency_hz", "time_remaining_fraction",
        ]
        self.observation_names.extend(f"previous_excitation/{name}" for name in self.muscle_names)
        for name, muscle in zip(self.muscle_names, self.muscles):
            self.observation_names.extend(
                f"{name}/{field}" for field in ("activation", "fiber_length_normalized", "fatigue")
            )
            self.observation_names.extend(
                f"{name}/pending_excitation/{i}" for i in range(len(muscle.delay.pending))
            )
        if self.config.task_mode != "reference":
            self._sensor_names = ["angle_centered_rad", "angular_velocity_scaled", "payload_mass_kg"]
            for name in self.muscle_names:
                self._sensor_names.extend(f"{name}/{field}" for field in (
                    "activation", "force_normalized", "fiber_length_normalized",
                    "tendon_length_normalized", "tendon_strain", "remaining_capacity"))
            self._command_names = ["target_angle_centered_rad", "target_velocity_scaled",
                                   "requested_target_centered_rad", "trigger"]
            self.observation_names = list(self._sensor_names) + list(self._command_names)
            for age in range(1, self.config.history_steps + 1):
                self.observation_names.extend(f"history/{age}/sensor/{n}" for n in self._sensor_names)
                self.observation_names.extend(f"history/{age}/excitation/{n}" for n in self.muscle_names)
                self.observation_names.extend(f"history/{age}/command/{n}" for n in self._command_names)
            for name, muscle in zip(self.muscle_names, self.muscles):
                self.observation_names.extend(f"{name}/pending_excitation/{i}" for i in range(len(muscle.delay.pending)))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32
        )

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reference_at(self, time: float) -> tuple[float, float]:
        """Analytic requested angle/velocity, available to the runtime controller."""
        if self.config.task_mode != "reference":
            # Future user inputs are never available to the controller/predictor.
            return self.command.target, self.command.velocity
        phase = 2 * np.pi * self._frequency * time + self._phase
        return (
            float(self.config.reference_center + self._amplitude * np.sin(phase)),
            float(2 * np.pi * self._frequency * self._amplitude * np.cos(phase)),
        )

    def muscle_lengths(self, angle: float | None = None) -> np.ndarray:
        if self.geometry:
            if angle is None:
                return self.geometry.lengths(self.data)
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            scratch.qpos[self.qpos_index] = float(angle)
            mujoco.mj_forward(self.model, scratch)
            return self.geometry.lengths(scratch)
        q = float(self.data.qpos[self.qpos_index]) if angle is None else float(angle)
        return self._reference_lengths - self.moment_arms * (q - self._length_reference_angle)

    def set_command(self, target_angle: float, trigger: float) -> None:
        """Override the curriculum with causal external commands; never change q."""
        if self.config.task_mode == "reference":
            raise ValueError("Use an interactive/curriculum config for user commands.")
        self.command.set(target_angle, trigger, self.angle_limits)
        self._manual_command = True

    def observe(self) -> np.ndarray:
        """Refresh current commands after input without advancing physics/history."""
        return self._get_obs()

    def _command_values(self) -> list[float]:
        return [self.command.target - self.config.reference_center, self.command.velocity / 4.0,
                self.command.requested - self.config.reference_center, self.command.trigger]

    def _state_outputs(self):
        lengths = self.muscle_lengths()
        velocity = float(self.data.qvel[self.dof_index])
        return [m.sample(float(length), float(-r * velocity))
                for m, length, r in zip(self.muscles, lengths, self.moment_arms)]

    def _sensor_values(self) -> list[float]:
        values = [float(self.data.qpos[self.qpos_index]) - self.config.reference_center,
                  float(self.data.qvel[self.dof_index]) / 4.0, self.payload_mass]
        for muscle, output in zip(self.muscles, self._state_outputs()):
            p = muscle.params
            values.extend((muscle.activation, output.tendon_force / p.max_isometric_force,
                           muscle.fiber_length / p.optimal_fiber_length,
                           output.tendon_length / p.tendon_slack_length,
                           output.tendon_strain, muscle.capacity))
        return values

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        allowed = {"q", "qvel", "reference_phase", "reference_amplitude", "reference_frequency",
                   "target_angle", "trigger", "payload_mass", "capacity"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {sorted(set(options) - allowed)}")
        mujoco.mj_resetData(self.model, self.data)
        curriculum = self.config.task_mode == "curriculum"
        self._manual_command = self.config.task_mode == "interactive"
        self.payload_mass = float(options.get("payload_mass", self.np_random.uniform(
            self.config.payload_mass_min, self.config.payload_mass_max) if curriculum else self.config.payload_mass))
        capacity = float(options.get("capacity", self.np_random.uniform(
            self.config.initial_capacity_min, 1.0) if curriculum else 1.0))
        if not np.isfinite(self.payload_mass) or self.payload_mass <= 0 or not 0 < capacity <= 1:
            raise ValueError("Reset mass must be positive and capacity in (0, 1].")
        self.model.body_mass[self.payload_id] = self.payload_mass
        self.model.body_inertia[self.payload_id] = 0.4 * self.payload_mass * 0.055**2
        mujoco.mj_setConst(self.model, self.data)
        randomized = self.config.randomize_reference
        self._phase = float(options.get("reference_phase", self.np_random.uniform(0, 2 * np.pi) if randomized else 0.0))
        self._amplitude = float(options.get("reference_amplitude", self.config.reference_amplitude * (self.np_random.uniform(0.8, 1.1) if randomized else 1.0)))
        self._frequency = float(options.get("reference_frequency", self.config.reference_frequency * (self.np_random.uniform(0.8, 1.2) if randomized else 1.0)))
        if not all(np.isfinite(v) for v in (self._phase, self._amplitude, self._frequency)):
            raise ValueError("Reference parameters must be finite.")
        if self._amplitude < 0 or self._frequency < 0:
            raise ValueError("Reference amplitude and frequency must be nonnegative.")
        if (self.config.reference_center - self._amplitude < self.angle_limits[0]
                or self.config.reference_center + self._amplitude > self.angle_limits[1]):
            raise ValueError("Reference is outside elbow limits.")
        self.command = ArmCommand(self.config.reference_center, self.config.reference_center,
                                  self.config.initial_trigger)
        self.command.set(float(options.get("target_angle", self.config.reference_center)),
                         float(options.get("trigger", self.config.initial_trigger)), self.angle_limits)
        self._next_command_time = 1.0
        target_q, _ = self.reference_at(0.0)
        q = float(options.get("q", target_q + self.np_random.uniform(-self.config.initial_angle_noise, self.config.initial_angle_noise)))
        qvel = float(options.get("qvel", 0.0))
        if not np.isfinite(q) or not np.isfinite(qvel) or not self.angle_limits[0] <= q <= self.angle_limits[1]:
            raise ValueError("Initial q/qvel must be finite and q inside elbow limits.")
        self.data.qpos[self.qpos_index] = q
        self.data.qvel[self.dof_index] = qvel
        mujoco.mj_forward(self.model, self.data)
        if self.geometry:
            self.moment_arms = self.geometry.moment_arms(self.data)
        self._muscle_outputs = [
            muscle.reset(float(length), activation=self.config.initial_activation, capacity=capacity)
            for muscle, length in zip(self.muscles, self.muscle_lengths())
        ]
        self._previous_excitation.fill(self.config.initial_activation)
        self._muscle_torque = 0.0
        self._step_count = 0
        self._reward_components = {}
        self._episode_done = False
        self._has_reset = True
        self._passive_torque = 0.0
        self._step_peaks = {}
        self._sensor_history.clear()
        self._action_history.clear()
        self._command_history.clear()
        if self.config.task_mode != "reference":
            for _ in range(self.config.history_steps):
                self._sensor_history.append(self._sensor_values())
                self._action_history.append(self._previous_excitation.copy())
                self._command_history.append(self._command_values())
        return self._get_obs(), self.telemetry()

    def _get_obs(self) -> np.ndarray:
        if self.config.task_mode != "reference":
            values = self._sensor_values() + self._command_values()
            for sensor, action, command in zip(reversed(self._sensor_history),
                                               reversed(self._action_history), reversed(self._command_history)):
                values.extend(sensor)
                values.extend(action)
                values.extend(command)
            for muscle in self.muscles:
                values.extend(muscle.delay.pending)
            observation = np.asarray(values, dtype=np.float32)
            if not np.isfinite(observation).all():
                raise FloatingPointError("Non-finite command observation.")
            return observation
        q, qvel = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        target_q, target_qvel = self.reference_at(self.time)
        phase = 2 * np.pi * self._frequency * self.time + self._phase
        values = [
            q - self.config.reference_center, qvel / 4.0,
            target_q - self.config.reference_center, target_qvel / 4.0,
            np.sin(phase), np.cos(phase), self._amplitude, self._frequency,
            max(0.0, 1.0 - self.time / self.config.episode_seconds),
            *self._previous_excitation,
        ]
        for muscle in self.muscles:
            values.extend((
                muscle.activation,
                muscle.fiber_length / muscle.params.optimal_fiber_length,
                1.0 - muscle.capacity,
                *muscle.delay.pending,
            ))
        observation = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Non-finite simulation state; inspect the muscle model and timestep.")
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._has_reset or self._episode_done:
            raise RuntimeError("Call reset() before stepping a new or finished episode.")
        excitation = np.asarray(action, dtype=np.float64)
        if excitation.shape != self.action_space.shape or not np.all(np.isfinite(excitation)):
            raise ValueError("Action must contain one finite excitation per muscle.")
        if np.any(excitation < 0.0) or np.any(excitation > 1.0):
            raise ValueError("Muscle excitations must lie in [0, 1].")
        previous = self._previous_excitation.copy()
        if self.config.task_mode != "reference":
            self._sensor_history.append(self._sensor_values())
            self._action_history.append(excitation.copy())
            self._command_history.append(self._command_values())
        self._step_peaks = {"tendon_force_N": 0.0, "tendon_strain": 0.0,
                            "summed_tendon_force_N": 0.0, "passive_limit_torque_Nm": 0.0,
                            "outward_limit_velocity_rad_s": 0.0, "joint_limit_reaction_Nm": 0.0,
                            "tissue_contact_force_N": 0.0, "tissue_contact_penetration_m": 0.0,
                            "tissue_contact_torque_Nm": 0.0}
        peak_risks = {name: 0.0 for name in self._risk_proxies()}
        for _ in range(self.config.frame_skip):
            # Single-step integrator: current kinematics, external muscle
            # update, generalized force, then rigid-body acceleration/step.
            mujoco.mj_step1(self.model, self.data)
            if self.geometry:
                self.moment_arms = self.geometry.moment_arms(self.data)
            endpoint_outputs = self._state_outputs()
            self._record_peaks(endpoint_outputs)
            for key, value in self._risk_proxies(endpoint_outputs).items():
                peak_risks[key] = max(peak_risks[key], value)
            velocity = float(self.data.qvel[self.dof_index])
            self._muscle_outputs = [
                muscle.step(float(u), float(length), float(-r * velocity))
                for muscle, u, length, r in zip(
                    self.muscles, excitation, self.muscle_lengths(), self.moment_arms
                )
            ]
            forces = np.asarray([output.tendon_force for output in self._muscle_outputs])
            if not np.all(np.isfinite(forces)):
                raise FloatingPointError("Muscle solver produced a non-finite tendon force.")
            self._muscle_torque = float(self.moment_arms @ forces)
            self._passive_torque = self.passive_limit_torque()
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_index] = self._muscle_torque + self._passive_torque
            for key, value in self._risk_proxies().items():
                peak_risks[key] = max(peak_risks[key], value)
            self._record_peaks()
            mujoco.mj_step2(self.model, self.data)
            self._record_constraint_peaks()
        # Make paths, body poses and surface inputs agree with the final q.
        mujoco.mj_forward(self.model, self.data)
        self._record_constraint_peaks()
        if self.geometry:
            self.moment_arms = self.geometry.moment_arms(self.data)
        endpoint_outputs = self._state_outputs()
        self._record_peaks(endpoint_outputs)
        for key, value in self._risk_proxies(endpoint_outputs).items():
            peak_risks[key] = max(peak_risks[key], value)
        if self.config.task_mode != "reference":
            self.command.advance(self.control_dt, self.config.target_max_speed, self.config.target_max_acceleration)
        self._step_count += 1
        self._previous_excitation = excitation.copy()
        q, qvel = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        target_q, target_qvel = self.reference_at(self.time)
        # Dimensionless optimization proxies, not metabolic energy, tissue
        # stress, pain, or a clinically valid joint-load estimate.
        position_reward = float(np.exp(-((q - target_q) / 0.25) ** 2))
        velocity_reward = float(np.exp(-((qvel - target_qvel) / 1.5) ** 2))
        effort_cost = float(0.01 * np.mean(excitation**2))
        smoothness_cost = float(0.002 * np.mean((excitation - previous) ** 2))
        fatigue_cost = float(0.005 * np.mean([1.0 - m.capacity for m in self.muscles]))
        risks = peak_risks
        tracking_weight = 1.0 if self.config.task_mode == "reference" else self.command.trigger
        self._reward_components = {
            "position_tracking": tracking_weight * 0.85 * position_reward,
            "velocity_tracking": tracking_weight * 0.15 * velocity_reward,
            "excitation_squared_cost": -effort_cost,
            "excitation_change_cost": -smoothness_cost,
            "fatigue_proxy_cost": -fatigue_cost,
            "rom_margin_proxy_cost": -self.config.rom_cost_weight * risks["rom_margin"],
            "summed_tendon_force_proxy_cost": -self.config.joint_load_cost_weight * risks["summed_tendon_force"],
            "fiber_stretch_proxy_cost": -self.config.fiber_stretch_cost_weight * risks["fiber_stretch"],
            "tendon_strain_proxy_cost": -self.config.tendon_stretch_cost_weight * risks["tendon_strain"],
        }
        if self.config.task_mode != "reference":
            self._reward_components["trigger_excess_cost"] = -self.config.trigger_excess_cost_weight * float(
                np.mean(np.maximum(excitation - self.command.trigger, 0.0) ** 2))
            self._reward_components["outward_limit_velocity_proxy_cost"] = -self.config.limit_velocity_cost_weight * (
                self._step_peaks["outward_limit_velocity_rad_s"] ** 2)
        terminated = bool(q < self.angle_limits[0] - 0.15 or q > self.angle_limits[1] + 0.15 or abs(qvel) > 50.0)
        if terminated:
            self._reward_components["physical_failure"] = -1.0
        reward = float(sum(self._reward_components.values()))
        truncated = bool(self.time + 1e-10 >= self.config.episode_seconds and not terminated)
        self._episode_done = terminated or truncated
        # A new unpredictable command is revealed only after scoring the action
        # chosen for the old command. It appears in the next causal observation.
        if self.config.task_mode == "curriculum" and not self._manual_command and self.time >= self._next_command_time:
            margin = max(self.config.rom_margin, 0.05)
            target = self.np_random.uniform(self.angle_limits[0] + margin,
                         min(self.angle_limits[1] - margin, self.config.curriculum_target_max))
            trigger = float(self.np_random.choice([0.0, 0.15, 0.4, 0.7, 1.0]))
            self.command.set(target, trigger, self.angle_limits)
            self._next_command_time = self.time + self.np_random.uniform(0.8, 2.0)
        observation = self._get_obs()
        info = self.telemetry()
        info["termination_reason"] = "joint_state_outside_guard" if terminated else None
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def baseline_action(self) -> np.ndarray:
        """Deterministic heuristic feedback through the same muscle actuator path.

        Uses a known reference preview to reduce motor-delay lag. It estimates
        requested torque, then allocates excitation between flexors/extensor;
        it never writes that torque directly or overrides the simulated pose.
        This controller is a reproducible comparison/data-collection baseline,
        not an optimal solution or a claim that PPO already learned the task.
        """
        if not self._has_reset:
            raise RuntimeError("Call reset() before requesting a baseline action.")
        q, qvel = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        preview_time = self.time + self.config.motor_delay + 0.025
        target_q, target_qvel = self.reference_at(preview_time)
        target_acc = (-(2 * np.pi * self._frequency) ** 2 * (target_q - self.config.reference_center)
                      if self.config.task_mode == "reference" else 0.0)
        velocity_gain = 9.0 if self.config.task_mode == "reference" else 14.0
        requested_acc = target_acc + 25.0 * (target_q - q) + velocity_gain * (target_qvel - qvel)
        # The fixed-base 1-DOF model has constant hinge inertia. Using the
        # reference diagonal also avoids relying on renamed mjData mass fields.
        inertia = self.model.dof_M0[self.dof_index]
        requested_torque = float(inertia * requested_acc + self.data.qfrc_bias[self.dof_index]
                                 + self.model.dof_damping[self.dof_index] * qvel - self.passive_limit_torque())
        capacities = []
        for muscle in self.muscles:
            normalized_length = muscle.fiber_length / muscle.params.optimal_fiber_length
            length_factor = max(0.0, 1.0 - ((normalized_length - 1.0) / muscle.params.force_length_width) ** 2)
            capacities.append(muscle.params.max_isometric_force * muscle.params.active_force_scale
                              * muscle.capacity * length_factor)
        torque_capacity = self.moment_arms * np.asarray(capacities)
        base = 0.015 if self.config.task_mode == "reference" else 0.0
        passive_torque = float(self.moment_arms @ np.asarray([o.passive_force for o in self._muscle_outputs]))
        residual_torque = requested_torque - passive_torque - base * float(torque_capacity.sum())
        result = np.full(len(self.muscles), base)
        selected = torque_capacity > 0 if residual_torque >= 0 else torque_capacity < 0
        available = float(np.abs(torque_capacity[selected]).sum())
        if available > 1e-8:
            result[selected] += abs(residual_torque) / available
        if self.config.task_mode != "reference":
            result = np.minimum(result, self.command.trigger)
            # Diagnostic braking goes through the ordinary delayed muscle path.
            # It is a comparison controller, never a learned-protection claim.
            stopping_distance = max(0.0, -qvel) * (self.config.motor_delay + 0.1)
            if q < self.angle_limits[0] + self.config.rom_margin + stopping_distance and qvel < 0:
                selected = torque_capacity > 0
                available = float(torque_capacity[selected].sum())
                # Scale a desired viscous deceleration by current muscle
                # strength. A fixed excitation gain launches a stronger arm
                # back upward instead of letting it settle after release.
                braking_torque = inertia * 15.0 * max(0.0, -qvel - 0.2)
                if available > 1e-8:
                    result[selected] = np.maximum(result[selected], braking_torque / available)
            if q > self.angle_limits[1] - self.config.rom_margin - max(0.0, qvel) * (self.config.motor_delay + 0.1) and qvel > 0:
                selected = torque_capacity < 0
                available = float(-torque_capacity[selected].sum())
                braking_torque = inertia * 15.0 * max(0.0, qvel - 0.2)
                if available > 1e-8:
                    result[selected] = np.maximum(result[selected], braking_torque / available)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def passive_limit_torque(self) -> float:
        """Continuous spring/damper near ROM; numerical limits remain a backstop."""
        if self.config.task_mode == "reference" and not self.geometry:
            return 0.0
        q, velocity = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        width = self.config.passive_limit_width
        low = max(0.0, self.angle_limits[0] + width - q)
        high = max(0.0, q - self.angle_limits[1] + width)
        spring = self.config.passive_limit_stiffness * (low * low - high * high)
        # Damping is dissipative during entry and exit; spring returns energy.
        damping = self.config.passive_limit_damping * min(1.0, (max(low, high) / width) ** 2)
        return float(spring - damping * velocity)

    def _record_peaks(self, outputs=None) -> None:
        q, velocity = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        outputs = self._muscle_outputs if outputs is None else outputs
        outward = 0.0
        if q < self.angle_limits[0] + self.config.rom_margin:
            outward = max(0.0, -velocity)
        elif q > self.angle_limits[1] - self.config.rom_margin:
            outward = max(0.0, velocity)
        values = {"tendon_force_N": max(o.tendon_force for o in outputs),
                  "tendon_strain": max(o.tendon_strain for o in outputs),
                  "summed_tendon_force_N": sum(o.tendon_force for o in outputs),
                  "passive_limit_torque_Nm": abs(self.passive_limit_torque()),
                  "outward_limit_velocity_rad_s": outward}
        for key, value in values.items():
            self._step_peaks[key] = max(self._step_peaks[key], float(value))

    def _constraint_torque(self, constraint_types: tuple[int, ...]) -> float:
        """Project selected solver reactions; contact is not a joint limit."""
        if not self.data.nefc:
            return 0.0
        force = np.where(np.isin(self.data.efc_type, constraint_types), self.data.efc_force, 0.0)
        torque = np.zeros(self.model.nv)
        mujoco.mj_mulJacTVec(self.model, self.data, torque, force)
        return float(torque[self.dof_index])

    def tissue_contact_metrics(self) -> dict[str, float | int]:
        """Static-envelope contact proxy, separate from tendon and joint reactions."""
        metrics = {"count": 0, "normal_force_N": 0.0,
                   "max_penetration_m": 0.0, "torque_Nm": 0.0}
        if self.geometry is None or not self.data.ncon:
            return metrics
        envelope_ids = self.geometry.contact_geom_ids
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 not in envelope_ids or contact.geom2 not in envelope_ids:
                continue
            metrics["count"] += 1
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            metrics["normal_force_N"] += max(0.0, float(wrench[0]))
            metrics["max_penetration_m"] = max(metrics["max_penetration_m"], -float(contact.dist))
        # Only the explicit tissue pair can collide in this model.
        if metrics["count"]:
            metrics["torque_Nm"] = self._constraint_torque((
                mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS,
                mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC,
            ))
        return metrics

    def _record_constraint_peaks(self) -> None:
        contact = self.tissue_contact_metrics()
        values = {
            "joint_limit_reaction_Nm": abs(self._constraint_torque((mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,))),
            "tissue_contact_force_N": contact["normal_force_N"],
            "tissue_contact_penetration_m": contact["max_penetration_m"],
            "tissue_contact_torque_Nm": abs(contact["torque_Nm"]),
        }
        for key, value in values.items():
            self._step_peaks[key] = max(self._step_peaks[key], float(value))

    def _risk_proxies(self, outputs=None) -> dict[str, float]:
        """Dimensionless engineering penalties; none estimates human pain.

        ROM uses squared fractional encroachment into a configured limit
        margin. Force uses squared excess of summed tendon tension relative
        to a chosen threshold; it is not anatomical joint contact force.
        Fiber and tendon penalties average squared *positive* threshold
        excesses. Shortening/slack never count as tensile overstretch.
        """
        q = float(self.data.qpos[self.qpos_index])
        clearance = min(q - self.angle_limits[0], self.angle_limits[1] - q)
        outputs = self._muscle_outputs if outputs is None else outputs
        force = sum(output.tendon_force for output in outputs)
        fiber_ratios = np.asarray([
            muscle.fiber_length / muscle.params.optimal_fiber_length
            for muscle in self.muscles
        ])
        tendon_strains = np.asarray([output.tendon_strain for output in outputs])
        return {
            "rom_margin": float(max(0.0, 1.0 - clearance / self.config.rom_margin) ** 2),
            "summed_tendon_force": float(max(0.0, force / self.config.joint_load_force_threshold - 1.0) ** 2),
            "fiber_stretch": float(np.mean(np.maximum(fiber_ratios / self.config.fiber_stretch_threshold - 1.0, 0.0) ** 2)),
            "tendon_strain": float(np.mean(np.maximum(tendon_strains / self.config.tendon_strain_threshold - 1.0, 0.0) ** 2)),
        }

    def telemetry(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot; no rendering or stepping side effects."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before reading telemetry.")
        target_q, target_qvel = self.reference_at(self.time)
        q, qvel = float(self.data.qpos[self.qpos_index]), float(self.data.qvel[self.dof_index])
        outputs = self._state_outputs()
        return {
            "t": self.time,
            "q": q,
            "qvel": qvel,
            "target_q": target_q,
            "target_qvel": target_qvel,
            "requested_target_q": self.command.requested if self.config.task_mode != "reference" else target_q,
            "trigger": self.command.trigger if self.config.task_mode != "reference" else 1.0,
            "command_source": ("reference" if self.config.task_mode == "reference" else
                               "manual" if self._manual_command else "curriculum"),
            "payload_mass_kg": self.payload_mass,
            "arm_mount_tilt_rad": self.config.arm_mount_tilt if self.geometry else None,
            "tissue_contact": self.tissue_contact_metrics(),
            "tracking_error": [q - target_q, qvel - target_qvel],
            "muscle_names": list(self.muscle_names),
            "excitation": self._previous_excitation.tolist(),
            "delayed_excitation": [float(o.delayed_excitation) for o in outputs],
            "activation": [float(m.activation) for m in self.muscles],
            "tendon_force_N": [float(o.tendon_force) for o in outputs],
            "tendon_strain": [float(o.tendon_strain) for o in outputs],
            "fiber_length_m": [float(m.fiber_length) for m in self.muscles],
            "fiber_velocity_m_s": [float(o.fiber_velocity) for o in outputs],
            "tendon_length_m": [float(o.tendon_length) for o in outputs],
            "musculotendon_length_m": self.muscle_lengths().tolist(),
            "moment_arm_m": self.moment_arms.tolist(),
            "joint_limits_rad": [float(self.angle_limits[0]),
                                 float(self.angle_limits[1]) if np.isfinite(self.angle_limits[1]) else None],
            "path_direction_reversal": bool(self.geometry is not None and (
                np.any(self.moment_arms[:3] < 0) or np.any(self.moment_arms[3:] > 0))),
            "passive_fiber_force_N": [float(o.passive_force) for o in outputs],
            "active_fiber_force_N": [float(o.active_force) for o in outputs],
            "damping_force_N": [float(o.damping_force) for o in outputs],
            "capacity": [float(m.capacity) for m in self.muscles],
            "fatigue": [float(1.0 - m.capacity) for m in self.muscles],
            "muscle_torque_Nm": float(self.moment_arms @ np.asarray([o.tendon_force for o in outputs])),
            "applied_muscle_torque_Nm": self._muscle_torque,
            "applied_tendon_force_N": [float(o.tendon_force) for o in self._muscle_outputs],
            "applied_equilibrium_residual_N": [float(o.equilibrium_residual) for o in self._muscle_outputs],
            "applied_force_time_s": max(0.0, self.time - self.config.physics_dt) if self._step_count else 0.0,
            "passive_limit_torque_Nm": self.passive_limit_torque(),
            "applied_passive_limit_torque_Nm": self._passive_torque,
            "substep_peaks": dict(self._step_peaks),
            "summed_tendon_force_proxy_N": float(sum(o.tendon_force for o in outputs)),
            "equilibrium_residual_N": [float(o.equilibrium_residual) for o in outputs],
            "fiber_at_bound": [bool(o.at_fiber_bound) for o in outputs],
            "pain_proxy_components": self._risk_proxies(outputs),
            "risk_proxy_inputs": {
                "rom_clearance_rad": float(min(q - self.angle_limits[0], self.angle_limits[1] - q)),
                "summed_tendon_force_N": float(sum(o.tendon_force for o in outputs)),
                "fiber_length_ratio": [float(m.fiber_length / m.params.optimal_fiber_length) for m in self.muscles],
                "tendon_strain": [float(o.tendon_strain) for o in outputs],
            },
            "reward_components": dict(self._reward_components),
            "episode_step": self._step_count,
            "reference_source": "analytic_sine" if self.config.task_mode == "reference" else "causal_commands",
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.config.render_height, width=self.config.render_width
                )
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
