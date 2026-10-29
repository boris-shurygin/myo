"""A small muscle-driven MuJoCo task with an analytic curl reference.

This is an imitation *pipeline baseline*, not a reproduction of MuscleMimic,
a calibrated human arm, or a biomechanical assessment. Four external
series-elastic muscles pull on one elbow DOF. Constant moment arms are an
explicit simplification; their length and torque maps obey virtual work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from myo.muscle import MuscleParameters, SeriesElasticMuscle


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
            self.fiber_stretch_cost_weight, self.tendon_stretch_cost_weight,
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
        xml = Path(__file__).parent / "assets" / "preacher_curl.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = self.config.physics_dt
        self.data = mujoco.MjData(self.model)
        self.elbow_id = self.model.joint("elbow").id
        self.qpos_index = int(self.model.jnt_qposadr[self.elbow_id])
        self.dof_index = int(self.model.jnt_dofadr[self.elbow_id])
        self.angle_limits = self.model.jnt_range[self.elbow_id].copy()
        largest_amplitude = self.config.reference_amplitude * (1.1 if self.config.randomize_reference else 1)
        if (self.config.reference_center - largest_amplitude < self.angle_limits[0]
                or self.config.reference_center + largest_amplitude > self.angle_limits[1]):
            raise ValueError("The full reference range must stay inside the elbow limits.")

        payload = self.model.body("payload").id
        self.model.body_mass[payload] = self.config.payload_mass
        self.model.body_inertia[payload] = 0.4 * self.config.payload_mass * 0.055**2
        mujoco.mj_setConst(self.model, self.data)
        maxima = (350.0, 300.0, 450.0, 600.0)
        fibers = (0.110, 0.105, 0.090, 0.100)
        tendons = (0.200, 0.180, 0.120, 0.170)
        self.muscles = [
            SeriesElasticMuscle(
                MuscleParameters(
                    name=name,
                    max_isometric_force=force,
                    optimal_fiber_length=fiber,
                    tendon_slack_length=tendon,
                    motor_delay=self.config.motor_delay,
                    fatigue_rate=self.config.fatigue_rate,
                    recovery_rate=self.config.recovery_rate,
                ),
                dt=self.config.physics_dt,
            )
            for name, force, fiber, tendon in zip(self.muscle_names, maxima, fibers, tendons)
        ]
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
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.observation_names),), dtype=np.float32
        )

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reference_at(self, time: float) -> tuple[float, float]:
        """Analytic requested angle/velocity, available to the runtime controller."""
        phase = 2 * np.pi * self._frequency * time + self._phase
        return (
            float(self.config.reference_center + self._amplitude * np.sin(phase)),
            float(2 * np.pi * self._frequency * self._amplitude * np.cos(phase)),
        )

    def muscle_lengths(self, angle: float | None = None) -> np.ndarray:
        q = float(self.data.qpos[self.qpos_index]) if angle is None else float(angle)
        return self._reference_lengths - self.moment_arms * (q - self._length_reference_angle)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        allowed = {"q", "qvel", "reference_phase", "reference_amplitude", "reference_frequency"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {sorted(set(options) - allowed)}")
        mujoco.mj_resetData(self.model, self.data)
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
        target_q, _ = self.reference_at(0.0)
        q = float(options.get("q", target_q + self.np_random.uniform(-self.config.initial_angle_noise, self.config.initial_angle_noise)))
        qvel = float(options.get("qvel", 0.0))
        if not np.isfinite(q) or not np.isfinite(qvel) or not self.angle_limits[0] <= q <= self.angle_limits[1]:
            raise ValueError("Initial q/qvel must be finite and q inside elbow limits.")
        self.data.qpos[self.qpos_index] = q
        self.data.qvel[self.dof_index] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._muscle_outputs = [
            muscle.reset(float(length), activation=self.config.initial_activation, capacity=1.0)
            for muscle, length in zip(self.muscles, self.muscle_lengths())
        ]
        self._previous_excitation.fill(self.config.initial_activation)
        self._muscle_torque = 0.0
        self._step_count = 0
        self._reward_components = {}
        self._episode_done = False
        self._has_reset = True
        return self._get_obs(), self.telemetry()

    def _get_obs(self) -> np.ndarray:
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
            raise ValueError("Action must contain four finite muscle excitations.")
        if np.any(excitation < 0.0) or np.any(excitation > 1.0):
            raise ValueError("Muscle excitations must lie in [0, 1].")
        previous = self._previous_excitation.copy()
        for _ in range(self.config.frame_skip):
            # Single-step integrator: current kinematics, external muscle
            # update, generalized force, then rigid-body acceleration/step.
            mujoco.mj_step1(self.model, self.data)
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
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_index] = self._muscle_torque
            mujoco.mj_step2(self.model, self.data)
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
        risks = self._risk_proxies()
        self._reward_components = {
            "position_tracking": 0.85 * position_reward,
            "velocity_tracking": 0.15 * velocity_reward,
            "excitation_squared_cost": -effort_cost,
            "excitation_change_cost": -smoothness_cost,
            "fatigue_proxy_cost": -fatigue_cost,
            "rom_margin_proxy_cost": -self.config.rom_cost_weight * risks["rom_margin"],
            "summed_tendon_force_proxy_cost": -self.config.joint_load_cost_weight * risks["summed_tendon_force"],
            "fiber_stretch_proxy_cost": -self.config.fiber_stretch_cost_weight * risks["fiber_stretch"],
            "tendon_strain_proxy_cost": -self.config.tendon_stretch_cost_weight * risks["tendon_strain"],
        }
        terminated = bool(q < self.angle_limits[0] - 0.15 or q > self.angle_limits[1] + 0.15 or abs(qvel) > 50.0)
        if terminated:
            self._reward_components["physical_failure"] = -1.0
        reward = float(sum(self._reward_components.values()))
        truncated = bool(self.time + 1e-10 >= self.config.episode_seconds and not terminated)
        self._episode_done = terminated or truncated
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
        target_acc = -(2 * np.pi * self._frequency) ** 2 * (target_q - self.config.reference_center)
        requested_acc = target_acc + 25.0 * (target_q - q) + 9.0 * (target_qvel - qvel)
        # The fixed-base 1-DOF model has constant hinge inertia. Using the
        # reference diagonal also avoids relying on renamed mjData mass fields.
        inertia = self.model.dof_M0[self.dof_index]
        requested_torque = float(inertia * requested_acc + self.data.qfrc_bias[self.dof_index] + 0.08 * qvel)
        capacities = []
        for muscle in self.muscles:
            normalized_length = muscle.fiber_length / muscle.params.optimal_fiber_length
            length_factor = max(0.0, 1.0 - ((normalized_length - 1.0) / muscle.params.force_length_width) ** 2)
            capacities.append(muscle.params.max_isometric_force * muscle.capacity * length_factor)
        torque_capacity = self.moment_arms * np.asarray(capacities)
        base = 0.015
        passive_torque = float(self.moment_arms @ np.asarray([o.passive_force for o in self._muscle_outputs]))
        residual_torque = requested_torque - passive_torque - base * float(torque_capacity.sum())
        result = np.full(4, base)
        selected = torque_capacity > 0 if residual_torque >= 0 else torque_capacity < 0
        available = float(np.abs(torque_capacity[selected]).sum())
        if available > 1e-8:
            result[selected] += abs(residual_torque) / available
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _risk_proxies(self) -> dict[str, float]:
        """Dimensionless engineering penalties; none estimates human pain.

        ROM uses squared fractional encroachment into a configured limit
        margin. Force uses squared excess of summed tendon tension relative
        to a chosen threshold; it is not anatomical joint contact force.
        Fiber and tendon penalties average squared *positive* threshold
        excesses. Shortening/slack never count as tensile overstretch.
        """
        q = float(self.data.qpos[self.qpos_index])
        clearance = min(q - self.angle_limits[0], self.angle_limits[1] - q)
        force = sum(output.tendon_force for output in self._muscle_outputs)
        fiber_ratios = np.asarray([
            muscle.fiber_length / muscle.params.optimal_fiber_length
            for muscle in self.muscles
        ])
        tendon_strains = np.asarray([output.tendon_strain for output in self._muscle_outputs])
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
        outputs = self._muscle_outputs
        return {
            "t": self.time,
            "q": q,
            "qvel": qvel,
            "target_q": target_q,
            "target_qvel": target_qvel,
            "tracking_error": [q - target_q, qvel - target_qvel],
            "muscle_names": list(self.muscle_names),
            "excitation": self._previous_excitation.tolist(),
            "delayed_excitation": [float(o.delayed_excitation) for o in outputs],
            "activation": [float(m.activation) for m in self.muscles],
            "tendon_force_N": [float(o.tendon_force) for o in outputs],
            "tendon_strain": [float(o.tendon_strain) for o in outputs],
            "fiber_length_m": [float(m.fiber_length) for m in self.muscles],
            "fiber_velocity_m_s": [float(o.fiber_velocity) for o in outputs],
            "capacity": [float(m.capacity) for m in self.muscles],
            "fatigue": [float(1.0 - m.capacity) for m in self.muscles],
            "muscle_torque_Nm": self._muscle_torque,
            "summed_tendon_force_proxy_N": float(sum(o.tendon_force for o in outputs)),
            "equilibrium_residual_N": [float(o.equilibrium_residual) for o in outputs],
            "fiber_at_bound": [bool(o.at_fiber_bound) for o in outputs],
            "pain_proxy_components": self._risk_proxies(),
            "risk_proxy_inputs": {
                "rom_clearance_rad": float(min(q - self.angle_limits[0], self.angle_limits[1] - q)),
                "summed_tendon_force_N": float(sum(o.tendon_force for o in outputs)),
                "fiber_length_ratio": [float(m.fiber_length / m.params.optimal_fiber_length) for m in self.muscles],
                "tendon_strain": [float(o.tendon_strain) for o in outputs],
            },
            "reward_components": dict(self._reward_components),
            "episode_step": self._step_count,
            "reference_source": "analytic_sine",
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
