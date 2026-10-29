"""Small, explicitly phenomenological series-elastic muscle for the V0 demo.

An active force-length element, a parallel passive spring, and a viscous fiber
element pull a separate tensile-only series tendon. The massless fiber/tendon
junction is integrated implicitly. This is not an implementation or validation
of OpenSim, Millard, MuJoCo's built-in muscle, or the MyoSuite fatigue model.
See docs/model.md for equations, units, assumptions, and numerical limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .delay import FixedStepDelay


@dataclass(frozen=True)
class MuscleParameters:
    name: str = "muscle"
    max_isometric_force: float = 800.0  # N; reference strength, not fitted anatomy.
    optimal_fiber_length: float = 0.10  # m
    tendon_slack_length: float = 0.20  # m
    tendon_strain_at_max_force: float = 0.04
    activation_time_constant: float = 0.015  # s
    deactivation_time_constant: float = 0.050  # s
    motor_delay: float = 0.040  # s; integer multiple of physics dt.
    fatigue_rate: float = 0.08  # 1/s
    recovery_rate: float = 0.02  # 1/s
    fiber_damping: float = 0.3  # Dimensionless, at reference_shortening_speed.
    reference_shortening_speed: float = 10.0  # Optimal fiber lengths / s.
    force_length_width: float = 0.5  # Half-width in optimal fiber lengths.
    passive_stiffness: float = 4.0  # Normalized quadratic force coefficient.

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("muscle name must be a nonempty string")
        positive = (
            "max_isometric_force", "optimal_fiber_length", "tendon_slack_length",
            "tendon_strain_at_max_force", "activation_time_constant",
            "deactivation_time_constant", "fiber_damping", "reference_shortening_speed",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("motor_delay", "fatigue_rate", "recovery_rate", "passive_stiffness"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.force_length_width) or not 0 < self.force_length_width < 1:
            raise ValueError("force_length_width must lie strictly between zero and one")

    @property
    def tendon_stiffness(self) -> float:
        """Linear tensile stiffness in N/m; independent of current fatigue."""
        return self.max_isometric_force / (
            self.tendon_slack_length * self.tendon_strain_at_max_force
        )

    @property
    def fiber_viscosity(self) -> float:
        """Physical viscosity in N s/m."""
        return self.max_isometric_force * self.fiber_damping / (
            self.reference_shortening_speed * self.optimal_fiber_length
        )


@dataclass(frozen=True)
class MuscleOutput:
    excitation: float
    delayed_excitation: float
    activation: float
    capacity: float
    fiber_length: float
    fiber_velocity: float
    tendon_length: float
    tendon_strain: float
    tendon_force: float
    active_force: float
    passive_force: float
    tendon_energy: float
    passive_energy: float
    damping_power: float
    mt_power: float
    equilibrium_residual: float
    solver_iterations: int
    at_fiber_bound: bool

    @property
    def fatigue(self) -> float:
        return 1.0 - self.capacity


class MuscleEquilibriumError(RuntimeError):
    """No admissible force-balance solution, or failure to converge."""


class SeriesElasticMuscle:
    """One muscle advanced at fixed physics dt, with its motor delay included.

    Call ``reset(mt_length)`` before stepping. Lengths are in meters; positive
    ``mt_velocity`` means lengthening. Tension is positive, so force delivered
    to a joint is ``-tendon_force * d(mt_length)/dq``. ``mt_velocity`` is used
    for power telemetry; the fiber solve holds the supplied path length fixed
    for this substep (first-order splitting with the rigid-body integrator).
    """

    def __init__(self, params: MuscleParameters, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.params = params
        self.dt = float(dt)
        # This conservative bound makes the implicit residual strictly
        # increasing, including on the descending active force-length limb.
        max_dt = params.fiber_viscosity * params.force_length_width * (
            params.optimal_fiber_length / (2 * params.max_isometric_force)
        )
        if dt >= max_dt:
            raise ValueError(
                f"physics dt must be less than {max_dt:g} s for this fiber damping; "
                "use smaller substeps or more damping"
            )
        self.delay = FixedStepDelay(dt, params.motor_delay)
        self.activation = 0.0
        self.capacity = 1.0
        self.fiber_length = params.optimal_fiber_length
        self.last_output: MuscleOutput | None = None
        self._initialized = False

    @property
    def fatigue(self) -> float:
        return 1.0 - self.capacity

    @property
    def minimum_fiber_length(self) -> float:
        return self.params.optimal_fiber_length * (1 - self.params.force_length_width)

    def _check_length(self, length: float) -> None:
        if not math.isfinite(length) or length < self.minimum_fiber_length:
            raise ValueError("mt_length must be finite and at least the minimum fiber length")

    @staticmethod
    def _check_unit(value: float, name: str) -> None:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    def _fiber_forces(self, length: float, activation: float, capacity: float):
        p = self.params
        normalized = length / p.optimal_fiber_length
        z = (normalized - 1.0) / p.force_length_width
        fl = max(0.0, 1.0 - z * z)
        active = p.max_isometric_force * capacity * activation * fl
        active_derivative = (
            -2 * z / (p.force_length_width * p.optimal_fiber_length)
            * p.max_isometric_force * capacity * activation
            if abs(z) < 1.0 else 0.0
        )
        stretch = max(0.0, normalized - 1.0)
        passive = p.max_isometric_force * p.passive_stiffness * stretch * stretch
        passive_derivative = (
            2 * p.max_isometric_force * p.passive_stiffness
            * stretch / p.optimal_fiber_length
        )
        return active, passive, active_derivative + passive_derivative

    def _tendon_force(self, mt_length: float, fiber_length: float) -> float:
        extension = max(0.0, mt_length - fiber_length - self.params.tendon_slack_length)
        return self.params.tendon_stiffness * extension

    def reset(self, mt_length: float, activation: float = 0.0,
              capacity: float = 1.0) -> MuscleOutput:
        """Reset to static fiber/tendon equilibrium and a constant command history."""
        self._check_length(mt_length)
        self._check_unit(activation, "activation")
        self._check_unit(capacity, "capacity")
        self.activation, self.capacity = float(activation), float(capacity)
        self.delay.reset(activation)
        p = self.params
        if activation * capacity == 0 and mt_length <= p.tendon_slack_length + p.optimal_fiber_length:
            length = max(self.minimum_fiber_length, mt_length - p.tendon_slack_length)
            iterations = 0
        else:
            lo, hi = self.minimum_fiber_length, mt_length
            # Static equilibrium can have an unloaded lower-bound solution.
            # Bisection gives a bracketed solution without dividing by activation.
            length = lo
            for iterations in range(1, 81):
                length = (lo + hi) * 0.5
                active, passive, _ = self._fiber_forces(length, activation, capacity)
                residual = self._tendon_force(mt_length, length) - active - passive
                if abs(residual) <= self._force_tolerance:
                    break
                if residual > 0:
                    lo = length
                else:
                    hi = length
            else:
                raise MuscleEquilibriumError("static fiber equilibrium did not converge")
        self.fiber_length = length
        self._initialized = True
        self.last_output = self._output(activation, activation, mt_length, 0.0, 0.0, iterations)
        return self.last_output

    @property
    def _force_tolerance(self) -> float:
        return 1e-10 * max(1.0, self.params.max_isometric_force)

    def step(self, excitation: float, mt_length: float, mt_velocity: float = 0.0,
             dt: float | None = None) -> MuscleOutput:
        if not self._initialized:
            raise RuntimeError("reset(mt_length) must be called before step")
        self._check_unit(excitation, "excitation")
        self._check_length(mt_length)
        if not math.isfinite(mt_velocity):
            raise ValueError("mt_velocity must be finite")
        if dt is not None and (not math.isfinite(dt) or not math.isclose(
            dt, self.dt, rel_tol=1e-12, abs_tol=0.0
        )):
            raise ValueError("step dt must match the fixed physics dt")
        delayed = self.delay.step(excitation)
        p = self.params
        tau = p.activation_time_constant if delayed > self.activation else p.deactivation_time_constant
        activation = delayed + (self.activation - delayed) * math.exp(-self.dt / tau)
        rate = p.recovery_rate + p.fatigue_rate * activation
        if rate:
            capacity_eq = p.recovery_rate / rate
            capacity = capacity_eq + (self.capacity - capacity_eq) * math.exp(-rate * self.dt)
        else:
            capacity = self.capacity
        old_length = self.fiber_length
        eta_dt = p.fiber_viscosity / self.dt

        def residual_and_slope(length: float):
            active, passive, dfiber = self._fiber_forces(length, activation, capacity)
            tendon = self._tendon_force(mt_length, length)
            residual = eta_dt * (length - old_length) - tendon + active + passive
            slope = eta_dt + dfiber + (p.tendon_stiffness if tendon > 0 else 0.0)
            return residual, slope

        lo, hi = self.minimum_fiber_length, mt_length
        r_lo, _ = residual_and_slope(lo)
        r_hi, _ = residual_and_slope(hi)
        if r_lo > self._force_tolerance or r_hi < -self._force_tolerance:
            raise MuscleEquilibriumError(
                "no fiber root inside physical length bounds; check path length changes "
                f"and substeps (residuals {r_lo:g}, {r_hi:g} N)"
            )
        length = min(hi, max(lo, old_length))  # Initial guess only, never a clipped solution.
        for iterations in range(1, 65):
            residual, slope = residual_and_slope(length)
            if abs(residual) <= self._force_tolerance:
                break
            if residual > 0:
                hi = length
            else:
                lo = length
            candidate = length - residual / slope
            length = candidate if lo < candidate < hi else 0.5 * (lo + hi)
        else:
            raise MuscleEquilibriumError("implicit fiber equilibrium did not converge")
        self.activation, self.capacity, self.fiber_length = activation, capacity, length
        velocity = (length - old_length) / self.dt
        self.last_output = self._output(excitation, delayed, mt_length, mt_velocity, velocity, iterations)
        return self.last_output

    def _output(self, excitation: float, delayed: float, mt_length: float,
                mt_velocity: float, fiber_velocity: float, iterations: int) -> MuscleOutput:
        p = self.params
        tendon_length = mt_length - self.fiber_length
        extension = max(0.0, tendon_length - p.tendon_slack_length)
        tension = p.tendon_stiffness * extension
        active, passive, _ = self._fiber_forces(self.fiber_length, self.activation, self.capacity)
        normalized_stretch = max(0.0, self.fiber_length / p.optimal_fiber_length - 1.0)
        return MuscleOutput(
            excitation=float(excitation), delayed_excitation=float(delayed),
            activation=self.activation, capacity=self.capacity,
            fiber_length=self.fiber_length, fiber_velocity=fiber_velocity,
            tendon_length=tendon_length, tendon_strain=extension / p.tendon_slack_length,
            tendon_force=tension, active_force=active, passive_force=passive,
            tendon_energy=0.5 * p.tendon_stiffness * extension * extension,
            passive_energy=p.max_isometric_force * p.passive_stiffness
            * p.optimal_fiber_length * normalized_stretch ** 3 / 3,
            damping_power=p.fiber_viscosity * fiber_velocity * fiber_velocity,
            mt_power=-tension * mt_velocity,
            equilibrium_residual=tension - active - passive - p.fiber_viscosity * fiber_velocity,
            solver_iterations=iterations,
            at_fiber_bound=(math.isclose(self.fiber_length, self.minimum_fiber_length, abs_tol=1e-10)
                            or math.isclose(self.fiber_length, mt_length, abs_tol=1e-10)),
        )

    def state_vector(self) -> np.ndarray:
        """[activation, fiber_length, capacity, pending commands...], independent copy."""
        return np.concatenate((
            np.array([self.activation, self.fiber_length, self.capacity], dtype=np.float64),
            self.delay.pending,
        ))
