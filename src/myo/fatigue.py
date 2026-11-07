"""Reduced compartment fatigue proxy, driven by existing muscle activation.

With available capacity c and activation a, the active, resting and fatigued
fractions are a*c, (1-a)*c and 1-c. Their fatigue balance gives
dc/dt = R_eff*(1-c) - F*a*c. This retains the existing activation dynamics;
it does not add the recruitment controller from the full 3CC model.

Elbow priors: Frey-Law et al. (2012), Table 1, F=.00912/s, R=.00094/s;
Looft et al. (2018), Table 4, rest multiplier 15. Applying joint-level priors
to individual muscles is an approximation. The activation threshold used for
rest is an engineering adaptation, not a measured physiological boundary.
See docs/calibration.md and ``python3 -m myo.fatigue_calibration``.
"""

from __future__ import annotations

import math


ELBOW_FATIGUE_RATE = 0.00912  # 1/s
ELBOW_RECOVERY_RATE = 0.00094  # 1/s during contraction
ELBOW_REST_MULTIPLIER = 15.0
REST_ACTIVATION_THRESHOLD = 0.01


def advance_capacity(
    capacity: float,
    activation: float,
    dt: float,
    fatigue_rate: float,
    recovery_rate: float,
    rest_multiplier: float = 1.0,
    rest_activation_threshold: float = REST_ACTIVATION_THRESHOLD,
) -> float:
    """Exact substep for fixed activation; inputs are validated by the muscle.

    Rest is determined from actual activation after transport delay and its
    activation/deactivation filter. A released trigger alone cannot activate
    faster recovery. With rest_multiplier=1 this is the historical V0 law.
    Capacity remains in [0,1] without a numerical clipping operation.
    """
    recovery = recovery_rate * (
        rest_multiplier if activation <= rest_activation_threshold else 1.0
    )
    rate = recovery + fatigue_rate * activation
    if rate == 0.0:
        return capacity
    equilibrium = recovery / rate
    return equilibrium + (capacity - equilibrium) * math.exp(-rate * dt)


def isometric_endurance(
    load: float,
    fatigue_rate: float = ELBOW_FATIGUE_RATE,
    recovery_rate: float = ELBOW_RECOVERY_RATE,
) -> float:
    """Ideal constant-load endurance in seconds, load / fresh isometric force.

    This isolated calibration experiment assumes an ideal controller raises
    activation to load/capacity while capacity >= load. Thus active fraction
    stays at load and dc/dt = R*(1-c)-F*load. Failure is capacity < load.
    It excludes geometry, recruitment dynamics, force-length/velocity effects,
    and rests. It is not the endurance time of the actual neural controller.
    """
    if not math.isfinite(load) or not 0.0 < load <= 1.0:
        raise ValueError("load must be in (0, 1]")
    if min(fatigue_rate, recovery_rate) < 0 or not all(
        math.isfinite(rate) for rate in (fatigue_rate, recovery_rate)
    ):
        raise ValueError("fatigue and recovery rates must be finite and nonnegative")
    if fatigue_rate == 0.0:
        return math.inf
    if load == 1.0:
        return 0.0
    if recovery_rate == 0.0:
        return (1.0 - load) / (fatigue_rate * load)
    fraction = recovery_rate * (1.0 - load) / (fatigue_rate * load)
    if fraction >= 1.0:
        return math.inf
    return -math.log1p(-fraction) / recovery_rate
