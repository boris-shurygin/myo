"""Regressions for fatigue time scale and delayed-state recovery selection."""

from dataclasses import replace

import pytest

from myo.fatigue import (
    ELBOW_FATIGUE_RATE,
    ELBOW_RECOVERY_RATE,
    ELBOW_REST_MULTIPLIER,
    advance_capacity,
)
from myo.fatigue_calibration import reference_endurance
from myo.muscle import MuscleParameters, SeriesElasticMuscle


def test_compensated_hold_has_minute_scale_low_load_endurance():
    # An ideal force controller compensates capacity loss. Holding activation
    # constant would decrease the load and is not an endurance experiment.
    measured = {}
    for load in (0.20, 0.80):
        capacity, seconds, dt = 1.0, 0.0, 0.02
        while capacity >= load and seconds < 900.0:
            capacity = advance_capacity(
                capacity, load / capacity, dt,
                ELBOW_FATIGUE_RATE, ELBOW_RECOVERY_RATE,
                ELBOW_REST_MULTIPLIER,
            )
            seconds += dt
        measured[load] = seconds
        # The reduced model agrees at these two source-curve anchor loads;
        # its much larger errors at other loads remain visible in the report.
        assert seconds == pytest.approx(reference_endurance(load), rel=0.15)
    assert measured[0.20] > 10 * measured[0.80]


def test_rest_multiplier_uses_activation_and_capacity_recovers_without_reset():
    capacity = advance_capacity(
        0.5, 0.0, 120.0, ELBOW_FATIGUE_RATE,
        ELBOW_RECOVERY_RATE, ELBOW_REST_MULTIPLIER,
    )
    assert 0.90 < capacity < 0.92
    # A still-active muscle gets no rest acceleration even if a user has
    # already released the trigger; only its actual activation is consulted.
    active = advance_capacity(
        0.5, 0.2, 120.0, ELBOW_FATIGUE_RATE,
        ELBOW_RECOVERY_RATE, ELBOW_REST_MULTIPLIER,
    )
    unmodified = advance_capacity(
        0.5, 0.2, 120.0, ELBOW_FATIGUE_RATE,
        ELBOW_RECOVERY_RATE, 1.0,
    )
    assert active == unmodified < 0.5
    params = MuscleParameters(
        fatigue_rate=ELBOW_FATIGUE_RATE,
        recovery_rate=ELBOW_RECOVERY_RATE,
        rest_recovery_multiplier=ELBOW_REST_MULTIPLIER,
    )
    accelerated = SeriesElasticMuscle(params, dt=0.001)
    original = SeriesElasticMuscle(replace(params, rest_recovery_multiplier=1.0), dt=0.001)
    for muscle in (accelerated, original):
        muscle.reset(0.30, activation=0.2, capacity=0.5)
    first_recovery_step = None
    for step in range(1, 301):
        accelerated.step(0.0, 0.30)
        original.step(0.0, 0.30)
        if accelerated.activation > params.rest_activation_threshold:
            assert accelerated.capacity == original.capacity
        elif first_recovery_step is None:
            first_recovery_step = step
    # Forty milliseconds of motor delay plus about 150 ms of deactivation
    # must pass before the rest multiplier can affect physical capacity.
    assert 189 <= first_recovery_step <= 192
    assert accelerated.capacity > original.capacity
