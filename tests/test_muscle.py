"""Behavioral checks for delay, compliance, energy and fatigue."""

from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from myo.muscle import MuscleEquilibriumError, MuscleParameters, SeriesElasticMuscle


def muscle(**overrides):
    params = replace(MuscleParameters(), **overrides)
    model = SeriesElasticMuscle(params, dt=0.001)
    model.reset(params.optimal_fiber_length + params.tendon_slack_length)
    return model


def test_motor_delay_is_separate_from_activation_filter():
    model = muscle(motor_delay=0.040, fatigue_rate=0.0)
    for _ in range(40):
        output = model.step(1.0, 0.30)
        assert output.delayed_excitation == 0.0
        assert output.activation == 0.0
        assert output.tendon_force == pytest.approx(0.0, abs=1e-8)
    output = model.step(1.0, 0.30)
    assert output.delayed_excitation == 1.0
    assert 0.0 < output.activation < 1.0
    assert output.activation == pytest.approx(1 - math.exp(-0.001 / 0.015))


def test_fixed_total_length_allows_fiber_shortening_and_tendon_extension():
    model = muscle(motor_delay=0.0, fatigue_rate=0.0)
    initial = model.last_output
    for _ in range(500):
        output = model.step(0.8, 0.30)
    assert output.fiber_length < initial.fiber_length
    assert output.tendon_length > initial.tendon_length
    assert output.fiber_length + output.tendon_length == pytest.approx(0.30)
    assert output.tendon_strain > 0.0
    assert output.tendon_force > 0.0
    assert output.tendon_energy > 0.0
    assert abs(output.equilibrium_residual) < 1e-6


def test_activation_alone_does_not_determine_force():
    model = muscle(motor_delay=0.0)
    short = model.reset(0.28, activation=0.7)
    long = model.reset(0.31, activation=0.7)
    fatigued = model.reset(0.31, activation=0.7, capacity=0.25)
    assert short.activation == long.activation == fatigued.activation
    assert abs(short.tendon_force - long.tendon_force) > 10
    assert fatigued.tendon_force < 0.5 * long.tendon_force


def test_unpowered_fixed_path_relaxation_cannot_create_elastic_energy():
    model = muscle(motor_delay=0.0)
    initial = model.reset(0.34)
    # Passive tension is part of this same unpowered relaxation experiment.
    assert initial.activation == initial.active_force == 0.0
    assert initial.passive_force > 0.0 and initial.tendon_force > 0.0
    energies = []
    for _ in range(200):
        output = model.step(0.0, 0.33)
        energies.append(output.tendon_energy + output.passive_energy)
        assert output.damping_power >= 0.0
        assert output.mt_power == 0.0
    assert np.max(np.diff(energies)) < 1e-9
    assert energies[-1] < energies[0]


def test_capacity_falls_under_sustained_activation_and_recovers_at_rest():
    model = muscle(motor_delay=0.0, fatigue_rate=0.8, recovery_rate=0.2)
    for _ in range(2000):
        loaded = model.step(1.0, 0.30)
    assert 0 < loaded.capacity < 0.5
    for _ in range(3000):
        rested = model.step(0.0, 0.30)
    assert loaded.capacity < rested.capacity < 1.0
    assert rested.activation < 1e-20
    assert model.fatigue == pytest.approx(1 - rested.capacity)


def _varying_path_run(dt):
    params = MuscleParameters()
    model = SeriesElasticMuscle(params, dt=dt)
    model.reset(0.30)
    for index in range(round(1.0 / dt)):
        time = index * dt
        length = 0.30 + 0.015 * math.sin(2 * math.pi * time)
        speed = 0.015 * 2 * math.pi * math.cos(2 * math.pi * time)
        command = 0.6 if 0.1 <= time < 0.8 else 0.0
        output = model.step(command, length, speed)
        assert np.isfinite(np.array(list(asdict(output).values()), dtype=float)).all()
        assert 0 <= output.activation <= 1
        assert 0 <= output.capacity <= 1
        assert output.tendon_length >= 0
        assert output.tendon_force >= 0
        assert output.tendon_energy >= 0
        assert abs(output.equilibrium_residual) < 1e-6
    return output


def test_moving_path_is_finite_and_converges_with_smaller_substeps():
    coarse = _varying_path_run(0.001)
    fine = _varying_path_run(0.0005)
    assert abs(coarse.fiber_length - fine.fiber_length) < 2e-4
    assert abs(coarse.tendon_force - fine.tendon_force) < 2.0


def test_impossible_geometry_raises_instead_of_clipping_to_a_fake_equilibrium():
    model = muscle(motor_delay=0.0)
    with pytest.raises(MuscleEquilibriumError, match="no fiber root"):
        model.step(0.0, 0.055)
