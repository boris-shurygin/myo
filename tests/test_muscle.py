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


def test_strength_calibration_changes_active_force_without_stiffening_passive_tissues():
    baseline = muscle(active_force_scale=1.0, fatigue_rate=0.0)
    stronger = muscle(active_force_scale=1.5, fatigue_rate=0.0)
    normal_active = baseline.reset(0.30, activation=0.8)
    scaled_active = stronger.reset(0.30, activation=0.8)
    assert scaled_active.active_force > 1.4 * normal_active.active_force
    assert scaled_active.tendon_force > 1.4 * normal_active.tendon_force
    assert normal_active.activation == scaled_active.activation == 0.8
    assert normal_active.capacity == scaled_active.capacity == 1.0

    normal_passive = baseline.reset(0.34)
    scaled_passive = stronger.reset(0.34)
    assert normal_passive.passive_force > 0
    assert scaled_passive.passive_force == normal_passive.passive_force
    assert scaled_passive.tendon_force == normal_passive.tendon_force
    # Sampling a small path extension without advancing fibers measures the
    # series tendon stiffness directly, independently of active equilibrium.
    normal_increment = baseline.sample(0.341).tendon_force - normal_passive.tendon_force
    scaled_increment = stronger.sample(0.341).tendon_force - scaled_passive.tendon_force
    assert normal_increment > 0
    assert scaled_increment == normal_increment

    for step in range(200):
        normal = baseline.step(0.0, 0.33)
        scaled = stronger.step(0.0, 0.33)
        assert normal.active_force == scaled.active_force == 0.0
        assert scaled.fiber_length == normal.fiber_length
        assert scaled.tendon_force == normal.tendon_force
        assert scaled.damping_force == normal.damping_force
        if step == 0:
            assert normal.damping_power > 0


def test_eccentric_active_force_and_passive_damping_have_separate_finite_limits():
    model = muscle(motor_delay=0.0, fatigue_rate=0.0, recovery_rate=0.0)
    p = model.params
    eccentric_multipliers = []
    for extension in (0.0001, 0.001, 0.01, 0.03, 0.08):
        isometric = model.reset(0.30, activation=0.8, capacity=0.6)
        assert isometric.damping_force == 0
        output = model.step(0.8, 0.30 + extension)
        assert output.fiber_velocity > 0
        active_limit = p.max_isometric_force * output.activation * output.capacity
        assert 0 <= output.active_force <= active_limit * p.max_eccentric_force
        assert 0 <= output.damping_force <= p.max_isometric_force * p.fiber_damping
        assert output.damping_power == pytest.approx(output.damping_force * output.fiber_velocity)
        assert abs(output.equilibrium_residual) < 1e-6
        # Normalize out the specified force-length curve so changes in fiber
        # length cannot mask an unbounded eccentric multiplier.
        normalized = output.fiber_length / p.optimal_fiber_length
        force_length = max(0.0, 1 - ((normalized - 1) / p.force_length_width) ** 2)
        if force_length > 0:
            eccentric_multipliers.append(output.active_force / (active_limit * force_length))
    assert 1 < eccentric_multipliers[0] < eccentric_multipliers[-1] < p.max_eccentric_force

    # Removing activation must remove active braking, while passive stretch
    # and passive dissipation remain distinct and measurable.
    model.reset(0.30)
    unpowered = model.step(0.0, 0.34)
    assert unpowered.active_force == 0
    assert unpowered.passive_force > 0
    assert 0 < unpowered.damping_force <= p.max_isometric_force * p.fiber_damping


def test_sampling_end_pose_recomputes_tendon_force_without_advancing_state():
    model = muscle(motor_delay=0.040, fatigue_rate=0.0)
    model.reset(0.30, activation=0.6)
    applied = model.step(0.7, 0.30)
    state = model.state_vector()
    sampled = model.sample(0.301, 0.05)
    np.testing.assert_array_equal(model.state_vector(), state)
    assert model.last_output is applied
    assert sampled.fiber_length == applied.fiber_length
    assert sampled.activation == applied.activation
    assert sampled.delayed_excitation == applied.delayed_excitation
    assert sampled.fiber_length + sampled.tendon_length == pytest.approx(0.301)
    force_difference = model.params.tendon_stiffness * 0.001
    assert sampled.tendon_force - applied.tendon_force == pytest.approx(force_difference)
    assert sampled.equilibrium_residual - applied.equilibrium_residual == pytest.approx(force_difference)
    assert sampled.mt_power == pytest.approx(-sampled.tendon_force * 0.05)


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
