"""Causal D1 history and release semantics through the delayed muscle path."""
import numpy as np
import pytest
from myo.env import PreacherCurlConfig, PreacherCurlEnv


def test_command_history_contains_sent_action_and_reset_does_not_leak_episode():
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive", initial_activation=0))
    try:
        first, _ = env.reset(seed=3)
        initial = dict(zip(env.observation_names, first))
        env.set_command(1.8, 0.0)
        before = env.observe()
        assert env.time == 0
        assert env.data.qpos[0] == pytest.approx(initial['angle_centered_rad'] + env.config.reference_center)
        action = np.full(4, 0.8)
        obs, _, _, _, info = env.step(action)
        values = dict(zip(env.observation_names, obs))
        assert values['history/1/excitation/biceps_long'] == pytest.approx(0.8)
        assert values['history/1/command/trigger'] == 0
        assert values['history/1/sensor/angle_centered_rad'] == before[0]
        assert info['excitation'] == pytest.approx(action)
        assert info['delayed_excitation'] == [0.0] * 4
        assert info['reward_components']['position_tracking'] == 0
        assert info['reward_components']['velocity_tracking'] == 0
        assert sum('pending_excitation' in n for n in env.observation_names) == sum(len(m.delay.pending) for m in env.muscles)
        reset, _ = env.reset(seed=3)
        np.testing.assert_array_equal(reset, first)
    finally:
        env.close()


def test_release_preserves_queued_excitation_and_passive_limit_reports_substep_peaks():
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive", initial_activation=0.4))
    try:
        env.reset(seed=0, options={'q': float(env.angle_limits[0] + 0.05), 'qvel': -0.2})
        env.set_command(1.5, 0)
        _, _, _, _, state = env.step(np.zeros(4))
        assert min(state['activation']) > 0.3
        assert min(state['delayed_excitation']) == pytest.approx(0.4)
        assert state['substep_peaks']['outward_limit_velocity_rad_s'] >= 0.2
        assert state['substep_peaks']['passive_limit_torque_Nm'] > 0
        assert state['applied_force_time_s'] < state['t']
        np.testing.assert_allclose(np.array(state['fiber_length_m']) + state['tendon_length_m'],
                                   state['musculotendon_length_m'], atol=1e-14)
    finally:
        env.close()


def test_asymmetric_training_fatigue_initializes_current_state_and_history_together():
    from myo.imitation import reset_training_capacity_profile

    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        env.reset(seed=8)
        pose, velocity = env.data.qpos.copy(), env.data.qvel.copy()
        reset_training_capacity_profile(env, 0.75, 1.0)
        state = env.telemetry()
        values = dict(zip(env.observation_names, env.observe()))
        assert state["capacity"] == [0.75, 0.75, 0.75, 1.0]
        assert state["activation"] == [env.config.initial_activation] * 4
        for name in env._sensor_names:
            for age in range(1, env.config.history_steps + 1):
                assert values[f"history/{age}/sensor/{name}"] == values[name]
        np.testing.assert_array_equal(env.data.qpos, pose)
        np.testing.assert_array_equal(env.data.qvel, velocity)
        assert env.time == 0.0
        env.step(env.baseline_action())
        with pytest.raises(RuntimeError, match="immediately after reset"):
            reset_training_capacity_profile(env, 1.0, 1.0)
    finally:
        env.close()


def test_unseen_curriculum_command_does_not_change_reward_for_previous_action():
    manual = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    scripted = PreacherCurlEnv(PreacherCurlConfig(task_mode="curriculum"))
    try:
        options = dict(q=1.3, qvel=0.0, payload_mass=1.0, capacity=1.0)
        manual.reset(seed=9, options=options)
        scripted.reset(seed=9, options=options)
        for _ in range(50):
            a = manual.step(np.full(4, 0.1))
            b = scripted.step(np.full(4, 0.1))
            assert a[1] == pytest.approx(b[1])
        assert manual.telemetry()['requested_target_q'] != scripted.telemetry()['requested_target_q']
    finally:
        manual.close(); scripted.close()


def test_target_reversal_respects_speed_and_acceleration_without_moving_joint():
    from myo.commands import ArmCommand
    command = ArmCommand(1.0, 1.8)
    dt, speed, acceleration = 0.02, 1.0, 3.0
    for i in range(300):
        if i == 20:
            command.set(0.4, 0.5, np.array([0.2, 2.2]))
        previous = command.velocity
        command.advance(dt, speed, acceleration)
        assert abs(command.velocity) <= speed + 1e-10
        assert abs(command.velocity - previous) <= acceleration * dt + 1e-10
    assert command.target == pytest.approx(0.4, abs=0.002)
    command.set(0.2, 0.5, np.array([0.2, 2.2]))
    for _ in range(300):
        previous = command.velocity
        command.advance(dt, speed, acceleration)
        assert command.target >= 0.2 - 1e-10
        assert abs(command.velocity - previous) <= acceleration * dt + 1e-10
