"""Environment contracts and actual muscle-to-rigid-body coupling."""

import json

import numpy as np
import pytest

from myo.env import PreacherCurlConfig, PreacherCurlEnv


def test_seeded_rollouts_repeat_and_observations_cover_delay_state():
    first, second = PreacherCurlEnv(), PreacherCurlEnv()
    try:
        obs_a, _ = first.reset(seed=7)
        obs_b, _ = second.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert len(first.observation_names) == obs_a.size
        assert first.observation_space.contains(obs_a)
        expected_delay_entries = sum(len(m.delay.pending) for m in first.muscles)
        assert sum("pending_excitation" in n for n in first.observation_names) == expected_delay_entries
        for _ in range(8):
            action = first.baseline_action()
            a, b = first.step(action), second.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
        json.dumps(first.telemetry(), allow_nan=False)
    finally:
        first.close()
        second.close()


def test_gravity_moves_forearm_while_upper_arm_remains_fixed():
    env = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0, motor_delay=0))
    try:
        env.reset(seed=1, options={"q": float(np.deg2rad(75)), "qvel": 0.0})
        upper_arm_id = env.model.body("upper_arm").id
        upper_arm_position = env.data.xpos[upper_arm_id].copy()
        initial_angle = env.telemetry()["q"]
        for _ in range(5):
            env.step(np.zeros(4, dtype=np.float32))
        assert env.model.nv == 1
        assert env.model.nu == 0
        assert env.telemetry()["q"] < initial_angle
        np.testing.assert_array_equal(env.data.xpos[upper_arm_id], upper_arm_position)
    finally:
        env.close()


def test_external_tendon_forces_produce_the_documented_elbow_torque():
    env = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0, motor_delay=0))
    try:
        env.reset(seed=2, options={"q": float(np.deg2rad(75)), "qvel": 0.0})
        for _ in range(4):
            _, _, _, _, info = env.step(np.array([0.3, 0.3, 0.3, 0.0], dtype=np.float32))
        expected = float(env.moment_arms @ np.asarray(info["tendon_force_N"]))
        assert expected > 0
        assert info["muscle_torque_Nm"] == pytest.approx(expected)
        assert env.data.qfrc_applied[env.dof_index] == pytest.approx(expected)
        assert max(info["tendon_strain"]) > 0
        assert max(info["activation"]) > 0
    finally:
        env.close()


def test_time_limit_is_truncation_and_finished_episode_requires_reset():
    env = PreacherCurlEnv(PreacherCurlConfig(episode_seconds=0.04))
    try:
        env.reset(seed=3)
        _, _, terminated, truncated, _ = env.step(env.baseline_action())
        assert not terminated and not truncated
        _, _, terminated, truncated, _ = env.step(env.baseline_action())
        assert not terminated and truncated
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(4))
    finally:
        env.close()


def test_rom_proxy_only_penalizes_encroachment_into_configured_margin():
    env = PreacherCurlEnv()
    try:
        env.reset(seed=5, options={"q": float(np.mean(env.angle_limits))})
        assert env._risk_proxies()["rom_margin"] == 0
        near_limit = float(env.angle_limits[0] + 0.25 * env.config.rom_margin)
        env.reset(seed=5, options={"q": near_limit})
        assert env._risk_proxies()["rom_margin"] == pytest.approx(0.75**2)
        _, _, _, _, info = env.step(np.zeros(4))
        assert info["pain_proxy_components"]["rom_margin"] > 0
        assert info["reward_components"]["rom_margin_proxy_cost"] < 0
    finally:
        env.close()


def test_static_coactivation_increases_force_proxy_at_identical_joint_angle():
    relaxed = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0))
    tense = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0.8))
    try:
        options = {"q": float(np.deg2rad(75)), "qvel": 0.0}
        relaxed.reset(seed=6, options=options)
        tense.reset(seed=6, options=options)
        assert relaxed.telemetry()["q"] == tense.telemetry()["q"]
        assert relaxed._risk_proxies()["summed_tendon_force"] == 0
        assert tense._risk_proxies()["summed_tendon_force"] > 0
        assert tense.telemetry()["summed_tendon_force_proxy_N"] > relaxed.telemetry()["summed_tendon_force_proxy_N"]
    finally:
        relaxed.close()
        tense.close()


def test_stretch_proxies_detect_positive_extension_and_not_shortening():
    env = PreacherCurlEnv(PreacherCurlConfig(
        initial_activation=0.8,
        fiber_stretch_threshold=1.01,
        tendon_strain_threshold=0.001,
    ))
    try:
        env.reset(seed=8, options={"q": float(np.deg2rad(75))})
        assert all(m.fiber_length < m.params.optimal_fiber_length for m in env.muscles)
        assert env._risk_proxies()["fiber_stretch"] == 0
        assert env._risk_proxies()["tendon_strain"] > 0
        env.reset(seed=8, options={"q": float(np.deg2rad(20))})
        assert env._risk_proxies()["fiber_stretch"] > 0
    finally:
        env.close()
