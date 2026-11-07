"""Acceleration rewards must use real elapsed motion and retain causal delay."""
import numpy as np
import pytest

from myo.vertical_acceleration import (
    REFERENCE, VerticalAccelerationEnv, desired_acceleration, load_acceleration_policy,
)
from myo.vertical_control import REWARD, VerticalControlEnv, load_policy, save_policy
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_reward_matches_acceleration_at_start_of_interval_without_changing_physics():
    config = VerticalConfig(variant=1)
    env, native = VerticalAccelerationEnv(config), VerticalMuscles(config)
    env.reset(options={"initial_position_m": 0., "target_m": .04})
    native.reset()
    target = desired_acceleration(0., 0., .04)
    command = native.command_for_force(config.mass_kg*(9.81+target), 5.)
    _, reward, _, _, info = env.step(command)
    np.testing.assert_array_equal(env.physical.state(), native.step(command))
    assert info["desired_acceleration_m_s2"] == target
    assert info["actual_acceleration_m_s2"] == pytest.approx(native.velocity/config.dt)
    assert info["reward_components"]["acceleration"] > .84
    assert "position" not in info["reward_components"]
    assert reward == pytest.approx(.1*info["raw_reward"])
    # Reference and Gaussian width preserve the original stationary position width.
    reference = desired_acceleration(0., 0., .04)
    assert np.exp(-(reference/REFERENCE["scale_m_s2"])**2) == pytest.approx(np.exp(-(.04/.08)**2))
    env.reset(options={"initial_position_m": 0., "target_m": .04})
    _, _, _, _, fallen = env.step(np.zeros(2))
    assert fallen["actual_acceleration_m_s2"] == pytest.approx(-9.81)
    assert fallen["reward_components"]["acceleration"] < info["reward_components"]["acceleration"]
    assert REWARD["position_scale_m"] == .08


def test_acceleration_feedback_cannot_see_new_command_before_motor_delay():
    config = VerticalConfig(variant=4)
    first, second, native = VerticalAccelerationEnv(config), VerticalAccelerationEnv(config), VerticalMuscles(config)
    for env in (first, second):
        env.reset(options={"initial_position_m": 0., "target_m": .04})
    native.reset()
    activation = native.activation.copy()
    for index in range(3):
        _, _, _, _, a = first.step(np.array([.4, .05]))
        _, _, _, _, b = second.step(np.array([.05, .4]))
        np.testing.assert_array_equal(first.physical.state(), native.step(np.array([.4, .05])))
        if index < 2:
            assert a["actual_acceleration_m_s2"] == b["actual_acceleration_m_s2"]
            assert a["reward_components"]["acceleration"] == b["reward_components"]["acceleration"]
            np.testing.assert_allclose(first.physical.activation, activation, atol=1e-15)
        else:
            assert a["actual_acceleration_m_s2"] > b["actual_acceleration_m_s2"]
            assert not np.allclose(first.physical.activation, first.physical.delivered)


def test_ppo_acceleration_checkpoint_rejects_position_reward_and_retains_direct_actions(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from myo.policy_beta import BetaExcitationPolicy
    torch.set_num_threads(1)
    env = VerticalAccelerationEnv(VerticalConfig(variant=4))
    baseline = VerticalControlEnv(env.physical.config)
    control = VerticalAccelerationEnv(env.physical.config, objective="position")
    options = {"initial_position_m": 0., "target_m": .04}
    baseline.reset(options=options)
    control.reset(options=options)
    a, r, _, _, _ = baseline.step(np.array([.2, .1]))
    b, s, _, _, _ = control.step(np.array([.2, .1]))
    np.testing.assert_array_equal(a, b)
    assert r == s
    model = sb3.PPO(BetaExcitationPolicy, env, seed=0, n_steps=32, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.learn(64)
    path = tmp_path/"actor.zip"
    save_policy(model, env, path, {"objective": "acceleration"})
    restored, restored_env, _ = load_acceleration_policy(path)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
    assert restored_env.contract() == env.contract()
    with pytest.raises(ValueError, match="contract mismatch"):
        load_policy(path, control)
