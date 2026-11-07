"""Direct PPO must preserve native delays, honest failures and portable commands."""
import json
import numpy as np
import pytest

from myo.vertical_control import (
    VerticalControlEnv, VALIDATION_CASES, evaluate_policy, load_policy, save_policy,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_control_wrapper_preserves_native_transport_and_target_change_is_not_a_reset():
    config = VerticalConfig(variant=4)
    wrapped, native = VerticalControlEnv(config), VerticalMuscles(config)
    wrapped.reset(options={"initial_position_m": 0., "target_m": .08})
    native.reset()
    initial_activation = native.activation.copy()
    for index in range(3):
        command = np.array([.4, .1])
        _, _, terminated, truncated, info = wrapped.step(command)
        np.testing.assert_array_equal(wrapped.physical.state(), native.step(command))
        if index < 2:
            np.testing.assert_allclose(wrapped.physical.activation, initial_activation, atol=1e-15)
        assert not terminated and not truncated
        assert info["error_m"] == pytest.approx(native.position-.08)
    state = wrapped.physical.state().copy()
    old_observation = wrapped.observation()
    wrapped.set_target(-.08)
    np.testing.assert_array_equal(state, wrapped.physical.state())
    np.testing.assert_array_equal(old_observation[:-1], wrapped.observation()[:-1])
    assert wrapped.observation()[-1] == pytest.approx(-.8)


def test_evaluation_rejects_a_goal_blind_hold_and_does_not_reward_an_early_fall():
    config = VerticalConfig(variant=1)
    native = VerticalMuscles(config)
    native.reset()

    class Constant:
        def __init__(self, action):
            self.action = action

        def predict(self, observation, deterministic=True):
            return self.action, None

    held = evaluate_policy(Constant(native.excitation.copy()), config, VALIDATION_CASES)
    assert not held["ready"]
    assert held["cases"][-1]["passed"]
    assert held["passed"] < held["total"]
    fallen = evaluate_policy(Constant(np.zeros(2)), config, ((0., .08),))
    assert fallen["failures"] == 1 and fallen["passed"] == 0
    assert fallen["score_m"] == .3
    assert fallen["cases"][0]["duration_s"] < 4.


def test_ppo_update_and_checkpoint_preserve_commands_and_reject_other_mechanics(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from gymnasium.utils.env_checker import check_env
    from myo.policy_beta import BetaExcitationPolicy
    torch.set_num_threads(1)
    env = VerticalControlEnv(VerticalConfig(variant=4))
    check_env(env, skip_render_check=True)
    model = sb3.PPO(BetaExcitationPolicy, env, seed=0, n_steps=32, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}})
    model.learn(64)
    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    assert np.all((action >= 0) & (action <= 1))
    json.dumps(evaluate_policy(model, env.physical.config, ((0., .04),), seconds=.1), allow_nan=False)
    path = tmp_path/"policy.zip"
    save_policy(model, env, path, {"seed": 0})
    restored, _ = load_policy(path, env)
    np.testing.assert_array_equal(restored.predict(observation, deterministic=True)[0], action)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_policy(path, VerticalControlEnv(VerticalConfig(variant=3)))
