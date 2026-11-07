"""A one-factor sweep must really change PPO settings and isolate reward scales."""
import copy

import numpy as np
import pytest

from myo.vertical_control import REWARD, VerticalControlEnv, save_policy
from myo.vertical_muscles import VerticalConfig
from myo.vertical_sweep_env import VerticalSweepEnv, load_sweep_policy


def test_reward_scale_does_not_change_physics_or_leak_between_branches(tmp_path):
    ordinary = VerticalControlEnv(VerticalConfig(variant=4))
    narrow = VerticalSweepEnv(VerticalConfig(variant=4), .02)
    options = {"initial_position_m": 0., "target_m": .04}
    ordinary.reset(options=options)
    narrow.reset(options=options)
    action = np.array([.3, .1])
    obs, base_reward, _, _, info = ordinary.step(action)
    other_obs, new_reward, _, _, other_info = narrow.step(action)
    np.testing.assert_array_equal(obs, other_obs)
    np.testing.assert_array_equal(ordinary.physical.state(), narrow.physical.state())
    expected = .1*.85*(np.exp(-(info["error_m"]/.02)**2)-np.exp(-(info["error_m"]/.08)**2))
    assert new_reward-base_reward == pytest.approx(expected)
    assert REWARD["position_scale_m"] == .08
    assert ordinary.contract()["reward"]["position_scale_m"] == .08
    assert narrow.contract()["reward"]["position_scale_m"] == .02


def test_configuration_preserves_parent_weights_and_adam_but_updates_buffer_and_schedules(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from myo.policy_beta import BetaExcitationPolicy
    from myo.train_vertical_sweep import actual_parameters, configure_model, optimizer_steps
    torch.set_num_threads(1)
    model = sb3.PPO(BetaExcitationPolicy, VerticalControlEnv(VerticalConfig(variant=2)),
                    seed=0, n_steps=32, batch_size=16, n_epochs=1, target_kl=None,
                    policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.learn(64)
    weights = copy.deepcopy(model.policy.state_dict())
    moment = model.policy.optimizer.state[model.policy.action_net.weight]["exp_avg"].clone()
    steps = optimizer_steps(model)
    env = VerticalSweepEnv(VerticalConfig(variant=2), .04)
    changes = {"learning_rate": 1e-4, "n_steps": 64, "gamma": .995,
               "gae_lambda": .9, "clip_range": .1, "n_epochs": 2}
    configure_model(model, env, changes)
    for name, value in weights.items():
        torch.testing.assert_close(value, model.policy.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(moment, model.policy.optimizer.state[model.policy.action_net.weight]["exp_avg"], rtol=0, atol=0)
    assert optimizer_steps(model) == steps
    assert model.rollout_buffer.buffer_size == 64
    assert model.rollout_buffer.gamma == .995 and model.rollout_buffer.gae_lambda == .9
    assert model.lr_schedule(.5) == 1e-4 and model.clip_range(.5) == .1
    model.learn(128, reset_num_timesteps=False)
    assert optimizer_steps(model)-steps == 16
    assert model.policy.optimizer.param_groups[0]["lr"] == 1e-4
    path = tmp_path/"trial.zip"
    save_policy(model, env, path, {"changes": changes})
    restored, restored_env, _ = load_sweep_policy(path)
    assert actual_parameters(restored, restored_env) == actual_parameters(model, env)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])


def test_every_candidate_changes_only_its_named_factor():
    pytest.importorskip("torch")
    from myo.train_vertical_sweep import candidates
    rows = candidates()
    assert rows[0] == ("control", {})
    assert len({name for name, _ in rows}) == len(rows)
    assert all(len(changes) == 1 for _, changes in rows[1:])
    assert {key for _, changes in rows for key in changes} == {
        "learning_rate", "n_steps", "ent_coef", "gae_lambda", "gamma", "batch_size",
        "n_epochs", "clip_range", "target_kl", "position_scale_m"}
