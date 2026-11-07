"""Shared gradients, causal transition labels and transactional PPO updates."""
import copy

import numpy as np
import pytest
import torch
from torch.distributions import Beta
from stable_baselines3.common.vec_env import DummyVecEnv

from myo.shared_vertical import SharedPolicy, SharedVerticalEnv, make_model, load_shared
from myo.train_shared_vertical import PredictionLabels
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import save_policy, load_policy
from myo.vertical_precision import VerticalPrecisionEnv
from myo.vertical_muscles import VerticalConfig


def test_all_three_heads_reach_shared_encoder_but_new_command_respects_delay():
    torch.set_num_threads(1)
    env = SharedVerticalEnv()
    first = make_model(env, 'shared', n_steps=32, batch_size=16, n_epochs=1)
    second = make_model(env, 'shared_prediction', n_steps=32, batch_size=16, n_epochs=1)
    assert parameter_hash(first) == parameter_hash(second)
    policy = second.policy
    with torch.no_grad():
        policy.prediction_net[-1].weight.fill_(.01)
    obs = torch.randn(8, 13)
    for kind in ('actor', 'critic', 'predictor'):
        policy.zero_grad()
        loss = (policy.get_distribution(obs).distribution.mean.square().mean() if kind == 'actor' else
                policy.predict_values(obs).square().mean() if kind == 'critic' else
                policy.normalized_delta(obs).square().mean())
        loss.backward()
        assert sum(float(p.grad.abs().sum()) for p in policy.mlp_extractor.encoder.parameters() if p.grad is not None) > 0
    action = torch.full((8, 2), .3, requires_grad=True)
    obs = obs.detach().requires_grad_(True)
    next_obs = policy.next_observation(obs, action)
    torch.testing.assert_close(next_obs[:, 6:8], obs[:, 8:10])
    torch.testing.assert_close(next_obs[:, 8:10], 2*action-1)
    derivative = torch.autograd.grad(next_obs[:, :6].sum(), action, allow_unused=True, retain_graph=True)[0]
    assert derivative is None or torch.count_nonzero(derivative) == 0
    # The third interval can depend on the currently issued command after exact FIFO shifts.
    second = policy.next_observation(next_obs, torch.full((8, 2), .2))
    derivative = torch.autograd.grad(second[:, :6].sum(), action, retain_graph=True)[0]
    assert torch.count_nonzero(derivative) == 0
    third = policy.next_observation(second, torch.full((8, 2), .2))
    assert torch.autograd.grad(third[:, :6].sum(), action)[0].abs().sum() > 0
    changed_goal = obs.detach().clone(); changed_goal[:, 12] += 2
    torch.testing.assert_close(policy.normalized_delta(obs), policy.normalized_delta(changed_goal))


def test_transition_labels_use_terminal_observation_and_normalizer_freezes():
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: SharedVerticalEnv(episode_seconds=.04)])
    model = make_model(env, 'shared_prediction', n_steps=32, batch_size=16, n_epochs=1)

    class Inspect(PredictionLabels):
        differences = 0
        def _on_step(self):
            super()._on_step()
            info = self.locals['infos'][0]
            if 'terminal_observation' in info:
                actual = self.model.rollout_buffer.next_observations[self.model.rollout_buffer.pos, 0]
                np.testing.assert_array_equal(actual, info['terminal_observation'])
                self.differences += int(not np.array_equal(actual, self.locals['new_obs'][0]))
            return True

    callback = Inspect()
    model.learn(64, callback=callback)
    assert callback.differences > 0
    assert bool(model.policy.delta_calibrated)
    scales = model.policy.delta_std.clone()
    model.learn(32, reset_num_timesteps=False, callback=Inspect())
    torch.testing.assert_close(scales, model.policy.delta_std, rtol=0, atol=0)
    assert model.last_audit['max_accepted_kl'] <= .03


def test_rejected_update_restores_weights_and_optimizer_and_checkpoint_continues(tmp_path):
    torch.set_num_threads(1)
    env = SharedVerticalEnv()
    model = make_model(env, 'shared_prediction', n_steps=32, batch_size=16, n_epochs=1)
    model.learn(64, callback=PredictionLabels())
    assert model.accepted_steps > 0
    obs = torch.as_tensor(np.stack([env.reset(options={'initial_position_m': 0., 'target_m': q})[0] for q in (0., .06)]))
    with torch.no_grad():
        old = model.policy.get_distribution(obs).distribution
        old = Beta(old.concentration1.clone(), old.concentration0.clone())
    before = parameter_hash(model)
    state = copy.deepcopy(model.policy.optimizer.state_dict())
    model.target_kl = 1e-12
    accepted, kl, _ = model.guarded_step(model.policy.action_net.bias.sum(), obs, old)
    assert not accepted and kl > model.target_kl
    assert parameter_hash(model) == before
    after = model.policy.optimizer.state_dict()
    assert state['param_groups'] == after['param_groups']
    for key, values in state['state'].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, after['state'][key][name], rtol=0, atol=0)
            else:
                assert value == after['state'][key][name]
    model.target_kl = .03
    path = tmp_path/'model.zip'
    save_policy(model, env, path, {'trial': 'shared_prediction'})
    restored, _, _ = load_shared(path)
    assert isinstance(restored.policy, SharedPolicy)
    assert parameter_hash(restored) == parameter_hash(model)
    for candidate in (model, restored):
        candidate.policy.optimizer.zero_grad()
        for parameter in candidate.policy.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        candidate.policy.optimizer.step()
    assert parameter_hash(restored) == parameter_hash(model)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_policy(path, VerticalPrecisionEnv(VerticalConfig(variant=4)))
