"""Separate gradient ownership, delayed causality and both optimizer states."""
import copy

import numpy as np
import pytest
import torch

from myo.detached_vertical import DetachedVerticalEnv, make_model, load_detached
from myo.shared_vertical import load_shared
from myo.train_shared_vertical import PredictionLabels
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import save_policy


def test_gradient_ownership_and_causal_prediction():
    torch.set_num_threads(1)
    frozen = make_model(DetachedVerticalEnv(trial='frozen'), 'frozen', n_steps=32, batch_size=16, n_epochs=1)
    online = make_model(DetachedVerticalEnv(), 'online', n_steps=32, batch_size=16, n_epochs=1)
    assert parameter_hash(frozen) == parameter_hash(online)
    policy = online.policy
    predictor_ids = {id(p) for p in policy.prediction_parameters()}
    assert not predictor_ids & {id(p) for g in policy.optimizer.param_groups for p in g['params']}
    with torch.no_grad():
        policy.prediction_net.weight.fill_(.01)
    obs = torch.randn(4, 13)
    for branch in ('actor', 'critic', 'predictor'):
        policy.zero_grad(set_to_none=True)
        loss = (policy.get_distribution(obs).distribution.mean.square().mean() if branch == 'actor' else
                policy.predict_values(obs).square().mean() if branch == 'critic' else
                policy.normalized_delta(obs).square().mean())
        loss.backward()
        active = {id(p) for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert active
        assert active <= predictor_ids if branch == 'predictor' else not active & predictor_ids
    action = torch.full((4, 2), .3, requires_grad=True)
    current = obs
    for step in range(3):
        current = policy.next_observation(current, action if step == 0 else torch.full_like(action, .2))
        derivative = torch.autograd.grad(current[:, :6].sum(), action, retain_graph=True)[0]
        assert derivative.abs().sum() > 0 if step == 2 else torch.count_nonzero(derivative) == 0
    changed = obs.clone(); changed[:, 8:] += 3
    torch.testing.assert_close(policy.normalized_delta(obs), policy.normalized_delta(changed))


def test_phase_updates_preserve_other_owner_and_measure_policy_drift(monkeypatch):
    torch.set_num_threads(1)
    env = DetachedVerticalEnv()
    model = make_model(env, 'online', n_steps=32, batch_size=16, n_epochs=1)
    original = model.update_predictor
    checked = []

    def inspect(observations):
        ids = {id(p) for p in model.policy.prediction_parameters()}
        private = [p for p in model.policy.parameters() if id(p) not in ids]
        before_private = [p.detach().clone() for p in private]
        before_predictor = [p.detach().clone() for p in model.policy.prediction_parameters()]
        result = original(observations)
        for before, after in zip(before_private, private, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert any(not torch.equal(before, after) for before, after in zip(before_predictor, model.policy.prediction_parameters(), strict=True))
        assert result['predictor_phase_kl'] > 0
        checked.append(result)
        return result

    monkeypatch.setattr(model, 'update_predictor', inspect)
    model.learn(64, callback=PredictionLabels())
    assert len(checked) == 2
    assert model.accepted_steps > 0 and model.predictor_steps == 20
    assert all(p.grad is None for p in model.policy.parameters())
    scales = model.policy.delta_std.clone()
    model.learn(32, reset_num_timesteps=False, callback=PredictionLabels())
    torch.testing.assert_close(scales, model.policy.delta_std, rtol=0, atol=0)
    assert model.last_audit['max_accepted_kl'] <= .03
    env.close()


def test_frozen_features_and_both_optimizers_survive_checkpoint(tmp_path):
    torch.set_num_threads(1)
    frozen = make_model(DetachedVerticalEnv(trial='frozen'), 'frozen', n_steps=32, batch_size=16, n_epochs=1)
    before = copy.deepcopy(frozen.policy.mlp_extractor.predictor.state_dict())
    frozen.learn(32, callback=PredictionLabels())
    for name, value in frozen.policy.mlp_extractor.predictor.state_dict().items():
        torch.testing.assert_close(before[name], value, rtol=0, atol=0)
    assert frozen.predictor_steps == 0 and frozen.last_audit['predictor_phase_kl'] == 0
    env = DetachedVerticalEnv()
    model = make_model(env, 'online', n_steps=32, batch_size=16, n_epochs=1)
    model.learn(64, callback=PredictionLabels())
    path = tmp_path/'model.zip'
    save_policy(model, env, path, {'trial': 'online'})
    restored, checked_env, _ = load_detached(path)
    assert parameter_hash(restored) == parameter_hash(model)
    assert restored.predictor_steps == model.predictor_steps
    np.testing.assert_array_equal(restored.predictor_rng.permutation(16), model.predictor_rng.permutation(16))
    for candidate in (model, restored):
        candidate.policy.zero_grad(set_to_none=True)
        for parameter in candidate.policy.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        candidate.policy.optimizer.step()
        candidate.policy.predictor_optimizer.step()
    assert parameter_hash(restored) == parameter_hash(model)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_shared(path)
    checked_env.close(); env.close()
