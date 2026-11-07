"""Whole-episode labels, unchanged predictor transfer and delayed-command causality."""
import numpy as np
import pytest
import torch

from myo.body_predictor import (BodyPredictor, collect, load_dataset, indices, motor_commands,
    save_predictor, load_predictor)
from myo.detached_vertical import DetachedVerticalEnv, make_model


def test_smooth_motor_data_keeps_episode_boundaries_and_native_labels(tmp_path):
    torch.set_num_threads(1)
    specs = [{"id": f'{split}/independent/{k}', "split": split, "family": "independent",
              "position_m": position, "limit": .5, "seed": 10+k}
             for k, (split, position) in enumerate((('train', -.08), ('train', .06), ('validation', 0.), ('test', .03)))]
    collect(tmp_path, seconds=.32, specs=specs)
    data, metadata = load_dataset(tmp_path)
    for episode in metadata['episodes']:
        start, stop = episode['start'], episode['stop']
        assert not episode['physical_failure'] and stop-start == 16
        np.testing.assert_array_equal(data['next_observations'][start:stop-1], data['observations'][start+1:stop])
        assert np.all((indices(metadata, episode['split'], 3) < start) |
                      (indices(metadata, episode['split'], 3) <= stop-3) |
                      (indices(metadata, episode['split'], 3) >= stop))
    env = DetachedVerticalEnv()
    obs, _ = env.reset(options={'initial_position_m': -.08, 'target_m': 0.})
    for k in range(16):
        np.testing.assert_array_equal(obs, data['observations'][k])
        obs, *_ = env.step(data['actions'][k])
        np.testing.assert_array_equal(obs, data['next_observations'][k])
    env.close()
    command = motor_commands('independent', 600, 17, .8, [.12, .04])
    np.testing.assert_array_equal(command, motor_commands('independent', 600, 17, .8, [.12, .04]))
    assert command.min() >= 0 and command.max() <= 1
    assert np.abs(np.diff(command, axis=0)).max() < .13
    assert np.abs(np.corrcoef(command.T)[0, 1]) < .8


def test_train_only_scaling_export_and_delayed_gradients():
    torch.set_num_threads(1)
    model = BodyPredictor.from_current_initialization()
    env = DetachedVerticalEnv()
    policy = make_model(env, 'online').policy
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, policy.mlp_extractor.predictor.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 51462
    observations = torch.randn(16, 13)
    labels = observations+.02*torch.randn_like(observations)
    model.fit_normalization(observations, labels)
    torch.testing.assert_close(model.delta_std, (labels[:, :6]-observations[:, :6]).std(0, unbiased=False))
    with pytest.raises(RuntimeError, match='already frozen'):
        model.fit_normalization(observations, labels+100.)
    with torch.no_grad():
        model.head.weight.fill_(.01)
    before = {key: value.clone() for key, value in policy.state_dict().items()
              if not key.startswith(('mlp_extractor.predictor.', 'prediction_net.', 'delta_'))}
    model.copy_into_policy(policy)
    for key, value in before.items():
        torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(model.normalized_delta(observations), policy.normalized_delta(observations), rtol=0, atol=0)
    action = torch.full((16, 2), .2, requires_grad=True)
    current = observations
    for step in range(3):
        current = model.next_observation(current, action if step == 0 else torch.full_like(action, .3))
        derivative = torch.autograd.grad(current[:, :6].sum(), action, retain_graph=True)[0]
        assert derivative.abs().sum() > 0 if step == 2 else torch.count_nonzero(derivative) == 0
    env.close()


def test_standalone_weights_and_rmsprop_continue_exactly(tmp_path):
    torch.set_num_threads(1)
    model = BodyPredictor.from_current_initialization()
    obs, labels = torch.randn(16, 13), torch.randn(16, 13)
    model.fit_normalization(obs, labels)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-4, eps=1e-5)
    loss = model.normalized_delta(obs).square().mean()+model.head.bias.sum()
    loss.backward(); optimizer.step()
    path = tmp_path/'predictor.pt'
    save_predictor(path, model, optimizer, 1, {'dataset_sha256': 'test'}, {'mode': 'unit test'})
    restored, state, metadata = load_predictor(path)
    assert metadata['update'] == 1
    resumed_optimizer = torch.optim.RMSprop(restored.parameters(), lr=3e-4, eps=1e-5)
    resumed_optimizer.load_state_dict(state)
    for candidate, opt in ((model, optimizer), (restored, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        for parameter in candidate.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        opt.step()
    for left, right in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_predictor(path)
