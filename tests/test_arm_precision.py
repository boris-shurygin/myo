"""Native arm transitions, delayed causality and fixed predictor ownership."""
import json

import numpy as np
import pytest
import torch

from myo.arm_precision import ArmPrecisionEnv
from myo.arm_learning import ArmBodyPredictor, make_model, save_arm, load_arm
from myo.env import PreacherCurlEnv
from myo.compact_protocol import initialize_command_at_reset
from myo.train_arm_precision import collect, load_data, starts, private_hash


def test_arm_observation_compression_preserves_native_dynamics_and_order():
    env = ArmPrecisionEnv()
    native = PreacherCurlEnv(env.unwrapped.config)
    obs, _ = env.reset(seed=0, options={'case': (45, 75)})
    native.reset(seed=0, options={'q': float(np.deg2rad(45)), 'target_angle': float(np.deg2rad(75)),
                                 'trigger': 1., 'capacity': 1., 'payload_mass': 1.})
    initialize_command_at_reset(native, 75, 1.)
    assert obs.shape == (153,) and len(env.physical_indices) == 20 and len(env.predictor_indices) == 27
    assert not any('target' in env.observation_names[i] or 'history' in env.observation_names[i]
                   for i in env.predictor_indices)
    for k in range(5):
        action = np.linspace(.01+.01*k, .08+.01*k, 6, dtype=np.float32)
        native.step(action)
        obs, _, failed, _, _ = env.step(action)
        assert not failed
        np.testing.assert_array_equal(env.unwrapped.data.qpos, native.data.qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, native.data.qvel)
        for left, right in zip(env.unwrapped.muscles, native.muscles, strict=True):
            assert left.activation == right.activation and left.fiber_length == right.fiber_length
            np.testing.assert_array_equal(left.delay.pending, right.delay.pending)
        raw = np.stack([m.delay.pending for m in native.muscles]).reshape(6, 2, 20)[:, :, 0].T
        np.testing.assert_allclose(obs[env.pending_indices].reshape(2, 6), 2*raw-1, atol=1e-7)
    q_before = env.unwrapped.data.qpos.copy()
    env.unwrapped.set_command(float(np.deg2rad(110)), 1.)
    env.observation()
    np.testing.assert_array_equal(env.unwrapped.data.qpos, q_before)
    env.close(); native.close()


def test_arm_motor_data_splits_whole_episodes_and_forecast_delay(tmp_path):
    torch.set_num_threads(1)
    specs = [{'split': split, 'family': 'independent', 'position_deg': 45+5*k,
              'capacity': 1., 'limit': .12, 'seed': 100+k}
             for k, split in enumerate(('train', 'validation', 'test'))]
    collect(tmp_path, specs=specs, steps=8)
    data, meta = load_data(tmp_path)
    for episode in meta['episodes']:
        a, b = episode['start'], episode['stop']
        np.testing.assert_array_equal(data['next_observations'][a:b-1], data['observations'][a+1:b])
        ids = starts(data, meta, episode['split'], 3)
        assert ids.min() == a and ids.max() == b-3
    body = ArmBodyPredictor()
    current = torch.randn(2, 20)
    queue = torch.randn(2, 2, 6)
    mass = torch.ones(2, 1)
    action = torch.full((2, 6), .1, requires_grad=True)
    for k in range(3):
        current = body.next_physical(torch.cat((current, queue[:, 0], mass), -1))
        # Keep graph connectivity to the queued command without physical leakage.
        derivative = torch.autograd.grad(current.sum()+0*action.sum(), action, retain_graph=True)[0]
        assert bool(derivative.abs().sum() > 0) == (k == 2)
        queue = torch.stack((queue[:, 1], 2*action-1 if k == 0 else torch.zeros_like(action)), 1)


def test_arm_fixed_features_and_policy_checkpoint_are_self_contained(tmp_path):
    torch.set_num_threads(1)
    plain_env, frozen_env = ArmPrecisionEnv(), ArmPrecisionEnv()
    plain = make_model(plain_env, 'none', n_steps=32, batch_size=16, n_epochs=1)
    frozen = make_model(frozen_env, 'frozen', n_steps=32, batch_size=16, n_epochs=1)
    assert private_hash(plain.policy) == private_hash(frozen.policy)
    body = ArmBodyPredictor()
    body.delta_calibrated.fill_(True)
    body.copy_into_policy(frozen.policy)
    before = {k: v.clone() for k, v in frozen.policy.state_dict().items()
              if k.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_'))}
    frozen.learn(32)
    assert frozen.optimizer_steps > 0
    for key, value in before.items():
        torch.testing.assert_close(value, frozen.policy.state_dict()[key], rtol=0, atol=0)
    path = tmp_path/'arm.zip'
    save_arm(frozen, frozen_env, path, {'trial': 'frozen', 'cases': ((45, 75),), 'predictor': {'hash': 'test'}})
    restored, checked, _ = load_arm(path)
    assert restored.optimizer_steps == frozen.optimizer_steps
    for candidate in (frozen, restored):
        candidate.policy.zero_grad(set_to_none=True)
        for parameter in candidate.policy.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, .001)
        candidate.policy.optimizer.step()
    assert private_hash(frozen.policy) == private_hash(restored.policy)
    sidecar = tmp_path/'arm.zip.metadata.json'
    meta = json.loads(sidecar.read_text()); meta['training']['predictor']['hash'] = 'changed'
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='embedded identity mismatch'):
        load_arm(path)
    checked.close(); plain_env.close(); frozen_env.close()
