"""Causal 60-ms labels, late action conditioning and incompatible checkpoint protection."""
import json

import numpy as np
import pytest
import torch

from myo.arm_anticipation import AnticipationPredictor, examples, identity, save_predictor, load_predictor
from myo.arm_precision import ArmPrecisionEnv


def test_sixty_ms_outcome_needs_current_action_but_not_next_issued_actions():
    env = ArmPrecisionEnv()
    physical = env.physical_indices
    trajectories = []
    # Change the current command, then independently change future commands.
    for first, future in ((.05, .05), (.7, .05), (.7, .9)):
        env.reset(seed=0, options={'case': (60, 75)})
        sequence = []
        for action in (first, future, future):
            obs, _, failed, _, _ = env.step(np.full(6, action, dtype=np.float32))
            assert not failed
            sequence.append(obs[physical])
        trajectories.append(np.array(sequence))
    np.testing.assert_array_equal(trajectories[0][:2], trajectories[1][:2])
    assert np.max(abs(trajectories[0][2]-trajectories[1][2])) > .01
    np.testing.assert_array_equal(trajectories[1], trajectories[2])
    env.close()


def test_direct_windows_stop_at_episode_boundary_and_do_not_read_future_actions():
    env = ArmPrecisionEnv()
    obs, _ = env.reset(seed=0, options={'case': (60, 75)})
    rows = {'observations': [], 'actions': [], 'next_observations': [], 'valid': []}
    for i in range(7):
        action = np.full(6, .02*i, dtype=np.float32)
        following, _, _, _, _ = env.step(action)
        for key, value in zip(rows, (obs, action, following, True), strict=True):
            rows[key].append(value)
        obs = following
    data = {k: np.array(v) for k, v in rows.items()}
    data['valid'][5] = False
    meta = {'contract': env.contract(), 'episodes': [
        {'start': 0, 'stop': 4, 'split': 'train'}, {'start': 4, 'stop': 7, 'split': 'test'}]}
    ids, x, commands, actual = examples(data, meta, 'train')
    np.testing.assert_array_equal(ids, [0, 1])
    assert x.shape == (2, 33)
    np.testing.assert_array_equal(actual, data['next_observations'][ids+2][:, env.physical_indices])
    copied = {k: v.copy() for k, v in data.items()}
    copied['actions'][2:] = 1.
    _, other_x, other_commands, _ = examples(copied, meta, 'train')
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(commands, other_commands)
    with pytest.raises(ValueError, match='No complete'):
        examples(data, meta, 'test')
    env.close()


def test_action_enters_head_and_does_not_change_transferable_encoder():
    torch.manual_seed(0)
    model = AnticipationPredictor()
    x = torch.randn(3, 33, requires_grad=True)
    action = torch.full((3, 6), .1, requires_grad=True)
    latent = model.encoder(x)
    changed = model.normalized_delta(x, action)
    torch.testing.assert_close(latent, model.encoder(x), rtol=0, atol=0)
    derivative = torch.autograd.grad(changed.square().sum(), action)[0]
    assert derivative.abs().sum() > 0
    assert not torch.equal(changed, model.normalized_delta(x, 1-action))


def test_direct_checkpoint_preserves_normalization_and_rejects_old_horizon(tmp_path):
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    model = AnticipationPredictor()
    model.calibrated.fill_(True)
    model.delta_std.fill_(.01)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    x, u = torch.randn(4, 33), torch.rand(4, 6)
    model.normalized_delta(x, u).square().mean().backward(); opt.step()
    path = tmp_path/'best.pt'
    save_predictor(path, model, opt, 1, identity(env.contract()), {'seed': 0})
    restored, _ = load_predictor(path)
    torch.testing.assert_close(model.next_physical(x, u), restored.next_physical(x, u), rtol=0, atol=0)
    metadata = path.with_suffix('.pt.metadata.json')
    spec = json.loads(metadata.read_text()); spec['identity']['horizon_steps'] = 1
    metadata.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='identity mismatch'):
        load_predictor(path)
    env.close()
