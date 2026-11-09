"""Changing the continuation LR must preserve moments and a loadable feature identity."""
import copy

import pytest
import torch

from myo.arm_conditioning import make_conditioned_model, conditioning_identity
from myo.arm_learning import ArmBodyPredictor, make_model, save_arm
from myo.arm_precision import ArmPrecisionEnv
from scripts.compare_arm_predictor_lr import load_checkpoint, override_learning_rate


@pytest.mark.parametrize('frozen', (False, True))
def test_lr_override_keeps_moments_schedule_and_checkpoint_identity(tmp_path, frozen):
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    if frozen:
        model = make_conditioned_model(env, 1e-4, .25, n_steps=32, batch_size=16, n_epochs=1)
        body = ArmBodyPredictor()
        body.delta_calibrated.fill_(True)
        body.copy_into_policy(model.policy)
    else:
        model = make_model(env, n_steps=32, batch_size=16, n_epochs=1)
    model.learn(32)
    training = {'trial': 'frozen' if frozen else 'none', 'ppo': {'learning_rate': model.learning_rate}}
    if frozen:
        training['conditioning'] = conditioning_identity(model.learning_rate, .25)
    original_training = copy.deepcopy(training)
    weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
    moments = copy.deepcopy(model.policy.optimizer.state_dict()['state'])
    training = override_learning_rate(model, training, 3e-5)
    assert model.lr_schedule(.25) == model.learning_rate == 3e-5
    assert all(g['lr'] == 3e-5 for g in model.policy.optimizer.param_groups)
    assert all(torch.equal(v, model.policy.state_dict()[k]) for k, v in weights.items())
    assert original_training['ppo']['learning_rate'] != training['ppo']['learning_rate']
    path = tmp_path/'continued.zip'
    save_arm(model, env, path, training)
    restored, checked, _ = load_checkpoint(path)
    assert restored.learning_rate == restored.lr_schedule(.5) == 3e-5
    assert restored.optimizer_steps == model.optimizer_steps
    for index, values in restored.policy.optimizer.state_dict()['state'].items():
        for name, value in values.items():
            assert torch.equal(value, moments[index][name])
    restored.set_env(checked)
    restored.learn(32, reset_num_timesteps=False)
    assert all(g['lr'] == 3e-5 for g in restored.policy.optimizer.param_groups)
    checked.close()
    env.close()
