"""Stage continuation must keep learned parameters/moments and change the real LR."""
from dataclasses import replace

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from myo.arm_curriculum import curriculum, passed_gate, set_learning_rate, update_progress
from myo.arm_timing import ArmTimingEnv, load_timing, make_fresh_model
from myo.arm_workload import save_workload


def test_stage_lr_continuation_keeps_weights_and_optimizer_moments(tmp_path):
    torch.set_num_threads(1)
    stage = curriculum()[0]
    spec = replace(stage.spec, seconds=.1)
    env = ArmTimingEnv(spec)
    vector = DummyVecEnv([lambda: env])
    second = replay = None
    try:
        model = make_fresh_model(vector, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
        set_learning_rate(model, 3e-4)
        model.learn(32)
        weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
        moments = {key: value['square_avg'].clone() for key, value in model.policy.optimizer.state.items()}
        second = DummyVecEnv([lambda: ArmTimingEnv(replace(spec, masses=(1.,)))])
        model.set_env(second, force_reset=True)
        set_learning_rate(model, 1e-4)
        assert all(torch.equal(weights[key], value) for key, value in model.policy.state_dict().items())
        assert all(torch.equal(moments[key], value['square_avg']) for key, value in model.policy.optimizer.state.items())
        path = tmp_path/'stage.zip'
        save_workload(model, second.envs[0], path, {'stage': 1, 'lr': 1e-4})
        restored, replay, _ = load_timing(path)
        assert restored.lr_schedule(.1) == restored.learning_rate == 1e-4
        assert all(group['lr'] == 1e-4 for group in restored.policy.optimizer.param_groups)
        assert restored.optimizer_steps == model.optimizer_steps
        restored.set_env(second)
        restored.learn(16, reset_num_timesteps=False)
        assert restored.optimizer_steps > model.optimizer_steps
        assert all(group['lr'] == 1e-4 for group in restored.policy.optimizer.param_groups)
        assert not np.array_equal(restored.policy.action_net.weight.detach().numpy(),
                                  model.policy.action_net.weight.detach().numpy())
    finally:
        vector.close()
        if second is not None:
            second.close()
        if replay is not None:
            replay.close()


def test_curriculum_gate_requires_real_accuracy_and_no_physical_failures():
    assert not passed_gate({'passed': 10, 'total': 10, 'failures': 1, 'score_deg': .2})
    assert not passed_gate({'passed': 9, 'total': 10, 'failures': 0, 'score_deg': .2}, strict=True)
    assert passed_gate({'passed': 9, 'total': 10, 'failures': 0, 'score_deg': 2.})


def test_successful_validation_does_not_hide_improving_incomplete_training_gate():
    validation = {'passed': 9, 'total': 9, 'failures': 0, 'score_deg': .5}
    train = {'passed': 20, 'total': 27, 'failures': 0, 'score_deg': 4.}
    reference, train_reference, stale = update_progress(validation, train, None, None, 0)
    _, _, stale = update_progress(validation, {**train, 'passed': 24, 'score_deg': 2.},
                                  reference, train_reference, 3)
    assert stale == 0
    _, _, stale = update_progress(validation, train, reference, train_reference, 3)
    assert stale == 4
