"""Continuation must reproduce the next update rather than restart optimizer/sample history."""
import copy

import numpy as np
import torch

from myo.arm_anticipation import AnticipationPredictor, identity, save_predictor
from myo.arm_precision import ArmPrecisionEnv
from scripts.continue_arm_anticipation import restore_training


def test_predictor_resume_reproduces_optimizer_samples_and_next_update(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(0)
    body = AnticipationPredictor(); body.calibrated.fill_(True)
    optimizer = torch.optim.RMSprop(body.parameters(), lr=1e-4, eps=1e-5, alpha=.99)
    rng = np.random.default_rng(0)
    x = torch.linspace(-1., 1., 33*256).reshape(256, 33)
    u = torch.linspace(0., 1., 6*256).reshape(256, 6)
    target = torch.sin(x[:, :20])

    def update(model, opt, random):
        draws = random.random(128)
        ids = (draws*256).astype(int)
        opt.zero_grad(set_to_none=True)
        (model.normalized_delta(x[ids], u[ids])-target[ids]).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5); opt.step()
        return draws

    for _ in range(7):
        update(body, optimizer, rng)
    torch_rng = torch.get_rng_state().clone()
    env = ArmPrecisionEnv(); path = tmp_path/'last.pt'
    save_predictor(path, body, optimizer, 7, identity(env.contract()),
                   {'seed': 0, 'learning_rate': 1e-4, 'batch_size': 128})
    env.close()
    expected_moments = copy.deepcopy(optimizer.state_dict())
    restored, opt, random, _ = restore_training(path)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in body.state_dict().items())
    assert opt.state_dict()['param_groups'] == expected_moments['param_groups']
    for index, values in opt.state_dict()['state'].items():
        for name, value in values.items():
            assert torch.equal(value, expected_moments['state'][index][name])
    np.testing.assert_array_equal(update(body, optimizer, rng), update(restored, opt, random))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in body.state_dict().items())
