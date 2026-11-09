"""A frozen feature branch must not break the warm parent's initial policy."""
import copy

import pytest
import torch

from myo.arm_anticipation import AnticipationPredictor, identity, save_predictor
from myo.arm_feature_comparison import (KINDS, architecture_identity, frozen_digest,
    load_comparison, make_comparison)
from myo.arm_learning import ArmBodyPredictor, make_model, save_arm
from myo.arm_precision import ArmPrecisionEnv
from myo.train_vertical_ppo import write_json


@pytest.mark.parametrize('kind', KINDS)
def test_warm_insertion_preserves_behavior_moments_and_frozen_checkpoint(tmp_path, kind):
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    parent = make_model(env, n_steps=32, batch_size=16, n_epochs=1)
    parent.learn(32)
    body_path = tmp_path/'body.pt'
    if kind == 'new60':
        body = AnticipationPredictor()
        body.calibrated.fill_(True)
        save_predictor(body_path, body, torch.optim.RMSprop(body.parameters()), 1,
                       identity(env.contract()), {})
    elif kind == 'old20':
        body = ArmBodyPredictor()
        body.delta_calibrated.fill_(True)
        torch.save({'state_dict': body.state_dict(), 'update': 1}, body_path)
        import hashlib
        write_json(str(body_path)+'.metadata.json', {'contract': env.contract(), 'update': 1,
            'checkpoint_sha256': hashlib.sha256(body_path.read_bytes()).hexdigest()})
    model, pred_meta = make_comparison(env, parent, kind, body_path,
                                     n_steps=32, batch_size=16, n_epochs=1)
    observation, _ = env.reset(seed=0, options={'case': (20, 75)})
    observation = torch.as_tensor(observation[None])
    with torch.no_grad():
        assert torch.equal(parent.policy.predict_values(observation), model.policy.predict_values(observation))
        assert torch.equal(parent.policy.get_distribution(observation).distribution.mean,
                           model.policy.get_distribution(observation).distribution.mean)
    parent_parameters = dict(parent.policy.named_parameters())
    for name, parameter in model.policy.named_parameters():
        if parameter.requires_grad and parameter in model.policy.optimizer.state:
            for field, value in model.policy.optimizer.state[parameter].items():
                assert torch.equal(value, parent.policy.optimizer.state[parent_parameters[name]][field])
    reference = frozen_digest(model.policy)
    private_before = copy.deepcopy(model.policy.mlp_extractor.actor_private.state_dict())
    model.learn(32, reset_num_timesteps=False)
    assert frozen_digest(model.policy) == reference
    assert any(not torch.equal(v, model.policy.mlp_extractor.actor_private.state_dict()[k])
               for k, v in private_before.items())
    training = {'trial': model.policy.feature_mode, 'predictor': pred_meta,
                'feature_comparison': architecture_identity(model.policy)}
    path = tmp_path/'policy.zip'
    save_arm(model, env, path, training)
    restored, checked, _ = load_comparison(path)
    assert frozen_digest(restored.policy) == reference
    assert all(torch.equal(v, restored.policy.state_dict()[k]) for k, v in model.policy.state_dict().items())
    checked.close()
    # A predictor silently replaced after training must fail the portable identity.
    with torch.no_grad():
        model.policy.mlp_extractor.encoder[0].weight.add_(.1)
    save_arm(model, env, tmp_path/'changed.zip', training)
    with pytest.raises(ValueError, match='comparison identity mismatch'):
        load_comparison(tmp_path/'changed.zip')
    env.close()
