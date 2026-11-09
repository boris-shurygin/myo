"""Regression: extra post-concat layers must train, persist and keep their identity."""
import pytest
import torch

from myo.arm_conditioning import make_conditioned_model, conditioning_identity
from myo.arm_depth import make_deep_model, depth_identity, load_deep_arm
from myo.arm_learning import ArmBodyPredictor, save_arm
from myo.arm_precision import ArmPrecisionEnv


def test_extra_layers_preserve_common_initialization_and_train_after_reload(tmp_path):
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    baseline = make_conditioned_model(env, 3e-4, .25, n_steps=32, batch_size=16, n_epochs=1)
    rng = torch.get_rng_state().clone()
    model = make_deep_model(env, n_steps=32, batch_size=16, n_epochs=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.policy.state_dict()[name])
               for name, value in baseline.policy.state_dict().items())
    assert sum(p.numel() for p in model.policy.parameters())-sum(p.numel() for p in baseline.policy.parameters()) == 66048
    body = ArmBodyPredictor()
    body.delta_calibrated.fill_(True)
    body.copy_into_policy(model.policy)
    extra = {name: p.clone() for name, p in model.policy.named_parameters()
             if name.startswith(tuple(f'mlp_extractor.{tail}.{i}.' for tail in ('policy_net', 'value_net') for i in (4, 6)))}
    assert len(extra) == 8
    optimizer_ids = {id(p) for group in model.policy.optimizer.param_groups for p in group['params']}
    assert all(id(p) in optimizer_ids for name, p in model.policy.named_parameters() if name in extra)
    frozen = {k: v.clone() for k, v in model.policy.state_dict().items()
              if k.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_'))}
    model.learn(32)
    for tail in ('policy_net', 'value_net'):
        assert any(not torch.equal(value, model.policy.state_dict()[name]) for name, value in extra.items() if tail in name)
    assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in frozen.items())
    training = {'trial': 'frozen', 'conditioning': conditioning_identity(3e-4, .25), 'depth': depth_identity()}
    path = tmp_path/'deep.zip'
    save_arm(model, env, path, training)
    restored, checked, _ = load_deep_arm(path)
    obs, _ = env.reset(options={'case': (45, 75)})
    torch.testing.assert_close(torch.from_numpy(model.predict(obs, deterministic=True)[0]),
                               torch.from_numpy(restored.predict(obs, deterministic=True)[0]), rtol=0, atol=0)
    for candidate in (model, restored):
        candidate.policy.zero_grad(set_to_none=True)
        for p in candidate.policy.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, .001)
        candidate.policy.optimizer.step()
    assert all(torch.equal(p, restored.policy.state_dict()[name]) for name, p in model.policy.state_dict().items())
    wrong = {**training, 'depth': {**training['depth'], 'actor_post_layers': [128]*2}}
    save_arm(restored, env, tmp_path/'wrong.zip', wrong)
    with pytest.raises(ValueError, match='depth identity mismatch'):
        load_deep_arm(tmp_path/'wrong.zip')
    checked.close()
    env.close()
