"""Protect the experimental feature gain from changing or disappearing on load."""
import pytest
import torch

from myo.arm_conditioning import make_conditioned_model, conditioning_identity, load_conditioned_arm
from myo.arm_learning import ArmBodyPredictor, make_model, save_arm
from myo.arm_precision import ArmPrecisionEnv
from myo.train_arm_precision import private_hash


def test_fixed_feature_gain_preserves_initialization_predictor_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    env = ArmPrecisionEnv()
    original = make_model(env, 'frozen', n_steps=32, batch_size=16, n_epochs=1)
    unit = make_conditioned_model(env, 1e-4, 1., n_steps=32, batch_size=16, n_epochs=1)
    scaled = make_conditioned_model(env, 3e-4, .25, n_steps=32, batch_size=16, n_epochs=1)
    assert private_hash(original.policy) == private_hash(unit.policy) == private_hash(scaled.policy)
    # Synthetic calibrated predictor keeps the regression independent of ignored runs/.
    body = ArmBodyPredictor()
    body.delta_calibrated.fill_(True)
    for model in (original, unit, scaled):
        body.copy_into_policy(model.policy)
    obs, _ = env.reset(options={'case': (45, 75)})
    x = torch.tensor(obs[None])
    with torch.no_grad():
        torch.testing.assert_close(original.policy.mlp_extractor.features(x),
                                   unit.policy.mlp_extractor.features(x), rtol=0, atol=0)
        torch.testing.assert_close(original.policy.mlp_extractor.features(x)*.25,
                                   scaled.policy.mlp_extractor.features(x), rtol=0, atol=0)
        torch.testing.assert_close(original.policy.get_distribution(x).distribution.mean,
                                   unit.policy.get_distribution(x).distribution.mean, rtol=0, atol=0)
    before = {k: v.clone() for k, v in scaled.policy.state_dict().items()
              if k.startswith(('mlp_extractor.encoder.', 'prediction_head.', 'delta_'))}
    scaled.learn(32)
    assert all(torch.equal(v, scaled.policy.state_dict()[k]) for k, v in before.items())
    training = {'trial': 'frozen', 'conditioning': conditioning_identity(3e-4, .25)}
    path = tmp_path/'scaled.zip'
    save_arm(scaled, env, path, training)
    restored, checked, _ = load_conditioned_arm(path)
    torch.testing.assert_close(torch.from_numpy(scaled.predict(obs, deterministic=True)[0]),
                               torch.from_numpy(restored.predict(obs, deterministic=True)[0]), rtol=0, atol=0)
    checked.close()
    # A gain that disagrees with its declared identity must not run silently.
    restored.policy.mlp_extractor.feature_gain.fill_(1.)
    save_arm(restored, env, tmp_path/'wrong.zip', training)
    with pytest.raises(ValueError, match='conditioning identity mismatch'):
        load_conditioned_arm(tmp_path/'wrong.zip')
    env.close()
