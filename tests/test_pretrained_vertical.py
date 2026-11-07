"""Transfer preserves private initialization, frozen normalization and runtime identity."""
import json

import pytest
import torch

from myo.body_predictor import BodyPredictor, save_predictor, load_predictor
from myo.pretrained_vertical import PretrainedVerticalEnv, make_model, load_pretrained, predictor_identity, TRIALS
from myo.train_pretrained_vertical import private_hash
from myo.train_shared_vertical import PredictionLabels
from myo.train_vertical_architecture import parameter_hash
from myo.vertical_control import save_policy


def test_transfer_freezing_adaptation_and_self_contained_checkpoint(tmp_path):
    torch.set_num_threads(1)
    body = BodyPredictor.from_current_initialization()
    x = torch.randn(32, 13)
    body.fit_normalization(x, x+.02*torch.randn_like(x))
    with torch.no_grad():
        body.head.weight.fill_(.01)
        next(body.encoder.parameters()).add_(.05)
    artifact = tmp_path/'body.pt'
    optimizer = torch.optim.RMSprop(body.parameters(), lr=3e-4, eps=1e-5)
    save_predictor(artifact, body, optimizer, 1, {'dataset_sha256': 'a'*64}, {'test': True})
    body, _, metadata = load_predictor(artifact)
    identity = predictor_identity(metadata)
    models = []
    for trial in TRIALS:
        env = PretrainedVerticalEnv(identity, trial)
        model = make_model(env, trial, body, identity, n_steps=32, batch_size=16, n_epochs=1)
        models.append((model, env))
    assert len({private_hash(m) for m, _ in models}) == 1
    assert parameter_hash(models[1][0]) == parameter_hash(models[2][0])
    for model, env in models:
        before = [p.detach().clone() for p in model.policy.prediction_parameters()]
        model.learn(32, callback=PredictionLabels())
        torch.testing.assert_close(model.policy.delta_std, body.delta_std, rtol=0, atol=0)
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.policy.prediction_parameters(), strict=True))
        assert changed == (env.transfer_trial == 'pretrained_online')
    original, env = models[2]
    path = tmp_path/'policy.zip'
    save_policy(original, env, path, {'trial': env.transfer_trial})
    artifact.unlink()  # Policy runtime must not need the external pretraining file.
    loaded, checked_env, _ = load_pretrained(path)
    assert parameter_hash(original) == parameter_hash(loaded)
    assert loaded.predictor_steps == original.predictor_steps == 10
    # Both independent RMSprop histories must resume to the same next weights.
    for candidate in (original, loaded):
        candidate.policy.zero_grad(set_to_none=True)
        for p in candidate.policy.parameters():
            p.grad = torch.full_like(p, .001)
        candidate.policy.optimizer.step()
        candidate.policy.predictor_optimizer.step()
    assert parameter_hash(original) == parameter_hash(loaded)
    meta_path = tmp_path/'policy.zip.metadata.json'
    altered = json.loads(meta_path.read_text())
    altered['contract']['predictor_identity']['pretraining']['checkpoint_sha256'] = 'b'*64
    meta_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='predictor identity/mode mismatch'):
        load_pretrained(path)
    checked_env.close()
    for _, env in models:
        env.close()
