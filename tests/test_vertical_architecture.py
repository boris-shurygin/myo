"""Architecture trials must isolate factors and resume their real optimizers."""
import hashlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("trial", ["control", "silu", "deep", "radam", "rmsprop", "muon"])
def test_trial_preserves_ppo_contract_and_resumes_optimizer_without_losing_moments(tmp_path, trial):
    torch = pytest.importorskip("torch")
    from myo.train_vertical_architecture import TRIALS, model_for_trial, parameter_hash, load_trial, optimizer_steps
    from myo.vertical_control import VerticalControlEnv, save_policy
    from myo.vertical_muscles import VerticalConfig
    import myo.optimizer_muon
    torch.set_num_threads(1)
    env = VerticalControlEnv(VerticalConfig(variant=4))
    reference = model_for_trial("control", env, rollout_steps=32, minibatch=16, epochs=1)
    reference_hash = parameter_hash(reference)
    model = model_for_trial(trial, env, rollout_steps=32, minibatch=16, epochs=1)
    assert sum(TRIALS[trial][key] != TRIALS["control"][key] for key in TRIALS[trial]) == (trial != "control")
    if trial != "deep":
        assert parameter_hash(model) == reference_hash
    model.learn(64)
    assert optimizer_steps(model) > 0
    obs, _ = env.reset(seed=0)
    action = model.predict(obs, deterministic=True)[0]
    assert np.all((action > 0) & (action < 1))
    path = tmp_path/"trial.zip"
    training = {"trial": TRIALS[trial], "optimizer_source_sha256":
                hashlib.sha256(Path(myo.optimizer_muon.__file__).read_bytes()).hexdigest()}
    save_policy(model, env, path, training)
    restored, _ = load_trial(path, env)
    np.testing.assert_array_equal(action, restored.predict(obs, deterministic=True)[0])
    before = parameter_hash(model)
    assert optimizer_steps(restored) == optimizer_steps(model)
    # A further deterministic gradient update detects detached inner optimizer
    # groups/state after SB3 deserialization (actions alone cannot detect this).
    for candidate in (model, restored):
        for group in candidate.policy.optimizer.param_groups:
            group["lr"] = 2e-4
        candidate.policy.optimizer.zero_grad()
        for parameter in candidate.policy.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        candidate.policy.optimizer.step()
    assert parameter_hash(model) != before
    assert parameter_hash(restored) == parameter_hash(model)
