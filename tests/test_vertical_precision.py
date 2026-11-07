"""Near-goal reward changes must preserve mechanics and keep test holds unseen."""
import numpy as np
import pytest

from myo.vertical_control import REWARD, TRAIN_CASES, TEST_CASES, VerticalControlEnv, load_policy, save_policy
from myo.vertical_muscles import VerticalConfig
from myo.vertical_precision import DENSE_CASES, VerticalPrecisionEnv, position_reward, evaluate_precision


def test_precision_changes_only_position_reward_and_expands_causal_tasks():
    assert len(DENSE_CASES) == 33
    assert set(TRAIN_CASES) <= set(DENSE_CASES)
    assert all((q, t) not in DENSE_CASES for q, t in TEST_CASES if q == t and t != 0.)
    baseline, precision = VerticalControlEnv(VerticalConfig(variant=4)), VerticalPrecisionEnv(VerticalConfig(variant=4))
    options = {"initial_position_m": -.03, "target_m": .06}
    np.testing.assert_array_equal(baseline.reset(options=options)[0], precision.reset(options=options)[0])
    for action in ([.3, .1], [.4, .2], [.2, .15], [.25, .12]):
        ob, _, tb, xb, ib = baseline.step(action)
        op, reward, tp, xp, ip = precision.step(action)
        np.testing.assert_array_equal(ob, op)
        np.testing.assert_array_equal(baseline.physical.state(), precision.physical.state())
        assert (tb, xb) == (tp, xp)
        for key in ("velocity", "effort", "command_change", "physical_failure"):
            assert ib["reward_components"][key] == ip["reward_components"][key]
        assert reward == pytest.approx(.1*sum(ip["reward_components"].values()))
    assert position_reward(0.) == pytest.approx(.85)
    assert position_reward(.08) == pytest.approx(.85*np.exp(-1))
    assert position_reward(-.005) == position_reward(.005)
    assert position_reward(0)-position_reward(.005) > 30*.85*(1-np.exp(-(.005/.08)**2))
    close_gain = position_reward(.004)-position_reward(.005)
    far_gain = position_reward(.04)-position_reward(.041)
    assert close_gain > far_gain > 0
    actual = []
    for i in range(len(DENSE_CASES)+1):
        precision.reset(seed=0 if i == 0 else None)
        actual.append((precision.physical.position, precision.target))
    np.testing.assert_allclose(actual[:-1], DENSE_CASES)
    assert actual[-1] == actual[0]


def test_precision_checkpoint_and_metrics_keep_reward_identity(tmp_path):
    import torch
    from myo.train_vertical_architecture import TRIALS, model_for_trial, load_trial
    import hashlib
    from pathlib import Path
    import myo.optimizer_muon
    torch.set_num_threads(1)
    # Exercise the requested architecture and optimizer without adding a new trial to the global registry.
    TRIALS["precision_test"] = {"hidden": [128]*4, "activation": "SiLU", "optimizer": "RMSprop"}
    try:
        env = VerticalPrecisionEnv(VerticalConfig(variant=4))
        model = model_for_trial("precision_test", env, rollout_steps=32, minibatch=16, epochs=1)
        model.learn(64)
        path = tmp_path/"model.zip"
        training = {"trial": TRIALS["precision_test"], "optimizer_source_sha256":
            hashlib.sha256(Path(myo.optimizer_muon.__file__).read_bytes()).hexdigest()}
        save_policy(model, env, path, training)
        restored, _ = load_trial(path, env)
        obs, _ = env.reset(options={"initial_position_m": 0., "target_m": 0.})
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
        with pytest.raises(ValueError, match="contract mismatch"):
            load_policy(path, VerticalControlEnv(VerticalConfig(variant=4)))
        result = evaluate_precision(restored, env.physical.config, ((0., 0.),), seconds=.1)
        replay = VerticalPrecisionEnv(env.physical.config, episode_seconds=.1)
        replay.reset(options={"initial_position_m": 0., "target_m": 0.})
        raw = 0.
        for t, q, v, target, u0, u1 in result["cases"][0]["samples"]:
            info = replay.step([u0, u1])[-1]
            raw += info["raw_reward"]
            np.testing.assert_array_equal([replay.physical.time, replay.physical.position, replay.physical.velocity], [t, q, v])
        assert raw == result["cases"][0]["raw_return"]
    finally:
        del TRIALS["precision_test"]
