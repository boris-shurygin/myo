"""Causality, data separation, and deployable predictor contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.predictor import (
    ErrorPredictor,
    OUTPUT_NAMES,
    PredictorObservationWrapper,
    collect_data,
    fit_predictor,
    make_episode_samples,
    split_episodes,
)
from myo.runtime import config_digest, dynamics_digest, load_controller


def _metadata(config, observation_names):
    return {
        "observation_names": list(observation_names),
        "horizon_s": 0.1,
        "config_sha256": config_digest(config),
        "dynamics_sha256": dynamics_digest(),
        "config": asdict(config),
        "behavior_controller": "heuristic",
    }


def _artifact(env):
    """An explicit nonconstant model fixture; no training required to load it."""
    width = len(env.observation_names)
    w1 = np.zeros((width, 2))
    w1[0, 0], w1[1, 1] = 0.5, -0.25
    return {
        "schema": "myo.error_predictor.v1",
        "model_type": "tanh_mlp",
        **_metadata(env.config, env.observation_names),
        "output_names": list(OUTPUT_NAMES),
        "x_mean": np.zeros(width).tolist(),
        "x_scale": np.ones(width).tolist(),
        "y_mean": [0.02, -0.03],
        "y_scale": [0.1, 0.2],
        "w1": w1.tolist(),
        "b1": [0.1, -0.2],
        "w2": [[1.0, 0.3], [-0.2, 0.7]],
        "b2": [0.05, 0.04],
    }


def _write_artifact(path, artifact):
    path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    return path


def test_future_labels_do_not_change_current_inputs_or_cross_episode_end():
    observations = np.arange(18, dtype=float).reshape(6, 3)
    errors = np.column_stack((np.arange(6), -np.arange(6)))
    x, future, current = make_episode_samples(observations, errors, steps=2)
    assert len(x) == 4
    np.testing.assert_array_equal(x[0], observations[0])
    np.testing.assert_array_equal(future[0], errors[2])
    np.testing.assert_array_equal(future[-1], errors[-1])
    np.testing.assert_array_equal(current[0], errors[0])
    changed_future = errors.copy()
    changed_future[2:] += 5000
    alternative_x, alternative_y, _ = make_episode_samples(observations, changed_future, 2)
    np.testing.assert_array_equal(x, alternative_x)
    assert not np.array_equal(future, alternative_y)
    observations[:] = 0
    assert np.any(x != 0), "Returned samples must not alias a mutable rollout buffer."
    with pytest.raises(ValueError, match="short"):
        make_episode_samples(np.zeros((2, 3)), np.zeros((2, 2)), 2)


def test_collector_labels_are_actual_future_errors_within_each_episode(tmp_path):
    config = PreacherCurlConfig(episode_seconds=0.2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    path = tmp_path / "pairs.npz"
    collect_data(SimpleNamespace(
        controller="heuristic", config=config_path, predictor=None,
        horizon=0.04, episodes=2, seed=91, output=path,
    ))
    with np.load(path, allow_pickle=False) as dataset:
        for episode in range(2):
            rows = np.flatnonzero(dataset["episode_id"] == episode)
            # 10 transitions, 11 observations, and H=2 give 9 valid labels.
            assert len(rows) == 9
            env = PreacherCurlEnv(config)
            try:
                first, info = env.reset(seed=91 + episode)
                np.testing.assert_array_equal(dataset["x"][rows[0]], first)
                np.testing.assert_array_equal(dataset["current_error"][rows[0]], info["tracking_error"])
                for _ in range(2):
                    _, _, _, _, info = env.step(env.baseline_action())
                np.testing.assert_allclose(dataset["y"][rows[0]], info["tracking_error"], rtol=0, atol=0)
                for _ in range(8):
                    _, _, _, _, info = env.step(env.baseline_action())
                np.testing.assert_allclose(dataset["y"][rows[-1]], info["tracking_error"], rtol=0, atol=0)
            finally:
                env.close()


def test_validation_outliers_cannot_leak_into_feature_or_target_normalization():
    rng = np.random.default_rng(5)
    # Interleaving also catches a row-wise split instead of a whole-episode split.
    ids = np.tile(np.arange(6), 8)
    train, validation = split_episodes(ids, seed=12)
    assert np.all(train ^ validation)
    assert train.any() and validation.any()
    assert set(ids[train]).isdisjoint(ids[validation])
    x = rng.normal(size=(len(ids), 3))
    x[:, 1] = 7.0  # A constant training feature must receive a finite scale.
    y = np.column_stack((0.1 * x[:, 0], np.full(len(ids), -0.25)))
    x[validation] += 100_000
    y[validation] += 10_000
    model, report = fit_predictor(
        x, y, ids, _metadata(PreacherCurlConfig(), ["a", "b", "c"]),
        epochs=2, hidden=4, seed=12,
    )
    np.testing.assert_allclose(model.x_mean, x[train].mean(axis=0))
    np.testing.assert_allclose(model.y_mean, y[train].mean(axis=0))
    assert model.x_scale[1] == 1.0
    assert model.y_scale[1] == 1.0
    assert not np.allclose(model.x_mean, x.mean(axis=0))
    assert set(report["train_episodes"]).isdisjoint(report["validation_episodes"])
    assert report["train_samples"] + report["validation_samples"] == len(x)
    assert np.isfinite(model.predict(x)).all()


def test_model_json_roundtrip_preserves_batch_and_single_predictions(tmp_path):
    env = PreacherCurlEnv()
    try:
        model = ErrorPredictor(_artifact(env))
        x = np.stack((np.zeros(len(env.observation_names)), np.ones(len(env.observation_names))))
        expected = model.predict(x)
        assert not np.allclose(expected[0], expected[1])
        path = _write_artifact(tmp_path / "predictor.json", model.artifact)
        reloaded = ErrorPredictor.load(path)
        np.testing.assert_array_equal(reloaded.predict(x), expected)
        np.testing.assert_array_equal(reloaded.predict(x[0]), model.predict(x[0]))
        # GEMV and batched GEMM may reduce floating-point products in different
        # orders on different BLAS/CPU builds. Serialization still stays exact
        # for the same input shape, while these two evaluation paths need a
        # double-precision numerical comparison rather than bitwise identity.
        np.testing.assert_allclose(reloaded.predict(x[0]), expected[0], rtol=1e-14, atol=1e-15)
    finally:
        env.close()


def test_wrapper_only_appends_prediction_and_never_advances_to_the_label(tmp_path, monkeypatch):
    base = PreacherCurlEnv()
    path = _write_artifact(tmp_path / "predictor.json", _artifact(base))
    wrapped = PredictorObservationWrapper(base, path)
    plain = PreacherCurlEnv()
    try:
        expected, _ = plain.reset(seed=44)
        observed, _ = wrapped.reset(seed=44)
        assert base.time == 0.0
        np.testing.assert_array_equal(observed[:-2], expected)
        assert wrapped.observation_names[-2:] == OUTPUT_NAMES
        assert wrapped.observation_space.contains(observed)

        def forbidden(*args, **kwargs):
            raise AssertionError("Prediction attempted to read/advance the environment.")

        with monkeypatch.context() as guard:
            guard.setattr(base, "step", forbidden)
            guard.setattr(base, "telemetry", forbidden)
            for _ in range(3):
                np.testing.assert_array_equal(wrapped.observation(expected), observed)
        assert base.time == 0.0
        action = plain.baseline_action()
        expected_next, _, _, _, _ = plain.step(action)
        actual_next, _, _, _, _ = wrapped.step(action)
        assert base.time == pytest.approx(base.control_dt)
        np.testing.assert_array_equal(actual_next[:-2], expected_next)
    finally:
        wrapped.close()
        plain.close()


def test_wrapper_rejects_feature_order_and_physics_configuration_mismatches(tmp_path):
    base = PreacherCurlEnv()
    changed = PreacherCurlEnv(replace(base.config, payload_mass=2.0))
    try:
        artifact = _artifact(base)
        path = _write_artifact(tmp_path / "predictor.json", artifact)
        with pytest.raises(ValueError, match="configuration"):
            PredictorObservationWrapper(changed, path)
        reordered = deepcopy(artifact)
        reordered["observation_names"][0:2] = reversed(reordered["observation_names"][0:2])
        _write_artifact(path, reordered)
        with pytest.raises(ValueError, match="observation order"):
            PredictorObservationWrapper(base, path)
    finally:
        base.close()
        changed.close()


def test_runtime_rejects_changed_dynamics_even_with_matching_config_and_features(tmp_path):
    base = PreacherCurlEnv()
    try:
        artifact = _artifact(base)
        artifact["dynamics_sha256"] = "0" * 64
        path = _write_artifact(tmp_path / "predictor.json", artifact)
        with pytest.raises(ValueError, match="dynamics"):
            PredictorObservationWrapper(base, path)
        checkpoint = tmp_path / "policy.zip"
        checkpoint.write_bytes(b"must reject fingerprint before loading weights")
        metadata = {
            "schema_version": 1,
            "config": asdict(base.config),
            "observation_names": list(base.observation_names),
            "dynamics_sha256": "0" * 64,
            "predictor_sha256": None,
        }
        Path(str(checkpoint) + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="dynamics"):
            load_controller(str(checkpoint))
    finally:
        base.close()


def test_policy_loader_rejects_predictor_hash_mismatch_before_loading_weights(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"intentionally not a policy archive")
    predictor = tmp_path / "predictor.json"
    predictor.write_text("{}", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "config": asdict(PreacherCurlConfig()),
        "dynamics_sha256": dynamics_digest(),
        "predictor_sha256": "0" * 64,
        "predictor_path": predictor.name,
    }
    Path(str(checkpoint) + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Predictor contents differ"):
        load_controller(str(checkpoint))


def test_optional_ppo_predictor_bundle_survives_relocation(tmp_path):
    pytest.importorskip("stable_baselines3")
    from myo.train import main as train_main

    config = PreacherCurlConfig(episode_seconds=0.12)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    base = PreacherCurlEnv(config)
    try:
        original_predictor = _write_artifact(tmp_path / "original.json", _artifact(base))
    finally:
        base.close()
    original_bytes = original_predictor.read_bytes()
    source = tmp_path / "training"
    output = source / "policy.zip"
    train_main([
        "--steps", "1", "--seed", "8", "--threads", "1",
        "--config", str(config_path), "--predictor", str(original_predictor),
        "--output", str(output),
    ])
    sidecar = Path(str(output) + ".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["dynamics_sha256"] == dynamics_digest()
    assert not Path(metadata["predictor_path"]).is_absolute()
    bundled = source / metadata["predictor_path"]
    assert bundled.read_bytes() == original_bytes
    assert metadata["predictor_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for item in (output, sidecar, bundled):
        shutil.copy2(item, relocated / item.name)
    original_predictor.unlink()
    shutil.rmtree(source)
    env, action, restored_metadata = load_controller(str(relocated / output.name))
    try:
        observation, _ = env.reset(seed=8)
        assert list(env.observation_names) == restored_metadata["observation_names"]
        control = action(observation)
        assert env.action_space.contains(control)
        next_observation, reward, _, _, _ = env.step(control)
        assert np.isfinite(next_observation).all() and np.isfinite(reward)
    finally:
        env.close()
