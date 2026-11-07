"""Protect temporal credit, episode boundaries and the predictor's contract."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.dynamics_prediction import (TransitionModel, compact_state, contract, load_predictor,
                                     save_predictor, sequence_windows)
from myo.env import PreacherCurlConfig, PreacherCurlEnv


def test_windows_do_not_cross_reset_and_normalization_uses_training_episodes_only():
    states = np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 1000
    episodes = [{"id": name, "states": states + offset,
                 "actions": np.full((7, 6), offset, dtype=np.float32)}
                for name, offset in (("train", 0.), ("validation", 100.))]
    windows = sequence_windows(episodes, 3)
    assert list(windows["episode_ids"]) == ["train"] * 5 + ["validation"] * 5
    np.testing.assert_array_equal(windows["targets"][4], states[5:8])
    np.testing.assert_array_equal(windows["initial"][5], states[0] + 100.)
    changed = copy.deepcopy(episodes)
    changed[1]["states"] += 10000
    model, other = TransitionModel(), TransitionModel()
    model.fit_normalization(episodes[:1])
    other.fit_normalization(changed[:1])
    torch.testing.assert_close(model.input_mean, other.input_mean, rtol=0, atol=0)
    torch.testing.assert_close(model.delta_scale, other.delta_scale, rtol=0, atol=0)


def test_rollout_bptt_crosses_transport_queue_and_checkpoint_rejects_different_contract(tmp_path):
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        env.reset(seed=0)
        meta = contract(env)
        muscles = len(env.muscles)
        model = TransitionModel(muscles=muscles, hidden=8)
        state = torch.tensor(compact_state(env))
        actions = torch.full((4, muscles), .2, requires_grad=True)
        # An explicit delivered-excitation dependence makes the timing test
        # independent of whether any particular random network learned it.
        with torch.no_grad():
            model.net[-1].weight[1].fill_(.1)
        predictions = model.rollout(state, actions)
        derivative = torch.autograd.functional.jacobian(lambda u: model.rollout(state, u)[:, 1], actions)
        torch.testing.assert_close(derivative[:2, 0], torch.zeros_like(derivative[:2, 0]), rtol=0, atol=0)
        assert derivative[2:, 0].abs().sum() > 0
        loss = predictions[-1, 1].square()
        full = torch.autograd.grad(loss, actions)[0]
        detached = model.rollout(state, actions, detach_between_steps=True)
        cut = torch.autograd.grad(detached[-1, 1].square(), actions)[0]
        assert full[0].abs().sum() > 0
        torch.testing.assert_close(cut[0], torch.zeros_like(cut[0]), rtol=0, atol=0)
        path = tmp_path / "predictor.pt"
        save_predictor(path, model, meta)
        restored, _ = load_predictor(path, meta)
        torch.testing.assert_close(restored.rollout(state, actions), predictions, rtol=0, atol=0)
        with pytest.raises(ValueError, match="control_dt_s"):
            load_predictor(path, {**meta, "control_dt_s": .01})
    finally:
        env.close()


def test_wide_predictor_preserves_three_hidden_layers_and_legacy_checkpoint(tmp_path):
    from myo.train_dynamics_wide import (WideTransitionModel, load_wide_predictor,
                                        save_wide_predictor, wide_contract)
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        env.reset(seed=0)
        base = contract(env)
        meta = wide_contract(base)
        muscles = len(env.muscles)
        model = WideTransitionModel(muscles=muscles, hidden_layers=(32, 24, 12))
        with torch.no_grad():
            model.net[-1].weight.fill_(.03)
        state = torch.tensor(compact_state(env))
        actions = torch.full((6, muscles), .2)
        path = tmp_path / "wide.pt"
        save_wide_predictor(path, model, meta)
        restored, _ = load_wide_predictor(path, meta)
        assert [(layer.in_features, layer.out_features) for layer in restored.net
                if isinstance(layer, torch.nn.Linear)] == [(18, 32), (32, 24), (24, 12), (12, 14)]
        torch.testing.assert_close(restored.rollout(state, actions), model.rollout(state, actions), rtol=0, atol=0)
        with pytest.raises(ValueError, match="base_source_sha256"):
            load_wide_predictor(path, {**meta, "base_source_sha256": "changed"})
        small = TransitionModel(muscles=muscles, hidden=8)
        small_path = tmp_path / "small.pt"
        save_predictor(small_path, small, base)
        previous, _ = load_predictor(small_path, base)
        torch.testing.assert_close(previous.rollout(state, actions), small.rollout(state, actions), rtol=0, atol=0)
    finally:
        env.close()


def test_acceleration_loss_is_equivalent_for_one_step_but_not_accumulated_velocity():
    from myo.train_dynamics_acceleration import acceleration_loss
    model = TransitionModel(hidden=8).double()
    initial = torch.randn(2, 32, dtype=torch.float64)
    target = torch.randn(2, 1, 32, dtype=torch.float64)
    prediction = torch.randn(2, 1, 32, dtype=torch.float64, requires_grad=True)
    velocity_loss = ((prediction[..., :20] - target[..., :20]) / model.delta_scale).square().mean()
    local_loss = acceleration_loss(initial, prediction, target, model, .02)
    torch.testing.assert_close(local_loss, velocity_loss, rtol=1e-12, atol=1e-12)
    velocity_gradient = torch.autograd.grad(velocity_loss, prediction, retain_graph=True)[0]
    local_gradient = torch.autograd.grad(local_loss, prediction)[0]
    torch.testing.assert_close(local_gradient, velocity_gradient, rtol=1e-12, atol=1e-12)
    # A constant velocity offset is penalized every step by the old loss,
    # but is an acceleration error only in the first interval.
    initial = torch.zeros(1, 32, dtype=torch.float64)
    target = torch.zeros(1, 3, 32, dtype=torch.float64)
    prediction = target.clone()
    prediction[..., 1] = .2
    velocity_loss = prediction[..., :20].square().mean()
    local_loss = acceleration_loss(initial, prediction, target, model, .02)
    torch.testing.assert_close(local_loss * 3, velocity_loss, rtol=1e-12, atol=1e-12)
