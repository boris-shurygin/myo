"""Regressions for physical-time comparisons at finer controller rates."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.dynamics_sensors import SensorTransitionModel, load_sensor_predictor, sensor_contract, sensor_state
from myo.dynamics_command_history import (HistoryTransitionModel, add_delivered_history,
                                         history_contract, load_history_predictor)
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.train_dynamics_timestep import evaluate_time, resample_episode, time_anchors
from myo.train_dynamics_wide import save_wide_predictor


@pytest.mark.parametrize("frame_skip", [5, 1])
def test_fine_queue_preserves_40ms_native_delay_and_checkpoint_period(frame_skip, tmp_path):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets")
    config = json.loads(Path("configs/d1_train.json").read_text())
    config["frame_skip"] = frame_skip
    env = PreacherCurlEnv(PreacherCurlConfig(**config))
    try:
        env.reset(seed=0)
        blocks = round(.04/env.control_dt)
        model = SensorTransitionModel(queue_steps=blocks, control_dt=env.control_dt, hidden_layers=(12, 8))
        model.configure_geometry(env)
        initial = torch.tensor(sensor_state(env))
        state = initial
        for step in range(blocks+2):
            action = torch.full((6,), .1 + .01*step)
            state = model(state, action)
            env.step(action.numpy())
            native = torch.tensor(sensor_state(env))
            torch.testing.assert_close(state[20:model.base_state_size], native[20:model.base_state_size], rtol=0, atol=0)
        # A learned nonzero transition must still have no path from a new
        # command to motion until the full physical delay has elapsed.
        with torch.no_grad():
            model.input_scale[-6:].fill_(1000.)
            model.delta_scale.fill_(.001)
            for layer in model.net:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.fill_(.01)
                    layer.bias.zero_()
        actions = torch.full((blocks+2, 6), .2)
        jac = torch.autograd.functional.jacobian(lambda u: model.rollout(initial, u)[[blocks-1, blocks], 1], actions)
        assert torch.count_nonzero(jac[0, 0]) == 0
        assert jac[1, 0].abs().sum() > 0
        path = tmp_path/"fine.pt"
        metadata = sensor_contract(env)
        save_wide_predictor(path, model, metadata)
        loaded, _ = load_sensor_predictor(path, metadata)
        torch.testing.assert_close(loaded.rollout(initial, actions), model.rollout(initial, actions), rtol=0, atol=0)
        with pytest.raises(ValueError, match="control_dt_s"):
            load_sensor_predictor(path, {**metadata, "control_dt_s": .02})
    finally:
        env.close()


def test_resampling_and_metrics_use_same_physical_times_and_20ms_acceleration():
    time = np.arange(501)*.001
    states = np.zeros((len(time), 260), dtype=np.float32)
    states[:, 0] = .5 + .3*time + .5*2*time**2
    states[:, 1] = .3 + 2*time
    actions = np.repeat(np.arange(25, dtype=np.float32)[:, None, None]*.01, 20*6).reshape(500, 6)
    # Use the exact oldest-first native queue for these held commands.
    history = np.concatenate((np.zeros((40, 6), dtype=np.float32), actions))
    states[:, 20:] = np.stack([history[i:i+40].ravel() for i in range(len(time))])
    episode = {"id": "synthetic", "states": states, "actions": actions}
    anchors = []
    for factor in (20, 5, 1):
        sampled = resample_episode(episode, factor)
        dt = factor*.001
        anchors.append([(i, round(s*dt, 6)) for i,s in time_anchors([sampled], .12, dt)])
        np.testing.assert_array_equal(sampled["states"][:, :20], states[::factor, :20])
        class ConstantAcceleration:
            def rollout(self, initial, commands):
                t = torch.arange(1, commands.shape[1]+1)*dt
                out = initial[:, None, :].expand(-1, len(t), -1).clone()
                out[..., 0] = initial[:, None, 0] + initial[:, None, 1]*t + t*t
                out[..., 1] = initial[:, None, 1] + 2*t
                return out
        metrics = evaluate_time(ConstantAcceleration(), [sampled], dt)
        for metric in metrics.values():
            assert metric["velocity_rmse_rad_s"] < 2e-7
            assert metric["last_20ms_acceleration_rmse_rad_s2"] < 1e-5
            assert metric["zero_20ms_acceleration_rmse_rad_s2"] == pytest.approx(2., abs=1e-5)
    assert anchors[0] == anchors[1] == anchors[2]


def test_delivered_history_is_causal_shifts_after_transport_and_survives_checkpoint(tmp_path):
    # Constant physical state makes the two command memories easy to separate.
    states = np.zeros((121, 266), dtype=np.float32)
    commands = np.repeat((np.arange(120, dtype=np.float32)/120)[:, None], 6, axis=1)
    source = np.concatenate((np.zeros((40, 6), dtype=np.float32), commands))
    states[:, 20:260] = np.stack([source[i:i+40].ravel() for i in range(121)])
    episode = {"states": states, "actions": commands}
    augmented = add_delivered_history(episode, .001)
    model = HistoryTransitionModel(queue_steps=40, control_dt=.001, hidden_layers=(12, 8))
    initial = torch.tensor(augmented["states"][0])
    with torch.no_grad():
        predicted = model.rollout(initial, torch.tensor(commands))
    # Both exact FIFOs agree, despite no physical dynamics being learned.
    torch.testing.assert_close(predicted[:, 20:260], torch.tensor(augmented["states"][1:, 20:260]), rtol=0, atol=0)
    torch.testing.assert_close(predicted[:, 266:], torch.tensor(augmented["states"][1:, 266:]), rtol=0, atol=0)
    features = model.physical_input(torch.tensor(augmented["states"]))
    for index, lag in enumerate((20, 40, 60)):
        np.testing.assert_array_equal(features[120, 38+index*6:44+index*6], states[120-lag, 20:26])
    changed = {**episode, "states": states.copy()}
    changed["states"][80:, 20:26] = .99
    np.testing.assert_array_equal(add_delivered_history(changed, .001)["states"][:80], augmented["states"][:80])
    # The existing toy environment suffices for a contract roundtrip.
    env = PreacherCurlEnv(PreacherCurlConfig(frame_skip=1))
    try:
        env.reset(seed=0)
        metadata = history_contract(env)
        small = HistoryTransitionModel(muscles=4, queue_steps=40, control_dt=.001, hidden_layers=(12, 8))
        small.configure_geometry(env)
        base = sensor_state(env)
        sample = torch.tensor(np.concatenate((base, np.zeros(60*4, dtype=np.float32))))
        path = tmp_path/"history.pt"
        save_wide_predictor(path, small, metadata)
        restored, _ = load_history_predictor(path, metadata)
        torch.testing.assert_close(small(sample, torch.zeros(4)), restored(sample, torch.zeros(4)), rtol=0, atol=0)
        with pytest.raises(ValueError, match="schema"):
            load_sensor_predictor(path, sensor_contract(env))
    finally:
        env.close()
