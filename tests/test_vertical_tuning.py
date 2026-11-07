"""Activation ablations must preserve initialization, output scale and delay."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.train_vertical_predictor import VerticalPredictor
from myo.train_vertical_tuning import TuningPredictor, load_model, save_model, tuning_contract
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


@pytest.mark.parametrize("hidden,output", [("tanh", "linear"), ("silu", "tanh_unit"), ("leaky_relu", "tanh_scaled")])
def test_tuning_preserves_linear_initialization_transport_and_checkpoint(tmp_path, hidden, output):
    env = VerticalMuscles(VerticalConfig(variant=4))
    actions = np.array([[.3, .1], [.2, .15], [.4, .05], [.1, .2]], dtype=np.float32)
    states = [env.reset()]
    for action in actions:
        states.append(env.step(action))
    episodes = [{"states": np.asarray(states), "actions": actions}]
    torch.manual_seed(0)
    baseline = VerticalPredictor(6, 2, hidden=8)
    torch.manual_seed(0)
    model = TuningPredictor(6, 2, hidden=8, hidden_activation=hidden, output_activation=output)
    for index in (0, 2, 4):
        torch.testing.assert_close(model.net[index].weight, baseline.net[index].weight, rtol=0, atol=0)
        torch.testing.assert_close(model.net[index].bias, baseline.net[index].bias, rtol=0, atol=0)
    model.fit_normalization(episodes)
    initial, commands = torch.tensor(states[0]), torch.tensor(actions)
    torch.testing.assert_close(model(initial, commands[0])[:6], initial[:6]+model.delta_mean, rtol=0, atol=0)
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(.01)
                layer.bias.zero_()
        model.input_scale.fill_(1.)
    jac = torch.autograd.functional.jacobian(lambda u: model.rollout(initial, u)[:, 1], commands)
    assert torch.count_nonzero(jac[:2, 0]) == 0
    assert jac[2, 0].abs().sum() > 0
    expected = tuning_contract(env, model, "test-data")
    path = tmp_path/"tuning.pt"
    save_model(path, model, expected, "bptt", .00003, 1, None)
    restored, _ = load_model(path, expected)
    torch.testing.assert_close(restored.rollout(initial, commands), model.rollout(initial, commands), rtol=0, atol=0)
    different = TuningPredictor(6, 2, hidden=8, hidden_activation=hidden,
                               output_activation="linear" if output != "linear" else "tanh_unit")
    with pytest.raises(ValueError, match="architecture"):
        load_model(path, tuning_contract(env, different, "test-data"))


def test_scaled_tanh_contains_train_increments_without_changing_zero_slope():
    states = np.array([[0., 0.], [.001, .1], [.003, -.2], [.002, .4]], dtype=np.float32)
    episode = {"states": states, "actions": np.full((3, 2), .2, dtype=np.float32)}
    model = TuningPredictor(2, 0, output_activation="tanh_scaled")
    model.fit_normalization([episode])
    z = (torch.tensor(np.diff(states, axis=0))-model.delta_mean)/model.delta_scale
    assert torch.all(z.abs() < model.net[-1].bound)
    slope = torch.autograd.functional.jacobian(model.net[-1], torch.zeros(2))
    torch.testing.assert_close(slope, torch.eye(2), rtol=0, atol=0)
    extreme = model.net[-1](torch.tensor([1e6, -1e6]))
    torch.testing.assert_close(extreme.abs(), model.net[-1].bound)
