"""Protect causal history construction, autonomous rollout and checkpoint identity."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.train_vertical_history import (
    HistoryPredictor, history_contract, load_history, save_history, with_history,
)
from myo.train_vertical_predictor import VerticalPredictor, load_predictor, predictor_contract
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_history_stays_inside_episode_and_rollout_shifts_predictions_with_gradient():
    episodes = [{"states": np.arange(24, dtype=np.float32).reshape(6, 4)+offset,
                 "actions": np.full((5, 2), .2, dtype=np.float32)} for offset in (0, 100)]
    augmented = with_history(episodes, 4, 3)
    for original, episode in zip(episodes, augmented):
        for t, state in enumerate(episode["states"]):
            np.testing.assert_array_equal(state[4:].reshape(3, 4),
                original["states"][[max(t-age, 0) for age in (1, 2, 3)]])
    torch.manual_seed(0)
    model = HistoryPredictor(4, 0, history_steps=3, hidden=8)
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(.01)
                layer.bias.zero_()
    initial = torch.tensor(augmented[0]["states"][2], requires_grad=True)
    prediction = model.rollout(initial, torch.full((3, 2), .2))
    torch.testing.assert_close(prediction[0, 4:8], initial[:4])
    torch.testing.assert_close(prediction[1, 4:8], prediction[0, :4])
    torch.testing.assert_close(prediction[2, 8:12], prediction[0, :4])
    gradient = torch.autograd.grad(prediction[-1, 1], initial)[0]
    assert torch.count_nonzero(gradient[4:]) == 12
    # A future target cannot affect the starting history.
    episodes[0]["states"][4:] += 1000
    np.testing.assert_array_equal(with_history(episodes, 4, 3)[0]["states"][2], initial.detach().numpy())


def test_history_preserves_initial_function_delay_normalization_and_checkpoint(tmp_path):
    env = VerticalMuscles(VerticalConfig(variant=4))
    states = [env.reset()]
    actions = np.array([[.3, .1], [.2, .15], [.4, .05], [.1, .2]], dtype=np.float32)
    for action in actions:
        states.append(env.step(action))
    episodes = [{"states": np.asarray(states), "actions": actions}]
    augmented = with_history(episodes, 6, 10)
    torch.manual_seed(0)
    baseline = VerticalPredictor(6, 2, hidden=8)
    torch.manual_seed(0)
    model = HistoryPredictor(6, 2, history_steps=10, hidden=8)
    baseline.fit_normalization(episodes)
    model.fit_normalization(augmented)
    torch.testing.assert_close(model.delta_scale, baseline.delta_scale, rtol=0, atol=0)
    torch.testing.assert_close(model.input_mean[:8], baseline.input_mean, rtol=0, atol=0)
    x = torch.tensor(augmented[0]["states"][0])
    commands = torch.tensor(actions)
    torch.testing.assert_close(model.rollout(x, commands)[:, :10],
                               baseline.rollout(x[:10], commands), rtol=0, atol=0)
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.fill_(.001)
                layer.bias.zero_()
    jac = torch.autograd.functional.jacobian(lambda u: model.rollout(x, u)[:, 1], commands)
    assert torch.count_nonzero(jac[:2, 0]) == 0
    assert jac[2, 0].abs().sum() > 0
    path = tmp_path/"history.pt"
    expected = history_contract(env, 10)
    save_history(path, model, expected, 1, "test")
    restored, _ = load_history(path, expected)
    torch.testing.assert_close(restored.rollout(x, commands), model.rollout(x, commands), rtol=0, atol=0)
    with pytest.raises(ValueError, match="history_steps|history_duration|state_names"):
        load_history(path, history_contract(env, 9))
    with pytest.raises(ValueError, match="schema"):
        load_predictor(path, predictor_contract(env))
