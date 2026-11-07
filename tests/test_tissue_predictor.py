"""Local tissue sensing, causal derivatives and consistent autonomous forecasts."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.train_tissue_predictor import (
    TissuePredictor, derivatives, load_tissue, native_to_tissues, save_tissue,
    tissue_contract, tissue_episodes,
)
from myo.train_vertical_history import load_history, history_contract
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_local_strains_preserve_series_elastic_state_and_ignore_global_velocity():
    config = VerticalConfig(variant=4)
    state = torch.tensor([.01, .2, .47, .49, .3, .1, .2, .1, .4, .15])
    sensor = native_to_tissues(state, config)
    torch.testing.assert_close(sensor[:2], state[2:4]/.48-1)
    torch.testing.assert_close(sensor[2:4], (torch.tensor([.59, .61])-state[2:4])/.12-1)
    changed = state.clone()
    changed[1] = -500.
    torch.testing.assert_close(native_to_tissues(changed, config), sensor, rtol=0, atol=0)
    changed[2] += .005
    shifted = native_to_tissues(changed, config)
    assert shifted[0] > sensor[0] and shifted[2] < sensor[2]
    assert shifted[2] < 0  # Geometrical slack signal is not clipped into force.
    rigid = native_to_tissues(state[:2], VerticalConfig(variant=1))
    torch.testing.assert_close(rigid[2:4], torch.zeros(2), rtol=0, atol=0)


def test_derivatives_and_history_use_only_past_frames_with_episode_local_reset():
    config = VerticalConfig(variant=4)
    states = np.tile(np.array([0., 0., .48, .48, .2, .1, .2, .1, .2, .1], dtype=np.float32), (7, 1))
    states[:, 2] += (np.arange(7)**2*.0001).astype(np.float32)
    episode = {"states": states, "actions": np.full((6, 2), .2, dtype=np.float32)}
    augmented = tissue_episodes([episode, episode], config)
    first = augmented[0]["states"]
    np.testing.assert_array_equal(first[0, 10:18], np.zeros(8))
    np.testing.assert_array_equal(first[0, 18:32], np.r_[first[0, :6], first[0, 10:18]])
    native = native_to_tissues(torch.tensor(states), config)
    for t in range(1, len(states)):
        expected = derivatives(native[t], native[t-1], torch.tensor(first[t-1, 10:14]), .02)
        torch.testing.assert_close(torch.tensor(first[t, 10:18]), expected)
    changed = states.copy()
    changed[4:] += .1
    modified = tissue_episodes([{**episode, "states": changed}, episode], config)
    np.testing.assert_array_equal(first[:4], modified[0]["states"][:4])
    np.testing.assert_array_equal(first, modified[1]["states"])


def test_rollout_keeps_derivatives_consistent_history_differentiable_and_delay_exact(tmp_path):
    env = VerticalMuscles(VerticalConfig(variant=4))
    states = [env.reset()]
    actions = np.array([[.3, .1], [.2, .15], [.4, .05], [.1, .2]], dtype=np.float32)
    for action in actions:
        states.append(env.step(action))
    episodes = tissue_episodes([{"states": np.asarray(states), "actions": actions}], env.config)
    model = TissuePredictor(6, 2, hidden=8)
    model.fit_normalization(episodes)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.001)
        # This tiny four-step sample has almost constant acceleration channels.
        # Test the structural delay path away from tanh saturation.
        model.input_scale.fill_(1.)
    state = torch.tensor(episodes[0]["states"][0], requires_grad=True)
    commands = torch.tensor(actions)
    forecast = model.rollout(state, commands)
    previous = state
    for predicted in forecast:
        torch.testing.assert_close(predicted[10:18], derivatives(predicted, previous, previous[10:14], .02))
        torch.testing.assert_close(predicted[18:32], model.sensations(previous))
        previous = predicted
    jac = torch.autograd.functional.jacobian(lambda u: model.rollout(state, u)[:, 0], commands)
    assert torch.count_nonzero(jac[:2, 0]) == 0
    assert jac[2, 0].abs().sum() > 0
    gradient = torch.autograd.grad(forecast[-1, 10:18].square().sum(), state)[0]
    assert gradient[18:].abs().sum() > 0
    expected = tissue_contract(env, model, 1.)
    assert all("position" not in name and "velocity_m_s" not in name for name in expected["neural_input_names"])
    path = tmp_path/"tissue.pt"
    save_tissue(path, model, expected, 1, "test")
    restored, _ = load_tissue(path, expected)
    torch.testing.assert_close(restored.rollout(state, commands), forecast, rtol=0, atol=0)
    with pytest.raises(ValueError, match="aux_weight"):
        load_tissue(path, tissue_contract(env, model, 0.))
    with pytest.raises(ValueError, match="schema"):
        load_history(path, history_contract(env, 3))
