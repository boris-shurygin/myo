"""Dense sensing must not increase the motor rate or peek into future frames."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.train_tissue_sampling import (
    SamplingPredictor, collect_dense, load_model, model_contract, sampling_windows, save_model, transition_windows,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles, heuristic


def test_dense_sensing_preserves_native_command_schedule_and_causal_history():
    config, episodes = collect_dense(4, .12)
    episode = episodes[0]
    env = VerticalMuscles(config)
    states = [env.reset(episode["initial_position_m"])]
    for _ in episode["actions"]:
        states.append(env.step(heuristic(env, episode["family"], episode["version"])))
    np.testing.assert_allclose(episode["native_physical"][::20], np.asarray(states)[:, :6], rtol=0, atol=1e-7)
    data = sampling_windows([episode], 5, 5, 60, True)
    np.testing.assert_array_equal(data["history"][0], np.repeat(episode["sensors"][:1], 13, axis=0))
    np.testing.assert_array_equal(data["history"][2], episode["sensors"][np.maximum(0, 40-np.arange(0, 61, 5))])
    np.testing.assert_array_equal(data["pending"][2], episode["actions"][:2])
    changed = {**episode, "sensors": episode["sensors"].copy()}
    changed["sensors"][41:] += 100
    modified = sampling_windows([changed, episode], 5, 5, 60, True)
    torch.testing.assert_close(modified["history"][:3], data["history"][:3], rtol=0, atol=0)
    torch.testing.assert_close(modified["history"][len(data["history"]):], data["history"], rtol=0, atol=0)
    # Upsampling commands repeats them; it does not introduce intermediate decisions.
    delivered = SamplingPredictor.delivered(data["pending"], data["actions"], 5, 60)
    torch.testing.assert_close(delivered[2], torch.cat((data["pending"][2], data["actions"][2, :1])).repeat_interleave(4, 0))
    fine = transition_windows([episode], 5, 5, True)
    assert len(fine["history"]) == 24  # Four phases per command, all included in warm-up.
    np.testing.assert_array_equal(fine["history"][1, 0], episode["sensors"][5])
    np.testing.assert_array_equal(fine["target"][1, 0], episode["sensors"][10])
    np.testing.assert_array_equal(fine["pending"][9, 0], episode["actions"][0])


def test_fine_rollout_uses_own_history_and_retains_delayed_gradient_and_contract(tmp_path):
    config, episodes = collect_dense(4, .12)
    data = sampling_windows(episodes[:1], 5, 5, 60, True)
    model = SamplingPredictor(6, 5, 5, hidden=8)
    model.fit_normalization(episodes[:1], True)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(.01)
        model.input_scale.fill_(1.)
    history = data["history"][1].clone().requires_grad_()
    pending, actions = data["pending"][1], data["actions"][1]
    forecast = model.rollout(history, pending, actions)
    rolling = history
    delivered = model.delivered(pending, actions, 5, 60)
    for i in range(12):
        following = model(rolling, delivered[i])
        torch.testing.assert_close(following, forecast[i], rtol=0, atol=0)
        rolling = torch.cat((following[None], rolling[:-1]))
    jac = torch.autograd.functional.jacobian(lambda u: model.rollout(history, pending, u)[:, 0], actions)
    assert torch.count_nonzero(jac[:8]) == 0  # 40 ms, even at a 5 ms predictor step.
    assert jac[8:, 0].abs().sum() > 0
    assert torch.autograd.grad(forecast[-1].sum(), history)[0][1:].abs().sum() > 0
    expected = model_contract(config, model, "test-data")
    path = tmp_path/"sampling.pt"
    save_model(path, model, expected, 1, "test")
    restored, _ = load_model(path, expected)
    torch.testing.assert_close(restored.rollout(history, pending, actions), forecast, rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture"):
        load_model(path, model_contract(config, SamplingPredictor(6, 1, 1, hidden=8), "test-data"))
    with pytest.raises(ValueError, match="sensory history tick"):
        SamplingPredictor(6, 5, 20).rollout(history, pending, actions)
