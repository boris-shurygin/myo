"""Reservoir memory must stay causal, frozen and differentiable through forecasts."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.train_vertical_predictor import load_predictor, predictor_contract
from myo.train_vertical_reservoir import (
    ReservoirPredictor, load_reservoir, reservoir_contract, save_reservoir, with_reservoir,
)
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_reservoir_replays_only_past_sensations_and_advances_with_own_predictions():
    states = np.array([[0., 0.], [.01, .03], [.02, .04], [.03, .02]], dtype=np.float32)
    episodes = [{"states": states.copy(), "actions": np.full((3, 2), .2, dtype=np.float32)}
                for _ in range(2)]
    model = ReservoirPredictor(2, 0, reservoir_size=32, input_density=1., hidden=8)
    model.fit_normalization(episodes[:1])
    augmented = with_reservoir(episodes, model)
    np.testing.assert_array_equal(augmented[0]["states"], augmented[1]["states"])
    episodes[0]["states"][2:] += 100
    changed = with_reservoir(episodes, model)
    np.testing.assert_array_equal(augmented[0]["states"][:2], changed[0]["states"][:2])
    np.testing.assert_array_equal(augmented[1]["states"], changed[1]["states"])
    x = torch.tensor(augmented[0]["states"][0], requires_grad=True)
    prediction = model.rollout(x, torch.full((3, 2), .2))
    torch.testing.assert_close(prediction[0, 2:], model.memory_step(x[2:], prediction[0, :2]))
    torch.testing.assert_close(prediction[1, 2:], model.memory_step(prediction[0, 2:], prediction[1, :2]))
    assert not torch.equal(prediction[0, 2:], torch.tensor(augmented[0]["states"][1, 2:]))
    gradient = torch.autograd.grad(prediction[-1, 2:].sum(), x)[0]
    assert gradient[:2].abs().sum() > 0 and gradient[2:].abs().sum() > 0
    radius = np.abs(np.linalg.eigvals(model.W_res.numpy().astype(float))).max()
    assert radius == pytest.approx(.8, abs=1e-6)


@pytest.mark.parametrize("readout", ["mlp", "linear"])
def test_reservoir_is_frozen_preserves_delay_and_roundtrips_checkpoint(tmp_path, readout):
    env = VerticalMuscles(VerticalConfig(variant=4))
    states = [env.reset()]
    actions = np.array([[.3, .1], [.2, .15], [.4, .05], [.1, .2]], dtype=np.float32)
    for action in actions:
        states.append(env.step(action))
    episodes = [{"states": np.asarray(states), "actions": actions}]
    model = ReservoirPredictor(6, 2, reservoir_size=32, hidden=8, readout=readout)
    model.fit_normalization(episodes)
    augmented = with_reservoir(episodes, model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.001)
    x = torch.tensor(augmented[0]["states"][0])
    commands = torch.tensor(actions, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda u: model.rollout(x, u)[:, 1], commands)
    assert torch.count_nonzero(jac[:2, 0]) == 0
    assert jac[2, 0].abs().sum() > 0
    frozen = {name: value.clone() for name, value in model.named_buffers() if name.startswith("W_")}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.rollout(x, commands)[-1, 1].backward()
    optimizer.step()
    assert model.W_res.grad is None and model.W_in.grad is None
    assert all(torch.equal(getattr(model, name), value) for name, value in frozen.items())
    path = tmp_path/"reservoir.pt"
    expected = reservoir_contract(env, model.architecture)
    save_reservoir(path, model, expected, 1, "test")
    restored, _ = load_reservoir(path, expected)
    torch.testing.assert_close(restored.rollout(x, commands), model.rollout(x, commands), rtol=0, atol=0)
    wrong = {**model.architecture, "warmup_steps": 21}
    with pytest.raises(ValueError, match="architecture"):
        load_reservoir(path, reservoir_contract(env, wrong))
    with pytest.raises(ValueError, match="schema"):
        load_predictor(path, predictor_contract(env))
