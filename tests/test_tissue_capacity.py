"""A deeper comparison must preserve and train every hidden layer on reload."""
import pytest

torch = pytest.importorskip("torch")

from myo.train_tissue_capacity import CapacityPredictor, capacity_contract, load_capacity
from myo.train_tissue_predictor import save_tissue, tissue_episodes
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def test_capacity_comparison_keeps_extra_layer_gradients_and_checkpoint(tmp_path):
    import numpy as np
    env = VerticalMuscles(VerticalConfig(variant=4))
    actions = np.array([[.3, .1], [.2, .15], [.4, .05], [.1, .2]], dtype=np.float32)
    states = [env.reset()]
    for action in actions:
        states.append(env.step(action))
    episodes = tissue_episodes([{"states": np.asarray(states), "actions": actions}], env.config)
    model = CapacityPredictor(6, 2, hidden_sizes=[16, 16, 8])
    model.fit_normalization(episodes)
    with torch.no_grad():
        model.net[-1].weight.fill_(.001)
        model.input_scale.fill_(1.)
    initial = torch.tensor(episodes[0]["states"][0])
    forecast = model.rollout(initial, torch.tensor(actions))
    forecast[-1, 0].backward()
    layers = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    assert len(layers) == 4
    assert all(layer.weight.grad.abs().sum() > 0 for layer in layers)
    path = tmp_path/"capacity.pt"
    expected = capacity_contract(env, model)
    save_tissue(path, model, expected, 1, "test")
    restored, _ = load_capacity(path, expected)
    torch.testing.assert_close(restored.rollout(initial, torch.tensor(actions)), forecast, rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture"):
        load_capacity(path, capacity_contract(env, CapacityPredictor(6, 2, hidden_sizes=[16, 16])))
