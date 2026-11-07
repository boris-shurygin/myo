"""Sensor features must be causal, differentiable and isolated from physics."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.dynamics_prediction import contract
from myo.dynamics_sensors import (SensorTransitionModel, augment_episodes,
                                  load_sensor_predictor, sensor_contract, sensor_state)
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.train_dynamics_wide import load_wide_predictor, save_wide_predictor, wide_contract


def test_sensor_values_match_native_force_and_causal_past_without_changing_physics():
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets")
    config = PreacherCurlConfig(**json.loads(Path("configs/d1_train.json").read_text()))
    env = PreacherCurlEnv(config)
    try:
        env.reset(seed=0, options={"q": float(np.deg2rad(75.)), "payload_mass": 1., "capacity": .8})
        before = env.observe().copy()
        model = SensorTransitionModel(hidden_layers=(16, 12, 8))
        model.configure_geometry(env)
        np.testing.assert_array_equal(env.observe(), before)
        states, actions = [], []
        previous = None
        for _ in range(5):
            state = sensor_state(env, previous)
            features = model.physical_input(torch.tensor(state)).detach().numpy()
            native_force = np.array([o.tendon_force for o in env._state_outputs()])
            np.testing.assert_allclose(features[-6:], native_force, atol=.02, rtol=1e-4)
            expected_velocity = np.zeros(6) if previous is None else (state[8:14] - previous) / env.control_dt
            np.testing.assert_allclose(features[26:32], expected_velocity, atol=1e-7, rtol=1e-6)
            states.append(state[:32])
            previous = state[8:14].copy()
            command = np.array([.2, .1, .3, .02, .03, .04])
            actions.append(command)
            env.step(command)
        states.append(sensor_state(env, previous)[:32])
        episode = {"id": "first", "states": np.asarray(states), "actions": np.asarray(actions)}
        augmented = augment_episodes([episode, {**episode, "id": "second"}])
        np.testing.assert_array_equal(augmented[0]["states"][0, -6:], states[0][8:14])
        np.testing.assert_array_equal(augmented[1]["states"][0, -6:], states[0][8:14])
        changed = copy.deepcopy(episode)
        changed["states"][3:, 8:14] += .01
        np.testing.assert_array_equal(augment_episodes([changed])[0]["states"][:3], augmented[0]["states"][:3])
        # Current force varies with current pose/fiber state, not future sensors.
        state = torch.tensor(sensor_state(env, previous))
        jacobian = torch.autograd.functional.jacobian(model.tendon_forces, state).numpy()
        forces = model.tendon_forces(state).detach().numpy()
        stiffness = model.tendon_stiffness.numpy()
        expected = np.where(forces > 0., -env.moment_arms * stiffness, 0.)
        np.testing.assert_allclose(jacobian[:, 0], expected, rtol=.03, atol=2.)
    finally:
        env.close()


def test_sensor_checkpoint_preserves_features_and_delay_and_rejects_legacy_loader(tmp_path):
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        env.reset(seed=0)
        muscles = len(env.muscles)
        model = SensorTransitionModel(muscles=muscles, hidden_layers=(16, 12, 8))
        model.configure_geometry(env)
        state = torch.tensor(sensor_state(env))
        actions = torch.full((6, muscles), .2)
        with torch.no_grad():
            # Keep this fixture out of tanh saturation: raw newton inputs and
            # arbitrary random weights can otherwise erase the tested path.
            model.input_scale[-muscles:].fill_(1000.)
            for layer in model.net:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.fill_(.02)
                    layer.bias.zero_()
        prediction = model.rollout(state, actions)
        torch.testing.assert_close(prediction[0, -muscles:], state[model.fiber_slice], rtol=0, atol=0)
        jacobian = torch.autograd.functional.jacobian(lambda u: model.rollout(state, u)[:, 1], actions)
        torch.testing.assert_close(jacobian[:2, 0], torch.zeros_like(jacobian[:2, 0]), rtol=0, atol=0)
        assert jacobian[2:, 0].abs().sum() > 0
        path = tmp_path / "sensors.pt"
        metadata = sensor_contract(env)
        save_wide_predictor(path, model, metadata)
        loaded, _ = load_sensor_predictor(path, metadata)
        torch.testing.assert_close(loaded.rollout(state, actions), prediction, rtol=0, atol=0)
        torch.testing.assert_close(loaded.physical_input(state), model.physical_input(state), rtol=0, atol=0)
        with pytest.raises(ValueError, match="schema"):
            load_wide_predictor(path, wide_contract(contract(env)))
    finally:
        env.close()
