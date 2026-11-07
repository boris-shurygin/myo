"""Acceleration supervision must match actual forces and provide a real gradient."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.actuator_ablation import make_environment
from myo.env import PreacherCurlConfig
from myo.supervised_actuator import AccelerationKernel, acceleration_tensor


@pytest.mark.parametrize("angle_deg", [30., 142.])
def test_acceleration_target_and_gradient_match_instant_rigid_physics(angle_deg):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets for the supervised comparison")
    config = PreacherCurlConfig(**json.loads(Path("configs/d1_train.json").read_text()))
    env, kernel = make_environment("instant_rigid", config), AccelerationKernel(config)
    try:
        env.reset(seed=0, options={"q": np.deg2rad(angle_deg), "qvel": .05,
            "target_angle": np.deg2rad(30), "capacity": 1., "payload_mass": 1., "trigger": 1.})
        command = np.array([.24, .21, .27, .12, .13, .11])
        state = kernel.capture(env)
        data = {k: np.asarray([v]) for k, v in state.items()}
        # Independently obtain forces using the actual muscle implementation.
        forces = []
        for m, length, arm, activation in zip(env.muscles, env.muscle_lengths(), env.moment_arms, command):
            m.activation = activation
            forces.append(m.sample(float(length), float(-arm * env.data.qvel[0])).tendon_force)
        native = mujoco.MjData(env.model)
        native.qpos[:] = env.data.qpos
        native.qvel[:] = env.data.qvel
        native.qfrc_applied[0] = env.moment_arms @ forces + env.passive_limit_torque()
        mujoco.mj_forward(env.model, native)
        expected = float(native.qacc[0])
        prediction = torch.tensor(command[None], dtype=torch.float64, requires_grad=True)
        actual = acceleration_tensor(prediction, data, np.array([0]), kernel)
        assert float(actual.detach()[0]) == pytest.approx(expected, abs=1e-6)
        actual.sum().backward()
        numeric = []
        for i in range(6):
            high, low = command.copy(), command.copy()
            high[i] += 1e-5
            low[i] -= 1e-5
            numeric.append((kernel.response(state, high) - kernel.response(state, low)) / 2e-5)
        np.testing.assert_allclose(prediction.grad.numpy()[0], numeric, rtol=2e-3, atol=2e-4)
        assert np.linalg.norm(numeric) > 1
        assert bool(state["free"]) == (angle_deg == 30.)
    finally:
        env.close()
        kernel.close()
