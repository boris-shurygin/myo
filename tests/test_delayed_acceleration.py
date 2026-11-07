"""Temporal credit must cross the real motor queue and compliant muscle state."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myo.compact_protocol import initialize_command_at_reset
from myo.delayed_acceleration import NativeResponse, capture, linear_response
from myo.env import PreacherCurlConfig, PreacherCurlEnv


@pytest.mark.parametrize("prefix_steps,angle_deg", [(0, 20.), (10, 20.), (0, 142.)])
def test_native_future_gradient_preserves_delay_compliance_and_source_state(prefix_steps, angle_deg):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets")
    config = PreacherCurlConfig(**json.loads(Path("configs/d1_train.json").read_text()))
    env, kernel = PreacherCurlEnv(config), NativeResponse(config)
    try:
        env.reset(seed=0, options={"q": float(np.deg2rad(angle_deg)), "target_angle": float(np.deg2rad(30)),
            "trigger": 1., "capacity": .8, "payload_mass": 1.})
        initialize_command_at_reset(env, 30., 1.)
        for _ in range(prefix_steps):
            env.step(env.baseline_action())
        before = env.observe().copy()
        muscle_before = [m.state_vector().copy() for m in env.muscles]
        state = capture(env)
        continuation, target = kernel.reference(state)
        # An independent complete clone, rather than another use of restore().
        independent = copy.deepcopy(env)
        expected = []
        try:
            for command in continuation:
                velocity = float(independent.data.qvel[independent.dof_index])
                independent.step(command)
                expected.append((float(independent.data.qvel[independent.dof_index]) - velocity) / env.control_dt)
        finally:
            independent.close()
        np.testing.assert_allclose(target, expected, rtol=0, atol=1e-10)
        anchor = np.array([.18, .17, .23, .04, .06, .05])
        acceleration, jacobian = kernel.linearize(state, continuation, anchor)
        np.testing.assert_allclose(jacobian[:2], 0., atol=1e-10, rtol=0)
        assert np.linalg.norm(jacobian[2:]) > 1
        command = torch.tensor(anchor[None], dtype=torch.float64, requires_grad=True)
        predicted = linear_response(command, torch.tensor(anchor[None]),
            torch.tensor(acceleration[None]), torch.tensor(jacobian[None]))
        loss = ((predicted[:, 2:] - torch.tensor(target[None, 2:])) ** 2).mean()
        loss.backward()
        numerical = []
        for i in range(6):
            high, low = anchor.copy(), anchor.copy()
            high[i] += .0005
            low[i] -= .0005
            hi = np.mean((kernel.response(state, continuation, high)[2:] - target[2:]) ** 2)
            lo = np.mean((kernel.response(state, continuation, low)[2:] - target[2:]) ** 2)
            numerical.append((hi - lo) / .001)
        np.testing.assert_allclose(command.grad.numpy()[0], numerical, rtol=.01, atol=.002)
        np.testing.assert_array_equal(env.observe(), before)
        for muscle, previous in zip(env.muscles, muscle_before):
            np.testing.assert_array_equal(muscle.state_vector(), previous)
        assert any(m.last_output.tendon_strain > 0 for m in env.muscles)
    finally:
        env.close()
        kernel.close()


def test_local_updates_decrease_whole_dataset_loss_inside_measured_neighborhood():
    from types import SimpleNamespace
    from gymnasium import spaces
    from myo.delayed_acceleration import mean_commands
    from myo.policy_beta import BetaExcitationPolicy
    from myo.train_delayed_acceleration import local_step

    policy = BetaExcitationPolicy(spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        spaces.Box(0., 1., shape=(6,), dtype=np.float32), lambda _: 3e-4,
        net_arch=[8], activation_fn=torch.nn.Tanh)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, betas=(0., .999))
    model = SimpleNamespace(policy=policy)
    observations = np.array([[0., 0.], [1., -1.], [-1., 1.]], dtype=np.float32)
    with torch.no_grad():
        anchors = mean_commands(model, observations).clone()
    gains = torch.tensor([40., 20., 70., -80., -70., -65.])
    tensors = {"anchors": anchors, "accelerations": torch.zeros(3, 6),
               "targets": torch.ones(3, 6) * .5, "jacobians": gains.expand(3, 6, 6),
               "sample_weights": torch.tensor([.1, 1., 1.9])}
    accepted = 0
    for _ in range(25):
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        result = local_step(model, observations, tensors, 2, .001)
        if result["accepted"]:
            accepted += 1
            assert result["loss_after"] < result["loss_before"]
            assert result["distance"] <= .001
        else:
            assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())
            break
    assert accepted > 0
