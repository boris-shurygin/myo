"""Command timing and episode reset, including the zero-delay ablation."""

import numpy as np

from myo.delay import FixedStepDelay


def test_command_arrives_after_exact_physics_delay():
    delay = FixedStepDelay(dt=0.002, delay=0.006)
    outputs = [delay.step(float(command)) for command in range(1, 7)]
    assert outputs == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    np.testing.assert_array_equal(delay.pending, [4, 5, 6])


def test_zero_delay_is_causal_passthrough():
    delay = FixedStepDelay(dt=0.001, delay=0.0)
    assert delay.step(0.8) == 0.8
    assert delay.pending.shape == (0,)


def test_reset_discards_episode_history():
    delay = FixedStepDelay(dt=0.001, delay=0.003)
    for _ in range(5):
        delay.step(1.0)
    delay.reset(0.2)
    assert [delay.step(0.8) for _ in range(4)] == [0.2, 0.2, 0.2, 0.8]
