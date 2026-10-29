import numpy as np
import pytest

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


def test_pending_state_can_reproduce_future_and_is_not_a_view():
    original = FixedStepDelay(dt=0.001, delay=0.004)
    for command in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        original.step(command)
    restored = FixedStepDelay(dt=0.001, delay=0.004)
    restored.restore(original.pending)
    copy = original.pending
    copy[:] = 999
    for command in [0.8, 0.1, 0.7, 0.0, 0.2]:
        assert restored.step(command) == original.step(command)


@pytest.mark.parametrize("dt,delay", [(0, 0.1), (-0.1, 0.1), (float("nan"), 0.1),
                                      (0.001, -0.1), (0.001, float("inf")),
                                      (0.002, 0.003)])
def test_invalid_or_fractional_delay_is_not_silently_rounded(dt, delay):
    with pytest.raises(ValueError):
        FixedStepDelay(dt=dt, delay=delay)


def test_invalid_command_and_restored_state_are_rejected():
    delay = FixedStepDelay(dt=0.001, delay=0.002)
    with pytest.raises(ValueError):
        delay.step(float("nan"))
    with pytest.raises(ValueError):
        delay.restore(np.zeros(3))
