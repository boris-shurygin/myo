"""A fixed-step, causal motor-command delay with observable pending state."""

from __future__ import annotations

import math

import numpy as np


class FixedStepDelay:
    """Delay a scalar by an exact integer number of physics steps.

    ``step(u[k])`` returns ``u[k - delay_steps]``. Before enough commands
    have arrived, the explicitly configured reset value is returned. Fractional
    delays are rejected instead of silently rounding the requested duration.
    ``pending`` lists commands in the order they will be delivered, making the
    buffer usable as part of a fully observed simulation state.
    """

    def __init__(self, dt: float, delay: float, initial_value: float = 0.0):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("delay must be finite and nonnegative")
        ratio = delay / dt
        steps = round(ratio)
        if not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("delay must be an integer multiple of physics dt")
        self.dt = float(dt)
        self.delay = float(delay)
        self.delay_steps = int(steps)
        self._buffer = np.empty(self.delay_steps, dtype=np.float64)
        self._head = 0
        self.reset(initial_value)

    def reset(self, value: float = 0.0) -> None:
        """Set the prehistory to a constant value and discard old commands."""
        if not math.isfinite(value):
            raise ValueError("delay reset value must be finite")
        self._buffer.fill(value)
        self._head = 0

    def step(self, command: float) -> float:
        if not math.isfinite(command):
            raise ValueError("delayed command must be finite")
        if not self.delay_steps:
            return float(command)
        output = float(self._buffer[self._head])
        self._buffer[self._head] = command
        self._head = (self._head + 1) % self.delay_steps
        return output

    @property
    def pending(self) -> np.ndarray:
        """Independent copy, next-to-deliver first; empty for zero delay."""
        return np.concatenate((self._buffer[self._head :], self._buffer[: self._head]))

    def restore(self, pending: np.ndarray) -> None:
        """Restore a chronological queue, e.g. for a counterfactual rollout."""
        values = np.asarray(pending, dtype=np.float64)
        if values.shape != (self.delay_steps,) or not np.isfinite(values).all():
            raise ValueError("pending state must have the expected shape and be finite")
        self._buffer[:] = values
        self._head = 0

    def __len__(self) -> int:
        return self.delay_steps
