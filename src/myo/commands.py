"""Causal user commands: bounded target motion, independent excitation preference."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ArmCommand:
    target: float
    requested: float
    trigger: float = 1.0
    velocity: float = 0.0

    def set(self, target: float, trigger: float, limits: np.ndarray) -> None:
        if not math.isfinite(target) or not math.isfinite(trigger) or not 0 <= trigger <= 1:
            raise ValueError("Command needs a finite target and trigger in [0, 1].")
        self.requested = float(np.clip(target, *limits))
        self.trigger = float(trigger)

    def advance(self, dt: float, speed: float, acceleration: float) -> None:
        distance = self.requested - self.target
        # Reserve two discrete control intervals for braking. The continuous
        # sqrt(2*a*d) envelope starts braking too late after sampled updates.
        margin = 2 * acceleration * dt
        stopping_speed = max(0.0, math.sqrt(2 * acceleration * abs(distance) + margin**2) - margin)
        desired = math.copysign(min(speed, stopping_speed), distance)
        # A target reversal decelerates the command, never the physical joint.
        new_velocity = self.velocity + float(np.clip(desired - self.velocity,
                                                     -acceleration * dt, acceleration * dt))
        step = 0.5 * (self.velocity + new_velocity) * dt
        if (distance * step >= 0 and abs(step) >= abs(distance)
                and abs(self.velocity) <= acceleration * dt):
            self.target, self.velocity = self.requested, 0.0
        else:
            self.target += step
            self.velocity = new_velocity
