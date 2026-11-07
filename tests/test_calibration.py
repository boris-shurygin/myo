"""The static audit must expose path reversals rather than rename the agonist."""

from pathlib import Path

import numpy as np
import pytest

from myo.calibration import measure_pose
from myo.env import PreacherCurlConfig, PreacherCurlEnv


@pytest.mark.skipif(not Path("data/myoarm/manifest.json").exists(), reason="Local MyoArm assets required")
def test_static_fixture_separates_gravity_passive_balance_and_extrapolated_paths():
    env = PreacherCurlEnv(PreacherCurlConfig(geometry="myoarm", task_mode="interactive"))
    env.reset(seed=0)
    before = [m.state_vector().copy() for m in env.muscles]
    try:
        below = measure_pose(env, np.deg2rad(140))
        above = measure_pose(env, np.deg2rad(150))
        invalid = measure_pose(env, np.deg2rad(170))
        assert below["gravity_compensation_Nm"] > 0 > above["gravity_compensation_Nm"]
        # The passive triceps can exceed the required extension moment just
        # beyond vertical: a negative gravity moment alone does not imply
        # active triceps recruitment.
        assert above["holding_group"] == "flexors"
        holding = sum(m["holding_tendon_force_N"] * m["moment_arm_m"] for m in above["muscles"])
        assert holding + above["passive_limit_torque_Nm"] == pytest.approx(
            above["gravity_compensation_Nm"], abs=1e-7)
        assert all(m["moment_arm_reversed"] for m in invalid["muscles"][3:])
        assert not invalid["holding_reachable"]
        for muscle, previous in zip(env.muscles, before):
            np.testing.assert_array_equal(muscle.state_vector(), previous)
    finally:
        env.close()
