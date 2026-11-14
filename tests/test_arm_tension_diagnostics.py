"""Avoid two misleading measures of antagonist effort."""
import numpy as np

from myo.arm_tension_diagnostics import antagonist_metrics


def test_alternating_bursts_are_not_simultaneous_coactivation():
    activation = np.array([[1.,1.,1.,0.,0.,0.], [0.,0.,0.,1.,1.,1.]])
    arms = np.array([.02,.03,.04,-.02,-.03,-.04])
    result = antagonist_metrics(activation, 100*activation, arms, np.zeros(2))
    assert result['activation_coactivation'].mean() == 0.
    assert result['opposing_muscle_torque_Nm'].mean() == 0.
    # Taking min after averaging would incorrectly report 50% overlap.
    assert min(result['flexor_activation_mean'].mean(), result['extensor_activation_mean'].mean()) == .5


def test_static_force_cancellation_is_not_mechanical_work_or_equal_activation():
    activation = np.full((1,6), .6)
    forces = np.array([[100.,0.,0.,200.,0.,0.]])
    arms = np.array([[.04,.03,.02,-.01,-.02,-.03]])
    result = antagonist_metrics(activation, forces, arms, np.zeros(1))
    assert result['activation_mean'][0] == .6
    assert np.isclose(result['activation_squared_mean'][0], .36)
    assert result['net_muscle_torque_Nm'][0] == 2.
    assert result['opposing_muscle_torque_Nm'][0] == 2.
    assert result['torque_cancelled_fraction'][0] == 2/3
    assert result['joint_muscle_power_W'][0] == result['absolute_joint_muscle_power_W'][0] == 0.
