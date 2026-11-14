"""Read-only antagonist metrics at one common physical timestamp."""
from __future__ import annotations

import numpy as np


def antagonist_metrics(activation, tendon_force, moment_arm, velocity):
    """Rows are time samples; the six anatomical channels are flexors then extensors.

    Torque cancellation uses actual signed r*F, not activation or group labels.
    Compute overlap before time averaging: alternating antagonist bursts must not
    be mistaken for simultaneous coactivation. These are engineering proxies.
    """
    activation = np.asarray(activation, dtype=float)
    torque = np.asarray(tendon_force, dtype=float)*np.asarray(moment_arm, dtype=float)
    flex = activation[:, :3].mean(axis=1)
    ext = activation[:, 3:].mean(axis=1)
    positive = np.maximum(torque, 0.).sum(axis=1)
    negative = np.maximum(-torque, 0.).sum(axis=1)
    absolute = positive+negative
    opposing = np.minimum(positive, negative)
    net = positive-negative
    cancelled_fraction = np.divide(2*opposing, absolute, out=np.zeros_like(absolute), where=absolute > 1e-12)
    return {
        'activation_mean': activation.mean(axis=1),
        'activation_squared_mean': np.square(activation).mean(axis=1),
        'flexor_activation_mean': flex,
        'extensor_activation_mean': ext,
        'activation_coactivation': np.minimum(flex, ext),
        'positive_muscle_torque_Nm': positive,
        'negative_muscle_torque_abs_Nm': negative,
        'net_muscle_torque_Nm': net,
        'opposing_muscle_torque_Nm': opposing,
        'torque_cancelled_fraction': cancelled_fraction,
        'summed_tendon_force_proxy_N': np.asarray(tendon_force, dtype=float).sum(axis=1),
        'joint_muscle_power_W': net*np.asarray(velocity),
        'absolute_joint_muscle_power_W': absolute*np.abs(np.asarray(velocity)),
    }
