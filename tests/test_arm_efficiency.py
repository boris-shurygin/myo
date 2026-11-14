"""Protect reward attribution, physical equivalence and quality-aware selection."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from myo.arm_efficiency import (ArmEfficiencyEnv, EfficiencyReward, load_efficiency,
    quality_gate, velocity_cost)
from myo.arm_timing import ArmTimingEnv, make_fresh_model
from myo.arm_workload import WorkloadSpec, save_workload


def test_stability_and_effort_use_real_state_without_changing_delayed_physics(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(.5,), tilts=(30.,), cases=((75.,75.),), seconds=.1)
    reward = EfficiencyReward(activation_squared_weight=.1)
    native, adapted = ArmTimingEnv(spec), ArmEfficiencyEnv(spec, reward=reward)
    try:
        np.testing.assert_array_equal(native.reset(seed=0)[0], adapted.reset(seed=0)[0])
        for command in (np.ones(6), np.zeros(6), np.full(6,.4)):
            obs, old_reward, old_fail, old_done, old = native.step(command)
            other, value, fail, done, info = adapted.step(command)
            np.testing.assert_array_equal(obs, other)
            assert (fail, done) == (old_fail, old_done)
            for name in ('q', 'qvel', 'activation', 'delayed_excitation', 'tendon_force_N', 'capacity'):
                np.testing.assert_array_equal(info[name], old[name])
            cost = -.1*np.mean(np.square(info['activation']))
            assert info['reward_components']['activation_squared_proxy_cost'] == pytest.approx(cost)
            assert value-old_reward == pytest.approx(.1*(cost+velocity_cost(
                info['q']-info['requested_target_q'], info['qvel'], reward)))
        model = make_fresh_model(adapted, adapted.contract(), n_steps=16, batch_size=16, n_epochs=1)
        model.learn(16)
        path = tmp_path/'efficiency.zip'
        save_workload(model, adapted, path, {'test': True})
        restored, env, _ = load_efficiency(path)
        try:
            obs = adapted.reset(seed=0)[0]
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          restored.predict(obs, deterministic=True)[0])
            assert env.contract() == adapted.contract()
            assert restored.optimizer_steps == model.optimizer_steps
            assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
        finally:
            env.close()
    finally:
        native.close()
        adapted.close()


def test_velocity_reference_and_gate_cannot_reward_jitter_or_sacrifice_individual_cases():
    reward = EfficiencyReward()
    assert velocity_cost(0., .3, reward, reference_velocity_rad_s=.3) == 0.
    assert velocity_cost(0., .3, reward) < velocity_cost(np.deg2rad(30), .3, reward) < 0.
    row = {'case': (55.,75.), 'payload_mass': .5, 'tilt_deg': 30., 'capacity': 1., 'seconds': 4.,
        'failure': False, 'passed': True, 'tail_rmse_deg': .5, 'tail_velocity_rms_deg_s': 1.,
        'settling_time_s': 1., 'overshoot_deg': .5}
    baseline = {'cases': [row, {**row, 'case': (95.,75.)}]}
    assert quality_gate(baseline, baseline) == (True, [])
    jitter = deepcopy(baseline)
    jitter['cases'][0].update(tail_rmse_deg=.1, tail_velocity_rms_deg_s=7.)
    assert not quality_gate(jitter, baseline)[0]
    slow = deepcopy(baseline)
    slow['cases'][0]['settling_time_s'] = 1.21
    assert not quality_gate(slow, baseline)[0]
