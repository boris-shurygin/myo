"""Protect the new transport, unchanged muscle dynamics, and fresh checkpoint."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from myo.arm_exploration import load_exploration
from myo.arm_timing import ArmTimingEnv, evaluation_job, load_timing, make_fresh_model
from myo.arm_workload import WorkloadSpec, save_workload
from myo.compact_protocol import initialize_command_at_reset
from myo.env import PreacherCurlEnv


def test_ten_ms_transport_is_separate_from_activation_and_queue_is_causal():
    spec = WorkloadSpec(masses=(1.,), tilts=(120.,), cases=((125., 125.),))
    low, high = ArmTimingEnv(spec), ArmTimingEnv(spec)
    try:
        low.reset(seed=0)
        high.reset(seed=0)
        assert low.unwrapped.control_dt == .01
        assert low.unwrapped.config.physics_dt == .001
        assert low.detector.dt == .01
        assert low.queue_steps == 10
        assert len(low.observation_names) == len(low.contract()['observation_scale']) == 150
        assert len(low.pending_indices) == 6
        for muscle in low.unwrapped.muscles:
            assert muscle.params.activation_time_constant == .015
            assert muscle.params.deactivation_time_constant == .05
        low.step(np.zeros(6))
        obs, _, _, _, _ = high.step(np.ones(6))
        np.testing.assert_array_equal(low.unwrapped.data.qpos, high.unwrapped.data.qpos)
        np.testing.assert_array_equal(low.unwrapped.data.qvel, high.unwrapped.data.qvel)
        np.testing.assert_array_equal(obs[high.pending_indices], np.ones(6))
        assert [m.activation for m in low.unwrapped.muscles] == [m.activation for m in high.unwrapped.muscles]
        low.step(np.zeros(6))
        high.step(np.zeros(6))
        assert all(a.activation < b.activation < 1. for a, b in zip(low.unwrapped.muscles, high.unwrapped.muscles))
        assert np.linalg.norm(low.unwrapped.data.qvel-high.unwrapped.data.qvel) > 1e-7
    finally:
        low.close()
        high.close()


def test_two_ten_ms_commands_equal_native_twenty_ms_with_same_delay():
    env = ArmTimingEnv(WorkloadSpec(masses=(1.,), tilts=(120.,), cases=((125., 125.),)))
    env.reset(seed=0)
    native = PreacherCurlEnv(replace(env.unwrapped.config, frame_skip=20))
    try:
        native.reset(seed=0, options={'q': np.deg2rad(125.), 'target_angle': np.deg2rad(125.),
                                     'payload_mass': 1., 'trigger': 1., 'capacity': 1.})
        initialize_command_at_reset(native, 125., 1.)
        for action in (np.array([.3,.1,.2,.05,.1,.15]), np.full(6,.2), np.zeros(6)):
            env.step(action)
            _, reward, _, _, info = env.step(action)
            native.step(action)
            np.testing.assert_array_equal(env.unwrapped.data.qpos, native.data.qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, native.data.qvel)
            for a, b in zip(env.unwrapped.muscles, native.muscles):
                assert (a.activation, a.fiber_length, a.capacity) == (b.activation, b.fiber_length, b.capacity)
            assert set(info['reward_components']) == {'position_log'}
            assert reward == .1*info['reward_components']['position_log']
    finally:
        env.close()
        native.close()


def test_fresh_ppo_and_timing_checkpoint_do_not_reuse_40ms_contract(tmp_path):
    torch.set_num_threads(1)
    spec = WorkloadSpec(masses=(1.,), tilts=(120.,), cases=((125.,125.),), seconds=.1)
    env = ArmTimingEnv(spec)
    model = make_fresh_model(env, env.contract(), n_steps=16, batch_size=16, n_epochs=1)
    try:
        assert not model.policy.optimizer.state
        assert model.num_timesteps == model.optimizer_steps == 0
        before = model.policy.action_net.weight.detach().clone()
        model.learn(32)
        assert not torch.equal(before, model.policy.action_net.weight)
        path = tmp_path/'timing.zip'
        save_workload(model, env, path, {'test': True})
        restored, replay, _ = load_timing(path)
        try:
            obs, _ = env.reset(seed=0)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          restored.predict(obs, deterministic=True)[0])
            assert restored.optimizer_steps == model.optimizer_steps
            assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
            result = evaluation_job(path, [{'case': (125.,125.), 'payload_mass': 1.,
                'tilt_deg': 120., 'capacity': 1., 'seconds': .1}])
            assert result['cases'][0]['steps'] == 10
            with pytest.raises(ValueError, match='Not an arm exploration'):
                load_exploration(path)
            meta_path = path.with_suffix('.zip.metadata.json')
            meta = json.loads(meta_path.read_text())
            meta['contract']['timing']['motor_delay_s'] = .02
            meta_path.write_text(json.dumps(meta))
            with pytest.raises(ValueError, match='identity mismatch'):
                load_timing(path)
        finally:
            replay.close()
    finally:
        env.close()
