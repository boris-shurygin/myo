"""Trajectory derivatives, causal deadlines and unchanged native muscle physics."""
import numpy as np
import pytest

from myo.arm_hierarchy import ArmHierarchyEnv, Motion, TRAIN_MOTIONS, HierarchySpec, upper_command
from myo.arm_workload import ArmWorkloadEnv, WorkloadSpec


def test_reference_derivatives_and_smooth_reversals():
    h = 1e-5
    for motion in TRAIN_MOTIONS:
        for t in (.213, .713, 1.313, 2.813, 4.513):
            current = motion.sample(t)
            numerical = (motion.sample(t+h)-motion.sample(t-h))/(2*h)
            np.testing.assert_allclose(current[1:], numerical[:2], atol=1e-7)
    motion = Motion('reverse', rise_s=.8, fall_s=2.4)
    for t in (.5, 1.3, 1.8, 4.2):
        np.testing.assert_allclose(motion.sample(t-1e-7), motion.sample(t+1e-7), atol=2e-5)


def test_deadlines_no_goal_input_and_exact_native_physics():
    env = ArmHierarchyEnv(motions=(Motion('test'),))
    native = ArmWorkloadEnv(WorkloadSpec(masses=(1.,), tilts=(120.,), cases=((120.,120.),)))
    try:
        obs, _ = env.reset(seed=0)
        native.reset(seed=0, options={'seconds': 6.04})
        assert len(obs) == len(env.observation_names) == 147
        assert not any('target' in n or '/command/' in n or 'trigger' in n for n in env.observation_names)
        env.tasks[:] = [11., 12., 13.]
        for i in range(3):
            velocity = float(env.unwrapped.data.qvel[env.unwrapped.dof_index])
            action = np.array([.2,.3,.1,.02,.03,.01], dtype=np.float32)
            _, _, _, _, info = env.step(action)
            native.step(action)
            np.testing.assert_array_equal(env.unwrapped.data.qpos, native.unwrapped.data.qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, native.unwrapped.data.qvel)
            assert info['hierarchy']['desired_acceleration'] == 11.+i
            assert info['hierarchy']['scored'] == (i >= 2)
            assert info['hierarchy']['issue_step'] == i-2
            assert info['hierarchy']['actual_acceleration'] == pytest.approx((info['qvel']-velocity)/.02)
        before = env.observation().copy()
        env.motion = Motion('different_unissued_future', 90., 100.)
        np.testing.assert_array_equal(before, env.observation())
        # Native task/trigger fields must not leak through history either.
        env.unwrapped.command.target = 1.
        env.unwrapped.command.requested = .2
        env.unwrapped.command.velocity = 3.
        env.unwrapped.command.trigger = .7
        for historical_command in env.unwrapped._command_history:
            historical_command[:] = [1., 2., 3., .7]
        np.testing.assert_array_equal(before, env.observation())
    finally:
        env.close()
        native.close()


def test_final_commands_are_flushed():
    env = ArmHierarchyEnv(motions=(Motion('short_hold', seconds=.1, kind='hold'),))
    try:
        env.reset(seed=0)
        due, issued = [], [env.tasks[-1]]
        while True:
            _, _, failed, done, info = env.step(np.full(6, .1, dtype=np.float32))
            assert not failed
            due.append(info['hierarchy']['desired_acceleration'])
            if env.step_index < 5:
                issued.append(env.tasks[-1])
            if done:
                break
        assert len(due) == 7
        np.testing.assert_array_equal(due[2:], issued)
    finally:
        env.close()


def test_lower_ppo_update_and_strict_checkpoint(tmp_path):
    import torch
    from myo.arm_learning import ArmPPO
    from myo.policy_beta import BetaExcitationPolicy
    from myo.arm_hierarchy import save_policy, load_policy
    torch.set_num_threads(1)
    env = ArmHierarchyEnv(motions=(Motion('short', seconds=.2, kind='hold'),))
    model = ArmPPO(BetaExcitationPolicy, env, seed=0, n_steps=16, batch_size=16, n_epochs=1,
                   policy_kwargs={'net_arch': {'pi':[16], 'vf':[16]}})
    before = model.policy.action_net.weight.detach().clone()
    model.learn(32)
    assert not torch.equal(before, model.policy.action_net.weight)
    path = tmp_path/'lower.zip'
    save_policy(model, env, path, {'test': True})
    restored, replay, _ = load_policy(path)
    try:
        obs, _ = env.reset(seed=0)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
        assert replay.contract() == env.contract()
        with path.open('ab') as stream:
            stream.write(b'bad')
        with pytest.raises(ValueError, match='mismatch'):
            load_policy(path)
    finally:
        env.close()
        replay.close()


def test_warm_start_removes_goals_but_preserves_fixed_goal_actor():
    import torch
    from dataclasses import asdict
    from myo.arm_exploration import ArmExplorationEnv, ExplorationPolicy, ExplorationSpec
    from myo.arm_hierarchy_transfer import transfer_fixed_goal_actor
    from myo.arm_learning import ArmPPO
    torch.set_num_threads(1)
    old = ArmExplorationEnv(WorkloadSpec(masses=(1.,), tilts=(120.,), cases=((120.,127.5),)))
    new = ArmHierarchyEnv()
    def model_for(env):
        return ArmPPO(ExplorationPolicy, env, seed=0, n_steps=16, batch_size=16,
            policy_kwargs={'predictor_indices': [0,1], 'physical_indices': [0,1], 'feature_mode': 'none',
                           'exploration': asdict(ExplorationSpec(mode='off'))})
    parent, model = model_for(old), model_for(new)
    try:
        source, _ = old.reset(seed=0)
        observation, _ = new.reset(seed=0)
        transfer_fixed_goal_actor(model, new, parent, old)
        for step in range(4):
            np.testing.assert_allclose(parent.predict(source, deterministic=True)[0],
                                       model.predict(observation, deterministic=True)[0], atol=2e-7)
            action = np.full(6, .13, dtype=np.float32)
            source, _, _, _, _ = old.step(action)
            observation, _, _, _, _ = new.step(action)
        assert not model.policy.optimizer.state
    finally:
        old.close()
        new.close()


def test_inverse_goal_interval_cannot_depend_on_unissued_future_actions():
    from myo.delayed_acceleration import NativeResponse, capture
    a, b = ArmHierarchyEnv(), ArmHierarchyEnv()
    try:
        a.reset(seed=0)
        b.reset(seed=0)
        first = np.array([0.,0.,0.,.3,.3,.3], dtype=np.float32)
        kernel = NativeResponse(a.unwrapped.config, horizon_steps=3)
        try:
            branch = kernel.response(capture(a.unwrapped), np.tile(first, (3,1)), first)
        finally:
            kernel.close()
        for i in range(4):
            _, _, _, _, ia = a.step(first if i == 0 else np.zeros(6, dtype=np.float32))
            _, _, _, _, ib = b.step(first if i == 0 else np.ones(6, dtype=np.float32))
            aa, ab = ia['hierarchy']['actual_acceleration'], ib['hierarchy']['actual_acceleration']
            if i <= 2:
                assert aa == ab
                assert aa == pytest.approx(branch[i], abs=1e-10)
            else:
                assert abs(aa-ab) > 1e-5
    finally:
        a.close()
        b.close()


def test_inverse_selection_prefers_native_command_execution_over_teacher_fit():
    from pathlib import Path
    import runpy
    rank = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/pretrain_arm_inverse.py'))['native_rank']
    partial = {'failures':0, 'passed':3, 'mean_position_rmse_deg':.8, 'mean_acceleration_rmse_rad_s2':.4}
    complete = dict(partial, passed=5, mean_position_rmse_deg=1.2)
    failed = dict(complete, failures=1, mean_position_rmse_deg=.1)
    assert rank(complete) < rank(partial) < rank(failed)


def test_trajectory_display_reference_does_not_change_native_command_or_state():
    from pathlib import Path
    import runpy
    display = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/demo_arm_hierarchy.py'))['TrajectoryDisplay']
    env = ArmHierarchyEnv()
    try:
        env.reset(seed=0, options={'motion':Motion('display',120.,135.)})
        before = env.unwrapped.observe().copy()
        proxy = display(env)
        env.motion = Motion('display_other',125.,135.)
        assert proxy.telemetry()['target_q'] == pytest.approx(np.deg2rad(125.))
        assert proxy.config.task_mode == 'reference'
        assert env.unwrapped.config.task_mode != 'reference'
        np.testing.assert_array_equal(before, env.unwrapped.observe())
    finally:
        env.close()
