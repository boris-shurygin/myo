"""Future acceleration tasks must shift causally and retain native muscle timing."""
import numpy as np
import pytest

from myo.vertical_control import VerticalControlEnv, load_policy, save_policy
from myo.vertical_muscles import VerticalConfig, VerticalMuscles
from myo.vertical_scheduled import (
    AccelerationCase, ScheduledAccelerationEnv, VALIDATION_CASES,
    evaluate_scheduled, load_scheduled_policy,
)


def test_issued_acceleration_is_scored_only_in_its_future_interval_with_native_delay():
    config = VerticalConfig(variant=4)
    env, native = ScheduledAccelerationEnv(config), VerticalMuscles(config)
    obs, _ = env.reset(options={"case": AccelerationCase("timing", .7, 5)})
    native.reset()
    np.testing.assert_array_equal(obs[-3:], [0., 0., np.float32(.7)])
    for index in range(3):
        velocity = native.velocity
        obs, _, terminated, truncated, info = env.step(np.array([.4, .05]))
        np.testing.assert_array_equal(env.physical.state(), native.step(np.array([.4, .05])))
        assert info["desired_acceleration_m_s2"] == (0. if index < 2 else .7)
        assert info["task_issue_step"] == index-2
        assert info["actual_acceleration_m_s2"] == pytest.approx((native.velocity-velocity)/config.dt)
        assert info["scored"] == (index >= 2)
        assert not terminated and not truncated
    # A change to an unissued future profile cannot change the current input.
    before = env.observation().copy()
    env.commands[env.step_index+1:] *= -1
    np.testing.assert_array_equal(before, env.observation())


def test_last_issued_tasks_are_flushed_and_goal_blind_hold_does_not_pass_commands():
    env = ScheduledAccelerationEnv(VerticalConfig(variant=1))
    case = AccelerationCase("flush", .5, 5)
    env.reset(options={"case": case})
    issued = env.commands.copy()
    # End in a nonzero command to detect premature truncation.
    env.commands[-1] = .75
    due = []
    for _ in range(len(issued)+2):
        force = env.physical.config.mass_kg*(9.81+env.tasks[0])
        action = env.physical.command_for_force(force, 5.)
        _, _, failed, truncated, info = env.step(action)
        assert not failed
        due.append(info["desired_acceleration_m_s2"])
    assert truncated and len(due) == len(issued)+2
    np.testing.assert_array_equal(due[2:], env.commands)

    class Constant:
        def predict(self, observation, deterministic=True):
            return np.array([.185125, .0625]), None
    result = evaluate_scheduled(Constant(), VerticalConfig(variant=1), VALIDATION_CASES)
    assert result["passed"] < result["total"] and not result["ready"]


def test_scheduled_ppo_update_and_checkpoint_cannot_reuse_position_policy(tmp_path):
    torch = pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from gymnasium.utils.env_checker import check_env
    from myo.policy_beta import BetaExcitationPolicy
    torch.set_num_threads(1)
    env = ScheduledAccelerationEnv(VerticalConfig(variant=3))
    check_env(env, skip_render_check=True)
    model = sb3.PPO(BetaExcitationPolicy, env, seed=0, n_steps=32, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}})
    model.learn(64)
    path = tmp_path/"scheduled.zip"
    save_policy(model, env, path, {"task": "acceleration_due_in_40ms"})
    restored, restored_env, _ = load_scheduled_policy(path)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
    assert restored_env.contract() == env.contract()
    with pytest.raises(ValueError, match="contract mismatch"):
        load_policy(path, VerticalControlEnv(VerticalConfig(variant=3)))


def test_one_zero_task_does_not_hide_lower_error_on_acceleration_profiles():
    pytest.importorskip("torch")
    from myo.train_vertical_scheduled import validation_rank
    zero_only = {"failures": 0, "ready": False, "passed": 1, "score_m_s2": .414}
    better_motion = {"failures": 0, "ready": False, "passed": 0, "score_m_s2": .399}
    all_ready = {"failures": 0, "ready": True, "passed": 5, "score_m_s2": .24}
    assert validation_rank(better_motion) < validation_rank(zero_only)
    assert validation_rank(all_ready) < validation_rank(better_motion)
