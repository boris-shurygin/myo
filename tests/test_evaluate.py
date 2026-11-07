"""A release comparison must preserve its prelude and refresh causal input."""

from dataclasses import replace

import numpy as np

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.evaluate import CommandEvent, Scenario, run_scenario


def test_release_branches_share_state_and_controller_sees_new_trigger():
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive", randomize_reference=False,
                                           reference_amplitude=0.0, episode_seconds=0.2))
    seen = []

    def act(observation):
        np.testing.assert_array_equal(observation, env.observe())
        seen.append(env.command.trigger)
        return np.full(env.action_space.shape, 0.2, dtype=np.float32)

    scenario = Scenario("release", 0.12, (CommandEvent(0, 75, 1), CommandEvent(0.06, 75, 0)),
                        1.0, branch_time_s=0.06)
    try:
        release = run_scenario(env, act, scenario)
        zero = run_scenario(env, act, replace(scenario, name="forced_zero", forced_zero_time_s=0.06))
        assert release["branch_state"]["sha256"] == zero["branch_state"]["sha256"]
        assert 0.0 in seen
        assert release["after_branch"]["tracking_samples"] == 0
        assert release["after_branch"]["angle_rmse_deg"] is None
        assert release["after_branch"]["trigger_excess_mean"] > 0.19
        assert zero["after_branch"]["trigger_excess_mean"] == 0.0
        np.testing.assert_allclose(release["after_branch"]["mean_excitation_by_muscle"], 0.2)
        np.testing.assert_array_equal(zero["after_branch"]["mean_excitation_by_muscle"], 0.0)
        assert release["trace_1hz"][-1]["trigger"] == 0.0
        assert release["trace_1hz"][-1]["time_s"] == env.time
    finally:
        env.close()


def test_long_hold_explicitly_crosses_time_limit_without_resetting_state():
    config = PreacherCurlConfig(task_mode="interactive", randomize_reference=False,
                               reference_amplitude=0.0, episode_seconds=0.04)
    env = PreacherCurlEnv(config)
    uninterrupted = PreacherCurlEnv(replace(config, episode_seconds=1.0))
    scenario = Scenario("hold", 0.12, (CommandEvent(0, 75, 1),), 1.0)
    action = lambda observation: np.full(env.action_space.shape, 0.2, dtype=np.float32)
    try:
        ordinary = run_scenario(env, action, scenario)
        assert ordinary["episode_truncated"] and not ordinary["completed_duration"]
        assert ordinary["overall"]["samples"] == 2

        extended = run_scenario(env, action, scenario, continue_after_time_limit=True)
        reference = run_scenario(uninterrupted, action, scenario)
        assert extended["completed_duration"] and not extended["physical_failure"]
        assert extended["overall"]["samples"] == reference["overall"]["samples"] == 6
        assert extended["hold_tail_last_5s"]["samples"] == 6
        assert min(extended["final_capacity"]) < 1.0
        # The full public observation covers capacities, activation, lengths,
        # pending commands and sensor/action history after crossing the limit.
        np.testing.assert_array_equal(env.observe(), uninterrupted.observe())
        np.testing.assert_array_equal(env.data.qpos, uninterrupted.data.qpos)
        np.testing.assert_array_equal(env.data.qvel, uninterrupted.data.qvel)
        assert env.time == uninterrupted.time
    finally:
        env.close()
        uninterrupted.close()
