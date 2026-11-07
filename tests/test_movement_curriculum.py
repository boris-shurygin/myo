"""Cold movement exercises must escape support and maintain reached positions."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myo.compact_protocol import compact_scenarios, initialize_command_at_reset
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.evaluate import CommandEvent, Scenario
from myo.movement_curriculum import MovementProtocol, stage_cases, summarize_movement_samples


def test_movement_curriculum_is_cold_causal_and_preserves_continuous_physics(monkeypatch):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets to check physical movement curriculum")
    assert stage_cases(5, 1) == compact_scenarios(1)
    for stage in range(5):
        cases = stage_cases(stage, 1)
        assert {(s.commands[0].trigger, s.capacity) for s in cases} == {
            (u, c) for u in (0.3, 0.6, 1.0) for c in (1.0, 0.8)}
        assert all(s.initial_angle_deg == 20 for s in cases)
        assert max(s.duration_s for s in cases) <= 24
    config = PreacherCurlConfig(geometry="myoarm", task_mode="curriculum", episode_seconds=0.04)
    env = MovementProtocol(PreacherCurlEnv(config), 2)
    reference = PreacherCurlEnv(replace(config, episode_seconds=10))
    # Shorten only this test's command times; the production schedule is checked above.
    case = Scenario("continuous", 0.12, (CommandEvent(0, 35, 0.3), CommandEvent(0.04, 20, 0.3),
                    CommandEvent(0.08, 35, 0.3)), 1, initial_angle_deg=20)
    env.scenarios = (case,)
    try:
        def no_teacher():
            raise AssertionError("Cold movement curriculum must never call a teacher")
        monkeypatch.setattr(env.unwrapped, "baseline_action", no_teacher)
        observation, info = env.reset(seed=0)
        reference.reset(seed=0, options={"q": np.deg2rad(20), "qvel": 0,
            "target_angle": np.deg2rad(35), "trigger": 0.3, "payload_mass": 1, "capacity": 1,
            "reference_phase": 0, "reference_amplitude": 0, "reference_frequency": 0})
        initialize_command_at_reset(reference, 35, 0.3)
        np.testing.assert_array_equal(observation, reference.observe())
        assert info["q"] == pytest.approx(np.deg2rad(20))
        for age in range(1, config.history_steps + 1):
            idx = env.unwrapped.observation_names.index(f"history/{age}/command/target_angle_centered_rad")
            assert observation[idx] == pytest.approx(np.deg2rad(35) - config.reference_center)
        for step in range(6):
            prior_goal = reference.command.requested
            action = np.full(env.action_space.shape, 0.1)
            observation, reward, terminated, truncated, info = env.step(action)
            reference.step(action)
            assert info["requested_target_q"] == prior_goal
            assert reward == sum(info["reward_components"].values())
            assert not terminated
            assert truncated == (step == 5)
            if not truncated:
                event = max((e for e in case.commands if e.time_s <= reference.time + 1e-10),
                            key=lambda e: e.time_s)
                reference.set_command(np.deg2rad(event.target_deg), event.trigger)
            np.testing.assert_array_equal(observation, reference.observe())
            np.testing.assert_array_equal(env.unwrapped.data.qpos, reference.data.qpos)
        assert env.unwrapped.time == pytest.approx(0.12)
    finally:
        env.close()
        reference.close()


def test_movement_gate_rejects_support_flythrough_and_missing_command_window():
    case = stage_cases(0, 1)[0]
    dt = 0.02
    def samples(angle, speed=0):
        return [{"time_s": (i + 1) * dt, "angle_deg": float(angle), "speed_deg_s": float(speed),
                 "target_deg": 30.0, "joint_reaction_Nm": 0.0, "contact_force_N": 0.0,
                 "excitation": [0.0] * 6, "reward": 0.0} for i in range(200)]
    held = samples(30)
    assert summarize_movement_samples(case, held, dt)["passed"]
    assert not summarize_movement_samples(case, samples(17), dt)["passed"]
    assert not summarize_movement_samples(case, samples(30, 30), dt)["passed"]
    # Reaching the target in the last sample alone is not a one-second hold.
    crossing = samples(17)
    crossing[-1]["angle_deg"] = 30
    assert not summarize_movement_samples(case, crossing, dt)["passed"]
    assert not summarize_movement_samples(case, held[:-1], dt)["passed"]
    assert not summarize_movement_samples(case, held, dt, terminated=True)["passed"]
    cycle = Scenario("reversal", 8, (CommandEvent(0, 30, 1), CommandEvent(4, 20, 1)), 1,
                     initial_angle_deg=20)
    final_window = [{**row, "time_s": row["time_s"] + 4, "angle_deg": 20.0, "target_deg": 20.0}
                    for row in held]
    failed_first = summarize_movement_samples(cycle, samples(17) + final_window, dt)
    assert failed_first["windows"][1]["passed"]
    assert not failed_first["passed"]
