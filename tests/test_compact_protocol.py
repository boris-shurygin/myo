"""Fixed exercises preserve physical continuity; fast crossings are not holds."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myo.compact_protocol import CompactProtocol, compact_scenarios, initialize_command_at_reset, result_reward
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.evaluate import CommandEvent, Scenario, command_response_metrics


def test_fixed_protocol_cycles_conditions_and_preserves_physics_across_commands(monkeypatch):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets to check the physical protocol")
    cases = compact_scenarios(1.0)
    assert len(cases) == 18
    assert {(s.commands[0].trigger, s.capacity) for s in cases} == {
        (u, c) for u in (0.3, 0.6, 1.0) for c in (1.0, 0.8)}
    sweep = cases[0]
    angles = [e.target_deg for e in sweep.commands]
    assert angles == list(range(15, 150, 10)) + [150] + list(range(145, 14, -10))
    assert max(s.duration_s for s in cases) == 29
    config = PreacherCurlConfig(geometry="myoarm", task_mode="curriculum", episode_seconds=0.04)
    env = CompactProtocol(PreacherCurlEnv(config))
    reference = PreacherCurlEnv(replace(config, episode_seconds=10))
    try:
        for s in cases:
            observation, info = env.reset(seed=7)
            assert info["protocol_scenario"] == s.name
            assert env.unwrapped.payload_mass == 1.0
            np.testing.assert_allclose(info["capacity"], s.capacity)
            assert env.unwrapped.command.target == np.deg2rad(s.initial_angle_deg)
            for age in range(1, config.history_steps + 1):
                name = f"history/{age}/command/target_angle_centered_rad"
                np.testing.assert_allclose(observation[env.unwrapped.observation_names.index(name)],
                                           np.deg2rad(s.initial_angle_deg) - config.reference_center)
        env.reset(seed=7)  # Wrap to the same first scenario, not a random draw.
        reference.reset(seed=7, options={"q": np.deg2rad(15), "qvel": 0,
            "target_angle": np.deg2rad(15), "trigger": 0.3, "payload_mass": 1,
            "capacity": 1, "reference_phase": 0, "reference_amplitude": 0, "reference_frequency": 0})
        initialize_command_at_reset(reference, 15, 0.3)
        def no_teacher():
            raise AssertionError("Result-based protocol must not call the teacher")
        monkeypatch.setattr(env.unwrapped, "baseline_action", no_teacher)
        action = np.full(env.action_space.shape, 0.1)
        for _ in range(105):
            old_goal = reference.command.requested
            observation, reward, terminated, truncated, info = env.step(action)
            reference.step(action)
            assert not terminated and not truncated
            assert info["requested_target_q"] == old_goal
            assert reward == sum(info["reward_components"].values())
            event = max((e for e in sweep.commands if e.time_s <= reference.time + 1e-10), key=lambda e: e.time_s)
            reference.set_command(np.deg2rad(event.target_deg), event.trigger)
            np.testing.assert_array_equal(observation, reference.observe())
            np.testing.assert_array_equal(env.unwrapped.data.qpos, reference.data.qpos)
        assert env.unwrapped.time > 2
        assert env.unwrapped.command.requested == np.deg2rad(35)
    finally:
        env.close()
        reference.close()


def test_result_reward_prefers_early_accurate_arrival_over_flythrough_or_compression():
    def info(q, velocity=0, force=0):
        return {"q": np.deg2rad(q), "qvel": velocity, "requested_target_q": np.deg2rad(75),
                "trigger": 0.3, "substep_peaks": {"tissue_contact_force_N": force},
                "reward_components": {"position_tracking": 999, "velocity_tracking": 999,
                                      "excitation_squared_cost": -0.01, "physical_failure": 0}}
    settled, terms = result_reward(info(75))
    crossing, _ = result_reward(info(75, 2))
    distant, _ = result_reward(info(20))
    closer, _ = result_reward(info(40))
    compressed, _ = result_reward(info(75, force=100))
    assert settled > crossing and settled > compressed and closer > distant
    assert "position_tracking" not in terms and terms["excitation_squared_cost"] == -0.01
    # Same final pose, but reaching it earlier earns more over the trajectory.
    assert 5 * distant + 15 * settled > 15 * distant + 5 * settled


def test_preparation_handoff_preserves_physics_and_never_calls_teacher_during_actor_steps(monkeypatch):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare local MyoArm assets to check handoff")
    from myo.prepared_protocol import PreparedProtocol

    config = PreacherCurlConfig(geometry="myoarm", task_mode="curriculum")
    case = Scenario("handoff", 0.08, (CommandEvent(0, 20, 1), CommandEvent(0.04, 30, 1)),
                    1, initial_angle_deg=20)
    env = PreparedProtocol(PreacherCurlEnv(config), [case], preparation_seconds=0.06)
    reference = PreacherCurlEnv(config)
    try:
        obs, info = env.reset(seed=0)
        reference.reset(seed=0, options={"q": np.deg2rad(20), "qvel": 0,
            "target_angle": np.deg2rad(20), "trigger": 1, "payload_mass": 1, "capacity": 1,
            "reference_phase": 0, "reference_amplitude": 0, "reference_frequency": 0})
        initialize_command_at_reset(reference, 20, 1)
        for _ in range(3):
            reference.step(reference.baseline_action())
        np.testing.assert_array_equal(obs, reference.observe())
        assert info["preparation"]["handoff_time_s"] == pytest.approx(0.06)
        assert min(info["preparation"]["handoff_capacity"]) < 1

        def no_teacher():
            raise AssertionError("Teacher must stop at policy handoff")
        monkeypatch.setattr(env.unwrapped, "baseline_action", no_teacher)
        action = np.full(env.action_space.shape, 0.1)
        for _ in range(2):
            obs, _, terminated, truncated, info = env.step(action)
            reference.step(action)
            assert not terminated and not truncated
            assert info["requested_target_q"] == np.deg2rad(20)
            np.testing.assert_array_equal(env.unwrapped.data.qpos, reference.data.qpos)
        assert env.unwrapped.time == pytest.approx(0.10)
        assert env.unwrapped.command.requested == np.deg2rad(30)
    finally:
        env.close()
        reference.close()


def test_command_metrics_distinguish_reaching_settling_and_unfinished_windows():
    scenario = Scenario("raise", 1.0, (CommandEvent(0, 75, 1),), 1, initial_angle_deg=20)
    samples = [({"t": (i + 1) * 0.1, "q": np.deg2rad(q), "qvel": v,
                 "substep_peaks": {"tissue_contact_force_N": 0, "tissue_contact_penetration_m": 0}}, [])
               for i, (q, v) in enumerate([(40, 1), (75, 1), (85, 0), (78, -1)] + [(75, 0)] * 6)]
    result = command_response_metrics(scenario, samples, 0.1)[0]
    assert result["first_reach_time_s"] == 0.2
    assert result["settling_time_s"] == 0.5
    np.testing.assert_allclose(result["overshoot_deg"], 10)
    assert command_response_metrics(scenario, samples[:-1], 0.1)[0]["settling_time_s"] is None


def test_plateau_requires_repeated_stagnation_and_counts_accumulated_progress():
    from myo.continue_compact import Plateau

    stop = Plateau(50)
    assert not stop.update(49.6)
    assert not stop.update(49.2)
    assert not stop.update(48.8)  # Combined gain is significant; patience resets.
    assert stop.stale == 0
    assert not stop.update(49)
    assert not stop.update(50)
    assert stop.update(48.7)


def test_demo_gate_does_not_hide_one_failed_direction_with_good_average():
    from myo.continue_compact import behavior_summary

    rows = []
    for scenario in compact_scenarios(1):
        rows.append({"scenario": {"name": scenario.name}, "physical_failure": False,
            "numerical_failure": None, "completed_duration": True,
            "overall": {"substep_peaks": {"tissue_contact_force_N": 50, "joint_limit_reaction_Nm": 0}},
            "final_angle_deg": scenario.commands[-1].target_deg,
            "hold_tail_last_5s": {"angle_std_deg": 0.1},
            "command_responses": [{"target_deg": e.target_deg, "final_error_deg": 0.1,
                "requested_angle_rmse_deg": 1, "settling_time_s": 1, "overshoot_deg": 0}
                for e in scenario.commands]})
    assert behavior_summary(rows)["demo_ready"]
    rows[-1]["command_responses"][-1]["settling_time_s"] = None
    result = behavior_summary(rows)
    assert result["moves_passed"] == 11 and not result["demo_ready"]
    rows[-1]["physical_failure"] = True
    assert behavior_summary(rows)["score"] > result["score"]


def test_compact_resume_does_not_silently_drop_reward_scaling(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    pytest.importorskip("stable_baselines3")
    from myo import continue_compact
    from myo.compact_protocol import protocol_metadata

    parent = {"config": {"geometry": "myoarm", "task_mode": "curriculum"},
              "training_protocol": json.loads(json.dumps(protocol_metadata(1.0))),
              "reward_scale": 0.01}
    monkeypatch.setattr(continue_compact, "load_controller", lambda _: (
        SimpleNamespace(close=lambda: None), None, parent))
    with pytest.raises(ValueError, match="reward scale"):
        continue_compact.main(["--controller", "scaled.zip", "--output-dir", str(tmp_path / "resume")])


def test_unit_sde_rescaling_features_preserves_sample_and_likelihood():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from myo.policy_exploration import UnitLatentNoise

    dist = UnitLatentNoise(6)
    _, log_std = dist.proba_distribution_net(128, log_std_init=-0.6)
    latent = torch.linspace(-1, 1, 128)[None, :]
    mean = torch.full((1, 6), 0.1)
    dist.proba_distribution(mean, log_std, latent)
    sample = dist.sample().detach().clone()
    std = dist.distribution.stddev.detach().clone()
    likelihood = dist.log_prob(sample).detach().clone()
    dist.proba_distribution(mean, log_std, 100 * latent)
    torch.testing.assert_close(dist.sample(), sample)
    torch.testing.assert_close(dist.distribution.stddev, std)
    torch.testing.assert_close(dist.log_prob(sample), likelihood)
