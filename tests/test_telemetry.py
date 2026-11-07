"""Renderer-facing units and separation of command, activation and force."""

import json

import numpy as np
import pytest

from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.runtime import config_digest, dynamics_digest
from myo.telemetry import encode_packet, make_packet


def test_packet_roundtrip_keeps_si_units_schema_and_actual_minus_target_error():
    env = PreacherCurlEnv()
    try:
        env.reset(seed=3, options={"q": 1.2, "qvel": 0.3})
        before = (env.time, env.data.qpos.copy(), env.data.qvel.copy())
        packet = make_packet(env, sequence=4, episode=2, run_id="проверка")
        decoded = json.loads(encode_packet(packet))
        assert decoded == packet
        assert env.time == before[0]
        np.testing.assert_array_equal(env.data.qpos, before[1])
        np.testing.assert_array_equal(env.data.qvel, before[2])
        assert packet["schema"] == "myo.character_state.v2"
        assert packet["config_sha256"] == config_digest(env.config)
        assert packet["dynamics_sha256"] == dynamics_digest()
        assert packet["coordinates"] == {
            "handedness": "right", "up": "Z", "length_unit": "m", "angle_unit": "rad",
        }
        assert packet["joints"][0]["angle_rad"] == 1.2
        assert packet["joints"][0]["angular_velocity_rad_s"] == 0.3
        assert packet["physics_dt_s"] == env.config.physics_dt
        assert packet["control_dt_s"] == env.control_dt
        assert packet["metrics"]["tracking_error_rad"] == pytest.approx(
            packet["joints"][0]["angle_rad"] - packet["joints"][0]["target_angle_rad"]
        )
        assert [muscle["name"] for muscle in packet["muscles"]] == list(env.muscle_names)
        assert all(0.04 < muscle["fiber_length_m"] < 0.25 for muscle in packet["muscles"])
        assert "summed_tendon_force_proxy_N" in packet["metrics"]
        assert "joint_contact_force_N" not in packet["metrics"]
    finally:
        env.close()


def test_passive_force_remains_distinct_from_zero_activation():
    env = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0.0))
    try:
        env.reset(seed=4, options={"q": float(np.deg2rad(20.0)), "qvel": 0.0})
        packet = make_packet(env, 0, 0, "passive")
        assert all(muscle["activation"] == 0 for muscle in packet["muscles"])
        assert max(muscle["tendon_force_N"] for muscle in packet["muscles"]) > 1.0
        for muscle in packet["muscles"]:
            assert muscle["force_normalized"] == pytest.approx(
                muscle["tendon_force_N"] / muscle["reference_max_force_N"]
            )
        assert packet["metrics"]["summed_tendon_force_proxy_N"] == pytest.approx(
            sum(muscle["tendon_force_N"] for muscle in packet["muscles"])
        )
    finally:
        env.close()


def test_command_delay_and_activation_are_separate_render_signals():
    env = PreacherCurlEnv(PreacherCurlConfig(initial_activation=0.0, motor_delay=0.04))
    try:
        env.reset(seed=5, options={"q": float(np.deg2rad(75.0))})
        env.step(np.full(4, 0.8, dtype=np.float32))
        packet = make_packet(env, 1, 0, "delay")
        assert packet["simulation_time_s"] == pytest.approx(0.02)
        for muscle in packet["muscles"]:
            assert muscle["excitation_command"] == pytest.approx(0.8)
            assert muscle["excitation_delayed"] == 0.0
            assert muscle["activation"] == 0.0
            assert muscle["remaining_capacity"] == 1.0
    finally:
        env.close()


def test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation():
    import socket
    from myo.control import LoopbackCommandReceiver
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode='interactive', initial_activation=0.4))
    receiver = LoopbackCommandReceiver(0, timeout=0.5)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        env.reset(seed=2)
        def send(sequence, episode, trigger):
            sender.sendto(json.dumps({'schema': 'myo.character_command.v1', 'run_id': 'test',
                                     'episode': episode, 'sequence': sequence,
                                     'target_angle_rad': 1.6, 'trigger': trigger}).encode(),
                          receiver.socket.getsockname())
        send(2, 0, 0.7)
        assert receiver.poll(env, 'test', 0, now=1.0)
        send(1, 0, 0.9)
        send(99, 1, 1.0)
        assert not receiver.poll(env, 'test', 0, now=1.2)
        assert env.command.trigger == 0.7
        before = env.observe().copy()
        assert receiver.poll(env, 'test', 0, now=1.6)
        assert env.command.trigger == 0.0 and receiver.timed_out
        assert env.time == 0
        assert [m.activation for m in env.muscles] == [0.4] * 4
        assert all(np.all(m.delay.pending == 0.4) for m in env.muscles)
        # Only the current command changes; no history or sensor is fabricated.
        after = env.observe()
        differing = np.flatnonzero(before != after)
        assert [env.observation_names[i] for i in differing] == ['trigger']
    finally:
        sender.close(); receiver.close(); env.close()
