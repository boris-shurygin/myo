"""Ten-second checks of the split muscle/rigid-body integration.

The control period and physical motor delay stay fixed while physics dt is
halved. These are convergence checks for two specified V0 trajectories, not
stability claims for arbitrary models, loads, controllers, or step sizes.
"""

from dataclasses import replace

import numpy as np
import pytest

from myo.env import PreacherCurlConfig, PreacherCurlEnv


CONTROL_DT = 0.02
MOTOR_DELAY = 0.04
FIXED_ACTION = np.array([0.15, 0.15, 0.15, 0.04], dtype=np.float32)
RESET_OPTIONS = {
    "q": float(np.deg2rad(75)),
    "qvel": 0.0,
    "reference_phase": 0.0,
    "reference_amplitude": float(np.deg2rad(30)),
    "reference_frequency": 0.18,
}


def _rollout(physics_dt, mode):
    config = replace(
        PreacherCurlConfig(), physics_dt=physics_dt,
        frame_skip=round(CONTROL_DT / physics_dt),
        episode_seconds=10.0, motor_delay=MOTOR_DELAY,
    )
    env = PreacherCurlEnv(config)
    last_path_lengths = np.zeros(4)
    last_path_velocities = np.zeros(4)

    # Capture the actual kinematics supplied to the last fiber substep.
    # Final telemetry q is after mj_step2; its forces were computed before
    # that rigid-body update. Comparing F against L(q_new) would be incorrect.
    for index, muscle in enumerate(env.muscles):
        original_step = muscle.step

        def capture(excitation, mt_length, mt_velocity=0.0, dt=None,
                    original_step=original_step, index=index):
            last_path_lengths[index] = mt_length
            last_path_velocities[index] = mt_velocity
            return original_step(excitation, mt_length, mt_velocity, dt)

        muscle.step = capture

    trace = {key: [] for key in ("t", "q", "qvel", "force", "activation", "capacity", "delayed")}
    maximum_geometry_offset = 0.0
    try:
        _, initial = env.reset(seed=31, options=RESET_OPTIONS)
        initial_state = [initial["q"], initial["qvel"], *initial["activation"],
                         *initial["fiber_length_m"], *initial["capacity"]]
        assert env.control_dt == pytest.approx(CONTROL_DT)
        for muscle in env.muscles:
            assert len(muscle.delay.pending) * physics_dt == pytest.approx(MOTOR_DELAY)
        for control_step in range(500):
            action = env.baseline_action() if mode == "baseline" else FIXED_ACTION
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated, info.get("termination_reason")
            assert truncated == (control_step == 499)
            outputs = env._muscle_outputs
            lengths_from_outputs = np.array([o.fiber_length + o.tendon_length for o in outputs])
            np.testing.assert_allclose(lengths_from_outputs, last_path_lengths, atol=1e-14, rtol=0)

            # Virtual work: MT shortening power must have the same sign and
            # magnitude as the torque delivered to the joint, at the same time.
            joint_velocities = -last_path_velocities / env.moment_arms
            np.testing.assert_allclose(joint_velocities, joint_velocities[0], atol=1e-12, rtol=0)
            assert sum(o.mt_power for o in outputs) == pytest.approx(
                info["muscle_torque_Nm"] * joint_velocities[0], abs=1e-10
            )
            energies_and_dissipation = np.array([
                [o.tendon_energy, o.passive_energy, o.damping_power] for o in outputs
            ])
            assert np.isfinite(energies_and_dissipation).all()
            assert (energies_and_dissipation >= 0).all()
            assert max(abs(value) for value in info["equilibrium_residual_N"]) < 1e-6

            maximum_geometry_offset = max(
                maximum_geometry_offset,
                float(abs(lengths_from_outputs - env.muscle_lengths()).max()),
            )
            for key, info_key in (
                ("t", "t"), ("q", "q"), ("qvel", "qvel"),
                ("force", "tendon_force_N"), ("activation", "activation"),
                ("capacity", "capacity"), ("delayed", "delayed_excitation"),
            ):
                trace[key].append(info[info_key])
        trace = {key: np.asarray(values) for key, values in trace.items()}
        assert all(np.isfinite(values).all() for values in trace.values())
        return trace, np.array(initial_state), maximum_geometry_offset
    finally:
        env.close()


def _check_pair(coarse_result, fine_result):
    coarse, coarse_initial, coarse_offset = coarse_result
    fine, fine_initial, fine_offset = fine_result
    np.testing.assert_array_equal(coarse_initial, fine_initial)
    np.testing.assert_allclose(coarse["t"], fine["t"], atol=5e-11, rtol=0)
    assert coarse["t"][-1] == pytest.approx(10.0)
    # Explicit absolute tolerances in SI units, comfortably above the measured
    # numerical differences. These are not physiological accuracy thresholds.
    np.testing.assert_allclose(coarse["q"], fine["q"], atol=0.003, rtol=0)
    np.testing.assert_allclose(coarse["qvel"], fine["qvel"], atol=0.02, rtol=0)
    np.testing.assert_allclose(coarse["force"], fine["force"], atol=0.5, rtol=0)
    np.testing.assert_allclose(coarse["capacity"], fine["capacity"], atol=1e-4, rtol=0)
    # The old-length/new-q telemetry offset should shrink with the split step.
    assert fine_offset <= 0.7 * coarse_offset + 1e-8
    return coarse, fine


def test_baseline_coupling_converges_with_fixed_control_period_and_motor_delay():
    _check_pair(_rollout(0.001, "baseline"), _rollout(0.0005, "baseline"))


def test_fixed_excitation_coupling_converges_without_changing_command_latency():
    coarse, fine = _check_pair(_rollout(0.001, "fixed"), _rollout(0.0005, "fixed"))
    # A constant command must produce the same activation history regardless
    # of the number of physics substeps, even though the body trajectories differ.
    np.testing.assert_allclose(coarse["activation"], fine["activation"], atol=1e-12, rtol=0)
    for trace in (coarse, fine):
        np.testing.assert_allclose(trace["delayed"][:2], 0.03, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            trace["delayed"][2:], np.broadcast_to(FIXED_ACTION, (498, 4)), atol=0, rtol=0
        )
        elapsed = np.maximum(trace["t"] - MOTOR_DELAY, 0.0)
        expected_activation = FIXED_ACTION.astype(np.float64) + (
            0.03 - FIXED_ACTION.astype(np.float64)
        ) * np.exp(-elapsed[:, None] / 0.015)
        np.testing.assert_allclose(trace["activation"], expected_activation, atol=2e-12, rtol=0)
