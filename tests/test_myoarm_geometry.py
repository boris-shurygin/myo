"""Geometry regressions against the locally prepared, licensed MyoArm assets."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myo.geometry import MyoArmGeometry


@pytest.fixture(scope="module")
def geometry():
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Run python3 scripts/prepare_myoarm.py to check real MyoArm assets")
    return MyoArmGeometry()


@pytest.mark.parametrize("degrees", [15, 45, 75, 105, 129])
def test_spatial_paths_obey_virtual_work_and_share_render_attachments(geometry, degrees):
    model = geometry.model
    data = mujoco.MjData(model)
    angle = np.deg2rad(degrees)
    data.qpos[0] = angle
    mujoco.mj_forward(model, data)
    epsilon = 1e-6
    finite_difference = (geometry.lengths_at(angle + epsilon)
                         - geometry.lengths_at(angle - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(geometry.moment_arms(data), -finite_difference, atol=1e-8)
    for tendon_id, points in zip(geometry.tendon_ids, geometry.path_points(data)):
        address = model.tendon_adr[tendon_id]
        count = model.tendon_num[tendon_id]
        first, last = model.wrap_objid[[address, address + count - 1]]
        np.testing.assert_allclose(points[0], data.site_xpos[first], atol=1e-12)
        np.testing.assert_allclose(points[-1], data.site_xpos[last], atol=1e-12)
    assert np.all(geometry.moment_arms(data)[:3] > 0)
    assert np.all(geometry.moment_arms(data)[3:] < 0)


def test_only_external_muscles_apply_forces_and_payload_is_hand_fixed(geometry):
    model = geometry.model
    assert (model.nq, model.nv, model.nu, model.na) == (1, 1, 0, 0)
    assert np.all(model.tendon_stiffness == 0)
    assert np.all(model.tendon_damping == 0)
    data = mujoco.MjData(model)
    hand = model.body("lunate_r").id
    positions = []
    for angle in (0.5, 1.5):
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qfrc_actuator, 0)
        relative = data.xmat[hand].reshape(3, 3).T @ (data.xpos[geometry.payload_id] - data.xpos[hand])
        np.testing.assert_allclose(relative, [0, -0.08, 0], atol=1e-12)
        positions.append(data.xpos[geometry.payload_id].copy())
    assert np.linalg.norm(positions[0] - positions[1]) > 0.2


def test_source_force_and_length_normalization_are_preserved(geometry):
    for parameter in geometry.muscle_parameters:
        assert parameter.max_isometric_force > 0
        assert parameter.motor_delay > 0
        assert parameter.tendon_slack_length > 0
    np.testing.assert_allclose([p.max_isometric_force for p in geometry.muscle_parameters],
                               [421.718, 270.82, 1168.76, 670.069, 682.224, 660.895])
    # A pose query must not advance or modify the live physics state.
    data = mujoco.MjData(geometry.model)
    before = data.qpos.copy()
    geometry.lengths_at(1.2)
    np.testing.assert_array_equal(data.qpos, before)


def test_unbounded_flexion_crosses_gravity_zero_without_an_upper_stop(geometry):
    from myo.env import PreacherCurlConfig, PreacherCurlEnv
    from myo.telemetry import make_packet

    env = PreacherCurlEnv(PreacherCurlConfig(geometry="myoarm", task_mode="curriculum"))
    try:
        assert np.isposinf(env.angle_limits[1])
        assert env.model.jnt_limited[env.elbow_id]
        # The curriculum samples a finite training range even though the joint
        # and user command have no upper bound.
        observation, _ = env.reset(seed=9)
        for _ in range(55):
            observation, _, terminated, _, _ = env.step(env.baseline_action())
            assert not terminated and np.isfinite(observation).all()
        assert np.isfinite(env.command.requested)
        torques = []
        for angle in (140, 150, 175):
            env.reset(options={"q": float(np.deg2rad(angle))})
            env.set_command(float(np.deg2rad(200)), 1.0)
            assert np.rad2deg(env.command.requested) == pytest.approx(200)
            assert env.passive_limit_torque() == 0.0
            assert env.data.qfrc_constraint[env.dof_index] == 0.0
            torques.append(env.data.qfrc_bias[env.dof_index])
            _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
            assert not terminated
            assert info["joint_limits_rad"][1] is None
            assert info["substep_peaks"]["joint_limit_reaction_Nm"] == 0.0
        assert torques[0] > 0 > torques[1]
        assert info["path_direction_reversal"]  # Report native path extrapolation; do not clamp it.
        packet = make_packet(env, 0, 0, "unbounded")
        assert packet["joints"][0]["limits_rad"][1] is None
        assert packet["metrics"]["path_direction_reversal"]
    finally:
        env.close()
