"""Geometry regressions against the locally prepared, licensed MyoArm assets."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from myo.geometry import CONTACT_GEOM_GROUP, CONTACT_GEOM_NAMES, MyoArmGeometry


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
    # An inverted mounting changes gravity, not the anatomical coordinate,
    # attachments, force transmission or local mass/inertia.
    inverted = MyoArmGeometry(arm_mount_tilt=np.deg2rad(210.0))
    inverted_data = mujoco.MjData(inverted.model)
    data.qpos[0] = inverted_data.qpos[0] = np.deg2rad(75.0)
    mujoco.mj_forward(geometry.model, data)
    mujoco.mj_forward(inverted.model, inverted_data)
    np.testing.assert_allclose(geometry.lengths(data), inverted.lengths(inverted_data), atol=1e-12)
    np.testing.assert_allclose(geometry.moment_arms(data), inverted.moment_arms(inverted_data), atol=1e-12)
    np.testing.assert_array_equal(geometry.model.body_mass, inverted.model.body_mass)
    np.testing.assert_array_equal(geometry.model.body_inertia, inverted.model.body_inertia)
    np.testing.assert_allclose(data.qfrc_bias, -inverted_data.qfrc_bias, atol=1e-12)


def test_unbounded_flexion_crosses_gravity_zero_without_an_upper_joint_limit(geometry):
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
            _, info = env.reset(options={"q": float(np.deg2rad(angle))})
            env.set_command(float(np.deg2rad(200)), 1.0)
            assert np.rad2deg(env.command.requested) == pytest.approx(200)
            assert env.passive_limit_torque() == 0.0
            # Tissue contact is now a physical obstruction. It must not be
            # mistaken for a restored upper angular limit in diagnostics.
            limits = env.data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
            assert not limits.any()
            torques.append(env.data.qfrc_bias[env.dof_index])
            assert info["joint_limits_rad"][1] is None
        assert torques[0] > 0 > torques[1]
        assert info["path_direction_reversal"]  # Report native path extrapolation; do not clamp it.
        packet = make_packet(env, 0, 0, "unbounded")
        assert packet["joints"][0]["limits_rad"][1] is None
        assert packet["metrics"]["path_direction_reversal"]
    finally:
        env.close()


def test_tissue_contact_prevents_pass_through_without_added_inertia_or_angle_limit(geometry):
    from myo.env import PreacherCurlConfig, PreacherCurlEnv
    from myo.telemetry import make_packet

    model = geometry.model
    # Compile the same imported skeleton without just the new contact geometry;
    # tissue mass must not be counted twice on top of the source inertials.
    root = ET.parse("data/myoarm/model.xml").getroot()
    root.find("compiler").set("meshdir", str(Path("data/myoarm/upstream/myo_sim/models").resolve()))
    root.remove(root.find("contact"))
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("name") in CONTACT_GEOM_NAMES:
                body.remove(geom)
    no_contact = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    np.testing.assert_array_equal(model.body_mass, no_contact.body_mass)
    np.testing.assert_allclose(model.body_inertia, no_contact.body_inertia, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(model.body_ipos, no_contact.body_ipos)
    assert np.isposinf(model.jnt_range[geometry.elbow_id, 1])
    assert np.all(model.geom_group[list(geometry.contact_geom_ids)] == CONTACT_GEOM_GROUP)

    data = mujoco.MjData(model)
    for degrees in (15, 45, 75, 105, 130, 135):
        data.qpos[0] = np.deg2rad(degrees)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
    # A mechanical stress test, not a runtime controller: apply sustained
    # +100 Nm from 125 degrees. Contact must stop the original pass-through
    # failure while allowing finite tissue compression and leaving q unbounded.
    maxima = []
    for candidate in (model, no_contact):
        state = mujoco.MjData(candidate)
        state.qpos[0] = np.deg2rad(125.0)
        state.qfrc_applied[0] = 100.0
        maximum = state.qpos[0]
        for _ in range(3000):
            mujoco.mj_step(candidate, state)
            maximum = max(maximum, state.qpos[0])
        maxima.append(float(np.rad2deg(maximum)))
        if candidate is model:
            assert state.ncon > 0
            assert state.qfrc_constraint[0] < -90.0
            assert 135.0 < np.rad2deg(state.qpos[0]) < 155.0
            assert 0 < max(-c.dist for c in state.contact) < 0.01
    assert 140.0 < maxima[0] < 160.0
    assert maxima[1] > 180.0

    # The ordinary delayed-muscle environment exposes contact separately from
    # the joint-limit reaction; the packet reports the same physical state.
    env = PreacherCurlEnv(PreacherCurlConfig(geometry="myoarm"))
    try:
        env.reset(options={"q": float(np.deg2rad(150.0))})
        _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        assert not terminated
        contact = env.tissue_contact_metrics()
        assert contact["count"] > 0
        assert contact["normal_force_N"] > 0
        assert contact["max_penetration_m"] > 0
        assert contact["torque_Nm"] < 0
        assert info["substep_peaks"]["joint_limit_reaction_Nm"] == 0
        packet = make_packet(env, 0, 0, "contact")
        assert packet["tissue_contact"] == contact
    finally:
        env.close()


def test_contact_softness_changes_response_without_moving_the_envelopes(geometry):
    from dataclasses import replace

    from myo.env import PreacherCurlConfig, PreacherCurlEnv

    base = PreacherCurlConfig(geometry="myoarm")
    configurations = (base, replace(base, tissue_contact_time_constant=0.05),
                       replace(base, tissue_contact_transition_depth=0.025))
    responses = []
    for config in configurations:
        env = PreacherCurlEnv(config)
        try:
            # Fixed-pose solver probe: remove gravity and native damping so
            # only the new contact law changes between measurements.
            env.model.opt.gravity[:] = 0.0
            env.model.dof_damping[:] = 0.0
            rows = []
            for angle, speed in ((142.0, 0.0), (145.0, 0.0), (145.0, 0.5)):
                env.data.qpos[env.qpos_index] = np.deg2rad(angle)
                env.data.qvel[env.dof_index] = speed
                env.data.qacc_warmstart[:] = 0.0
                mujoco.mj_forward(env.model, env.data)
                rows.append(env.tissue_contact_metrics())
            assert all(row["normal_force_N"] > 0 and row["torque_Nm"] < 0 for row in rows)
            assert rows[1]["normal_force_N"] > rows[0]["normal_force_N"]
            assert rows[2]["normal_force_N"] > rows[1]["normal_force_N"]
            responses.append(rows)
        finally:
            env.close()
    for softer in responses[1:]:
        for standard, changed in zip(responses[0], softer):
            assert changed["max_penetration_m"] == pytest.approx(standard["max_penetration_m"], abs=1e-12)
            assert changed["normal_force_N"] < standard["normal_force_N"]
