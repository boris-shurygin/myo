"""Display invariants on real MuJoCo scenes; GUI-free CLI control-loop checks."""

import copy
import json
import pickle
import queue
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myo import cli, demo
from myo.env import PreacherCurlConfig, PreacherCurlEnv
from myo.visualization import (
    activation_rgba,
    append_arm_geometry,
    configure_camera,
    configure_options,
    muscle_color_values,
    schematic_anchors,
    tissue_paths,
    tendon_rgba,
)


def _physics_snapshot(env):
    # MuJoCo's pickle includes integration state and derived/cache fields.
    # A stray mj_forward on the real data must not pass as a read-only draw.
    return (
        pickle.dumps(env.model),
        pickle.dumps(env.data),
        tuple(muscle.state_vector().tobytes() for muscle in env.muscles),
        tuple(muscle.last_output for muscle in env.muscles),
        env._previous_excitation.tobytes(),
        env._step_count,
    )


def _scene_arrays(scene):
    geoms = [scene.geoms[index] for index in range(scene.ngeom)]
    return {
        "type": np.array([geom.type for geom in geoms]),
        **{field: np.array([getattr(geom, field).copy() for geom in geoms])
           for field in ("pos", "mat", "size", "rgba")},
    }


def _restart_snapshot(env):
    # Serialized MjData also contains workspace bytes that need not be identical
    # after reuse. Compare the actual integration state and complete observation.
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, specification))
    mujoco.mj_getState(env.model, env.data, state, specification)
    return state.tobytes(), env._get_obs().tobytes(), tuple(
        muscle.state_vector().tobytes() for muscle in env.muscles
    )


def test_real_scene_and_legend_helpers_leave_the_physics_snapshot_unchanged():
    env = PreacherCurlEnv()
    try:
        env.reset(seed=12)
        for _ in range(4):
            env.step(env.baseline_action())
        before = _physics_snapshot(env)
        scene = mujoco.MjvScene(env.model, maxgeom=512)
        append_arm_geometry(scene, env)
        configure_camera(env.model, mujoco.MjvCamera())
        configure_options(mujoco.MjvOption())
        overlays = demo.text_overlays(env, paused=True)
        strip = demo.activation_strip()
        assert scene.ngeom > 0
        assert any("PAUSED" in overlay[2] for overlay in overlays)
        assert strip.dtype == np.uint8 and strip.ndim == 3 and strip.shape[2] == 3
        assert _physics_snapshot(env) == before
    finally:
        env.close()


def test_fixed_pose_activation_changes_colors_without_moving_or_scaling_geometry():
    env = PreacherCurlEnv()
    try:
        env.reset(seed=3, options={"q": float(np.deg2rad(75)), "qvel": 0.0})
        body_state = pickle.dumps(env.data)
        arrays = []
        for activation in (0.10, 0.90):
            for muscle in env.muscles:
                muscle.activation = activation
            scene = mujoco.MjvScene(env.model, maxgeom=512)
            append_arm_geometry(scene, env)
            arrays.append(_scene_arrays(scene))
        low, high = arrays
        assert pickle.dumps(env.data) == body_state
        for field in ("type", "pos", "mat", "size"):
            np.testing.assert_array_equal(low[field], high[field])
        changed = np.any(low["rgba"] != high["rgba"], axis=1)
        assert np.count_nonzero(changed) >= len(env.muscles)
        np.testing.assert_allclose(
            low["rgba"][changed],
            np.broadcast_to(activation_rgba(0.10), (int(changed.sum()), 4)),
            atol=1e-7, rtol=0,
        )
        np.testing.assert_allclose(
            high["rgba"][changed],
            np.broadcast_to(activation_rgba(0.90), (int(changed.sum()), 4)),
            atol=1e-7, rtol=0,
        )
        # Activation edits do not silently relabel the independent tendon state.
        force_colors = muscle_color_values(env, "force")
        strain_colors = muscle_color_values(env, "strain")
        for muscle in env.muscles:
            name = muscle.params.name
            assert force_colors[name] == pytest.approx(
                muscle.last_output.tendon_force / muscle.params.max_isometric_force,
            )
            assert strain_colors[name] == pytest.approx(muscle.last_output.tendon_strain / 0.04)
        assert force_colors != muscle_color_values(env, "activation")
    finally:
        env.close()


def test_keyboard_commands_preserve_pose_and_delayed_muscle_state():
    env = PreacherCurlEnv(PreacherCurlConfig(task_mode="interactive"))
    try:
        env.reset(seed=7)
        env.step(env.baseline_action())
        before = _physics_snapshot(env)
        initial_target = env.telemetry()["requested_target_q"]
        viewer = demo.DemoViewer.__new__(demo.DemoViewer)
        viewer.env = env
        viewer._keys = queue.SimpleQueue()
        viewer.paused = False
        viewer.color_mode = "activation"
        viewer.selected_muscle = 0
        for key in (265, ord("0"), ord("3"), ord("M"), ord(" ")):
            viewer._keys.put(key)
        assert not viewer.poll()
        command = env.telemetry()
        assert command["requested_target_q"] == pytest.approx(initial_target + np.deg2rad(5))
        assert command["trigger"] == 0.0
        assert viewer.paused and viewer.color_mode == "force" and viewer.selected_muscle == 1
        assert _physics_snapshot(env) == before
        # No new key leaves the persistent target in place.
        assert not viewer.poll()
        assert env.telemetry()["requested_target_q"] == command["requested_target_q"]
    finally:
        env.close()


@pytest.mark.skipif(not Path("data/myoarm/manifest.json").is_file(), reason="local MyoArm assets absent")
def test_native_bones_and_wrapped_paths_render_on_a_private_snapshot():
    env = PreacherCurlEnv(PreacherCurlConfig(geometry="myoarm", task_mode="interactive"))
    try:
        env.reset(seed=2, options={"q": float(np.deg2rad(15)), "qvel": 0.0})
        before = _physics_snapshot(env)
        model = copy.copy(env.model)
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, env.data)
        mujoco.mj_forward(model, data)
        camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
        configure_camera(model, camera)
        configure_options(options, model)
        scene = mujoco.MjvScene(model, maxgeom=2048)
        mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        native = [scene.geoms[i] for i in range(scene.ngeom)]
        assert not any(g.objtype == mujoco.mjtObj.mjOBJ_TENDON for g in native)
        assert any(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in native)
        paths = tissue_paths(env, kinematic_data=data)
        for path, mechanical_points, length in zip(paths, env.geometry.path_points(data), env.muscle_lengths(), strict=True):
            np.testing.assert_allclose(path.points[[0, -1]], mechanical_points[[0, -1]], atol=2e-7, rtol=0)
            arc_length = np.linalg.norm(np.diff(path.points, axis=0), axis=1).sum()
            assert arc_length == pytest.approx(length, abs=1e-5)
        # BRA wraps at this pose: preserving native arc subdivisions must
        # give more points than merely joining its tangent pair by a chord.
        assert len(paths[2].points) > len(env.geometry.path_points(data)[2])
        initial = scene.ngeom
        append_arm_geometry(scene, env, kinematic_data=data)
        tissue_geoms = [scene.geoms[i] for i in range(initial, scene.ngeom)]
        assert len(tissue_geoms) > 2 * len(env.muscles)
        for path in paths:
            assert any(np.allclose(g.rgba, path.muscle_rgba) for g in tissue_geoms)
            assert any(np.allclose(g.rgba, path.tendon_rgba) for g in tissue_geoms)
        assert _physics_snapshot(env) == before
    finally:
        env.close()


def test_fixed_pose_series_contraction_changes_shape_and_independent_tissue_colors():
    env = PreacherCurlEnv()
    try:
        env.reset(seed=3, options={"q": float(np.deg2rad(75)), "qvel": 0.0})
        muscle = env.muscles[0]
        before = tissue_paths(env)[0]
        # Color follows activation without inventing a simultaneous stretch.
        muscle.activation = 0.8
        activated = tissue_paths(env)[0]
        assert not np.array_equal(activated.muscle_rgba, before.muscle_rgba)
        np.testing.assert_array_equal(activated.tendon_rgba, before.tendon_rgba)
        assert activated.belly_radius == before.belly_radius
        # Isometric fixture: integrate only the series-elastic muscle against
        # fixed anchors; the fiber can shorten even at identical bone pose.
        body = pickle.dumps(env.data)
        for _ in range(200):
            muscle.step(0.8, env.muscle_lengths()[0], 0.0, env.config.physics_dt)
        contracted = tissue_paths(env)[0]
        assert pickle.dumps(env.data) == body
        assert contracted.fiber_length < before.fiber_length
        assert contracted.tendon_length > before.tendon_length
        assert contracted.belly_radius > before.belly_radius
        assert contracted.tendon_radius < before.tendon_radius
        assert contracted.belly_start > before.belly_start
        assert contracted.belly_end < before.belly_end
        assert contracted.belly_radius**2 * contracted.fiber_length == pytest.approx(
            before.belly_radius**2 * before.fiber_length)
        state = env.telemetry()
        np.testing.assert_allclose(contracted.tendon_rgba, tendon_rgba(
            state["tendon_force_N"][0] / muscle.params.max_isometric_force))
        assert not np.allclose(contracted.tendon_rgba, before.tendon_rgba)
        snapshot = _physics_snapshot(env)
        scene = mujoco.MjvScene(env.model, maxgeom=512)
        append_arm_geometry(scene, env)
        assert _physics_snapshot(env) == snapshot
        # A completely slack series element is an admissible boundary state;
        # it must disappear without an infinite radius or division by zero.
        muscle.fiber_length = env.muscle_lengths()[0]
        slack = tissue_paths(env)[0]
        assert slack.tendon_length == 0
        assert np.isfinite(slack.tendon_radius)
        scene.ngeom = 0
        append_arm_geometry(scene, env)
        assert np.isfinite(_scene_arrays(scene)["size"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("angle_degrees", [45, 75, 105])
def test_brachialis_insertion_touches_the_drawn_ulna(angle_degrees):
    env = PreacherCurlEnv()
    try:
        angle = float(np.deg2rad(angle_degrees))
        env.reset(seed=3, options={"q": angle, "qvel": 0.0})
        scene = mujoco.MjvScene(env.model, maxgeom=512)
        append_arm_geometry(scene, env)
        anchors = schematic_anchors(env.model, angle)
        endpoint = next(path for path in anchors.muscles
                        if path.name == "brachialis").control_points[-1]
        geoms = [scene.geoms[index] for index in range(scene.ngeom)]
        markers = [geom for geom in geoms
                   if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE
                   and np.allclose(geom.pos, endpoint, atol=1e-7, rtol=0)]
        # Identify the positive-Y forearm shaft without relying on geom order
        # or recomputing its endpoints with the helper used by the renderer.
        shafts = [geom for geom in geoms
                  if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE
                  and np.isclose(geom.size[0], 0.0075, atol=1e-8, rtol=0)
                  and (anchors.forearm_rotation.T @ (geom.pos - anchors.elbow))[1] > 0]
        assert len(markers) == len(shafts) == 1
        marker, shaft = markers[0], shafts[0]
        axis = shaft.mat.reshape(3, 3)[:, 2]
        # MjvGeom stores capsule half-length in size[2], unlike MjModel.
        along = np.clip(np.dot(marker.pos - shaft.pos, axis),
                        -shaft.size[2], shaft.size[2])
        nearest = shaft.pos + along * axis
        # The marker's center must lie on the actual capsule surface, including
        # its endcaps; a detached marker must fail even if its anchor is valid.
        distance = np.linalg.norm(marker.pos - nearest)
        assert distance == pytest.approx(float(shaft.size[0]), abs=1e-7)
    finally:
        env.close()


def _observe_real_cli_environment(monkeypatch):
    """Keep actual dynamics/controller; only record public loop operations."""
    observed = {"reset_seeds": [], "reset_states": [], "reset_infos": [], "steps": 0,
                "policy_calls": 0, "env_closed": False, "sleeps": [],
                "step_observations": [], "policy_observations": []}
    original_load = cli.load_controller

    def load(*args, **kwargs):
        env, action, metadata = original_load(*args, **kwargs)
        observed["env"] = env
        original_reset, original_step, original_close = env.reset, env.step, env.close

        def reset(*reset_args, **reset_kwargs):
            result = original_reset(*reset_args, **reset_kwargs)
            observed["reset_seeds"].append(reset_kwargs.get("seed"))
            observed["reset_states"].append(_restart_snapshot(env.unwrapped))
            observed["reset_infos"].append(result[1])
            return result

        def step(command):
            observed["steps"] += 1
            result = original_step(command)
            observed["step_observations"].append(result[0].copy())
            return result

        def close():
            observed["env_closed"] = True
            original_close()

        def policy(observation):
            observed["policy_calls"] += 1
            observed["policy_observations"].append(observation.copy())
            return action(observation)

        env.reset, env.step, env.close = reset, step, close
        return env, policy, metadata

    monkeypatch.setattr(cli, "load_controller", load)
    monkeypatch.setattr(cli.time, "sleep", lambda duration: observed["sleeps"].append(duration))
    return observed


def test_demo_curriculum_starts_fresh_at_nominal_load_including_restart(
    monkeypatch, capsys, tmp_path,
):
    observed = _observe_real_cli_environment(monkeypatch)
    config = tmp_path / "curriculum.json"
    config.write_text(json.dumps({"task_mode": "curriculum", "payload_mass": 1.0}),
                      encoding="utf-8")

    class RestartingViewer:
        paused = False

        def __init__(self, env):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return self.poll_count == 2

        def is_running(self):
            return self.poll_count < 3

        def draw(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(demo, "DemoViewer", RestartingViewer)
    cli.main(["demo", "--config", str(config)])
    capsys.readouterr()
    assert observed["reset_seeds"] == [0, 0]
    for info in observed["reset_infos"]:
        assert info["payload_mass_kg"] == 1.0
        np.testing.assert_array_equal(info["capacity"], 1.0)
    # Headless evaluation retains the original curriculum distribution.
    cli.main(["rollout", "--config", str(config), "--seconds", "0.02"])
    capsys.readouterr()
    info = observed["reset_infos"][-1]
    assert info["payload_mass_kg"] != 1.0
    assert np.all(np.asarray(info["capacity"]) < 1.0)
    assert observed["env"].config.task_mode == "curriculum"


def test_cli_pause_restart_and_close_preserve_control_loop_semantics(monkeypatch, capsys):
    observed = _observe_real_cli_environment(monkeypatch)

    class ScriptedViewer:
        def __init__(self, env):
            observed["viewer"] = self
            self.env = env
            self.paused = False
            self.running = True
            self.poll_times = []
            self.draws = []
            self.closed = False

        def poll(self):
            self.poll_times.append(self.env.time)
            index = len(self.poll_times)
            self.paused = index in (1, 2, 4, 5)
            if index == 7:
                self.running = False
            return index == 4  # Restart while paused, after exactly one step.

        def is_running(self):
            return self.running

        def draw(self):
            self.draws.append((len(self.poll_times), self.env.time, self.paused))

        def close(self):
            self.running = False
            self.closed = True

    monkeypatch.setattr(demo, "DemoViewer", ScriptedViewer)
    cli.main(["demo", "--seed", "31"])
    summary = json.loads(capsys.readouterr().out)
    viewer = observed["viewer"]
    assert observed["steps"] == observed["policy_calls"] == 2
    assert observed["reset_seeds"] == [31, 31]
    assert observed["reset_states"][0] == observed["reset_states"][1]
    np.testing.assert_allclose(viewer.poll_times, [0, 0, 0, 0.02, 0, 0, 0.02], atol=1e-12, rtol=0)
    assert all(time == 0 for _, time, paused in viewer.draws if paused)
    assert all(index < 7 for index, _, _ in viewer.draws)
    assert viewer.closed and observed["env_closed"]
    assert summary["simulated_seconds"] == pytest.approx(0.04)
    assert summary["episodes"] == 2
    assert summary["mean_reward"] is not None


def test_cli_close_before_first_step_emits_valid_zero_step_summary(monkeypatch, capsys):
    observed = _observe_real_cli_environment(monkeypatch)

    class ImmediatelyClosedViewer:
        paused = False

        def __init__(self, env):
            self.closed = False
            observed["viewer"] = self

        def poll(self):
            return True  # A close must win even over a queued restart request.

        def is_running(self):
            return False

        def draw(self):
            pytest.fail("A closed viewer must not trigger a draw")

        def close(self):
            self.closed = True

    monkeypatch.setattr(demo, "DemoViewer", ImmediatelyClosedViewer)
    cli.main(["demo"])
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert "NaN" not in output and "Infinity" not in output
    assert observed["steps"] == observed["policy_calls"] == 0
    assert observed["reset_seeds"] == [0]
    assert observed["sleeps"] == []
    assert observed["viewer"].closed and observed["env_closed"]
    assert summary["simulated_seconds"] == 0
    assert summary["physical_failures"] == 0
    for field in ("angle_rmse_deg", "angular_velocity_rmse_rad_s", "mean_reward",
                  "policy_mean_ms", "env_step_mean_ms", "viewer_mean_ms"):
        assert summary[field] is None


@pytest.mark.parametrize("arguments", [["demo"], ["rollout", "--viewer"], ["rollout"]])
def test_interactive_window_keeps_state_across_training_horizon(
    monkeypatch, capsys, tmp_path, arguments,
):
    observed = _observe_real_cli_environment(monkeypatch)
    config = tmp_path / "short_episode.json"
    config.write_text(json.dumps({"task_mode": "interactive", "episode_seconds": 0.04}),
                      encoding="utf-8")
    output = tmp_path / "state.jsonl"
    windowed = arguments != ["rollout"]

    class OpenViewer:
        paused = False
        stop_reason = None

        def __init__(self, env):
            self.env = env
            self.first_poll = True

        def poll(self):
            if self.first_poll:
                self.env.set_command(1.5, 0.7)
                self.first_poll = False
            return False

        def is_running(self):
            return True

        def draw(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(demo, "DemoViewer", OpenViewer)
    cli.main([*arguments, "--config", str(config), "--seconds", "0.12", "--output", str(output)])
    summary = json.loads(capsys.readouterr().out)
    packets = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["simulated_seconds"] == pytest.approx(0.12)
    assert observed["env"].config.episode_seconds == 0.04
    assert summary["physical_failures"] == 0
    if windowed:
        assert observed["reset_seeds"] == [0]
        assert summary["episodes"] == 1
        np.testing.assert_allclose([packet["simulation_time_s"] for packet in packets],
                                   np.arange(7) * 0.02, atol=1e-12, rtol=0)
        assert {packet["episode"] for packet in packets} == {0}
        assert all(packet["command"]["trigger"] == 0.7 for packet in packets[1:])
        assert all(packet["command"]["requested_target_angle_rad"] == 1.5
                   for packet in packets[1:])
        # Complete observations include history, activation and pending delays.
        # The next policy call sees precisely the preceding step's result even
        # across the horizon, with no artificial reset or time jump.
        np.testing.assert_array_equal(observed["policy_observations"][1:],
                                      observed["step_observations"][:-1])
    else:
        assert observed["reset_seeds"] == [0, 1, 2]
        assert summary["episodes"] == 3
        assert {packet["episode"] for packet in packets} == {0, 1, 2}
        assert max(packet["simulation_time_s"] for packet in packets) == pytest.approx(0.04)


def test_physical_failure_stops_window_until_explicit_restart(monkeypatch, capsys):
    observed = _observe_real_cli_environment(monkeypatch)

    class FailureViewer:
        def __init__(self, env):
            observed["viewer"] = self
            self.env = env
            self.paused = False
            self.stop_reason = None
            self.poll_count = 0
            self.running = True
            self.stopped_states = []
            self.statuses = []

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                # A real out-of-guard transition, rather than a mocked done flag.
                self.env.data.qvel[self.env.dof_index] = 100.0
            elif self.poll_count in (2, 3):
                self.stopped_states.append(_restart_snapshot(self.env))
                assert self.stop_reason == "joint_state_outside_guard"
                self.paused = False  # Space cannot bypass a physical stop.
            elif self.poll_count == 4:
                self.running = False
            return self.poll_count == 3

        def is_running(self):
            return self.running

        def draw(self):
            self.statuses.append(self.stop_reason)

        def close(self):
            self.running = False

    monkeypatch.setattr(demo, "DemoViewer", FailureViewer)
    cli.main(["demo", "--seed", "3"])
    summary = json.loads(capsys.readouterr().out)
    viewer = observed["viewer"]
    assert observed["steps"] == observed["policy_calls"] == 2
    assert observed["reset_seeds"] == [3, 3]
    assert viewer.stopped_states[0] == viewer.stopped_states[1]
    assert viewer.statuses[:2] == ["joint_state_outside_guard"] * 2
    assert viewer.statuses[-1] is None
    assert summary["physical_failures"] == 1
    assert summary["episodes"] == 2
