"""Display invariants on real MuJoCo scenes; GUI-free CLI control-loop checks."""

import json
import pickle

import mujoco
import numpy as np
import pytest

from myo import cli, demo
from myo.env import PreacherCurlEnv
from myo.visualization import (
    activation_rgba,
    append_arm_geometry,
    configure_camera,
    configure_options,
    schematic_anchors,
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
    observed = {"reset_seeds": [], "reset_states": [], "steps": 0,
                "policy_calls": 0, "env_closed": False, "sleeps": []}
    original_load = cli.load_controller

    def load(*args, **kwargs):
        env, action, metadata = original_load(*args, **kwargs)
        observed["env"] = env
        original_reset, original_step, original_close = env.reset, env.step, env.close

        def reset(*reset_args, **reset_kwargs):
            result = original_reset(*reset_args, **reset_kwargs)
            observed["reset_seeds"].append(reset_kwargs.get("seed"))
            observed["reset_states"].append(_restart_snapshot(env.unwrapped))
            return result

        def step(command):
            observed["steps"] += 1
            return original_step(command)

        def close():
            observed["env_closed"] = True
            original_close()

        def policy(observation):
            observed["policy_calls"] += 1
            return action(observation)

        env.reset, env.step, env.close = reset, step, close
        return env, policy, metadata

    monkeypatch.setattr(cli, "load_controller", load)
    monkeypatch.setattr(cli.time, "sleep", lambda duration: observed["sleeps"].append(duration))
    return observed


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
