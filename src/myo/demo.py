"""Diagnostic arm viewer with explicit target/trigger commands and signal colors."""

from __future__ import annotations

import copy
from functools import lru_cache
import os
import queue
import sys

import mujoco
import numpy as np

from .visualization import (
    COLOR_LABELS,
    COLOR_MODES,
    activation_rgba,
    tendon_rgba,
    append_arm_geometry,
    configure_camera,
    configure_options,
)


@lru_cache(maxsize=4)
def activation_strip(width: int = 220, height: int = 12) -> np.ndarray:
    """The same fixed 0..1 color scale used by the muscles, left to right."""
    row = np.array([activation_rgba(float(a))[:3] for a in np.linspace(0, 1, width)])
    return np.repeat(np.round(row[None, :, :] * 255).astype(np.uint8), height, axis=0)


@lru_cache(maxsize=4)
def tendon_strip(width: int = 220, height: int = 12) -> np.ndarray:
    row = np.array([tendon_rgba(float(force))[:3] for force in np.linspace(0, 1, width)])
    return np.repeat(np.round(row[None, :, :] * 255).astype(np.uint8), height, axis=0)


def legend_images(color_mode: str, left: int = 0, bottom: int = 0):
    if color_mode == "tissue":
        return ((mujoco.MjrRect(left + 10, bottom + 102, 220, 12), activation_strip()),
                (mujoco.MjrRect(left + 10, bottom + 38, 220, 12), tendon_strip()))
    return ((mujoco.MjrRect(left + 10, bottom + 38, 220, 12), activation_strip()),)


def text_overlays(
    env, paused: bool = False, color_mode: str = "tissue", selected_muscle: int = 0,
    stop_reason: str | None = None,
) -> list[tuple[int, int, str, str]]:
    """Display each signal separately; percentages never stand in for force."""
    state = env.telemetry()
    index = selected_muscle % len(env.muscles)
    muscle = env.muscles[index]
    angle = np.rad2deg(state["q"])
    requested = np.rad2deg(state.get("requested_target_q", state["target_q"]))
    target = np.rad2deg(state["target_q"])
    task = getattr(env.config, "task_mode", "reference")
    geometry = getattr(env.config, "geometry", "toy")
    labels = (f"{muscle.params.name} [{index + 1}/{len(env.muscles)}]"
              "\nExcitation\nDelayed excitation\nActivation\nTendon force"
              "\nTendon strain\nFiber length\nTendon length\nCapacity proxy")
    values = (f"\n{100 * state['excitation'][index]:.1f}%"
              f"\n{100 * state['delayed_excitation'][index]:.1f}%"
              f"\n{100 * state['activation'][index]:.1f}%"
              f"\n{state['tendon_force_N'][index]:.1f} N"
              f"\n{100 * state['tendon_strain'][index]:.2f}%"
              f"\n{1000 * state['fiber_length_m'][index]:.1f} mm"
              f"\n{1000 * state['tendon_length_m'][index]:.1f} mm"
              f"\n{100 * state['capacity'][index]:.1f}%")
    status = "STOPPED" if stop_reason else "PAUSED" if paused else "ARM DIAGNOSTIC"
    command_help = ("Up/Down target | +/- trigger | 0 release\n" if task != "reference"
                    else "Scripted commands (read-only)\n")
    return [
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
         f"{status} | {geometry}\nElbow {angle:.0f} deg | Load {state['payload_mass_kg']:g} kg"
         f"\nTarget {target:.0f} deg | Requested {requested:.0f} deg"
         f"\nTrigger {100 * state.get('trigger', 1.0):.0f}% | {state.get('command_source', task)}"
         + ("\nMuscle path reversal: outside source anatomy range" if state.get("path_direction_reversal") else "")
         + (f"\n{stop_reason} | R to reset" if stop_reason else ""), ""),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, labels, values),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
         COLOR_LABELS[color_mode], ""),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
         command_help + "1 tissues | 2 activation | 3 force | 4 strain\nM muscle | "
         "Space pause | R restart | F fit\nMouse rotate/zoom | Esc close", ""),
    ]


class DemoViewer:
    """Render a separate model/data copy; GUI edits never enter the simulation."""

    def __init__(self, env, *, color_mode: str = "tissue", selected_muscle: int = 0) -> None:
        if color_mode not in COLOR_MODES:
            raise ValueError(f"Unknown muscle color mode: {color_mode}")
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise RuntimeError("The demo needs a graphical desktop. Use rollout without --viewer for headless runs.")
        from mujoco import viewer

        self.env = env
        self.model = copy.copy(env.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, env.data)
        self._keys: queue.SimpleQueue[int] = queue.SimpleQueue()
        self.paused = False
        self.stop_reason = None
        self.color_mode = color_mode
        self.selected_muscle = selected_muscle % len(env.muscles)
        self._path_scene = mujoco.MjvScene(self.model, maxgeom=512)
        self.handle = viewer.launch_passive(
            self.model, self.data, key_callback=self._keys.put,
            show_left_ui=False, show_right_ui=False,
        )
        try:
            with self.handle.lock():
                configure_camera(self.model, self.handle.cam)
                configure_options(self.handle.opt, self.model)
            self.draw()
        except Exception:
            self.handle.close()
            raise

    def is_running(self) -> bool:
        return self.handle.is_running()

    def poll(self) -> bool:
        """Process GUI requests on the simulation thread; return a reset request."""
        restart = False
        while True:
            try:
                key = self._keys.get_nowait()
            except queue.Empty:
                break
            if key == ord(" "):
                if not getattr(self, "stop_reason", None):
                    self.paused = not self.paused
            elif key in (ord("R"), ord("r")):
                restart = True
            elif key in (ord("F"), ord("f")):
                with self.handle.lock():
                    configure_camera(self.model, self.handle.cam)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                self.color_mode = COLOR_MODES[key - ord("1")]
            elif key in (ord("M"), ord("m")):
                self.selected_muscle = (self.selected_muscle + 1) % len(self.env.muscles)
            elif getattr(self.env.config, "task_mode", "reference") != "reference" and key in (
                264, 265, ord("+"), ord("="), 334, ord("-"), 333, ord("0"),
            ):
                state = self.env.telemetry()
                target = state.get("requested_target_q", state["target_q"])
                trigger = state["trigger"]
                if key in (264, 265):  # GLFW down / up; no GUI import in helpers.
                    target += np.deg2rad(5.0) * (1 if key == 265 else -1)
                elif key in (ord("+"), ord("="), 334):
                    trigger += 0.1
                elif key in (ord("-"), 333):
                    trigger -= 0.1
                else:
                    trigger = 0.0
                self.env.set_command(
                    float(np.clip(target, *self.env.angle_limits)),
                    float(np.clip(trigger, 0.0, 1.0)),
                )
            elif key == 256:  # GLFW_KEY_ESCAPE; avoid another GUI import in helpers.
                self.close()
        return restart

    def draw(self) -> None:
        if not self.is_running():
            return
        with self.handle.lock():
            mujoco.mj_copyData(self.data, self.model, self.env.data)
            mujoco.mj_forward(self.model, self.data)
            configure_options(self.handle.opt, self.model)
            self.handle.user_scn.ngeom = 0
            append_arm_geometry(
                self.handle.user_scn, self.env, self.color_mode, kinematic_data=self.data,
                path_scene=self._path_scene,
            )
        self.handle.set_texts(text_overlays(
            self.env, self.paused, self.color_mode, self.selected_muscle, self.stop_reason,
        ))
        viewport = self.handle.viewport
        if viewport is not None and viewport.width >= 260 and viewport.height >= 160:
            self.handle.set_images(list(legend_images(self.color_mode, viewport.left, viewport.bottom)))
        else:
            self.handle.clear_images()
        # MuJoCo transfers GUI changes during sync. Both arguments belong to this
        # viewer, so mouse perturbations/options cannot modify the real env.
        self.handle.sync()

    def close(self) -> None:
        self.handle.close()


def render_frame(
    env, width: int = 960, height: int = 720, *,
    color_mode: str = "tissue", selected_muscle: int = 0,
) -> np.ndarray:
    """Render the same scene/legend offscreen for visual QA (MUJOCO_GL=egl)."""
    model = copy.copy(env.model)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, env.data)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
    configure_camera(model, camera)
    configure_options(options, model)
    scene = mujoco.MjvScene(model, maxgeom=2048)
    gl = mujoco.GLContext(width, height)
    context = None
    try:
        gl.make_current()
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        append_arm_geometry(scene, env, color_mode, kinematic_data=data)
        mujoco.mjr_render(viewport, scene, context)
        for font, position, text, values in text_overlays(
            env, color_mode=color_mode, selected_muscle=selected_muscle,
        ):
            mujoco.mjr_overlay(font, position, viewport, text, values, context)
        if width >= 260 and height >= 160:
            for rectangle, strip in legend_images(color_mode):
                mujoco.mjr_drawPixels(strip.ravel(), None, rectangle, context)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, viewport, context)
        return np.flipud(pixels).copy()
    finally:
        if context is not None:
            context.free()
        gl.free()
