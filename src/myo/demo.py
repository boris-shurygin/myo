"""A compact, read-only view of the arm and its muscle activations."""

from __future__ import annotations

import copy
import os
import queue
import sys

import mujoco
import numpy as np

from .visualization import activation_rgba, append_arm_geometry, configure_camera, configure_options


def activation_strip(width: int = 220, height: int = 12) -> np.ndarray:
    """The same fixed 0..1 color scale used by the muscles, left to right."""
    row = np.array([activation_rgba(float(a))[:3] for a in np.linspace(0, 1, width)])
    return np.repeat(np.round(row[None, :, :] * 255).astype(np.uint8), height, axis=0)


def text_overlays(env, paused: bool = False) -> list[tuple[int, int, str, str]]:
    """Only pose, four activations, a legend, and the essential controls."""
    angle = np.rad2deg(env.data.qpos[env.qpos_index])
    labels = "Biceps long\nBiceps short\nBrachialis\nTriceps"
    values = "\n".join(f"{100 * muscle.activation:5.1f}%" for muscle in env.muscles)
    status = "PAUSED" if paused else "PREACHER CURL"
    return [
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
         f"{status}\nElbow {angle:.0f} deg | Load {env.config.payload_mass:g} kg", ""),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, labels, values),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
         "Activation 0-100%", ""),
        (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
         "Space pause | R restart | F fit\nMouse rotate/zoom | Esc close", ""),
    ]


class DemoViewer:
    """Render a separate model/data copy; GUI edits never enter the simulation."""

    def __init__(self, env) -> None:
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
        self._strip = activation_strip()
        self.handle = viewer.launch_passive(
            self.model, self.data, key_callback=self._keys.put,
            show_left_ui=False, show_right_ui=False,
        )
        try:
            with self.handle.lock():
                configure_camera(self.model, self.handle.cam)
                configure_options(self.handle.opt)
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
                self.paused = not self.paused
            elif key in (ord("R"), ord("r")):
                restart = True
            elif key in (ord("F"), ord("f")):
                with self.handle.lock():
                    configure_camera(self.model, self.handle.cam)
            elif key == 256:  # GLFW_KEY_ESCAPE; avoid another GUI import in helpers.
                self.close()
        return restart

    def draw(self) -> None:
        if not self.is_running():
            return
        with self.handle.lock():
            mujoco.mj_copyData(self.data, self.model, self.env.data)
            mujoco.mj_forward(self.model, self.data)
            configure_options(self.handle.opt)
            self.handle.user_scn.ngeom = 0
            append_arm_geometry(self.handle.user_scn, self.env)
        self.handle.set_texts(text_overlays(self.env, self.paused))
        viewport = self.handle.viewport
        if viewport is not None and viewport.width >= 260 and viewport.height >= 160:
            rect = mujoco.MjrRect(viewport.left + 10, viewport.bottom + 38, 220, 12)
            self.handle.set_images((rect, self._strip))
        else:
            self.handle.clear_images()
        # MuJoCo transfers GUI changes during sync. Both arguments belong to this
        # viewer, so mouse perturbations/options cannot modify the real env.
        self.handle.sync()

    def close(self) -> None:
        self.handle.close()


def render_frame(env, width: int = 960, height: int = 720) -> np.ndarray:
    """Render the same scene/legend offscreen for visual QA (MUJOCO_GL=egl)."""
    model = copy.copy(env.model)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, env.data)
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    camera, options = mujoco.MjvCamera(), mujoco.MjvOption()
    configure_camera(model, camera)
    configure_options(options)
    scene = mujoco.MjvScene(model, maxgeom=512)
    gl = mujoco.GLContext(width, height)
    context = None
    try:
        gl.make_current()
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        append_arm_geometry(scene, env)
        mujoco.mjr_render(viewport, scene, context)
        for font, position, text, values in text_overlays(env):
            mujoco.mjr_overlay(font, position, viewport, text, values, context)
        if width >= 260 and height >= 160:
            mujoco.mjr_drawPixels(activation_strip().ravel(), None, mujoco.MjrRect(10, 38, 220, 12), context)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, viewport, context)
        return np.flipud(pixels).copy()
    finally:
        if context is not None:
            context.free()
        gl.free()
