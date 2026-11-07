"""MuJoCo 3.17 passive viewer with an owned, joinable rendering thread.

The public launch_passive handle only requests shutdown; it does not expose
the daemon thread to join. Keep the same upstream GUI but own its thread so
GLFW termination cannot race simulate.destroy during interpreter shutdown.
"""
from __future__ import annotations

import copy
import queue
import threading

import mujoco

from .demo import legend_images
from .vertical_visualization import append_tissues, configure_camera, configure_model, configure_options, text_overlays


class VerticalViewer:
    def __init__(self, env):
        from mujoco import viewer

        self.model = copy.copy(env.model)
        configure_model(self.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data,self.model,env.data)
        handles = queue.Queue(1)
        # Pinned MuJoCo 3.17 adapter: this is the same launch target and
        # arguments as launch_passive on Linux, but the thread is not detached.
        self._thread = threading.Thread(target=viewer._launch_internal,
            kwargs={"model":self.model,"data":self.data,"run_physics_thread":False,
                    "handle_return":handles,"show_left_ui":False,"show_right_ui":False},
            name="myo-vertical-viewer",daemon=False)
        self._thread.start()
        self.handle = handles.get()
        try:
            with self.handle.lock():
                configure_camera(self.handle.cam,env)
                configure_options(self.handle.opt)
        except BaseException:
            self.close()
            raise

    def is_running(self):
        return self.handle.is_running()

    def sync(self, env):
        with self.handle.lock():
            mujoco.mj_copyData(self.data,self.model,env.data)
            configure_options(self.handle.opt)
            self.handle.user_scn.ngeom = 0
            append_tissues(self.handle.user_scn,env)
        self.handle.set_texts(text_overlays(env))
        viewport = self.handle.viewport
        if viewport is not None and viewport.width >= 300 and viewport.height >= 200:
            self.handle.set_images(list(legend_images("tissue",viewport.left,viewport.bottom)))
        else:
            self.handle.clear_images()
        self.handle.sync()

    def close(self):
        self.handle.close()
        # Waiting for the exit request alone is insufficient: render_loop,
        # reload-thread join and native destroy must all finish before exit.
        self._thread.join()
