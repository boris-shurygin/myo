"""Wait for render-thread cleanup, and keep GUI edits out of native physics."""
import contextlib
import queue
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

from myo.vertical_muscles import VerticalMuscles
from myo.vertical_viewer import VerticalViewer


def test_close_joins_native_cleanup_and_viewer_owns_display_state(monkeypatch):
    from mujoco import viewer

    requested,finish_cleanup,close_returned = (threading.Event() for _ in range(3))
    display = queue.Queue()
    class Handle:
        cam = SimpleNamespace(lookat=np.zeros(3),distance=0.,elevation=0.)
        opt = mujoco.MjvOption()
        viewport = None
        def lock(self):
            return contextlib.nullcontext()
        def is_running(self):
            return not requested.is_set()
        def sync(self):
            # MuJoCo sync can transfer slider edits/perturbations to its data.
            display.get_nowait().qpos[0] = .2
        def close(self):
            requested.set()
        def set_texts(self, texts):
            pass
        def clear_images(self):
            pass
    def launch(**kwargs):
        display.put(kwargs["data"])
        handle = Handle()
        handle.user_scn = mujoco.MjvScene(kwargs["model"],maxgeom=16)
        kwargs["handle_return"].put(handle)
        finish_cleanup.wait()
    monkeypatch.setattr(viewer,"_launch_internal",launch)
    env = VerticalMuscles()
    before = env.reset().copy()
    window = VerticalViewer(env)
    window.sync(env)
    np.testing.assert_array_equal(env.state(),before)
    assert window.data.qpos[0] == .2
    def close():
        window.close()
        close_returned.set()
    closing = threading.Thread(target=close)
    try:
        closing.start()
        assert requested.wait(2.)
        assert not close_returned.is_set()
    finally:
        finish_cleanup.set()
        closing.join(2.)
        window.close()
    assert close_returned.is_set()
    assert not window._thread.is_alive()
