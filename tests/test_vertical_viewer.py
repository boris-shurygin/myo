"""Wait for render-thread cleanup, and keep GUI edits out of native physics."""
import contextlib
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myo.demo import DemoViewer
from myo.env import PreacherCurlEnv
from myo.vertical_muscles import VerticalMuscles
from myo.vertical_viewer import VerticalViewer


@pytest.mark.parametrize('arm', (False, True))
def test_close_joins_native_cleanup_and_viewer_owns_display_state(monkeypatch, arm):
    from mujoco import viewer

    requested,finish_cleanup,close_returned = (threading.Event() for _ in range(3))
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
            self.data.qpos[0] = .2
        def close(self):
            requested.set()
        def set_texts(self, texts):
            pass
        def clear_images(self):
            pass
    def launch(**kwargs):
        handle = Handle()
        handle.data = kwargs['data']
        handle.user_scn = mujoco.MjvScene(kwargs["model"],maxgeom=512)
        kwargs["handle_return"].put(handle)
        finish_cleanup.wait()
    monkeypatch.setattr(viewer,"_launch_internal",launch)
    if arm:
        monkeypatch.setenv('DISPLAY', ':test')
        env = PreacherCurlEnv()
        env.reset(seed=0)
        before = env.data.qpos.copy()
        window = DemoViewer(env)
        window.draw()
        np.testing.assert_array_equal(env.data.qpos,before)
    else:
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
    if arm:
        env.close()
