"""Render the four bone fixtures in a real editor viewport, then close it."""

import json
from pathlib import Path
import time
import traceback

import unreal

unreal.EditorPythonScripting.set_keep_python_script_alive(True)

OUTPUT = Path(unreal.Paths.project_saved_dir()) / "Myo"
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT / "bones-capture.json"
REPORT = {"ok": False, "images": []}
ANGLES = iter((15, 45, 90, 120))
TASK = None
CURRENT = None
HANDLE = None
READY_AT = time.monotonic() + 15
DEADLINE = time.monotonic() + 300


def finish():
    unreal.unregister_slate_post_tick_callback(HANDLE)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    unreal.log(f"MYO_BONES_CAPTURE {REPORT_PATH}: ok={REPORT['ok']}")
    unreal.SystemLibrary.quit_editor()


def tick(delta_time):
    global TASK, CURRENT, READY_AT
    try:
        if time.monotonic() > DEADLINE:
            raise RuntimeError("Timed out waiting for editor screenshots")
        if time.monotonic() < READY_AT:
            return
        if TASK is not None:
            if not TASK.is_task_done():
                return
            if not CURRENT.is_file():
                raise RuntimeError(f"Screenshot was not written: {CURRENT}")
            REPORT["images"].append(str(CURRENT))
            TASK = None
        angle = next(ANGLES, None)
        if angle is None:
            REPORT["ok"] = True
            finish()
            return
        unreal.EditorLoadingAndSavingUtils.load_map(f"/Game/Generated/Bones_v1/L_Bones_{angle}")
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
        camera = next(actor for actor in actors if isinstance(actor, unreal.CameraActor))
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(camera.get_actor_location(), camera.get_actor_rotation())
        CURRENT = OUTPUT / f"bones-{angle}.png"
        CURRENT.unlink(missing_ok=True)
        TASK = unreal.AutomationLibrary.take_high_res_screenshot(1280, 720, str(CURRENT), camera=camera, delay=5.0)
        if TASK is None or not TASK.is_valid_task():
            raise RuntimeError("Editor did not accept the screenshot task")
        READY_AT = time.monotonic() + 6
    except Exception:
        REPORT["error"] = traceback.format_exc()
        unreal.log_error(REPORT["error"])
        finish()


HANDLE = unreal.register_slate_post_tick_callback(tick)
