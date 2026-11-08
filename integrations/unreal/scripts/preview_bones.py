"""Select the saved arm camera; optionally capture editor and verify Play view."""

import json
import os
from pathlib import Path
import time
import traceback

import unreal

unreal.EditorPythonScripting.set_keep_python_script_alive(True)

REPORT_PATH = Path(os.environ["MYO_PREVIEW_REPORT"])
GEOMETRY_PATH = Path(__file__).resolve().parents[3] / "data/unreal/myoarm/geometry.json"
FRAME_PATH = REPORT_PATH.with_name("frame.png")
REPORT = {"ok": False, "editor_framed": False, "play_camera_verified": False}
CAPTURE = os.environ.get("MYO_PREVIEW_CAPTURE") == "1"
WIDTH = int(os.environ.get("MYO_PREVIEW_WIDTH", "640"))
HEIGHT = int(os.environ.get("MYO_PREVIEW_HEIGHT", "360"))
STATE = "frame"
READY_AT = time.monotonic() + 2
DEADLINE = time.monotonic() + 75
TASK = None
HANDLE = None
NAVIGATION = None
INITIAL_CAMERA = None
INITIAL_YAW = None
SCENE_POSES = None


def scene_poses(world):
    result = {}
    for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.StaticMeshActor):
        p, r = actor.get_actor_location(), actor.get_actor_rotation()
        result[actor.get_name()] = [p.x, p.y, p.z, r.pitch, r.yaw, r.roll]
    return result


def save():
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")


def finish():
    unreal.unregister_slate_post_tick_callback(HANDLE)
    unreal.unregister_slate_post_tick_callback(DRIVE_HANDLE)
    save()
    unreal.log(f"MYO_PREVIEW {REPORT_PATH}: {REPORT}")
    if CAPTURE or not REPORT["ok"]:
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def tick(delta_time):
    global STATE, READY_AT, TASK, NAVIGATION, INITIAL_CAMERA, INITIAL_YAW, SCENE_POSES
    try:
        if time.monotonic() > DEADLINE:
            raise RuntimeError(f"Preview timed out in {STATE}")
        if time.monotonic() < READY_AT:
            return
        editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if STATE == "frame":
            actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
            priorities = [a.light_component.get_editor_property("forward_shading_priority")
                          for a in actors if isinstance(a, unreal.DirectionalLight)]
            REPORT["light_priorities"] = priorities
            if not priorities or priorities.count(max(priorities)) != 1:
                raise RuntimeError("Directional lights do not have a unique forward shading priority")
            geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
            # All four kinematic fixtures share the same named bone roster.
            expected_bones = {entry["name"] for entry in geometry["poses"][0]["geoms"]}
            REPORT["bone_components"] = []
            for actor in actors:
                if isinstance(actor, unreal.StaticMeshActor):
                    component = actor.static_mesh_component
                    mesh = component.static_mesh
                    if mesh and actor.get_actor_label() in expected_bones:
                        REPORT["bone_components"].append({
                            "name": actor.get_actor_label(), "mesh": mesh.get_path_name(),
                            "visible": component.is_visible(),
                            "hidden_in_game": component.get_editor_property("hidden_in_game"),
                            "owner_hidden": actor.get_editor_property("hidden"),
                            "render_main_pass": component.get_editor_property("render_in_main_pass"),
                            "sections": mesh.get_num_sections(0), "vertices": mesh.get_num_vertices(0),
                            "triangles": mesh.get_num_triangles(0),
                            "material": component.get_material(0).get_path_name() if component.get_material(0) else None,
                        })
            loaded_bones = {entry["name"] for entry in REPORT["bone_components"]}
            REPORT["missing_bones"] = sorted(expected_bones - loaded_bones)
            if REPORT["missing_bones"]:
                raise RuntimeError(f"Bone mesh references did not survive reload: {REPORT['missing_bones']}")
            if os.environ.get("MYO_PREVIEW_VIEW") == "muscles":
                REPORT["muscle_actors"] = [a.get_actor_label() for a in actors if isinstance(a, unreal.StaticMeshActor)
                                          and a.get_actor_label().startswith("Muscle_") and a.static_mesh_component.static_mesh]
                if len(REPORT["muscle_actors"]) != 6:
                    raise RuntimeError("The muscle view must contain six separate surfaces")
            camera = next(a for a in actors if isinstance(a, unreal.CameraActor) and a.get_actor_label() == "MyoPreviewCamera")
            editor.pilot_level_actor(camera)
            editor.set_exact_camera_view(True)
            editor.editor_set_game_view(True)
            REPORT["editor_framed"] = editor.get_pilot_level_actor() == camera
            REPORT["camera"] = camera.get_path_name()
            if not REPORT["editor_framed"]:
                raise RuntimeError("The viewport did not select the preview camera")
            save()
            if not CAPTURE:
                REPORT["ok"] = True
                finish()
                return
            TASK = unreal.AutomationLibrary.take_high_res_screenshot(WIDTH, HEIGHT, str(FRAME_PATH), camera=camera, delay=3.0)
            if TASK is None or not TASK.is_valid_task():
                raise RuntimeError("Screenshot task was not accepted")
            STATE, READY_AT = "screenshot", time.monotonic() + 4
        elif STATE == "screenshot":
            if not TASK.is_task_done():
                return
            if not FRAME_PATH.is_file():
                raise RuntimeError("Screenshot was not written")
            REPORT["frame"] = str(FRAME_PATH)
            editor.editor_request_begin_play()
            STATE, READY_AT = "play", time.monotonic() + 3
        elif STATE == "play":
            world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
            if world is None or not editor.is_in_play_in_editor():
                return
            controller = unreal.GameplayStatics.get_player_controller(world, 0)
            if controller is None:
                return
            target = controller.get_view_target()
            if not isinstance(target, unreal.CameraActor) or target.get_auto_activate_player_index() != 0:
                raise RuntimeError(f"Play selected the wrong view target: {target}")
            REPORT["play_camera_verified"] = True
            REPORT["play_view_target"] = target.get_path_name()
            pawn = controller.get_controlled_pawn()
            if pawn is None or target.get_attach_parent_actor() != pawn:
                raise RuntimeError("Play camera is not attached to the controlled navigator")
            pawns = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Pawn)
            if len(pawns) != 1 or any(c.static_mesh is not None for c in pawn.get_components_by_class(unreal.StaticMeshComponent)):
                raise RuntimeError("An extra pawn or visible navigation sphere is present")
            NAVIGATION = pawn
            INITIAL_CAMERA = target.get_actor_location()
            INITIAL_YAW = controller.get_control_rotation().yaw
            SCENE_POSES = scene_poses(world)
            REPORT["navigation_pawn"] = pawn.get_path_name()
            REPORT["navigation_meshes"] = 0
            REPORT["pawn_count"] = len(pawns)
            REPORT["navigation_speed_cm_s"] = pawn.get_movement_component().get_editor_property("max_speed")
            STATE, READY_AT = "navigate", time.monotonic() + 1.0
        elif STATE == "navigate":
            world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
            controller = unreal.GameplayStatics.get_player_controller(world, 0)
            target = controller.get_view_target()
            distance = (target.get_actor_location() - INITIAL_CAMERA).length()
            yaw_delta = abs(controller.get_control_rotation().yaw - INITIAL_YAW)
            if distance < 0.1 or yaw_delta < 1:
                raise RuntimeError(f"Camera controls did not move/turn the view: {distance=}, {yaw_delta=}")
            if scene_poses(world) != SCENE_POSES:
                raise RuntimeError("Camera movement changed the arm or payload pose")
            REPORT["navigation_verified"] = True
            REPORT["camera_translation_cm"] = distance
            REPORT["camera_yaw_delta_deg"] = yaw_delta
            REPORT["scene_unchanged_after_navigation"] = True
            editor.editor_request_end_play()
            REPORT["ok"] = True
            finish()
    except Exception:
        REPORT["error"] = traceback.format_exc()
        unreal.log_error(REPORT["error"])
        finish()


save()
HANDLE = unreal.register_slate_post_tick_callback(tick)


def drive_navigation(delta_time):
    # Exercise the same native callbacks used by WASD/E and mouse/arrow input.
    if STATE == "navigate" and time.monotonic() < READY_AT:
        NAVIGATION.move_forward(0.5)
        NAVIGATION.move_up_world(0.2)
        NAVIGATION.turn_at_rate(0.5)


DRIVE_HANDLE = unreal.register_slate_post_tick_callback(drive_navigation)
