"""Build local bone import fixtures with explicit coordinates, bypassing FBX axes."""

import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import unreal

# This helper uses only the standard library, including in UE's embedded Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from myo.render_normals import NORMALS_STYLE, smooth_corner_normals

ROOT = "/Game/Generated/Bones_v1"


def material(name, color, two_sided=False):
    path = f"{ROOT}/{name}"
    result = unreal.load_asset(path)
    if result is None:
        result = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, ROOT, unreal.Material, unreal.MaterialFactoryNew())
        editing = unreal.MaterialEditingLibrary
        base = editing.create_material_expression(result, unreal.MaterialExpressionConstant3Vector, -300, 0)
        base.set_editor_property("constant", unreal.LinearColor(*color, 1.0))
        editing.connect_material_property(base, "", unreal.MaterialProperty.MP_BASE_COLOR)
        roughness = editing.create_material_expression(result, unreal.MaterialExpressionConstant, -300, 150)
        roughness.set_editor_property("r", 0.75)
        editing.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
        editing.recompile_material(result)
        if not unreal.EditorAssetLibrary.save_loaded_asset(result):
            raise RuntimeError(f"Cannot save {path}")
    if result.get_editor_property("two_sided") != two_sided:
        result.set_editor_property("two_sided", two_sided)
        unreal.MaterialEditingLibrary.recompile_material(result)
        unreal.EditorAssetLibrary.save_loaded_asset(result)
    return result


def frame_camera(actors, camera):
    low, high = [float("inf")] * 3, [float("-inf")] * 3
    for actor in actors:
        center, extent = actor.get_actor_bounds(False)
        for i, axis in enumerate(("x", "y", "z")):
            low[i] = min(low[i], getattr(center, axis) - getattr(extent, axis))
            high[i] = max(high[i], getattr(center, axis) + getattr(extent, axis))
    center = [(a + b) / 2 for a, b in zip(low, high)]
    radius = math.sqrt(sum(((b - a) / 2) ** 2 for a, b in zip(low, high)))
    # Fit the complete arm/payload bounding sphere in the narrower vertical FOV.
    vertical_half_fov = math.atan(math.tan(math.radians(40 / 2)) * 9 / 16)
    distance = radius / math.sin(vertical_half_fov) * 1.1
    direction = [1.0, -1.5, 0.35]
    norm = math.sqrt(sum(v * v for v in direction))
    position = unreal.Vector(*(c + v / norm * distance for c, v in zip(center, direction)))
    rotation = unreal.MathLibrary.find_look_at_rotation(position, unreal.Vector(*center))
    camera.set_actor_location(position, False, True)
    camera.set_actor_rotation(rotation, False)
    camera.camera_component.set_field_of_view(40.0)
    camera.camera_component.set_aspect_ratio(16 / 9)
    camera.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
    settings = unreal.PostProcessSettings()
    settings.set_editor_property("override_auto_exposure_method", True)
    settings.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    settings.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    settings.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", 0.0)
    camera.camera_component.set_editor_property("post_process_settings", settings)
    camera.camera_component.set_editor_property("post_process_blend_weight", 1.0)
    return {"min": low, "max": high, "camera_position": [position.x, position.y, position.z]}


def transform(pose, scale=1.0):
    return unreal.Transform(location=unreal.Vector(*pose["position_cm"]),
                            rotation=unreal.Quat(*pose["quaternion_xyzw"]).rotator(),
                            scale=unreal.Vector(scale, scale, scale))


def viewer_game_mode():
    path = ROOT + "/BP_MyoViewerGameMode"
    blueprint = unreal.load_asset(path)
    if blueprint is None:
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", unreal.GameModeBase)
        blueprint = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "BP_MyoViewerGameMode", ROOT, unreal.Blueprint, factory)
    defaults = unreal.get_default_object(blueprint.generated_class())
    # The level owns its camera pawn. Never spawn an extra DefaultPawn sphere.
    defaults.set_editor_property("default_pawn_class", None)
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    if not unreal.EditorAssetLibrary.save_loaded_asset(blueprint, only_if_is_dirty=False):
        raise RuntimeError("Cannot save viewer GameMode")
    return blueprint.generated_class()


def add_camera_navigation(actors, camera):
    pawn = actors.spawn_actor_from_class(unreal.DefaultPawn, camera.get_actor_location(),
                                        camera.get_actor_rotation())
    if pawn is None:
        raise RuntimeError("Cannot create camera navigator")
    pawn.set_actor_label("MyoViewerNavigator - invisible camera controls")
    pawn.set_editor_property("auto_possess_player", unreal.AutoReceiveInput.PLAYER0)
    pawn.set_editor_property("use_controller_rotation_yaw", True)
    pawn.set_editor_property("use_controller_rotation_pitch", True)
    pawn.set_actor_enable_collision(False)
    pawn.mesh_component.set_static_mesh(None)
    pawn.mesh_component.set_visibility(False)
    pawn.mesh_component.set_editor_property("cast_shadow", False)
    movement = pawn.get_movement_component()
    # Units here are UE cm, sized for a single arm rather than a large game level.
    movement.set_editor_property("max_speed", 40.0)
    movement.set_editor_property("acceleration", 200.0)
    movement.set_editor_property("deceleration", 240.0)
    if any(c.static_mesh is not None for c in pawn.get_components_by_class(unreal.StaticMeshComponent)):
        raise RuntimeError("The camera navigator must not have a visible mesh")
    if not camera.attach_to_actor(pawn, "None", unreal.AttachmentRule.KEEP_WORLD,
                                  unreal.AttachmentRule.KEEP_WORLD,
                                  unreal.AttachmentRule.KEEP_WORLD, False):
        raise RuntimeError("Cannot attach the preview camera to its navigator")
    # A real PlayerStart also preserves the pitch/yaw on game restart.
    start = actors.spawn_actor_from_class(unreal.PlayerStart, camera.get_actor_location(),
                                         camera.get_actor_rotation())
    start.set_actor_label("MyoViewerStart")
    return {"pawn": "DefaultPawn without mesh", "render_meshes": 0, "speed_cm_s": 40.0,
            "camera_attached": camera.get_attach_parent_actor() == pawn}


def create_mesh(directory, entry, asset_root=ROOT):
    path = asset_root + f"/SM_{entry['id']}_{entry['sha256'][:12]}_{NORMALS_STYLE}"
    payload = (directory / entry["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Mesh checksum mismatch: {entry['name']}")
    buffers = json.loads(payload)
    mesh = unreal.load_asset(path)
    if mesh is None:
        dynamic = unreal.DynamicMesh()
        # Some upstream STL surfaces are nonmanifold. Keep every triangle via
        # independent render vertices instead of letting DynamicMesh drop faces.
        vertices = [buffers["vertices_cm"][index] for triangle in buffers["triangles"] for index in triangle]
        normals = smooth_corner_normals(buffers["vertices_cm"], buffers["triangles"])
        data = unreal.GeometryScriptSimpleMeshBuffers(
            vertices=[unreal.Vector(*v) for v in vertices],
            normals=[unreal.Vector(*n) for n in normals],
            # Bone STL has no UVs. UE's StaticMesh build requires at least UV0;
            # this simple diagnostic projection is not an anatomical texture UV.
            uv0=[unreal.Vector2D(v[0] * 0.1, v[1] * 0.1) for v in vertices],
            triangles=[unreal.IntVector(i, i + 1, i + 2) for i in range(0, len(vertices), 3)])
        unreal.GeometryScript_MeshEdits.append_buffers_to_mesh(dynamic, data)
        options = unreal.GeometryScriptCreateNewStaticMeshAssetOptions(
            enable_recompute_normals=False, enable_recompute_tangents=True, enable_nanite=False)
        mesh, outcome = unreal.GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh(
            dynamic, path, options)
        if mesh is None or outcome != unreal.GeometryScriptOutcomePins.SUCCESS:
            raise RuntimeError(f"Cannot build bone mesh: {entry['name']}")
    # Verify no automatic importer axis conversion or unit scale was introduced.
    bounds = mesh.get_bounds()
    expected = [(max(v[i] for v in buffers["vertices_cm"]) - min(v[i] for v in buffers["vertices_cm"])) / 2
                for i in range(3)]
    actual = [bounds.box_extent.x, bounds.box_extent.y, bounds.box_extent.z]
    if any(abs(a - b) > 0.001 for a, b in zip(actual, expected)):
        raise RuntimeError(f"Imported mesh scale/axes mismatch: {entry['name']}")
    if mesh.get_num_triangles(0) != len(buffers["triangles"]):
        raise RuntimeError(f"Imported mesh lost triangles: {entry['name']}")
    # GeometryScript registers a dirty asset but does not save its package.
    # Saving only the level leaves dangling mesh references in the next process.
    if not unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False):
        raise RuntimeError(f"Cannot persist bone mesh: {entry['name']}")
    asset_file = Path(unreal.Paths.project_content_dir()) / (path.removeprefix("/Game/") + ".uasset")
    if not asset_file.is_file():
        raise RuntimeError(f"Bone mesh package was not written: {asset_file}")
    return mesh


def prepare():
    directory = Path(__file__).resolve().parents[3] / "data/unreal/myoarm"
    geometry = json.loads((directory / "geometry.json").read_text(encoding="utf-8"))
    if geometry["schema"] != "myo.unreal_geometry.v1" or geometry["physics_feedback"]:
        raise RuntimeError("Unsupported geometry export")
    meshes = {entry["id"]: create_mesh(directory, entry) for entry in geometry["meshes"]}
    bone_material = material("M_BonePreview_v1", (0.78, 0.70, 0.52), two_sided=True)
    payload_material = material("M_PayloadPreview_v1", (0.12, 0.20, 0.30))
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    game_mode = viewer_game_mode()
    muscle_dir = directory.parent / "muscles"
    muscle_export = json.loads((muscle_dir / "muscles.json").read_text(encoding="utf-8")) if (muscle_dir / "muscles.json").exists() else None
    if muscle_export and (muscle_export["schema"] != "myo.unreal_muscles.v1" or
                          muscle_export["geometry_sha256"] != geometry["geometry_sha256"] or
                          muscle_export["physics_feedback"]):
        raise RuntimeError("Muscle fixtures do not match the bone geometry")
    muscle_meshes = {}
    if muscle_export:
        for pose in muscle_export["poses"]:
            for entry in pose["meshes"]:
                muscle_meshes[entry["id"]] = create_mesh(muscle_dir, entry, "/Game/Generated/Muscles_v1")
    muscle_material = material("M_MusclePreview_v1", (.42, .045, .035), two_sided=False) if muscle_export else None
    sphere = unreal.load_asset("/Engine/BasicShapes/Sphere")
    maps = []
    muscle_maps = []
    views = []
    for pose in geometry["poses"]:
        unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
        render_actors = []
        for entry in pose["geoms"]:
            actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
            actor.set_actor_label(entry["name"])
            component = actor.static_mesh_component
            component.set_static_mesh(meshes[entry["mesh_id"]])
            component.set_material(0, bone_material)
            component.set_simulate_physics(False)
            component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
            actor.set_actor_transform(transform(entry), False, True)
            render_actors.append(actor)
        actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
        actor.set_actor_label("payload - MuJoCo pose fixture")
        actor.static_mesh_component.set_static_mesh(sphere)
        actor.static_mesh_component.set_material(0, payload_material)
        actor.static_mesh_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actor.set_actor_transform(transform(pose["payload"], pose["payload_radius_cm"] / 50.0), False, True)
        render_actors.append(actor)
        light = actor_subsystem.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 200),
                                                      unreal.Rotator(pitch=-35, yaw=-45, roll=0))
        light.set_actor_label("Preview key - movable, no light bake")
        light.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
        light.light_component.set_intensity(3.0)
        light.light_component.set_forward_shading_priority(1)
        fill = actor_subsystem.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 200),
                                                     unreal.Rotator(pitch=-20, yaw=130, roll=0))
        fill.set_actor_label("Preview fill - movable, no light bake")
        fill.light_component.set_mobility(unreal.ComponentMobility.MOVABLE)
        fill.light_component.set_intensity(1.5)
        fill.light_component.set_forward_shading_priority(0)
        camera = actor_subsystem.spawn_actor_from_class(unreal.CameraActor, unreal.Vector())
        camera.set_actor_label("MyoPreviewCamera")
        view = {"angle_deg": pose["angle_deg"], **frame_camera(render_actors, camera),
                "auto_activate_player_index": camera.get_auto_activate_player_index()}
        view["navigation"] = add_camera_navigation(actor_subsystem, camera)
        view["light_priorities"] = [light.light_component.get_editor_property("forward_shading_priority"),
                                    fill.light_component.get_editor_property("forward_shading_priority")]
        views.append(view)
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        world.get_world_settings().set_editor_property("default_game_mode", game_mode)
        map_path = ROOT + f"/L_Bones_{pose['angle_deg']}"
        if not unreal.EditorLoadingAndSavingUtils.save_map(world, map_path):
            raise RuntimeError(f"Cannot save {map_path}")
        maps.append(map_path)
        if muscle_export:
            muscle_pose = next(p for p in muscle_export["poses"] if p["angle_deg"] == pose["angle_deg"])
            for entry in muscle_pose["meshes"]:
                actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector())
                actor.set_actor_label(f"Muscle_{entry['name']} - BodyParts3D pose fixture")
                actor.static_mesh_component.set_static_mesh(muscle_meshes[entry["id"]])
                actor.static_mesh_component.set_material(0, muscle_material)
                actor.static_mesh_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
                actor.static_mesh_component.set_simulate_physics(False)
            muscle_path = f"/Game/Generated/Muscles_v1/L_Muscles_{pose['angle_deg']}"
            if not unreal.EditorLoadingAndSavingUtils.save_map(world, muscle_path):
                raise RuntimeError(f"Cannot save {muscle_path}")
            muscle_maps.append(muscle_path)
    return {"geometry_sha256": geometry["geometry_sha256"], "meshes": len(meshes),
            "maps": maps, "views": views, "movable_lights": True, "camera_navigation": "freefly_v1",
            "render_normals": NORMALS_STYLE, "surface_geometry_changed": False,
            "muscle_maps": muscle_maps, "muscle_meshes": len(muscle_meshes),
            "muscle_mesh_files": [str(Path(unreal.Paths.project_content_dir()) / (value.get_path_name().split(".")[0].removeprefix("/Game/") + ".uasset")) for value in muscle_meshes.values()],
            "mesh_material_slots": {str(key): len(value.get_editor_property("static_materials")) for key, value in meshes.items()},
            "mesh_sections": {str(key): value.get_num_sections(0) for key, value in meshes.items()},
            "mesh_files": [str(Path(unreal.Paths.project_content_dir()) / (value.get_path_name().split(".")[0].removeprefix("/Game/") + ".uasset")) for value in meshes.values()],
            "scope": geometry["scope"], "ok": True}


def main():
    report_path = Path(unreal.Paths.project_saved_dir()) / "Myo/bones-import.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": False}
    try:
        report = prepare()
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        unreal.log(f"MYO_BONES_IMPORT {report_path}: ok={report['ok']}")


if __name__ == "__main__":
    main()
