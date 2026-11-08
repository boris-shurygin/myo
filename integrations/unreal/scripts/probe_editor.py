"""Run with UE's Python commandlet; no project Python dependencies required."""

import json
from pathlib import Path
import traceback

import unreal


def prepare_tetrahedral_specimen():
    """Build a small passive test volume, distinct from anatomical muscle assets."""
    root = "/Game/Generated/FleshProbe_v1"
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    graph = unreal.load_asset(f"{root}/DF_Probe")
    if graph is None:
        graph = tools.create_asset("DF_Probe", root, unreal.Dataflow, unreal.DataflowAssetFactory())
        library = unreal.DataflowEditorBlueprintLibrary
        nodes = []
        for node_type, name, properties in (
            ("FRadialTetrahedronDataflowNodes", "TetVolume", {
                "InnerRadius": "1.0", "OuterRadius": "3.0", "Height": "20.0",
                "RadialSample": "2", "AngularSample": "8", "VerticalSample": "6",
                "BulgeRatio": "0.2",
            }),
            ("FSetFleshDefaultPropertiesNode", "Material", {"VertexDamping": "0.05"}),
            ("FFleshAssetTerminalDataflowNode", "FleshAssetTerminal", {}),
        ):
            node = library.add_dataflow_node(graph, node_type, name, unreal.Vector2D(350 * len(nodes), 0))
            if str(node) == "None":
                raise RuntimeError(f"Cannot create {node_type}")
            for key, value in properties.items():
                if not library.set_dataflow_node_property(graph, node, key, value):
                    raise RuntimeError(f"Cannot set {name}.{key}")
            if nodes and not library.connect_dataflow_nodes(graph, nodes[-1], "Collection", node, "Collection"):
                raise RuntimeError(f"Cannot connect {name}")
            nodes.append(node)
        if not unreal.EditorAssetLibrary.save_loaded_asset(graph):
            raise RuntimeError("Cannot save the specimen graph")
    flesh = unreal.load_asset(f"{root}/Flesh_Probe")
    if flesh is None:
        flesh = tools.create_asset("Flesh_Probe", root, unreal.FleshAsset, unreal.FleshAssetFactory())
    flesh.set_dataflow_asset(graph)
    flesh.set_dataflow_terminal("FleshAssetTerminal")
    # Bind the result explicitly for the graph evaluator. In the installed
    # 5.8.4 source EvaluateDataflow returns bHasError, opposite to its comment.
    # The owner/terminal helpers call a base Evaluate overload with an ensure.
    if not unreal.DataflowEditorBlueprintLibrary.set_dataflow_node_property(
        graph, "FleshAssetTerminal", "FleshAsset", flesh.get_path_name()
    ):
        raise RuntimeError("Cannot bind the Flesh terminal asset")
    if unreal.DataflowBlueprintLibrary.evaluate_dataflow(graph, flesh):
        raise RuntimeError("Dataflow reported an evaluation error")
    if not unreal.EditorAssetLibrary.save_loaded_asset(flesh):
        raise RuntimeError("Cannot save the evaluated specimen")
    # A nonzero scene bound checks that evaluation produced geometry, rather
    # than accepting an empty asset merely because a graph exists.
    unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actors.spawn_actor_from_class(unreal.FleshActor, unreal.Vector(0, 0, 0))
    actor.set_actor_label("D0 passive tet specimen - not an anatomical muscle")
    component = actor.get_editor_property("flesh_component")
    component.set_rest_collection(flesh)
    actor.set_actor_location(unreal.Vector(0, 0, 100), False, True)
    origin, extent, radius = unreal.SystemLibrary.get_component_bounds(component)
    if radius <= 0:
        raise RuntimeError("Evaluated Flesh specimen has empty bounds")
    level_path = f"{root}/L_FleshProbe"
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if not unreal.EditorLoadingAndSavingUtils.save_map(world, level_path):
        raise RuntimeError("Cannot save the specimen level")
    return {
        "asset": flesh.get_path_name(), "graph": graph.get_path_name(), "level": level_path,
        "bounds_extent_cm": [extent.x, extent.y, extent.z],
        "scope": "Passive tetrahedral geometry only; solver motion and activation curves untested.",
    }


def main():
    specimen_requested = "-myo-flesh-probe" in unreal.SystemLibrary.get_command_line()
    filename = "flesh-probe.json" if specimen_requested else "editor-probe.json"
    report_path = Path(unreal.Paths.project_saved_dir()) / "Myo" / filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "engine_version": unreal.SystemLibrary.get_engine_version()}
    try:
        required = (
            "FleshAsset", "FleshActor", "FleshComponent", "DeformableSolverActor",
            "Dataflow", "DataflowAssetFactory", "DataflowEditorBlueprintLibrary",
        )
        report["classes"] = {name: hasattr(unreal, name) for name in required}
        if not all(report["classes"].values()):
            raise RuntimeError("Required Flesh/Dataflow Python types are missing")
        templates = (
            "/ChaosFlesh/Templates/DF_Muscle.DF_Muscle",
            "/ChaosFlesh/Templates/DF_SimpleFleshSetup.DF_SimpleFleshSetup",
            "/ChaosFlesh/Deformers/DG_FleshDeformer.DG_FleshDeformer",
        )
        report["engine_assets"] = {
            path: unreal.load_asset(path) is not None for path in templates
        }
        if not all(report["engine_assets"].values()):
            raise RuntimeError("Required engine templates are unavailable")
        # Check actual reflected API, not just the presence of the plugin folder.
        graph = unreal.new_object(unreal.Dataflow)
        node = unreal.DataflowEditorBlueprintLibrary.add_dataflow_node(
            graph, "FFleshAssetTerminalDataflowNode", "FleshAssetTerminal", unreal.Vector2D(0, 0)
        )
        if str(node) == "None":
            raise RuntimeError("Could not create a Flesh terminal node")
        report["terminal_node"] = str(node)
        if specimen_requested:
            report["specimen"] = prepare_tetrahedral_specimen()
        report["scope"] = "Plugin loading and graph authoring; tissue simulation is not established."
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        unreal.log_error(report["error"])
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        unreal.log(f"MYO_EDITOR_PROBE {report_path}: ok={report['ok']}")


if __name__ == "__main__":
    main()
