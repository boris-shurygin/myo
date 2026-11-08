"""Protect the renderer boundary from double mesh offsets and mirrored poses."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myo.geometry import MyoArmGeometry
from myo.unreal_geometry import export_geometry


def test_exported_bones_preserve_world_geometry_in_four_poses(tmp_path):
    if not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare the licensed local MyoArm assets first")
    exported = export_geometry(Path("configs/d1_myoarm.json"), tmp_path)
    geometry = MyoArmGeometry()
    model = geometry.model
    data = mujoco.MjData(model)
    basis = np.diag([1., -1., 1.])
    meshes = {entry["id"]: json.loads((tmp_path / entry["file"]).read_text())
              for entry in exported["meshes"]}
    for pose in exported["poses"]:
        data.qpos[geometry.qpos_index] = np.deg2rad(pose["angle_deg"])
        mujoco.mj_forward(model, data)
        for item in pose["geoms"]:
            geom_id, mesh_id = model.geom(item["name"]).id, item["mesh_id"]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start:start + count].astype(float)
            source_world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            xyzw = np.array(item["quaternion_xyzw"])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, xyzw[[3, 0, 1, 2]])
            actual_world = np.asarray(meshes[mesh_id]["vertices_cm"]) @ rotation.reshape(3, 3).T + item["position_cm"]
            np.testing.assert_allclose(actual_world, source_world @ basis * 100, atol=1e-8)
            fa = model.mesh_faceadr[mesh_id]
            count = model.mesh_facenum[mesh_id]
            np.testing.assert_array_equal(meshes[mesh_id]["triangles"], model.mesh_face[fa:fa + count][:, [0, 2, 1]])
    assert [pose["angle_deg"] for pose in exported["poses"]] == [15, 45, 90, 120]
    assert not exported["physics_feedback"]
