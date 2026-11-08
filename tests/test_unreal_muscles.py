"""Check the reviewed arm subset and preserve topology at the UE boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

from myo.geometry import MYOARM_MUSCLES
from myo.unreal_muscles import export_muscles, read_obj


def test_bodyparts_surfaces_keep_identity_topology_and_units(tmp_path):
    root = Path("data/bodyparts3d")
    if not (root / "manifest.json").exists() or not Path("data/myoarm/manifest.json").exists():
        pytest.skip("Prepare the licensed local MyoArm and BodyParts3D assets first")
    report = export_muscles(Path("configs/d1_myoarm.json"), root, tmp_path)
    assert not report["physics_feedback"] and not report["active_deformation"]
    entries = {entry["myo_id"]: entry for entry in report["source"]["assets"]}
    assert len({entries[name]["fma_id"] for name in MYOARM_MUSCLES}) == 6
    for fit in report["fits"].values():
        assert np.linalg.det(fit["rotation"]) == pytest.approx(1)
        assert .0005 < fit["scale_m_per_mm"] < .002
    assert [pose["angle_deg"] for pose in report["poses"]] == [15, 45, 90, 120]
    for pose in report["poses"]:
        assert {mesh["name"] for mesh in pose["meshes"]} == set(MYOARM_MUSCLES)
        for mesh in pose["meshes"]:
            source_vertices, source_faces = read_obj(root / "source" / (entries[mesh["name"]]["element_id"] + ".obj"))
            exported = json.loads((tmp_path / mesh["file"]).read_text())
            vertices = np.asarray(exported["vertices_cm"])
            assert vertices.shape == source_vertices.shape
            assert np.isfinite(vertices).all()
            # An arm muscle spans centimetres, never the millimetre source range.
            assert 5 < np.linalg.norm(np.ptp(vertices, axis=0)) < 70
            np.testing.assert_array_equal(exported["triangles"], source_faces[:, [0, 2, 1]])
