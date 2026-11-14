"""Offline render geometry for UE; no change to physical state or telemetry V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from myo.geometry import MyoArmGeometry, MYOARM_COMMIT
from myo.runtime import read_config, config_digest


class MyoToUnreal:
    """Named renderer boundary: metres/right-Z-up -> cm/left-Z-up, reflect Y."""

    basis = np.diag([1.0, -1.0, 1.0])

    @classmethod
    def positions(cls, values):
        return np.asarray(values) @ cls.basis.T * 100.0

    @classmethod
    def pose(cls, position, rotation):
        matrix = cls.basis @ np.asarray(rotation).reshape(3, 3) @ cls.basis.T
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, matrix.ravel())
        return {"position_cm": cls.positions(position).tolist(),
                "quaternion_xyzw": quaternion[[1, 2, 3, 0]].tolist()}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def export_geometry(config_path: Path, output: Path):
    config = read_config(config_path)
    if config.geometry != "myoarm":
        raise ValueError("The UE geometry export requires MyoArm.")
    geometry = MyoArmGeometry(config.asset_dir, physics_dt=config.physics_dt,
                              payload_mass=config.payload_mass, arm_mount_tilt=config.arm_mount_tilt,
                              tissue_contact_time_constant=config.tissue_contact_time_constant,
                              tissue_contact_transition_depth=config.tissue_contact_transition_depth)
    model = geometry.model
    output.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)  # Dedicated kinematic fixture; never step the environment.
    geoms = [i for i in range(model.ngeom)
             if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[i] == 0]
    meshes = []
    for mesh_id in sorted({int(model.geom_dataid[i]) for i in geoms}):
        va, fa = model.mesh_vertadr[mesh_id], model.mesh_faceadr[mesh_id]
        vertices = model.mesh_vert[va:va + model.mesh_vertnum[mesh_id]]
        faces = model.mesh_face[fa:fa + model.mesh_facenum[mesh_id]]
        # Compiled vertices include MuJoCo's mesh preprocessing. geom_pos/quat
        # below include its matching principal-frame offset: do not apply it twice.
        filename = f"mesh_{mesh_id}.json"
        write_json(output / filename, {"vertices_cm": MyoToUnreal.positions(vertices).tolist(),
                                       "triangles": faces[:, [0, 2, 1]].tolist()})
        meshes.append({"id": mesh_id, "name": model.mesh(mesh_id).name, "file": filename,
                       "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
    poses = []
    for angle in (15, 45, 90, 120):
        data.qpos[geometry.qpos_index] = np.deg2rad(angle)
        mujoco.mj_forward(model, data)
        poses.append({"angle_deg": angle, "geoms": [
            {"name": model.geom(i).name, "mesh_id": int(model.geom_dataid[i]),
             **MyoToUnreal.pose(data.geom_xpos[i], data.geom_xmat[i])} for i in geoms],
            "payload": MyoToUnreal.pose(data.xpos[geometry.payload_id], data.xmat[geometry.payload_id]),
            "payload_radius_cm": float(model.geom_size[model.geom("payload").id, 0] * 100.0)})
    packet = {
        "schema": "myo.unreal_geometry.v1", "source_commit": MYOARM_COMMIT,
        "geometry_sha256": geometry.fingerprint, "config_sha256": config_digest(config),
        "boundary": {"name": "MyoToUnreal", "basis": MyoToUnreal.basis.tolist(),
                     "length_scale": 100, "quaternion_order": "xyzw", "triangle_winding_reversed": True},
        "muscle_ids": list(geometry.muscle_names), "meshes": meshes, "poses": poses,
        "bodies_si": [{"name": model.body(i).name,
                       "parent": model.body(int(model.body_parentid[i])).name,
                       "local_position_m": model.body_pos[i].tolist(),
                       "local_quaternion_wxyz": model.body_quat[i].tolist()} for i in range(1, model.nbody)],
        "mesh_offsets_si": [{"name": model.geom(i).name, "body": model.body(int(model.geom_bodyid[i])).name,
                             "position_m": model.geom_pos[i].tolist(),
                             "quaternion_wxyz": model.geom_quat[i].tolist()} for i in geoms],
        "physics_feedback": False,
        "scope": "Kinematic bone import fixtures; no muscle surfaces, rig, live controller or tissue deformation.",
        "provenance": "Derived MyoArm geometry for local noncommercial use. See data/myoarm/manifest.json and docs/myoarm.md.",
    }
    write_json(output / "geometry.json", packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_myoarm.json"))
    parser.add_argument("--output", type=Path, default=Path("data/unreal/myoarm"))
    args = parser.parse_args()
    packet = export_geometry(args.config, args.output)
    print(f"Exported {len(packet['meshes'])} bone meshes and {len(packet['poses'])} kinematic poses to {args.output}")


if __name__ == "__main__":
    main()
