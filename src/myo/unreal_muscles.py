"""Preliminary BodyParts3D registration and kinematic surface fixtures, not Flesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from myo.geometry import MyoArmGeometry, MYOARM_MUSCLES
from myo.runtime import read_config
from myo.unreal_geometry import MyoToUnreal, write_json


def read_obj(path):
    vertices, triangles = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields and fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"Expected triangular source faces: {path}")
            triangles.append([int(value.split("/")[0]) - 1 for value in fields[1:]])
    vertices = np.asarray(vertices, dtype=float)
    # OBJ normal seams repeat geometrically identical vertices. Join only exact
    # duplicates before reporting open edges or preparing a future tet surface.
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    return unique, inverse[np.asarray(triangles, dtype=int)]


def surface_report(vertices, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return {"vertices": len(vertices), "triangles": len(faces),
            "boundary_edges": int(np.count_nonzero(counts == 1)),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "degenerate_triangles": int(np.count_nonzero(np.linalg.norm(np.cross(b - a, c - a), axis=1) < 1e-12)),
            "signed_volume_mm3": float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)}


def nearest(source, target):
    indices, distances = [], []
    for start in range(0, len(source), 128):
        delta = source[start:start + 128, None, :] - target[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        closest = squared.argmin(axis=1)
        indices.extend(closest)
        distances.extend(squared[np.arange(len(closest)), closest])
    return np.asarray(indices), np.sqrt(distances)


def anatomical_frame(points, proximal_hint, anterior_hint):
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    y = axes[0] * (1 if axes[0] @ proximal_hint > 0 else -1)
    x = np.asarray(anterior_hint) - y * (y @ anterior_hint)
    x /= np.linalg.norm(x)
    return np.column_stack([x, y, np.cross(x, y)])


def fit_bone(source_mm, target_m):
    # Anatomical orientation constrains the initialization: no mirrored fits.
    source = source_mm[np.linspace(0, len(source_mm) - 1, min(700, len(source_mm)), dtype=int)]
    target = target_m[np.linspace(0, len(target_m) - 1, min(1600, len(target_m)), dtype=int)]
    sf = anatomical_frame(source, [0, 0, 1], [0, -1, 0])
    tf = anatomical_frame(target, [0, 1, 0], [1, 0, 0])
    rotation = tf @ sf.T
    scale = np.ptp(target @ tf[:, 1]) / np.ptp(source @ sf[:, 1])
    initial_scale = scale
    translation = target.mean(axis=0) - scale * source.mean(axis=0) @ rotation.T
    for _ in range(25):
        indices, distances = nearest(scale * source @ rotation.T + translation, target)
        keep = distances <= np.quantile(distances, .85)
        x, y = source[keep], target[indices[keep]]
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        u, singular, vt = np.linalg.svd((x - xm).T @ (y - ym))
        signs = np.array([1., 1., np.linalg.det(vt.T @ u.T)])
        rotation = vt.T @ np.diag(signs) @ u.T
        scale = np.clip((singular * signs).sum() / np.square(x - xm).sum(), .8 * initial_scale, 1.2 * initial_scale)
        translation = ym - scale * xm @ rotation.T
    _, distances = nearest(scale * source @ rotation.T + translation, target)
    return {"scale_m_per_mm": float(scale), "rotation": rotation.tolist(), "translation_m": translation.tolist(),
            "mean_vertex_gap_mm": float(distances.mean() * 1000), "p95_vertex_gap_mm": float(np.quantile(distances, .95) * 1000),
            "method": "Anatomical PCA initialization + trimmed similarity ICP; preliminary, not anatomical calibration"}


def apply_fit(vertices, fit):
    return vertices @ np.asarray(fit["rotation"]).T * fit["scale_m_per_mm"] + fit["translation_m"]


def export_muscles(config_path, source_dir, output):
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source, topology = {}, {}
    for entry in manifest["assets"]:
        path = source_dir / "source" / (entry["element_id"] + ".obj")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed BodyParts3D surface: {path}")
        vertices, faces = read_obj(path)
        source[entry["myo_id"]] = (vertices, faces)
        topology[entry["myo_id"]] = surface_report(vertices, faces)
    config = read_config(config_path)
    if config.geometry != "myoarm":
        raise ValueError("BodyParts3D registration requires the MyoArm geometry")
    geometry = MyoArmGeometry(config.asset_dir, physics_dt=config.physics_dt,
                              payload_mass=config.payload_mass, arm_mount_tilt=config.arm_mount_tilt,
                              tissue_contact_time_constant=config.tissue_contact_time_constant,
                              tissue_contact_transition_depth=config.tissue_contact_transition_depth)
    model, fits = geometry.model, {}
    for name in ("humerus_r", "radius_r", "ulna_r"):
        geom = model.geom(name).id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        local = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        fits[name] = fit_bone(source[name][0], local)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    humerus = model.body("humerus_r").id
    elbow = (data.xanchor[geometry.elbow_id] - data.xpos[humerus]) @ data.xmat[humerus].reshape(3, 3)
    output.mkdir(parents=True, exist_ok=True)
    poses, weights = [], {}
    for angle in (15, 45, 90, 120):
        data.qpos[geometry.qpos_index] = np.deg2rad(angle)
        mujoco.mj_forward(model, data)
        meshes = []
        for name in MYOARM_MUSCLES:
            vertices, faces = source[name]
            proximal = apply_fit(vertices, fits["humerus_r"])
            distal_name = "radius_r" if name.startswith("BIC") else "ulna_r"
            distal = apply_fit(vertices, fits[distal_name])
            distal_id = model.body(distal_name).id
            # Temporary pose skinning at the elbow, not activation or a tissue law.
            t = np.clip((elbow[1] - proximal[:, 1] + .025) / .045, 0, 1)
            weight = (t * t * (3 - 2 * t))[:, None]
            weights[name] = {"distal_min": float(weight.min()), "distal_max": float(weight.max()),
                             "blend_width_m": .045, "proximal_body": "humerus_r", "distal_body": distal_name}
            p = proximal @ data.xmat[humerus].reshape(3, 3).T + data.xpos[humerus]
            d = distal @ data.xmat[distal_id].reshape(3, 3).T + data.xpos[distal_id]
            world = (1 - weight) * p + weight * d
            # Preserve the source topology; OBJ files already use outward winding.
            filename = f"{name}_{angle}.json"
            write_json(output / filename, {"vertices_cm": MyoToUnreal.positions(world).tolist(),
                                           "triangles": faces[:, [0, 2, 1]].tolist()})
            meshes.append({"id": f"{name}_{angle}", "name": name, "file": filename,
                           "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
        poses.append({"angle_deg": angle, "meshes": meshes})
    report = {"schema": "myo.unreal_muscles.v1", "geometry_sha256": geometry.fingerprint,
              "source": manifest, "fits": fits, "source_topology": topology, "pose_skinning": weights,
              "poses": poses, "physics_feedback": False, "active_deformation": False,
              "scope": "Six BodyParts3D surfaces with preliminary bone registration and pose skinning only; no Flesh, activation, calibrated attachments or skin.",
              "modifications": ["Weld exact duplicate OBJ vertex positions, preserving faces", "OBJ mm to MyoArm SI using per-bone similarity registration", "Temporary elbow blend for four static pose fixtures", "MyoToUnreal cm/axis/winding boundary"]}
    write_json(output / "muscles.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d1_myoarm.json"))
    parser.add_argument("--source", type=Path, default=Path("data/bodyparts3d"))
    parser.add_argument("--output", type=Path, default=Path("data/unreal/muscles"))
    args = parser.parse_args()
    report = export_muscles(args.config, args.source, args.output)
    print(json.dumps({"fits": report["fits"], "source_topology": report["source_topology"]}, indent=2))


if __name__ == "__main__":
    main()
