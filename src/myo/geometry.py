"""MyoArm spatial paths shared by external muscle dynamics and rendering.

Third-party assets remain outside the Python package. Run
``python3 scripts/prepare_myoarm.py`` once before selecting MyoArm geometry.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from myo.muscle import MuscleParameters

MYOARM_COMMIT = "93b0ca8f4ec90c9899ee7f05fee561e9911da91b"
MYOARM_MUSCLES = ("BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed")


def prepare_model(directory: Path) -> None:
    """Create a clearly marked derivative; leave downloaded originals intact."""
    assets = directory / "upstream/myo_sim/models/arm/assets"
    source = ET.parse(assets / "myoarm_r_assets.xml").getroot()
    root = ET.Element("mujoco", model="myoarm_preacher_curl")
    root.append(ET.Comment(f" Derived from MyoSim {MYOARM_COMMIT}; see manifest.json and upstream/LICENSE. "))
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="auto",
                  balanceinertia="true", boundmass="0.001", boundinertia="0.0001",
                  meshdir="upstream/myo_sim/models")
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 -9.81",
                  integrator="Euler", jacobian="dense")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.8 0.8 0.8")
    ET.SubElement(root, "statistic", center="0.2 -0.13 0.98", extent="0.55")
    for tag in ("default", "asset"):
        root.append(copy.deepcopy(source.find(tag)))
    # Native muscle defaults are harmless without actuators, but are removed to
    # make the single external force path evident in the prepared XML as well.
    for node in root.iter("default"):
        for child in list(node):
            if child.tag in ("general", "muscle"):
                node.remove(child)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "camera", name="overview", pos="0.75 -1.1 1.1",
                  xyaxes="0.877 0.480 0 -0.030 0.056 0.998")
    # Ry(-30 deg) * Rx(90 deg): native Y-up anatomy becomes Z-up, upper
    # arm inclines forward. This explicit mount belongs to mechanics, not a
    # hidden renderer conversion; all geometric distances remain in metres.
    mount = ET.SubElement(world, "body", name="arm_mount", pos="0 0 1.15",
                          quat="0.683012701892 0.683012701892 -0.183012701892 0.183012701892")
    chain = ET.parse(assets / "myoarm_r_chain.xml").getroot()
    for body in chain:
        mount.append(copy.deepcopy(body))
    for body in mount.iter("body"):
        for joint in list(body.findall("joint")):
            if joint.get("name") == "elbow_flexion_r":
                joint.set("name", "elbow")
                # Keep the extension constraint but remove the upper flexion
                # stop entirely. MuJoCo supports a one-sided infinite range.
                joint.set("range", f"{np.deg2rad(15)} inf")
            else:
                body.remove(joint)
        # All geom collisions are disabled in this constrained arm. Source
        # inertials are retained; render geometry cannot add contact forces.
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    hand = mount.find(".//body[@name='lunate_r']")
    assert hand is not None
    payload = ET.SubElement(hand, "body", name="payload", pos="0 -0.08 0")
    ET.SubElement(payload, "geom", name="payload", type="sphere", size="0.055",
                  mass="1", rgba="0.2 0.3 0.4 1", contype="0", conaffinity="0")
    tendon = ET.SubElement(root, "tendon")
    upstream_tendons = ET.parse(assets / "myoarm_r_tendon.xml").getroot()
    upstream_muscles = ET.parse(assets / "myoarm_r_muscle.xml").getroot()
    parameters = []
    for name in MYOARM_MUSCLES:
        path = copy.deepcopy(upstream_tendons.find(f".//spatial[@name='{name}_tendon']"))
        assert path is not None
        path.set("stiffness", "0")
        path.set("damping", "0")
        path.set("width", "0.003")
        tendon.append(path)
        actuator = upstream_muscles.find(f".//general[@name='{name}']")
        assert actuator is not None
        gain = np.fromstring(actuator.attrib["gainprm"], sep=" ")
        length_range = np.fromstring(actuator.attrib["lengthrange"], sep=" ")
        # MuJoCo's normalized length = (L_MT - L_T)/L_0. Preserve its
        # geometric length normalization and force scale only. Our compliant
        # law, passive curve, delay and activation are intentionally separate.
        optimal = float(np.diff(length_range)[0] / (gain[1] - gain[0]))
        slack = float(length_range[0] - gain[0] * optimal)
        parameters.append({"name": name, "max_isometric_force": float(gain[2]),
                           "optimal_fiber_length": optimal, "tendon_slack_length": slack,
                           "upstream_gainprm": gain.tolist(),
                           "upstream_lengthrange_m": length_range.tolist()})
    ET.indent(root)
    ET.ElementTree(root).write(directory / "model.xml", encoding="unicode")
    (directory / "muscles.json").write_text(json.dumps(parameters, indent=2) + "\n")


class MyoArmGeometry:
    """Verified local model, with six explicitly external series-elastic muscles."""

    muscle_names = MYOARM_MUSCLES

    def __init__(self, asset_dir: str | Path = "data/myoarm", *,
                 physics_dt: float = 0.001, payload_mass: float = 1.0):
        directory = Path(asset_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("MyoArm assets missing; run python3 scripts/prepare_myoarm.py first.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("commit") != MYOARM_COMMIT or not manifest.get("derived"):
            raise ValueError("MyoArm source revision or preparation is incompatible; prepare assets again.")
        for entry in manifest["derived"]:
            if hashlib.sha256((directory / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Prepared MyoArm asset changed: {entry['path']}")
        # Geometry meshes also affect rendering and inertia; never silently use
        # modified files under the same checkpoint fingerprint.
        for entry in manifest["files"]:
            if hashlib.sha256((directory / "upstream" / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"MyoArm source asset changed: {entry['path']}")
        self.fingerprint = f"myoarm:{MYOARM_COMMIT}:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
        self.model.opt.timestep = physics_dt
        self.elbow_id = self.model.joint("elbow").id
        if not np.isposinf(self.model.jnt_range[self.elbow_id, 1]):
            raise ValueError("Prepared MyoArm still has a flexion limit; run python3 scripts/prepare_myoarm.py again.")
        self.qpos_index = int(self.model.jnt_qposadr[self.elbow_id])
        self.dof_index = int(self.model.jnt_dofadr[self.elbow_id])
        self.payload_id = self.model.body("payload").id
        self.model.body_mass[self.payload_id] = payload_mass
        self.model.body_inertia[self.payload_id] = 0.4 * payload_mass * 0.055**2
        self.tendon_ids = np.asarray([self.model.tendon(f"{name}_tendon").id for name in self.muscle_names])
        self.muscle_parameters = tuple(MuscleParameters(**{
            key: value for key, value in entry.items() if not key.startswith("upstream_")
        }) for entry in json.loads((directory / "muscles.json").read_text()))
        if self.model.nq != 1 or self.model.nv != 1 or self.model.nu != 0:
            raise ValueError("MyoArm must have exactly one DOF and no native actuators.")
        self._scratch = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self._scratch)

    def lengths(self, data: mujoco.MjData) -> np.ndarray:
        return data.ten_length[self.tendon_ids].copy()

    def moment_arms(self, data: mujoco.MjData) -> np.ndarray:
        return -data.ten_J.reshape(self.model.ntendon, self.model.nv)[self.tendon_ids, self.dof_index].copy()

    def lengths_at(self, angle: float) -> np.ndarray:
        """Query geometry without changing the live MuJoCo state."""
        self._scratch.qpos[self.qpos_index] = angle
        mujoco.mj_forward(self.model, self._scratch)
        return self.lengths(self._scratch)

    def path_points(self, data: mujoco.MjData) -> tuple[np.ndarray, ...]:
        """MuJoCo's attachment/wrap tangent points in world metres.

        Wrap tangent pairs delimit curved segments; use MuJoCo native tendon
        rendering for the arcs rather than replacing these with straight lines.
        """
        points = data.wrap_xpos.reshape(-1, 3)
        return tuple(points[int(data.ten_wrapadr[i]):int(data.ten_wrapadr[i] + data.ten_wrapnum[i])].copy()
                     for i in self.tendon_ids)
