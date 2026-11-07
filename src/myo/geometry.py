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
CONTACT_GEOM_NAMES = ("upperarm_tissue_contact", "forearm_tissue_contact")
CONTACT_GEOM_GROUP = 5
CONTACT_ENVELOPE_VERSION = "landmark_profiles_v1"


def _add_contact_profiles(root: ET.Element, mount: ET.Element) -> dict:
    """Build two continuous convex envelopes from landmark-indexed sections.

    Native -Y runs from shoulder to elbow and from elbow toward wrist; X is
    approximately anterior, Z transverse. Longitudinal stations use the source
    joint offsets. Transverse offsets/radii below are explicit engineering
    choices in metres, not measurements of BodyParts3D or an individual human.
    Narrow ends deliberately leave the shared hinge/crease out of self-contact.
    Each segment has one convex hull, avoiding overlapping primitive contacts
    on internal seams. A solver may still create a distributed contact patch.
    These fixed shapes add neither mass nor activation-dependent deformation.
    """
    ulna = mount.find(".//body[@name='ulna_r']")
    radius = mount.find(".//body[@name='radius_r']")
    wrist = mount.find(".//body[@name='lunate_r']")
    assert ulna is not None and radius is not None and wrist is not None
    elbow_anchor = np.fromstring(ulna.attrib["pos"], sep=" ")
    wrist_anchor = (np.fromstring(radius.attrib["pos"], sep=" ")
                    + np.fromstring(wrist.attrib["pos"], sep=" "))
    # label, fraction of longitudinal bone span, x/z centre, x/z radii (m).
    layouts = (
        ("humerus_r", CONTACT_GEOM_NAMES[0], -elbow_anchor[1], (
            ("proximal_cap", -.05, 0., 0., .012, .018),
            ("proximal_bulk", .12, .002, 0., .030, .035),
            ("belly_proximal", .345, .005, 0., .043, .043),
            ("belly_distal", .55, .005, 0., .043, .041),
            ("distal_taper", .74, .001, 0., .030, .030),
            ("elbow_taper", .86, -.006, 0., .016, .023),
            ("elbow_crease", .937, -.008, -.007, .010, .018),
        )),
        ("ulna_r", CONTACT_GEOM_NAMES[1], -wrist_anchor[1], (
            ("elbow_crease", .10, -.005, .012, .014, .020),
            ("proximal_bulk", .255, .001, .018, .028, .035),
            ("belly", .435, .006, .021, .031, .033),
            ("distal_taper", .63, .010, .025, .027, .028),
            ("wrist_taper", .83, .014, .030, .020, .023),
            ("wrist_cap", .97, .018, .039, .012, .016),
        )),
    )
    asset = root.find("asset")
    assert asset is not None
    profiles = []
    for body_name, name, span, layout in layouts:
        vertices, sections = [], []
        for region, fraction, x, z, rx, rz in layout:
            center = [x, -fraction * span, z]
            sections.append({"region": region, "longitudinal_fraction": fraction,
                             "center_m": center, "radii_xz_m": [rx, rz]})
            for azimuth in np.linspace(0., 2. * np.pi, 16, endpoint=False):
                vertices.extend((x + rx * np.cos(azimuth), center[1],
                                 z + rz * np.sin(azimuth)))
        ET.SubElement(asset, "mesh", name=f"{name}_profile",
                      vertex=" ".join(f"{v:.9g}" for v in vertices))
        body = mount.find(f".//body[@name='{body_name}']")
        assert body is not None
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=f"{name}_profile",
                      mass="0", contype="0", conaffinity="0", margin="0",
                      group=str(CONTACT_GEOM_GROUP), rgba="0.8 0.45 0.2 0.35")
        profiles.append({"body": body_name, "geom": name, "longitudinal_span_m": float(span),
                         "sections": sections, "ring_vertices": 16})
    return {"version": CONTACT_ENVELOPE_VERSION,
            "source": "project-generated engineering profiles; no anatomical surface import",
            "elbow_in_humerus_m": elbow_anchor.tolist(),
            "wrist_in_ulna_m": wrist_anchor.tolist(), "profiles": profiles}


def prepare_model(directory: Path) -> dict:
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
        # Source bone/wrap/legacy collision geometry remains noncontact.
        # Only the separate, named tissue envelopes below generate contact.
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    contact_profiles = _add_contact_profiles(root, mount)
    contact = ET.SubElement(root, "contact")
    # Normal compliance permits compression; no artificial contact friction or
    # angular stop is added. These solver parameters are uncalibrated proxies.
    ET.SubElement(contact, "pair", name="upperarm_forearm_tissue",
                  geom1=CONTACT_GEOM_NAMES[0], geom2=CONTACT_GEOM_NAMES[1],
                  condim="1", margin="0", solref="0.03 1", solimp="0.1 0.95 0.015 0.5 2")
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
    return contact_profiles


class MyoArmGeometry:
    """Verified local model, with six explicitly external series-elastic muscles."""

    muscle_names = MYOARM_MUSCLES

    def __init__(self, asset_dir: str | Path = "data/myoarm", *,
                 physics_dt: float = 0.001, payload_mass: float = 1.0,
                 arm_mount_tilt: float = np.deg2rad(30.0),
                 tissue_contact_time_constant: float = 0.03,
                 tissue_contact_transition_depth: float = 0.015):
        directory = Path(asset_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("MyoArm assets missing; run python3 scripts/prepare_myoarm.py first.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("commit") != MYOARM_COMMIT or not manifest.get("derived"):
            raise ValueError("MyoArm source revision or preparation is incompatible; prepare assets again.")
        if manifest.get("tissue_contact_profiles", {}).get("version") != CONTACT_ENVELOPE_VERSION:
            raise ValueError("Prepared MyoArm contact profiles are outdated; run python3 scripts/prepare_myoarm.py again.")
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
        self.contact_geom_ids = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                                      for name in CONTACT_GEOM_NAMES)
        if (min(self.contact_geom_ids) < 0 or self.model.npair != 1
                or set(self.contact_geom_ids) != {int(self.model.pair_geom1[0]), int(self.model.pair_geom2[0])}
                or any(self.model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH for i in self.contact_geom_ids)):
            raise ValueError("Prepared MyoArm lacks tissue contact envelopes; run python3 scripts/prepare_myoarm.py again.")
        self.contact_pair_ids = (0,)
        self.model.pair_solref[0] = [tissue_contact_time_constant, 1.0]
        self.model.pair_solimp[0] = [0.1, 0.95, tissue_contact_transition_depth, 0.5, 2.0]
        # Rotate the complete fixed arm mount, including attachments and inertia;
        # joint angle still measures the same anatomical elbow coordinate.
        half_tilt = float(arm_mount_tilt) / 2.0
        self.model.body_quat[self.model.body("arm_mount").id] = np.asarray([
            np.cos(half_tilt), np.cos(half_tilt), -np.sin(half_tilt), np.sin(half_tilt),
        ]) / np.sqrt(2.0)
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
