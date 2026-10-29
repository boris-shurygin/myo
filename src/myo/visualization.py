"""Presentation-only geometry for the V0 arm, independent of its dynamics.

The bones, attachment markers, and curved muscle paths are a schematic. Their
drawn lengths are not the constant-moment-arm paths used by ``env.py`` and must
not be used to calculate force, tendon strain, or joint torque. Activation
changes color only: this module does not imply a volumetric muscle simulation.

All geometry is appended to an existing ``MjvScene``. No model or simulation
data is written, and no forward-dynamics or kinematics call is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .env import PreacherCurlEnv


# A fixed perceptual scale, deliberately more distinct at low activations.
# These are absolute activation values, never per-frame minima or maxima.
# The same public stops/helper can be used by a viewer's color legend.
ACTIVATION_COLOR_STOPS = (
    (0.00, (0.28, 0.43, 0.57)),
    (0.08, (0.14, 0.52, 0.74)),
    (0.25, (0.95, 0.58, 0.10)),
    (0.60, (1.00, 0.36, 0.12)),
    (1.00, (0.90, 0.12, 0.055)),
)

_BONE = (0.88, 0.86, 0.77, 1.0)
_BONE_END = (0.96, 0.94, 0.85, 1.0)
_TENDON = (0.91, 0.92, 0.89, 1.0)
_ATTACHMENT = (1.0, 0.91, 0.63, 1.0)
_JOINT = (0.23, 0.30, 0.36, 1.0)
_JOINT_FACE = (0.70, 0.77, 0.79, 1.0)
_WEIGHT = (0.24, 0.29, 0.34, 1.0)
_HANDLE = (0.60, 0.66, 0.69, 1.0)
_PATH_SEGMENTS = 20
_BELLY_START = 0.15
_BELLY_END = 0.80
_FOREARM_BONES = {"radius": (-0.009, -0.014), "ulna": (0.009, 0.014)}


@dataclass(frozen=True)
class SchematicMusclePath:
    """Four world-space cubic Bezier points, not a mechanical muscle path."""

    name: str
    control_points: np.ndarray
    belly_radius: float


@dataclass(frozen=True)
class ArmAnchors:
    """A read-only calculation from the model's fixed frames and current q."""

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    upper_rotation: np.ndarray
    forearm_rotation: np.ndarray
    forearm_origin: np.ndarray
    muscles: tuple[SchematicMusclePath, ...]


def activation_rgba(activation: float) -> np.ndarray:
    """Map absolute activation to RGBA; finite values are clipped to [0, 1]."""
    value = float(activation)
    if not np.isfinite(value):
        raise ValueError("Activation color requires a finite value.")
    value = float(np.clip(value, 0.0, 1.0))
    levels = [stop[0] for stop in ACTIVATION_COLOR_STOPS]
    colors = np.asarray([stop[1] for stop in ACTIVATION_COLOR_STOPS])
    return np.asarray(
        [*(np.interp(value, levels, colors[:, channel]) for channel in range(3)), 1.0],
        dtype=np.float32,
    )


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def forearm_bone_axis(name: str, length: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one schematic shaft's endpoints/radius in the elbow-local frame.

    Both drawing and attachments use this geometry. The local frame has the
    forearm's orientation and points down the shaft along negative Z.
    """
    if not np.isfinite(length) or length <= 0.060:
        raise ValueError("The schematic forearm length must exceed 60 mm.")
    x, y = _FOREARM_BONES[name]
    return np.array((x, y, -0.030)), np.array((0.75 * x, 0.75 * y, -length + 0.030)), 0.0075


def forearm_bone_surface(name: str, length: float, toward_local: np.ndarray) -> np.ndarray:
    """Find the closest capsule surface toward an illustrative attachment point."""
    start, end, radius = forearm_bone_axis(name, length)
    target = np.asarray(toward_local, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("The attachment reference must be a finite 3D point.")
    axis = end - start
    fraction = np.clip(np.dot(target - start, axis) / np.dot(axis, axis), 0.0, 1.0)
    nearest = start + fraction * axis
    direction = target - nearest
    distance = np.linalg.norm(direction)
    if distance < 1e-12:
        raise ValueError("The attachment reference must specify a side of the bone.")
    return nearest + radius * direction / distance


def schematic_anchors(model: mujoco.MjModel, elbow_angle: float) -> ArmAnchors:
    """Calculate V0 display anchors without changing ``model`` or ``MjData``.

    This reads the fixed upper-arm frame and rotates the forearm about its
    hinge. Using q directly avoids using potentially one-substep-old ``xpos``
    after ``mj_step2``. The attachment choices below are illustrative and
    deliberately separated so four muscle signals remain visible.
    """
    angle = float(elbow_angle)
    if not np.isfinite(angle):
        raise ValueError("The displayed elbow angle must be finite.")
    upper_id = model.body("upper_arm").id
    forearm_id = model.body("forearm").id
    payload_id = model.body("payload").id
    elbow_id = model.joint("elbow").id
    if int(model.body_parentid[upper_id]) != 0:
        raise ValueError("The V0 schematic expects a fixed upper arm attached to world.")

    shoulder = model.body_pos[upper_id].copy()
    upper_rotation = _quaternion_matrix(model.body_quat[upper_id])
    base_rotation = upper_rotation @ _quaternion_matrix(model.body_quat[forearm_id])
    base_origin = shoulder + upper_rotation @ model.body_pos[forearm_id]
    joint_position = model.jnt_pos[elbow_id]
    elbow = base_origin + base_rotation @ joint_position
    q_index = int(model.jnt_qposadr[elbow_id])
    forearm_rotation = base_rotation @ _axis_rotation(
        model.jnt_axis[elbow_id], angle - float(model.qpos0[q_index])
    )
    forearm_origin = elbow - forearm_rotation @ joint_position
    wrist = forearm_origin + forearm_rotation @ model.body_pos[payload_id]

    def upper(point):
        return shoulder + upper_rotation @ np.asarray(point, dtype=np.float64)

    def forearm(point):
        return forearm_origin + forearm_rotation @ np.asarray(point, dtype=np.float64)

    # Brachialis attaches to the displayed ulna surface. Intermediate controls
    # fan the schematic paths apart without moving that endpoint off its bone.
    brachialis_insertion = elbow + forearm_rotation @ forearm_bone_surface(
        "ulna", float(np.linalg.norm(wrist - elbow)), np.array((-0.008, 0.015, -0.060))
    )
    paths = (
        SchematicMusclePath("biceps_long", np.asarray((
            upper((-0.009, -0.012, -0.020)),
            upper((-0.100, -0.050, -0.110)),
            forearm((-0.060, -0.045, -0.055)),
            forearm((-0.010, -0.015, -0.120)),
        )), 0.011),
        SchematicMusclePath("biceps_short", np.asarray((
            upper((-0.012, 0.000, -0.065)),
            upper((-0.065, -0.005, -0.140)),
            forearm((-0.042, -0.003, -0.040)),
            forearm((-0.010, -0.005, -0.090)),
        )), 0.0095),
        SchematicMusclePath("brachialis", np.asarray((
            upper((-0.010, 0.009, -0.130)),
            upper((-0.036, 0.036, -0.205)),
            forearm((-0.026, 0.026, -0.025)),
            brachialis_insertion,
        )), 0.008),
        SchematicMusclePath("triceps", np.asarray((
            upper((0.012, 0.007, -0.040)),
            upper((0.066, 0.032, -0.140)),
            upper((0.062, 0.030, -0.305)),
            forearm((0.016, 0.014, -0.022)),
        )), 0.011),
    )
    return ArmAnchors(
        shoulder, elbow, wrist, upper_rotation, forearm_rotation,
        forearm_origin, paths,
    )


def sample_muscle_path(path: SchematicMusclePath, parameters: np.ndarray) -> np.ndarray:
    """Sample an illustrative curve at parameters in [0, 1]."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Curve parameters must be a finite one-dimensional array in [0, 1].")
    t = values[:, None]
    p0, p1, p2, p3 = path.control_points
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def configure_camera(model: mujoco.MjModel, camera: mujoco.MjvCamera) -> None:
    """Set a close, slightly oblique side view covering the full V0 curl."""
    reference = schematic_anchors(model, np.deg2rad(75.0))
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = reference.elbow + np.array((-0.060, 0.0, 0.055))
    camera.distance = 0.85
    camera.azimuth = 105.0
    camera.elevation = -12.0


def configure_options(options: mujoco.MjvOption) -> None:
    """Hide the original V0 shapes/sites and decorative debug overlays."""
    options.geomgroup[0] = 0
    options.sitegroup[:] = 0
    options.label = mujoco.mjtLabel.mjLABEL_NONE
    options.frame = mujoco.mjtFrame.mjFRAME_NONE
    for name in ("mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_COM", "mjVIS_CONTACTPOINT",
                 "mjVIS_CONTACTFORCE", "mjVIS_CONSTRAINT", "mjVIS_TENDON"):
        flag = getattr(mujoco.mjtVisFlag, name, None)
        if flag is not None:
            options.flags[flag] = 0


@dataclass(frozen=True)
class _Shape:
    kind: int
    first: np.ndarray
    second: np.ndarray | None
    radius: float
    rgba: tuple[float, ...] | np.ndarray


def _arm_shapes(anchors: ArmAnchors, activations: dict[str, float]) -> list[_Shape]:
    shapes: list[_Shape] = []

    def sphere(point, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_SPHERE, point, None, radius, color))

    def capsule(start, end, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_CAPSULE, start, end, radius, color))

    def cylinder(start, end, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_CYLINDER, start, end, radius, color))

    def spindle(start, end, radius, color):
        shapes.append(_Shape(mujoco.mjtGeom.mjGEOM_ELLIPSOID, start, end, radius, color))

    shoulder, elbow, wrist = anchors.shoulder, anchors.elbow, anchors.wrist
    upper_axis = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
    hinge_axis = anchors.upper_rotation[:, 1]
    forearm_axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    capsule(shoulder + 0.015 * upper_axis, elbow - 0.018 * upper_axis, 0.012, _BONE)
    sphere(shoulder, 0.019, _BONE_END)
    sphere(elbow - 0.023 * upper_axis, 0.017, _BONE_END)

    # Radius and ulna are two separate thin shafts, rather than a solid arm tube.
    for name in _FOREARM_BONES:
        start, end, radius = forearm_bone_axis(name, float(np.linalg.norm(wrist - elbow)))
        proximal = elbow + anchors.forearm_rotation @ start
        distal = elbow + anchors.forearm_rotation @ end
        capsule(proximal, distal, radius, _BONE)
        sphere(proximal, 0.0105, _BONE_END)
        sphere(distal, 0.009, _BONE_END)
    capsule(wrist - 0.035 * forearm_axis, wrist, 0.010, _BONE_END)

    # A visible mechanical hinge identifies the single simulated degree of freedom.
    cylinder(elbow - 0.027 * hinge_axis, elbow + 0.027 * hinge_axis, 0.025, _JOINT)
    for side in (-1, 1):
        face = elbow + side * 0.028 * hinge_axis
        cylinder(face, face + side * 0.003 * hinge_axis, 0.020, _JOINT_FACE)
        sphere(face + side * 0.004 * hinge_axis, 0.0065, _JOINT)

    values = np.linspace(0.0, 1.0, _PATH_SEGMENTS + 1)
    for path in anchors.muscles:
        color = activation_rgba(activations[path.name])
        points = sample_muscle_path(path, values)
        for index in range(_PATH_SEGMENTS):
            middle = 0.5 * (values[index] + values[index + 1])
            if not _BELLY_START <= middle <= _BELLY_END:
                capsule(points[index], points[index + 1], 0.0028, _TENDON)
        # One smooth spindle avoids a beaded chain of overlapping capsules.
        # Its endpoints coincide with the two sampled tendon sections.
        ends = sample_muscle_path(path, np.array((_BELLY_START, _BELLY_END)))
        spindle(ends[0], ends[1], path.belly_radius + 0.0025, color)
        sphere(points[0], 0.0045, _ATTACHMENT)
        sphere(points[-1], 0.0045, _ATTACHMENT)

    # A compact dumbbell symbol marks the payload; it does not change its inertia.
    weight_axis = anchors.forearm_rotation[:, 1]
    cylinder(wrist - 0.065 * weight_axis, wrist + 0.065 * weight_axis, 0.007, _HANDLE)
    for side in (-1, 1):
        center = wrist + side * 0.047 * weight_axis
        cylinder(center - 0.012 * weight_axis, center + 0.012 * weight_axis, 0.036, _WEIGHT)
        cylinder(center + side * 0.012 * weight_axis,
                 center + side * 0.014 * weight_axis, 0.012, _HANDLE)
    return shapes


def append_arm_geometry(scene: mujoco.MjvScene, env: PreacherCurlEnv) -> int:
    """Append the schematic and return its geom count, preserving earlier geoms.

    Call after the scene's ordinary update, once per displayed frame. A caller
    owns scene clearing/lifetime. Insufficient capacity raises before appending
    anything, avoiding a misleading partial muscle drawing.
    """
    anchors = schematic_anchors(env.model, float(env.data.qpos[env.qpos_index]))
    activations = {muscle.params.name: muscle.activation for muscle in env.muscles}
    shapes = _arm_shapes(anchors, activations)
    initial = scene.ngeom
    if initial + len(shapes) > scene.maxgeom:
        raise ValueError(f"Arm display needs {len(shapes)} free geoms; scene has {scene.maxgeom - initial}.")
    identity = np.eye(3, dtype=np.float64).ravel()
    for offset, shape in enumerate(shapes):
        geom = scene.geoms[initial + offset]
        size = np.full(3, shape.radius, dtype=np.float64)
        position = np.asarray(shape.first, dtype=np.float64)
        rotation = identity
        if shape.kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            direction = shape.second - shape.first
            size[2] = 0.5 * np.linalg.norm(direction)
            position = 0.5 * (shape.first + shape.second)
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_quatZ2Vec(quaternion, direction)
            rotation = _quaternion_matrix(quaternion).ravel()
        mujoco.mjv_initGeom(
            geom, shape.kind, size, position, rotation,
            np.asarray(shape.rgba, dtype=np.float32),
        )
        if shape.second is not None and shape.kind != mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            mujoco.mjv_connector(geom, shape.kind, shape.radius, shape.first, shape.second)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        geom.objid = -1
        geom.segid = -1
        geom.specular = 0.15
        geom.shininess = 0.25
    scene.ngeom = initial + len(shapes)
    return len(shapes)
